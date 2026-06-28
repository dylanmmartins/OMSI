# -*- coding: utf-8 -*-
"""
OMSI/deconv.py

Entry points and Ray-parallel dispatch for MCMC spike deconvolution.

Functions
---------
_compute_otsu_threshold
    Scans every split point and picks the threshold maximizing between-class variance.
_process_cell
    Ray remote task: runs MCMC sampler on one cell and returns results dict.
deconv
    Dispatches one _process_cell task per cell and assembles output arrays.
_compute_dff
    Computes dF/F with neuropil subtraction and near-zero baseline clamping.
_spikes_to_train
    Converts per-cell spike time arrays to a (n_cells, n_frames) uint8 spike train.
_spikes_to_padded
    Packs per-cell spike time arrays into a NaN-padded (n_cells, max_spikes) array.
_save_results
    Writes spike_inference{tag}.npz and optionally a .mat file.
deconv_from_array
    Entry point for numpy array inputs.
deconv_from_suite2p
    Entry point for suite2p output directories.
deconv_from_caiman
    Entry point for CaImAn HDF5 files.
_build_parser
    Builds the argparse CLI parser.
main
    CLI entry point, dispatches to the appropriate deconv_from_* function.


DMM, Feb 2026
"""


import argparse
import glob
import logging
import os
import sys
import textwrap
import h5py
import numpy as np
import ray
from tqdm import tqdm
from scipy.signal import find_peaks
import time
from scipy.ndimage import gaussian_filter1d
import numba

from . import helpers
from .sampler import cont_ca_sampler
from .make_mean_sample import make_mean_sample


@numba.jit(nopython=True, cache=True)
def _compute_otsu_threshold(data):
    """Otsu threshold via exhaustive between-class variance scan.

    Sorts data, then walks every possible split point and tracks the split
    maximizing weighted between-class variance. Used to separate noise peaks
    from real spike peaks in the probability trace.

    Parameters
    ----------
    data : np.ndarray
        1-D array of peak heights.

    Returns
    -------
    float
        Threshold midpoint between the best split pair.
    """

    n_orig = len(data)
    sorted_data = np.zeros(n_orig + 1, dtype=data.dtype)
    sorted_data[:n_orig] = data

    sorted_data.sort()
    n = len(sorted_data)

    cum_sum = np.cumsum(sorted_data)
    total_sum = cum_sum[-1]

    max_var = -1.0
    best_idx = 0

    for i in range(n - 1):
        w0 = (i + 1) / n
        w1 = 1.0 - w0

        mu0 = cum_sum[i] / (i + 1)
        mu1 = (total_sum - cum_sum[i]) / (n - (i + 1))

        var_between = w0 * w1 * (mu0 - mu1)**2

        if var_between > max_var:
            max_var = var_between
            best_idx = i

    threshold = (sorted_data[best_idx] + sorted_data[best_idx+1]) / 2.0

    return threshold


# max_calls=1 tells Ray to kill and restart the worker after each cell,
# preventing memory from accumulating across cells in long sessions.
@ray.remote(max_calls=1)
def _process_cell(Y_cell, cell_idx, params, true_spikes_cell, fs, n_frames, lag_s=0.0):
    """Ray remote task: run MCMC sampler on one cell and return results dict.

    Runs cont_ca_sampler, builds the spike probability trace from posterior
    samples, calls spike selection (last/MAP/prob methods), and optionally
    computes precision/recall/F1 if ground truth is provided.

    Parameters
    ----------
    Y_cell : np.ndarray
        Fluorescence trace for one cell, shape (n_frames,).
    cell_idx : int
        Cell index, used to sort results after parallel collection.
    params : dict
        Sampler parameters forwarded to cont_ca_sampler.
    true_spikes_cell : np.ndarray or None
        Ground-truth spike times in seconds for this cell, or None.
    fs : float
        Frame rate in Hz.
    n_frames : int
        Number of frames in the recording.
    lag_s : float
        Indicator rise-time lag in seconds to subtract from inferred spike times.

    Returns
    -------
    dict
        Keys: cell_idx, calcium, prob, spikes, precision, recall, F1,
        final_tau, sn_mad, final_sg, n_samples, time.
    """

    t0 = time.time()
    SAMPLES = cont_ca_sampler(Y_cell, params)
    time_taken = time.time() - t0

    final_tau = SAMPLES['g'][-1]
    sn_mad    = SAMPLES.get('sn_mad', 0.0)
    final_sg  = float(np.mean(np.sqrt(SAMPLES['sn2']))) if 'sn2' in SAMPLES else 0.0

    calcium = make_mean_sample(SAMPLES, Y_cell)

    ss = SAMPLES['ss']

    # Count how often each frame had a spike across all posterior samples,
    # then normalize by sample count to get per-frame spike probability.
    prob_trace  = np.zeros(n_frames, dtype=np.float32)
    for sp_times in ss:
        if len(sp_times) > 0:
            # sp_times are in continuous frame units (dt=1). A spike at position
            # 100.7 belongs to frame 100.
            idx = np.clip(sp_times.astype(int), 0, n_frames - 1)
            np.add.at(prob_trace, idx, 1)
    prob_trace /= max(1, len(ss))

    lag_frames = lag_s * fs
    spike_method = params.get('spike_method', 'map') if params else 'map'

    if spike_method == 'last' and len(ss) > 0:
        # Return the final posterior sample directly -- mirrors CaImAn's
        # cont_ca_sampler which outputs samples{end} without any post-hoc
        # scoring. After burn-in the chain draws from the posterior;
        # taking the last sample avoids MAP's double-penalisation of spike
        # density (sparse prior deflates expected_n, then MAP penalises
        # anything above that already-deflated expectation).
        sp_best    = np.asarray(ss[-1], dtype=np.float64)
        spikes_sec = np.clip(sp_best - lag_frames, 0, n_frames - 1) / fs

    elif spike_method == 'map' and len(ss) > 0:
        # MAP spike calling: select the single posterior sample whose spike
        # times best agree with the full posterior consensus.
        expected_n  = float(np.sum(prob_trace))
        best_i      = 0
        best_score  = -np.inf
        for i, sp_times in enumerate(ss):
            idxs  = (np.clip(sp_times.astype(int), 0, n_frames - 1)
                     if len(sp_times) > 0 else np.array([], dtype=int))
            score = (float(np.sum(prob_trace[idxs]))
                     - 0.5 * max(0.0, len(sp_times) - expected_n))
            if score > best_score:
                best_score = score
                best_i     = i

        sp_best    = np.asarray(ss[best_i], dtype=np.float64)
        spikes_sec = np.clip(sp_best - lag_frames, 0, n_frames - 1) / fs

    else:
        # Prob spike calling: smooth the probability trace and apply Otsu
        # thresholding to find peaks. Kept as a fallback/alternative.
        prob_smooth = gaussian_filter1d(prob_trace, sigma=max(1.5, 0.020 * fs))

        peaks, properties = find_peaks(prob_smooth, height=0)
        peak_heights = (properties['peak_heights']
                        if 'peak_heights' in properties else np.array([]))

        MIN_THRESH = 0.001
        if len(peak_heights) >= 2 and peak_heights.max() > 1e-6:
            otsu_thresh = _compute_otsu_threshold(peak_heights)
            noise_peaks = peak_heights[peak_heights < otsu_thresh]
            if len(noise_peaks) >= 2:
                prob_thresh = noise_peaks.mean() - 5.0 * noise_peaks.std()
            else:
                prob_thresh = otsu_thresh
            prob_thresh = max(MIN_THRESH, prob_thresh)
        else:
            prob_thresh = MIN_THRESH

        min_dist_frames = max(1, int(0.1 * fs))
        spikes_frames, _ = find_peaks(
            prob_smooth, height=prob_thresh, distance=min_dist_frames
        )
        spikes_sec = np.clip(spikes_frames - lag_frames, 0, n_frames - 1) / fs

    prec = rec = f1 = 0.0
    if true_spikes_cell is not None:
        p_arr, r_arr, f_arr = helpers.compute_accuracy_strict(
            [true_spikes_cell], [spikes_sec]
        )
        prec, rec, f1 = float(p_arr[0]), float(r_arr[0]), float(f_arr[0])

    return {
        'cell_idx':  cell_idx,
        'calcium':   calcium,
        'prob':      prob_trace,
        'spikes':    spikes_sec,
        'precision': prec,
        'recall':    rec,
        'F1':        f1,
        'final_tau': final_tau,
        'sn_mad':    sn_mad,
        'final_sg':  final_sg,
        'n_samples': len(ss),
        'time':      time_taken,
    }


def deconv(Y, params=None, true_spikes=None, benchmark=False, lag_s=None):
    """Initialize Ray, dispatch one _process_cell task per cell, collect results.

    Assembles output arrays from per-cell result dicts. Returns a simple dict
    when benchmark=False, or a full benchmark dict with per-cell accuracy
    metrics when benchmark=True.

    Parameters
    ----------
    Y : np.ndarray
        Fluorescence array, shape (n_cells, n_frames).
    params : dict, optional
        Sampler parameters. Must include 'f' (frame rate in Hz).
    true_spikes : list of np.ndarray, optional
        Per-cell ground-truth spike times in seconds. Required for benchmark mode.
    benchmark : bool
        If True, return per-cell precision/recall/F1 alongside calcium and spikes.
    lag_s : float, optional
        Indicator rise-time lag in seconds. Derived from params['defg'] if omitted,
        otherwise defaults to 45 ms.

    Returns
    -------
    dict
        benchmark=False: Ca_trace, prob_trace, spikes, spike_train.
        benchmark=True: optim_F1, optim_precision, optim_recall, optim_calcium,
        optim_prob, optim_spikes, optim_nsamples, optim_times_per_cell.
    """

    if ray.is_initialized():
        ray.shutdown()

    from OMSI._config import get_path
    ray_dir = get_path(
        key='ray_dir',
        prompt='Select a directory for Ray temporary files.\n'
               'A fast local drive (e.g. SSD scratch space) is recommended.',
    )

    # runtime_env.env_vars only reaches task/actor workers, not the
    # raylet/dashboard-agent/driver core-worker processes ray.init() spawns
    # itself -- set these in the actual process env so the dashboard-less
    # setup doesn't log connection errors on every run.
    os.environ.setdefault('GLOG_minloglevel', '3')
    os.environ.setdefault('RAY_enable_metrics_collection', '0')

    ray.init(
        _temp_dir=ray_dir,
        ignore_reinit_error=False,
        include_dashboard=False,
        log_to_driver=False,
        logging_level=logging.FATAL,
        runtime_env={
            "env_vars": {
                # Pin each worker to a single thread so Ray processes don't
                # fight each other for CPU cores (Ray parallelizes at the
                # process level).
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_enable_metrics_collection": "0",
                # Suppress C++ glog messages from gcs_server, raylet, core_worker.
                "GLOG_minloglevel": "3",
            }
        },
        _metrics_export_port=0,
    )

    Y = np.atleast_2d(Y)
    n_cells, n_frames = Y.shape

    fs = params['f'] if params and 'f' in params else 1.0

    # Derive indicator lag from the rise time constant if available,
    # otherwise fall back to 45 ms -- a reasonable default for GCaMP6.
    if lag_s is None:
        defg = params.get('defg', []) if params else []
        if len(defg) > 0 and 0.0 < defg[0] < 1.0:
            import math
            lag_s = -1.0 / (fs * math.log(defg[0]))
        else:
            lag_s = 0.045

    futures = []
    for i in range(n_cells):
        p_copy   = params.copy() if params else {}

        if isinstance(p_copy.get('init'), list):
            p_copy['init'] = p_copy['init'][i]

        if 'auto_stop' not in p_copy:
            p_copy['auto_stop'] = True

        ts_cell  = true_spikes[i] if true_spikes is not None else None
        futures.append(
            _process_cell.remote(Y[i].copy(), i, p_copy, ts_cell, fs, n_frames, lag_s)
        )

    results_list = []
    pending = futures

    del futures

    with tqdm(total=len(pending), desc="Processing cells") as pbar:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            results_list.append(ray.get(ready[0]))
            pbar.update(1)

    results_list.sort(key=lambda r: r['cell_idx'])

    tradpy_calcium    = np.zeros((n_cells, n_frames), dtype=np.float32)
    tradpy_prob       = np.zeros((n_cells, n_frames), dtype=np.float32)
    tradpy_spikes     = []
    tradpy_F1         = np.zeros(n_cells)
    tradpy_precision  = np.zeros(n_cells)
    tradpy_recall     = np.zeros(n_cells)
    tradpy_nsamples   = np.zeros(n_cells, dtype=int)
    tradpy_times_per_cell = np.zeros(n_cells)

    for res in results_list:
        i = res['cell_idx']
        tradpy_calcium[i]   = res['calcium']
        tradpy_prob[i]      = res['prob']
        tradpy_spikes.append(res['spikes'])
        tradpy_F1[i]        = res['F1']
        tradpy_precision[i] = res['precision']
        tradpy_recall[i]    = res['recall']
        tradpy_nsamples[i]  = res['n_samples']
        tradpy_times_per_cell[i] = res['time']

    spikes_obj = np.empty(len(tradpy_spikes), dtype=object)
    for _i, _sp in enumerate(tradpy_spikes):
        spikes_obj[_i] = np.asarray(_sp, dtype=float)

    if benchmark is False:
        return {
            'Ca_trace': tradpy_calcium,
            'prob_trace': tradpy_prob,
            'spikes': spikes_obj,
            'spike_train': _spikes_to_train(res['spikes'], n_frames, fs)
        }
    elif benchmark is True:
        return {
            'optim_F1':        tradpy_F1        if true_spikes is not None else None,
            'optim_precision': tradpy_precision  if true_spikes is not None else None,
            'optim_recall':    tradpy_recall     if true_spikes is not None else None,
            'optim_calcium':   tradpy_calcium,
            'optim_prob':      tradpy_prob,
            'optim_spikes':    spikes_obj,
            'optim_nsamples':  tradpy_nsamples,
            'optim_times_per_cell': tradpy_times_per_cell,
        }


def _compute_dff(f, fneu=None, f_corr=0.7, baseline_pct=8):
    """Compute dF/F with neuropil subtraction and near-zero baseline clamping.

    Subtracts f_corr * fneu from f, computes baseline as the baseline_pct
    percentile, then clamps near-zero baselines to 1 to avoid divide-by-zero.
    Uses abs in the denominator so dF/F stays positive when baseline goes
    negative.

    Parameters
    ----------
    f : np.ndarray
        Raw fluorescence, shape (n_cells, n_frames).
    fneu : np.ndarray, optional
        Neuropil fluorescence, same shape as f. Skipped if None.
    f_corr : float
        Neuropil correction coefficient.
    baseline_pct : float
        Percentile of the corrected trace used as the dF/F baseline.

    Returns
    -------
    np.ndarray
        dF/F array, same shape as f.
    """

    f_corr_traces = f - f_corr * fneu if fneu is not None else f.copy()
    baseline = np.percentile(f_corr_traces, baseline_pct, axis=1, keepdims=True)

    # Near-zero baseline causes division to blow up -- clamp to 1.
    baseline = np.where(np.abs(baseline) < 1.0, 1.0, baseline)

    return (f_corr_traces - baseline) / np.abs(baseline)


def _spikes_to_train(spike_times_list, n_frames, hz):
    """Convert per-cell spike time arrays to a (n_cells, n_frames) uint8 spike train.

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        Per-cell spike times in seconds.
    n_frames : int
        Number of frames in the recording.
    hz : float
        Frame rate in Hz.

    Returns
    -------
    np.ndarray
        Binary spike train, shape (n_cells, n_frames), dtype uint8.
    """

    n_cells = len(spike_times_list)
    train = np.zeros((n_cells, n_frames), dtype=np.uint8)

    for i, sp in enumerate(spike_times_list):
        sp = np.asarray(sp, dtype=float)
        sp = sp[np.isfinite(sp)]
        if len(sp) == 0:
            continue
        frames = np.clip(np.round(sp * hz).astype(int), 0, n_frames - 1)
        np.add.at(train[i], frames, 1)

    return train


def _spikes_to_padded(spike_times_list):
    """Pack per-cell spike time arrays into a NaN-padded (n_cells, max_spikes) array.

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        Per-cell spike times in seconds.

    Returns
    -------
    np.ndarray
        Float64 array of shape (n_cells, max_spikes), NaN-padded.
    """

    n_cells = len(spike_times_list)
    lengths = [len(np.asarray(sp)) for sp in spike_times_list]
    max_n   = max(lengths) if lengths else 0
    out = np.full((n_cells, max(max_n, 1)), np.nan, dtype=np.float64)

    for i, sp in enumerate(spike_times_list):
        sp = np.asarray(sp, dtype=np.float64)
        out[i, :len(sp)] = sp

    return out


def _save_results(results, dFF, hz, outdir, tag='', save_mat=False):
    """Write spike_inference{tag}.npz and optionally a .mat file.

    Saved arrays
    ------------
    dFF          : (n_cells, n_frames) dF/F input used for inference.
    Ca_trace     : (n_cells, n_frames) MCMC-reconstructed calcium signal.
    prob_trace   : (n_cells, n_frames) per-frame spike-probability trace.
    spike_train  : (n_cells, n_frames) uint8 binary spike train.
    spike_times  : (n_cells, max_spikes) spike times in seconds, NaN-padded.
    n_spikes     : (n_cells,) number of detected spikes per cell.
    hz           : scalar frame rate used during inference.

    Parameters
    ----------
    results : dict
        Output of deconv -- must contain Ca_trace, prob_trace, spike_train, spikes.
    dFF : np.ndarray
        dF/F array passed to deconv, saved alongside inference outputs.
    hz : float
        Frame rate in Hz.
    outdir : str
        Directory to write output files.
    tag : str
        String appended to the output filename stem.
    save_mat : bool
        If True, also write a MATLAB-compatible .mat file.

    Returns
    -------
    str
        Absolute path to the saved .npz file.
    """

    os.makedirs(outdir, exist_ok=True)

    n_cells, n_frames = results['Ca_trace'].shape
    spike_times_list  = list(results['spikes'])

    spike_times  = _spikes_to_padded(spike_times_list)
    n_spikes     = np.array([len(np.asarray(sp)) for sp in spike_times_list],
                            dtype=np.int32)

    save_dict = dict(
        dFF         = dFF.astype(np.float32),
        Ca_trace    = results['Ca_trace'].astype(np.float32),
        prob_trace  = results['prob_trace'].astype(np.float32),
        spike_train = results['spike_train'].astype(np.uint8),
        spike_times = spike_times,
        n_spikes    = n_spikes,
        hz          = np.float64(hz),
    )

    npz_path = os.path.join(outdir, 'spike_inference{}.npz'.format(tag))
    np.savez_compressed(npz_path, **save_dict)
    print('Results saved to {}.'.format(npz_path))

    if save_mat:
        import scipy.io
        spike_cell = np.empty((n_cells, 1), dtype=object)
        for i, sp in enumerate(spike_times_list):
            spike_cell[i, 0] = np.asarray(sp, dtype=np.float64).reshape(1, -1)

        mat_dict = {k: v for k, v in save_dict.items()
                    if k != 'spike_times'}
        mat_dict['spike_times'] = spike_cell

        mat_path = os.path.join(outdir, 'spike_inference{}.mat'.format(tag))
        scipy.io.savemat(mat_path, mat_dict)
        print('Results saved to {}.'.format(mat_path))

    return npz_path


def deconv_from_array(dff=None, f=None, fneu=None, hz=0, f_corr=0.7,
                      outdir=None, tag='', save_mat=False, params=None):
    """Entry point for numpy array inputs.

    Computes dF/F if not provided, runs deconv, and optionally saves results.

    Parameters
    ----------
    dff : np.ndarray, optional
        Pre-computed dF/F array, shape (n_cells, n_frames). If None, computed
        from f and fneu.
    f : np.ndarray, optional
        Raw fluorescence array. Used only when dff is None.
    fneu : np.ndarray, optional
        Neuropil fluorescence array. Used only when dff is None.
    hz : float
        Imaging frame rate in Hz. Must be positive.
    f_corr : float
        Neuropil correction coefficient passed to _compute_dff.
    outdir : str, optional
        Directory to write result files. If None, results are not saved.
    tag : str
        Tag appended to output filename stem.
    save_mat : bool
        If True, also save a MATLAB-compatible .mat file.
    params : dict, optional
        Sampler parameters forwarded to deconv.

    Returns
    -------
    dict
        Output of deconv -- Ca_trace, prob_trace, spikes, spike_train.
    """

    if hz <= 0:
        raise ValueError('hz must be a positive frame rate in Hz.')

    if dff is None:
        if f is None:
            raise ValueError('Provide either dFF or f (raw fluorescence).')
        dff = _compute_dff(f, fneu=fneu, f_corr=f_corr)

    dff = np.atleast_2d(dff).astype(np.float32)
    run_params = dict(params) if params else {}
    run_params['f'] = float(hz)

    print('Running deconvolution on {} cells x {} frames at {} Hz...'.format(
        dff.shape[0], dff.shape[1], hz))
    results = deconv(dff, params=run_params)

    if outdir is not None:
        _save_results(results, dff, hz, outdir, tag=tag, save_mat=save_mat)

    return results


def deconv_from_suite2p(datadir, hz=None, f_corr=0.7, planes=None,
                        cells_only=True, outdir=None, save_mat=False, params=None):
    """Entry point for suite2p output directories.

    Loads F.npy, Fneu.npy, iscell.npy, and ops.npy per plane, then calls
    deconv_from_array for each plane.

    Parameters
    ----------
    datadir : str
        Path to the data directory. Searches for suite2p/plane*/ subdirectories,
        then plane*/ directly, then F.npy in datadir itself.
    hz : float, optional
        Imaging frame rate in Hz. Read from ops.npy when omitted.
    f_corr : float
        Neuropil correction coefficient.
    planes : list of int, optional
        Plane indices to process. Defaults to all planes found.
    cells_only : bool
        If True, keep only ROIs classified as cells by suite2p (iscell[:,0]==1).
    outdir : str, optional
        Directory to write result files. Defaults to each plane subdirectory.
    save_mat : bool
        If True, also save a MATLAB-compatible .mat file per plane.
    params : dict, optional
        Sampler parameters forwarded to deconv_from_array.

    Returns
    -------
    dict
        Mapping of plane name to deconv_from_array output dict.
    """

    suite2p_root = os.path.join(datadir, 'suite2p')
    search_root = suite2p_root if os.path.isdir(suite2p_root) else datadir

    plane_dirs = sorted(glob.glob(os.path.join(search_root, 'plane*')))

    if not plane_dirs:
        if os.path.isfile(os.path.join(datadir, 'F.npy')):
            plane_dirs = [datadir]
        else:
            raise FileNotFoundError(
                '[fMCSI] No suite2p plane directories found under {}.\n'.format(datadir) +
                'Expected: <datadir>/suite2p/plane*/ or <datadir>/plane*/ '
                'or F.npy directly in <datadir>.'
            )

    if planes is not None:
        plane_dirs = [p for p in plane_dirs
                      if any(os.path.basename(p) == 'plane{}'.format(i) for i in planes)]
        if not plane_dirs:
            raise FileNotFoundError(
                '[fMCSI] No plane directories match --plane {} under {}.'.format(
                    planes, search_root)
            )

    all_results = {}

    for plane_dir in plane_dirs:
        plane_name = os.path.basename(plane_dir)
        print('\n{} ({})'.format(plane_name, plane_dir))

        f_path      = os.path.join(plane_dir, 'F.npy')
        fneu_path   = os.path.join(plane_dir, 'Fneu.npy')
        iscell_path = os.path.join(plane_dir, 'iscell.npy')
        ops_path    = os.path.join(plane_dir, 'ops.npy')

        for req in [f_path, fneu_path]:
            if not os.path.isfile(req):
                raise FileNotFoundError('[fMCSI] Required file not found: {}'.format(req))

        F    = np.load(f_path,    allow_pickle=True).astype(np.float32)
        Fneu = np.load(fneu_path, allow_pickle=True).astype(np.float32)

        # iscell[:,0]: binary cell/not-cell classification from suite2p.
        # Column 1 is classifier probability -- not needed here.
        if cells_only and os.path.isfile(iscell_path):
            iscell = np.load(iscell_path, allow_pickle=True)
            cell_mask = iscell[:, 0].astype(bool)
            F    = F[cell_mask]
            Fneu = Fneu[cell_mask]
            print('  Cells: {} / {} ROIs'.format(cell_mask.sum(), len(cell_mask)))
        else:
            if cells_only and not os.path.isfile(iscell_path):
                print('  iscell.npy not found -- using all {} ROIs'.format(F.shape[0]))
            else:
                print('  Using all {} ROIs (--all-rois)'.format(F.shape[0]))

        if hz and hz > 0:
            fs = float(hz)
        elif os.path.isfile(ops_path):
            ops = np.load(ops_path, allow_pickle=True).item()
            fs = float(ops.get('fs', ops.get('fs2', 0)))
            if fs <= 0:
                raise ValueError(
                    'Frame rate read from ops.npy is {}. '
                    'Pass --hz explicitly.'.format(fs)
                )
        else:
            raise ValueError(
                'ops.npy not found and --hz not provided. '
                'Cannot determine the frame rate.'
            )

        print('  Frame rate: {} Hz  |  shape: {}'.format(fs, F.shape))

        tag   = '_{}'.format(plane_name) if len(plane_dirs) > 1 else ''
        out_d = outdir if outdir else plane_dir

        results = deconv_from_array(
            f=F, fneu=Fneu, hz=fs, f_corr=f_corr,
            outdir=out_d, tag=tag, save_mat=save_mat, params=params,
        )
        all_results[plane_name] = results

    return all_results


def deconv_from_caiman(datadir, hz=None, outdir=None, save_mat=False, params=None):
    """Entry point for CaImAn HDF5 files.

    Reads F_dff directly if present, otherwise computes dF/F from the raw
    component traces C. Calls deconv_from_array with the result.

    Parameters
    ----------
    datadir : str
        Directory containing a CaImAn .hdf5 or .h5 result file.
    hz : float, optional
        Imaging frame rate in Hz. Read from HDF5 metadata when omitted.
    outdir : str, optional
        Directory to write result files. Defaults to datadir.
    save_mat : bool
        If True, also save a MATLAB-compatible .mat file.
    params : dict, optional
        Sampler parameters forwarded to deconv_from_array.

    Returns
    -------
    dict
        Output of deconv_from_array -- Ca_trace, prob_trace, spikes, spike_train.
    """

    candidates = (
        glob.glob(os.path.join(datadir, '*.hdf5')) +
        glob.glob(os.path.join(datadir, '*.h5'))
    )
    if not candidates:
        raise FileNotFoundError(
            '[fMCSI] No .hdf5 or .h5 files found in {}. '.format(datadir) +
            'CaImAn saves results via cnmf.save("path.hdf5").'
        )
    if len(candidates) > 1:
        print('Multiple HDF5 files found -- using {}.'.format(
            os.path.basename(candidates[0])))
    caiman_file = candidates[0]
    print('Loading CaImAn file: {}...'.format(caiman_file))

    dFF = None
    fs  = float(hz) if (hz and hz > 0) else None

    with h5py.File(caiman_file, 'r') as hf:

        if fs is None:
            for key_path in ['params/data/fr', 'params/init/fr', 'params/motion/fr']:
                if key_path in hf:
                    try:
                        fs = float(np.squeeze(hf[key_path][()]))
                        break
                    except (TypeError, ValueError):
                        pass
            if fs is None or fs <= 0:
                raise ValueError(
                    'Could not read frame rate from CaImAn file. '
                    'Pass --hz explicitly.'
                )

        # Prefer F_dff if CaImAn already computed it, otherwise fall back to
        # raw component traces C and compute dF/F manually.
        if 'estimates/F_dff' in hf:
            raw = hf['estimates/F_dff'][()]
            if raw is not None and np.ndim(raw) == 2 and raw.shape[0] > 0:
                dFF = raw.astype(np.float32)
                print('  Source: estimates/F_dff  shape={}'.format(dFF.shape))

        if dFF is None:
            if 'estimates/C' not in hf:
                raise KeyError(
                    'Could not find estimates/F_dff or estimates/C in the '
                    'CaImAn file. Make sure you saved a completed CNMF object.'
                )
            C = hf['estimates/C'][()].astype(np.float32)
            # 8th percentile as rough baseline estimate -- same logic as _compute_dff.
            baseline = np.percentile(C, 8, axis=1, keepdims=True)
            baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
            dFF = (C - baseline) / np.abs(baseline)
            print('  Source: estimates/C, dF/F  shape={}'.format(dFF.shape))

    print('  Frame rate: {} Hz'.format(fs))
    out_d = outdir if outdir else datadir
    results = deconv_from_array(dFF=dFF, hz=fs, outdir=out_d, save_mat=save_mat, params=params)
    return results


def _build_parser():
    """Build the argparse CLI parser with suite2p/caiman/array source flags and all options.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser for the fMCSI CLI.
    """

    parser = argparse.ArgumentParser(
        prog='fMCSI',
        description=(
            'Optimized MCMC spike deconvolution.\n\n'
            'Provide one source flag (--suite2p, --caiman, or --array) to '
            'select the input format, then supply the data directory with -dir.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # suite2p output directory (frame rate auto-read from ops.npy)
            python -m fMCSI.deconv --suite2p -dir /data/mouse1/suite2p

            # suite2p, explicit frame rate, two planes, save elsewhere
            python -m fMCSI.deconv --suite2p -dir /data/mouse1 -hz 30 --plane 0 1 --outdir /results

            # CaImAn HDF5 file
            python -m fMCSI.deconv --caiman -dir /data/mouse1 -hz 30

            # Raw numpy arrays (pass paths as positional arguments)
            python -m fMCSI.deconv --array -dir /data/mouse1 -hz 30
        """),
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        '--suite2p', action='store_true',
        help='Load data from suite2p output directory.',
    )
    src.add_argument(
        '--caiman', action='store_true',
        help='Load data from a CaImAn HDF5 result file.',
    )
    src.add_argument(
        '--array', action='store_true',
        help='Load raw fluorescence from numpy .npy files in -dir. '
             'Expects F.npy (and optionally Fneu.npy, dFF.npy).',
    )

    parser.add_argument(
        '-dir', '--datadir', type=str, required=True,
        metavar='DIR',
        help='Path to the data directory.',
    )
    parser.add_argument(
        '-hz', '--sample_rate', type=float, default=None,
        metavar='HZ',
        help='Imaging frame rate in Hz.  For --suite2p and --caiman this is '
             'read from the saved metadata when omitted.',
    )
    parser.add_argument(
        '--outdir', type=str, default=None,
        metavar='DIR',
        help='Directory to write result files.  Defaults to the input '
             'directory (or plane sub-directory for suite2p).',
    )
    parser.add_argument(
        '--mat', action='store_true', default=False,
        help='In addition to the .npz file, also save a MATLAB-compatible '
             '.mat file (requires scipy).',
    )

    s2p = parser.add_argument_group('suite2p options')
    s2p.add_argument(
        '--f-corr', type=float, default=0.7,
        metavar='COEFF',
        help='Neuropil correction coefficient: F_corr = F - COEFF * Fneu '
             '(default: 0.7).',
    )
    s2p.add_argument(
        '--plane', type=int, nargs='+', default=None,
        metavar='N',
        help='Plane index/indices to process (e.g. --plane 0 1).  '
             'Defaults to all planes found.',
    )
    s2p.add_argument(
        '--all-rois', action='store_true', default=False,
        help='Process all ROIs, including those not classified as cells by '
             'suite2p.  Default behavior keeps only cells (iscell[:,0]==1).',
    )

    return parser


def main(argv=None):
    """CLI entry point, dispatches to the appropriate deconv_from_* function.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to sys.argv[1:] when None.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.isdir(args.datadir):
        parser.error('[fMCSI] Data directory not found: {}'.format(args.datadir))

    hz       = args.sample_rate
    save_mat = args.mat

    if args.suite2p:
        deconv_from_suite2p(
            datadir=args.datadir,
            hz=hz,
            f_corr=args.f_corr,
            planes=args.plane,
            cells_only=not args.all_rois,
            outdir=args.outdir,
            save_mat=save_mat,
        )

    elif args.caiman:
        deconv_from_caiman(
            datadir=args.datadir,
            hz=hz,
            outdir=args.outdir,
            save_mat=save_mat,
        )

    elif args.array:

        dff_path  = os.path.join(args.datadir, 'dFF.npy')
        f_path    = os.path.join(args.datadir, 'F.npy')
        fneu_path = os.path.join(args.datadir, 'Fneu.npy')

        if os.path.isfile(dff_path):
            dFF  = np.load(dff_path, allow_pickle=True).astype(np.float32)
            f    = None
            fneu = None
            print('Loaded dFF.npy: {}.'.format(dFF.shape))
        elif os.path.isfile(f_path):
            dFF  = None
            f    = np.load(f_path, allow_pickle=True).astype(np.float32)
            fneu = (np.load(fneu_path, allow_pickle=True).astype(np.float32)
                    if os.path.isfile(fneu_path) else None)
            print('Loaded F.npy: {}'.format(f.shape)
                  + (', Fneu.npy: {}.'.format(fneu.shape) if fneu is not None else '.'))
        else:
            parser.error(
                '[fMCSI] --array requires F.npy or dFF.npy in {}.'.format(args.datadir)
            )

        if hz is None or hz <= 0:
            parser.error('--array requires --hz (frame rate in Hz).')

        deconv_from_array(
            dFF=dFF, f=f, fneu=fneu,
            hz=hz,
            f_corr=args.f_corr,
            outdir=args.outdir if args.outdir else args.datadir,
            save_mat=save_mat,
        )


if __name__ == '__main__':

    main()
