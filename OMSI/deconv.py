# -*- coding: utf-8 -*-
"""
Runnable modules for fast Markov chain Monte Carlo spike inference.

Written Feb 2026, DMM
"""


import argparse
import glob
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


# otsu thresholding: scans every possible split point and picks the one where
# the two resulting groups have the most separated means (weighted by group size).
# used here to separate "noise" peaks from "real" spike peaks in the prob trace
@numba.jit(nopython=True, cache=True)
def _compute_otsu_threshold(data):

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


# max_calls=1 tells ray to kill and restart the worker after each cell.
# prevents memory from accumulating across cells in long sessions
@ray.remote(max_calls=1)
def _process_cell(Y_cell, cell_idx, params, true_spikes_cell, fs, n_frames, lag_s=0.0):

    t0 = time.time()
    SAMPLES = cont_ca_sampler(Y_cell, params)
    time_taken = time.time() - t0

    final_tau = SAMPLES['g'][-1]
    sn_mad    = SAMPLES.get('sn_mad', 0.0)
    final_sg  = float(np.mean(np.sqrt(SAMPLES['sn2']))) if 'sn2' in SAMPLES else 0.0

    calcium = make_mean_sample(SAMPLES, Y_cell)

    ss = SAMPLES['ss']

    # build the spike probability trace by counting how often each frame
    # had a spike across all posterior samples, then normalize by sample count
    prob_trace  = np.zeros(n_frames, dtype=np.float32)
    for sp_times in ss:
        if len(sp_times) > 0:
            # sp_times are in continuous frame units (dt=1).  A spike at position
            # 100.7 belongs to frame 100.
            idx = np.clip(sp_times.astype(int), 0, n_frames - 1)
            np.add.at(prob_trace, idx, 1)
    prob_trace /= max(1, len(ss))

    prob_smooth = gaussian_filter1d(prob_trace, sigma=1.5)

    peaks, properties = find_peaks(prob_smooth, height=0)
    peak_heights = properties['peak_heights'] if 'peak_heights' in properties else np.array([])

    # use otsu to split peaks into noise and signal clusters.
    # then set the threshold well below the noise cluster (5 std below its mean)
    # so we only call peaks that are clearly above the noise floor
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

    # shift spike times back by the indicator lag so they line up with the actual spikes.
    # the calcium signal peaks after the spike, so we subtract that delay here
    lag_frames = lag_s * fs
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

    if ray.is_initialized():
        ray.shutdown()

    from OMSI._config import get_path
    ray_dir = get_path(
        key='ray_dir',
        prompt='Select a directory for Ray temporary files.\n'
               'A fast local drive (e.g. SSD scratch space) is recommended.',
    )

    ray.init(
        _temp_dir=ray_dir,
        ignore_reinit_error=False,
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                # pin each worker to a single thread so ray processes dont fight each other
                # for cpu cores (ray already parallelizes at the process level)
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_enable_metrics_collection": "0",
            }
        },
        _metrics_export_port=0,
    )

    Y = np.atleast_2d(Y)
    n_cells, n_frames = Y.shape

    fs = params['f'] if params and 'f' in params else 1.0
    
    # derive the indicator lag from the rise time constant if we have it,
    # otherwise fall back to 45ms which is a reasonable default for GCaMP6
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

    f_corr_traces = f - f_corr * fneu if fneu is not None else f.copy()
    baseline = np.percentile(f_corr_traces, baseline_pct, axis=1, keepdims=True)

    # if baseline is near zero the division blows up, so clamp it to 1.
    # use abs in the denominator so dff stays positive when baseline goes negative
    baseline = np.where(np.abs(baseline) < 1.0, 1.0, baseline)

    return (f_corr_traces - baseline) / np.abs(baseline)


def _spikes_to_train(spike_times_list, n_frames, hz):

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

    n_cells = len(spike_times_list)
    lengths = [len(np.asarray(sp)) for sp in spike_times_list]
    max_n   = max(lengths) if lengths else 0
    out = np.full((n_cells, max(max_n, 1)), np.nan, dtype=np.float64)

    for i, sp in enumerate(spike_times_list):
        sp = np.asarray(sp, dtype=np.float64)
        out[i, :len(sp)] = sp

    return out


def _save_results(results, dFF, hz, outdir, tag='', save_mat=False):
    """ Write to npz file.

    Saved arrays

    dFF          - (n_cells, n_frames) dF/F input used for inference
    Ca_trace     - (n_cells, n_frames) MCMC-reconstructed calcium signal
    prob_trace   - (n_cells, n_frames) per-frame spike-probability trace
    spike_train  - (n_cells, n_frames) uint8 Otsu-thresholded binary spike train
    spike_times  - (n_cells, max_spikes) spike times in seconds, NaN-padded
    n_spikes     - (n_cells,) number of detected spikes per cell
    hz           - scalar frame rate used during inference

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

    npz_path = os.path.join(outdir, f'spike_inference{tag}.npz')
    np.savez_compressed(npz_path, **save_dict)
    print(f'Results saved -> {npz_path}')

    if save_mat:
        import scipy.io
        spike_cell = np.empty((n_cells, 1), dtype=object)
        for i, sp in enumerate(spike_times_list):
            spike_cell[i, 0] = np.asarray(sp, dtype=np.float64).reshape(1, -1)

        mat_dict = {k: v for k, v in save_dict.items()
                    if k != 'spike_times'}
        mat_dict['spike_times'] = spike_cell

        mat_path = os.path.join(outdir, f'spike_inference{tag}.mat')
        scipy.io.savemat(mat_path, mat_dict)
        print(f'Results saved -> {mat_path}')

    return npz_path


def deconv_from_array(dff=None, f=None, fneu=None, hz=0, f_corr=0.7,
                      outdir=None, tag='', save_mat=False):
    if hz <= 0:
        raise ValueError('hz must be a positive frame rate in Hz.')

    if dff is None:
        if f is None:
            raise ValueError('Provide either dFF or f (raw fluorescence).')
        dff = _compute_dff(f, fneu=fneu, f_corr=f_corr)

    dff = np.atleast_2d(dff).astype(np.float32)
    params = {'f': float(hz)}

    print(f'Running deconvolution on {dff.shape[0]} cells x {dff.shape[1]} frames '
          f'at {hz} Hz...')
    results = deconv(dff, params=params)

    if outdir is not None:
        _save_results(results, dff, hz, outdir, tag=tag, save_mat=save_mat)

    return results


def deconv_from_suite2p(datadir, hz=None, f_corr=0.7, planes=None,
                        cells_only=True, outdir=None, save_mat=False):

    suite2p_root = os.path.join(datadir, 'suite2p')
    search_root = suite2p_root if os.path.isdir(suite2p_root) else datadir

    plane_dirs = sorted(glob.glob(os.path.join(search_root, 'plane*')))

    if not plane_dirs:
        if os.path.isfile(os.path.join(datadir, 'F.npy')):
            plane_dirs = [datadir]
        else:
            raise FileNotFoundError(
                f'No suite2p plane directories found under {datadir}.\n'
                'Expected: <datadir>/suite2p/plane*/ or <datadir>/plane*/ '
                'or F.npy directly in <datadir>.'
            )

    if planes is not None:
        plane_dirs = [p for p in plane_dirs
                      if any(os.path.basename(p) == f'plane{i}' for i in planes)]
        if not plane_dirs:
            raise FileNotFoundError(
                f'No plane directories match --plane {planes} under {search_root}.'
            )

    all_results = {}

    for plane_dir in plane_dirs:
        plane_name = os.path.basename(plane_dir)
        print(f'\n--- {plane_name} ({plane_dir}) ---')

        f_path      = os.path.join(plane_dir, 'F.npy')
        fneu_path   = os.path.join(plane_dir, 'Fneu.npy')
        iscell_path = os.path.join(plane_dir, 'iscell.npy')
        ops_path    = os.path.join(plane_dir, 'ops.npy')

        for req in [f_path, fneu_path]:
            if not os.path.isfile(req):
                raise FileNotFoundError(f'Required file not found: {req}')

        F    = np.load(f_path,    allow_pickle=True).astype(np.float32)
        Fneu = np.load(fneu_path, allow_pickle=True).astype(np.float32)

        # iscell[:,0] is the binary cell/not-cell classification from suite2p.
        # column 1 is the classifier probability... don't need this
        if cells_only and os.path.isfile(iscell_path):
            iscell = np.load(iscell_path, allow_pickle=True)  # (n_rois, 2)
            cell_mask = iscell[:, 0].astype(bool)
            F    = F[cell_mask]
            Fneu = Fneu[cell_mask]
            print(f'  Cells: {cell_mask.sum()} / {len(cell_mask)} ROIs')
        else:
            if cells_only and not os.path.isfile(iscell_path):
                print(f'  iscell.npy not found - using all {F.shape[0]} ROIs')
            else:
                print(f'  Using all {F.shape[0]} ROIs (--all-rois)')

        if hz and hz > 0:
            fs = float(hz)
        elif os.path.isfile(ops_path):
            ops = np.load(ops_path, allow_pickle=True).item()
            fs = float(ops.get('fs', ops.get('fs2', 0)))
            if fs <= 0:
                raise ValueError(
                    f'Frame rate read from ops.npy is {fs}. '
                    'Pass --hz explicitly.'
                )
        else:
            raise ValueError(
                'ops.npy not found and --hz not provided. '
                'Cannot determine the frame rate.'
            )

        print(f'  Frame rate: {fs} Hz  |  shape: {F.shape}')

        tag   = f'_{plane_name}' if len(plane_dirs) > 1 else ''
        out_d = outdir if outdir else plane_dir

        results = deconv_from_array(
            f=F, fneu=Fneu, hz=fs, f_corr=f_corr,
            outdir=out_d, tag=tag, save_mat=save_mat,
        )
        all_results[plane_name] = results

    return all_results


def deconv_from_caiman(datadir, hz=None, outdir=None, save_mat=False):

    candidates = (
        glob.glob(os.path.join(datadir, '*.hdf5')) +
        glob.glob(os.path.join(datadir, '*.h5'))
    )
    if not candidates:
        raise FileNotFoundError(
            f'No .hdf5 or .h5 files found in {datadir}. '
            'CaImAn saves results via cnmf.save("path.hdf5").'
        )
    if len(candidates) > 1:
        print(f'Multiple HDF5 files found; using {os.path.basename(candidates[0])}')
    caiman_file = candidates[0]
    print(f'Loading CaImAn file: {caiman_file}')

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

        # prefer F_dff if caiman already computed it, otherwise fall back to
        # the raw component traces C and compute dff manually
        if 'estimates/F_dff' in hf:
            raw = hf['estimates/F_dff'][()]
            if raw is not None and np.ndim(raw) == 2 and raw.shape[0] > 0:
                dFF = raw.astype(np.float32)
                print(f'  Source: estimates/F_dff  shape={dFF.shape}')

        if dFF is None:
            if 'estimates/C' not in hf:
                raise KeyError(
                    'Could not find estimates/F_dff or estimates/C in the '
                    'CaImAn file.  Make sure you saved a completed CNMF object.'
                )
            C = hf['estimates/C'][()].astype(np.float32)
            # 8th percentile as a rough baseline estimate (same logic as _compute_dff)
            baseline = np.percentile(C, 8, axis=1, keepdims=True)
            baseline = np.where(np.abs(baseline) < 1e-6, 1e-6, baseline)
            dFF = (C - baseline) / np.abs(baseline)
            print(f'  Source: estimates/C -> dF/F  shape={dFF.shape}')

    print(f'  Frame rate: {fs} Hz')
    out_d = outdir if outdir else datadir
    results = deconv_from_array(dFF=dFF, hz=fs, outdir=out_d, save_mat=save_mat)
    return results


def _build_parser():
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
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.isdir(args.datadir):
        parser.error(f'Data directory not found: {args.datadir}')

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
            print(f'Loaded dFF.npy: {dFF.shape}')
        elif os.path.isfile(f_path):
            dFF  = None
            f    = np.load(f_path, allow_pickle=True).astype(np.float32)
            fneu = (np.load(fneu_path, allow_pickle=True).astype(np.float32)
                    if os.path.isfile(fneu_path) else None)
            print(f'Loaded F.npy: {f.shape}'
                  + (f', Fneu.npy: {fneu.shape}' if fneu is not None else ''))
        else:
            parser.error(
                f'--array requires F.npy or dFF.npy in {args.datadir}.'
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
