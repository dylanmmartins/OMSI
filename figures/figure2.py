# -*- coding: utf-8 -*-
"""
figures/figure2.py

Scaling and sensitivity benchmarks for OMSI.

Functions
---------
_oasis_spikes_from_s
    Detect spikes from an OASIS deconvolved signal using threshold or peak-finding.
_run_cascade_inference
    Run CASCADE spike inference in a subprocess and return results.
_metrics
    Compute all accuracy metrics for predicted spikes against ground truth.
_row
    Build a result record dict from metric outputs and metadata.
_save_records
    Save a list of record dicts to a .npz file, merging with existing data.
_load_records
    Load a .npz record table and return it as a dict of arrays.
_tbl_len
    Return the number of rows in a table dict.
_tbl_filter
    Filter a table dict by matching a column value.
_tbl_sort
    Sort a table dict by a numeric column.
_records_to_tbl
    Convert a list of record dicts to a columnar table dict.
_tbl_concat
    Concatenate a list of table dicts along the row axis.
_oasis_spikes
    Run OASIS deconvolution on all cells and return spikes and calcium traces.
_trad_mcmc_from_json
    Load CaImAn MCMC records from a JSON benchmark file.
_load_external_matlab_data
    Load all precomputed external MATLAB benchmark records.
benchmark_sweeps
    Benchmark accuracy and speed as a function of MCMC sweep count.
benchmark_scalability
    Benchmark inference speed across cell counts and recording durations.
benchmark_params
    Benchmark accuracy across calcium decay time constants and frame rates.
benchmark_noise_sensitivity
    Benchmark accuracy across a range of SNR levels.
benchmark_firing_rate_sensitivity
    Benchmark accuracy as a function of per-cell firing rate.
benchmark_cascade_sample_rate
    Compare CASCADE accuracy at 7.5 Hz vs 30 Hz sampling rates.
run_test
    Run all benchmark functions and save results to data_dir.
_fbeta
    Compute F-beta score from precision and recall arrays.
_mad
    Compute the median absolute deviation, ignoring NaNs.
_fit_scaling
    Fit linear and polynomial models to timing data and compare R-squared.
_set_three_ticks_x
    Set three evenly spaced x-axis ticks based on plotted data range.
_filter_cascade_shared_x
    Restrict CASCADE rows to x-values shared with non-CASCADE methods.
_rebuild_cascade_sample_rate_data
    Recompute the CASCADE sample-rate comparison file from saved outputs.
_plot_cascade_comparison
    Plot violin comparison of CASCADE accuracy at 7.5 Hz vs 30 Hz.
_running_median_mad
    Compute running median and median absolute deviation over a sliding window.
_load_benchmark_tables
    Load and merge the scaling/sensitivity benchmark tables.
_load_noise_cells
    Load the cell-level noise sensitivity table.
plot_figure
    Load benchmark results and render figure 2 panels A and B.
_print_experiment
    Print one row per model and x value for a benchmark experiment.
print_stats
    Print the values plotted in figure 2 without rendering the figure.


To run inference
    $ python figure2.py --mode test --data-dir /path/to/results

To create figure:
    $ python figure2.py --mode plot --data-dir /path/to/results

To print the plotted values:
    $ python figure2.py --mode print --data-dir /path/to/results

To re-run only the sample-rate sweep (keeps all other results):
    $ python figure2.py --mode fs-sensitivity --no-matlab --data-dir /path/to/results

DMM, March 2026
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from oasis.functions import deconvolve

import OMSI
import OMSI.helpers as helpers
from run_pnev_MCMC import run_matlab_pnevMCMC
from simulation_helpers import generate_synthetic_data
from OMSI._win_perf import no_power_throttling

_MATLAB_PRECOMPUTED_DIR    = '/home/dylan/Fast2/spike_deconv/sweeping_benchmarks/all_other_methods'
_CASCADE_DURATION_ALT_DIR  = '/home/dylan/Fast2/spike_deconv/sweeping_benchmarks/cascade'

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig2')

mpl.rcParams['axes.spines.top']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

np.random.seed(3)

BETA = 0.5
USE_STRICT_ACCURACY = False  # Hungarian one-to-one matching (compute_accuracy_strict).

COLORS = {
    'OMSI':        '#4C72B0',
    'CaImAn MCMC':  '#DD8452',
    'OASIS':        '#55A868',
    'CASCADE_GPU':  '#8172B3',
    'CASCADE_CPU':  '#B39DDB',
}

OASIS_SPIKE_DETECTION = 'peaks'


def _oasis_spikes_from_s(s, sigma, fs, height=1.0):
    """
    Detect spikes from an OASIS deconvolved signal.

    Uses peak-finding or thresholding depending on OASIS_SPIKE_DETECTION.

    Parameters
    ----------
    s : ndarray
        Deconvolved spike signal from OASIS.
    sigma : float
        Noise standard deviation estimate for the trace.
    fs : float
        Sampling rate in Hz.
    height : float, optional
        Threshold multiplier applied to sigma (default 1.0).

    Returns
    -------
    ndarray
        Spike times in seconds.
    """
    thresh = height * sigma
    if OASIS_SPIKE_DETECTION == 'peaks':
        min_dist = max(1, int(0.05 * fs))
        peaks, _ = find_peaks(s, height=thresh, distance=min_dist)
        return peaks / fs
    return np.where(s > thresh)[0] / fs


def _run_cascade_inference(dff, fs, data_dir, prefix, device='gpu', max_cells_per_call=None):
    """
    Run CASCADE spike inference in a subprocess.

    Saves input data, calls the CASCADE subprocess, and reads back results.

    Parameters
    ----------
    dff : ndarray, shape (n_cells, n_frames)
        Delta F over F fluorescence traces.
    fs : float
        Sampling rate in Hz.
    data_dir : str
        Directory for temporary input/output files.
    prefix : str
        Filename prefix for the temporary .npz files.
    device : str, optional
        Compute device passed to CASCADE ('gpu' or 'cpu').
    max_cells_per_call : int or None, optional
        When set, forwarded to run_cascade_subprocess.py as
        --max-cells-per-call: splits the call into multiple cascade.predict()
        calls of at most this many cells each (summing their times), so a
        single call's input tensor doesn't exceed GPU memory for very large
        inputs. Default None: unchanged single-call behavior.

    Returns
    -------
    cascade_probs : ndarray
        Per-frame spike probability output from CASCADE.
    cascade_spikes : list of ndarray
        Spike times in seconds for each cell.
    cascade_time : float
        Wall-clock inference time in seconds.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
    cmd = [shutil.which('conda') or 'conda', 'run', '-n', 'cascade_gpu', 'python', script,
           '--mode', 'inference', '--input', input_path, '--output', output_path,
           '--device', device]
    if max_cells_per_call:
        cmd += ['--max-cells-per-call', str(max_cells_per_call)]
    subprocess.run(cmd, check=True)
    result = np.load(output_path, allow_pickle=True)
    cascade_probs  = result['cascade_probs']
    cascade_spikes = list(result['cascade_spikes'])
    cascade_time   = float(result['cascade_time'])
    return cascade_probs, cascade_spikes, cascade_time


class _MetricTuple(tuple):
    """ Tuple of mean metrics that also carries the per-cell values.

    Behaves exactly like the plain tuple returned previously, so existing
    unpacking is unchanged. The per-cell arrays, keyed by metric name, are in
    the per_cell attribute.
    """
    per_cell = None


def _metrics(true_spk, pred_spk, true_ev, fs_):
    """
    Compute all accuracy metrics for predicted spikes.

    Parameters
    ----------
    true_spk : list of ndarray
        Ground-truth spike times in seconds for each cell.
    pred_spk : list of ndarray
        Predicted spike times in seconds for each cell.
    true_ev : list of ndarray
        Ground-truth event times (burst-collapsed) for each cell.
    fs_ : float
        Sampling rate in Hz, used for CosMIC computation.

    Returns
    -------
    dict
        Median across cells of the strict, window, and event metrics and
        CosMIC and per-cell F-beta (keys 'F1', 'Precision', ..., 'COSMIC',
        'Fbeta', 'Fbeta_window'), plus the matching
        median absolute deviation under each key with a '_mad' suffix.
    """
    prec,   rec,   f1   = OMSI.compute_accuracy_strict(true_spk, pred_spk, tolerance=0.1)
    prec_w, rec_w, f1_w = helpers.compute_accuracy_window(true_spk, pred_spk)
    prec_e, rec_e, f1_e = helpers.compute_accuracy_window(true_ev,  pred_spk)
    cosmic = helpers.compute_cosmic(true_spk, pred_spk, fs_)
    per_cell = {
        'F1':        f1,     'Precision':        prec,   'Recall':        rec,
        'F1_window': f1_w,   'Precision_window': prec_w, 'Recall_window': rec_w,
        'F1_event':  f1_e,   'Precision_event':  prec_e, 'Recall_event':  rec_e,
        'COSMIC':    cosmic,
        'Fbeta':        _fbeta(prec, rec),
        'Fbeta_window': _fbeta(prec_w, rec_w),
    }
    out = {}
    for key, vals in per_cell.items():
        out[key]          = float(np.nanmedian(vals))
        out[key + '_mad'] = float(_mad(vals))
    return out


def _mad(x, axis=None):
    """ Compute the median absolute deviation, ignoring NaNs.

    Parameters
    ----------
    x : array-like
        Input values.
    axis : int or None, optional
        Axis along which to compute; None flattens the input.

    Returns
    -------
    float or np.ndarray
        Median of |x - median(x)| along axis.
    """
    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _row(exp, model, tau_, fs_, time_, m, sweeps=0, n_cells=None, duration=None,
         mean_kurtosis=None, **extra):
    """
    Build a result record dict from metrics and metadata.

    Parameters
    ----------
    exp : str
        Experiment name (e.g. 'Sweeps', 'Tau_Sensitivity').
    model : str
        Algorithm name (e.g. 'OMSI', 'OASIS').
    tau_ : float
        Calcium decay time constant in seconds.
    fs_ : float
        Sampling rate in Hz.
    time_ : float
        Total inference wall-clock time in seconds.
    m : dict
        Metric dict returned by _metrics.
    sweeps : int, optional
        Number of MCMC sweeps (0 for non-MCMC methods).
    n_cells : int, optional
        Number of cells processed.
    duration : float, optional
        Recording duration in seconds.
    mean_kurtosis : float, optional
        Mean kurtosis of the fluorescence traces.
    **extra
        Additional key-value pairs merged into the record.

    Returns
    -------
    dict
        Record mapping field names to scalar values.
    """
    d = {
        'Experiment': exp, 'Model': model, 'Tau': tau_, 'Fs': fs_,
        'Time': time_, 'Sweeps': sweeps,
        **m,
    }
    if n_cells is not None:
        d['N_Cells'] = n_cells
    if duration is not None:
        d['Duration'] = duration
    if mean_kurtosis is not None:
        d['Mean_Kurtosis'] = mean_kurtosis
    per_cell = getattr(m, 'per_cell', None)
    if per_cell:
        # JSON string keeps the columnar record table (float or string
        # columns only) intact. Read back with _per_cell_metrics.
        d['PerCell'] = json.dumps(
            {k: [round(float(x), 6) for x in np.asarray(v, dtype=float).ravel()]
             for k, v in per_cell.items()})
    d.update(extra)
    return d

def _save_records(records, path, experiment=None):
    """
    Save a list of record dicts to a .npz file.

    Merges new records with existing data, replacing rows for models already
    present in the new records.

    Parameters
    ----------
    records : list of dict
        Records to save.
    path : str
        Output .npz file path.
    experiment : str or None, optional
        If given, only existing rows of this experiment are replaced; rows of
        other experiments are kept even for models present in the new records.
    """
    if not records:
        if not os.path.exists(path):
            np.savez(path)
        return

    new_tbl = _records_to_tbl(records)

    if os.path.exists(path):
        try:
            existing = _load_records(path)
            if existing and 'Model' in existing and 'Model' in new_tbl:
                new_models = set(str(m) for m in new_tbl['Model'])
                mask = np.array([str(m) not in new_models for m in existing['Model']], dtype=bool)
                if experiment is not None:
                    mask |= existing['Experiment'] != experiment
                if mask.sum() > 0:
                    existing_filtered = {k: v[mask] for k, v in existing.items()}
                    combined = _tbl_concat([existing_filtered, new_tbl])
                else:
                    combined = new_tbl
            else:
                combined = new_tbl
        except Exception:
            combined = new_tbl
    else:
        combined = new_tbl

    np.savez(path, **combined)


def _load_records(path):
    """
    Load a .npz record table and return it as a dict of arrays.

    Parameters
    ----------
    path : str
        Path to a .npz file saved by _save_records.

    Returns
    -------
    dict
        Mapping of column names to numpy arrays.
    """
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _tbl_len(tbl):
    """Return the number of rows in a table dict."""
    return len(next(iter(tbl.values()))) if tbl else 0


def _tbl_filter(tbl, col, val):
    """
    Filter a table dict by matching a column value.

    Parameters
    ----------
    tbl : dict
        Columnar table as returned by _load_records.
    col : str
        Column name to filter on.
    val : scalar
        Value to match.

    Returns
    -------
    dict
        Subset of rows where tbl[col] == val.
    """
    mask = tbl[col] == val
    return {k: v[mask] for k, v in tbl.items()}


def _tbl_sort(tbl, col):
    """
    Sort a table dict by a numeric column.

    Parameters
    ----------
    tbl : dict
        Columnar table as returned by _load_records.
    col : str
        Column name to sort by.

    Returns
    -------
    dict
        Table with rows sorted in ascending order of tbl[col].
    """
    idx = np.argsort(tbl[col].astype(float))
    return {k: v[idx] for k, v in tbl.items()}


def _records_to_tbl(records):
    """
    Convert a list of record dicts to a columnar table dict.

    Parameters
    ----------
    records : list of dict
        Records to convert.

    Returns
    -------
    dict
        Mapping of column names to numpy arrays.
    """
    if not records:
        return {}
    keys = list(dict.fromkeys(k for r in records for k in r))
    out = {}
    for k in keys:
        vals = [r.get(k, None) for r in records]
        if any(isinstance(v, str) for v in vals if v is not None):
            out[k] = np.array([str(v) if v is not None else '' for v in vals], dtype=object)
        else:
            out[k] = np.array([float(v) if v is not None else np.nan for v in vals],
                               dtype=np.float64)
    return out


def _tbl_concat(tbls):
    """
    Concatenate a list of table dicts along the row axis.

    Parameters
    ----------
    tbls : list of dict
        Tables to concatenate; missing columns are filled with NaN or ''.

    Returns
    -------
    dict
        Combined table with all rows from all input tables.
    """
    tbls = [t for t in tbls if t]
    if not tbls:
        return {}
    all_keys = list(dict.fromkeys(k for t in tbls for k in t))
    result = {}
    for k in all_keys:
        parts = []
        for t in tbls:
            if k in t:
                parts.append(t[k])
            else:
                n = _tbl_len(t)
                ref = next((t2[k] for t2 in tbls if k in t2), None)
                if ref is not None and ref.dtype == object:
                    parts.append(np.array([''] * n, dtype=object))
                else:
                    parts.append(np.full(n, np.nan))
        if any(p.dtype == object for p in parts):
            result[k] = np.concatenate([p.astype(object) for p in parts])
        else:
            result[k] = np.concatenate([p.astype(np.float64) for p in parts])
    return result


def _oasis_spikes(dff, fs, tau, n_cells):
    """
    Run OASIS deconvolution on all cells.

    Parameters
    ----------
    dff : ndarray, shape (n_cells, n_frames)
        Delta F over F fluorescence traces.
    fs : float
        Sampling rate in Hz.
    tau : float
        Calcium decay time constant in seconds.
    n_cells : int
        Number of cells.

    Returns
    -------
    spikes : list of ndarray
        Spike times in seconds for each cell.
    calcium : list of ndarray
        Denoised calcium traces for each cell.
    """
    diff = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    spikes, calcium = [], []
    for i in range(n_cells):
        g = np.exp(-1 / (fs * tau))
        c, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        spikes.append(_oasis_spikes_from_s(s, sigmas[i], fs))
        calcium.append(c)
    return spikes, calcium


def _trad_mcmc_from_json(json_path):
    """
    Load CaImAn MCMC records from a JSON benchmark file.

    Parameters
    ----------
    json_path : str
        Path to a JSON file containing benchmark records.

    Returns
    -------
    list of dict
        Records with model name normalized to 'CaImAn MCMC'.
    """
    if not os.path.exists(json_path):
        print('  WARNING: precomputed file not found: {}'.format(json_path))
        return []
    with open(json_path) as f:
        data = json.load(f)
    out = []
    for r in data:
        key = 'Model' if 'Model' in r else 'model'
        if r.get(key) == 'Trad MCMC':
            r = dict(r)
            r[key] = 'CaImAn MCMC'
            out.append(r)
    return out


def _load_external_matlab_data():
    """
    Load all precomputed external MATLAB benchmark records.

    Returns
    -------
    dict
        Keys are benchmark names ('sweeps', 'scalability', 'params',
        'noise_sensitivity', 'firing_rate') mapping to lists of record dicts,
        plus paths to precomputed trace .npz files.
    """
    d = _MATLAB_PRECOMPUTED_DIR
    return {
        'sweeps':            _trad_mcmc_from_json(os.path.join(d, 'benchmark_sweeps_partial.json')),
        'scalability':       _trad_mcmc_from_json(os.path.join(d, 'benchmark_scalability_partial.json')),
        'params':            _trad_mcmc_from_json(os.path.join(d, 'benchmark_params_partial.json')),
        'noise_sensitivity': [],
        'firing_rate':       _trad_mcmc_from_json(os.path.join(d, 'firing_rate_sensitivity_partial.json')),
        'sweeps_traces_npz':      os.path.join(d, 'benchmark_sweeps_traces.npz'),
        'firing_rate_traces_npz': os.path.join(d, 'firing_rate_sensitivity_traces.npz'),
    }


def benchmark_sweeps(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True, matlab_records=None):
    """
    Benchmark accuracy and speed as a function of MCMC sweep count.

    Generates synthetic data and runs each enabled method across a range of
    sweep counts, saving results incrementally.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_mine : bool, optional
        Whether to run OMSI.
    run_cascade : bool, optional
        Whether to run CASCADE.
    matlab_records : list of dict or None, optional
        Precomputed CaImAn MCMC records to inject instead of running MATLAB.
    """
    n_cells     = 50
    duration    = 600
    fs          = 30.0
    tau         = 1.2
    sweeps_list = [10, 50, 100, 250, 500, 1000, 2000, 3000]

    print('Generating synthetic data (n_cells={}, duration={}s)...'.format(n_cells, duration))
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    n_frames   = dff.shape[1]
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

    results   = []
    npz_spikes = {'true': true_spikes}
    npz_calcium = {}

    partial_path = os.path.join(data_dir, 'benchmark_sweeps_partial.npz')

    if run_oasis:
        print('\nRunning OASIS (baseline)...')
        t0 = time.time()
        oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
        time_oasis = time.time() - t0
        results.append(_row('Sweeps', 'OASIS', tau, fs, time_oasis,
                            _metrics(true_spikes, oas_spk, true_events, fs),
                            sweeps=0, n_cells=n_cells, duration=duration,
                            Samples_per_sec=np.nan))
        npz_spikes['oasis'] = oas_spk
        npz_calcium['oasis'] = np.array(oas_cal)
        print('  OASIS: {:.1f}s  F1={:.3f} ± {:.3f}'.format(time_oasis, results[-1]['F1'], results[-1]['F1_mad']))
        _save_records(results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print('\nRunning CASCADE (subprocess, {}, baseline)...'.format(_dev.upper()))
            _, cascade_spikes, time_cascade = _run_cascade_inference(
                dff, fs, data_dir, f'bench_sweeps_cascade_baseline_{_dev}', device=_dev
            )
            results.append(_row('Sweeps', _model, tau, fs, time_cascade,
                                _metrics(true_spikes, cascade_spikes, true_events, fs),
                                sweeps=0, n_cells=n_cells, duration=duration,
                                Samples_per_sec=np.nan))
            npz_spikes[f'cascade_{_dev}'] = cascade_spikes
            print('  CASCADE ({}): {:.1f}s  F1={:.3f} ± {:.3f}'.format(_dev.upper(), time_cascade, results[-1]['F1'], results[-1]['F1_mad']))
            _save_records(results, partial_path)

    print('\n--- Varying sweeps: {} ---'.format(sweeps_list))
    for s in sweeps_list:
        print('  sweeps={}...'.format(s))
        if run_mine:
            try:
                t0 = time.time()
                burn_in = int(s * 0.25)
                params  = {'f': fs, 'p': 2, 'Nsamples': s - burn_in, 'B': burn_in, 'auto_stop': False}
                res = OMSI.deconv(dff, params=params, benchmark=True)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'OMSI', tau, fs, elapsed,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['my_method'] = res['optim_spikes']
                print('    OMSI: {:.1f}s  F1={:.3f} ± {:.3f}'.format(elapsed, results[-1]['F1'], results[-1]['F1_mad']))
            except Exception as exc:
                print('    OMSI failed: {}'.format(exc))

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps=s)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'CaImAn MCMC', tau, fs, elapsed,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['trad_mcmc'] = trad_spikes
                print('    CaImAn MCMC: {:.1f}s  F1={:.3f} ± {:.3f}'.format(elapsed, results[-1]['F1'], results[-1]['F1_mad']))
            except Exception as exc:
                print('    CaImAn MCMC failed: {}'.format(exc))

        _save_records(results, partial_path)

    if run_matlab and matlab_records is not None:
        print('\nInjecting {} precomputed CaImAn MCMC (sweeps) records...'.format(len(matlab_records)))
        results.extend(matlab_records)
        _save_records(results, partial_path)

    npz_save = {'dff': dff, 'fs': fs, 'tau': tau}
    for k, v in npz_spikes.items():
        npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
    for k, v in npz_calcium.items():
        npz_save[f'calcium_{k}'] = v
    np.savez(os.path.join(data_dir, 'benchmark_sweeps_traces.npz'), **npz_save)
    return


def benchmark_scalability(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                           run_cascade=True, matlab_records=None):
    """
    Benchmark inference speed across cell counts and recording durations.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_mine : bool, optional
        Whether to run OMSI.
    run_cascade : bool, optional
        Whether to run CASCADE.
    matlab_records : list of dict or None, optional
        Precomputed CaImAn MCMC records to inject instead of running MATLAB.
    """
    fs  = 30.0
    tau = 1.2
    cell_counts    = [50, 200, 500, 1000, 2000, 3000]
    fixed_duration = 300.0
    durations      = [300, 1800, 3600, 7200]
    fixed_cells    = 100

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_scalability_partial.npz')

    print('\n--- Cell-count scaling ---')
    for n_cells in cell_counts:
        print('  n_cells={}...'.format(n_cells))
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=fixed_duration, tau=tau
            )
        except MemoryError:
            print('  Skipping n_cells={}: MemoryError'.format(n_cells))
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * n_cells * n_frames) / t_my
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'OMSI',
                                'N_Cells': n_cells, 'Time': t_my, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print('    OMSI: {:.1f}s'.format(t_my))
            except Exception as exc:
                print('    OMSI failed: {}'.format(exc))

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * n_cells * n_frames) / t_trad
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'CaImAn MCMC',
                                'N_Cells': n_cells, 'Time': t_trad, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print('    CaImAn MCMC: {:.1f}s'.format(t_trad))
            except Exception as exc:
                print('    CaImAn MCMC failed: {}'.format(exc))

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'OASIS',
                                'N_Cells': n_cells, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print('    OASIS: {:.1f}s'.format(t_oasis))
            except Exception as exc:
                print('    OASIS failed: {}'.format(exc))

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_cells_{n_cells}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Cell_Scaling', 'Model': _model,
                                    'N_Cells': n_cells, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'Duration': fixed_duration, 'Frames': n_frames})
                    print('    CASCADE ({}): {:.1f}s'.format(_dev.upper(), t_cascade))
                except Exception as exc:
                    print('    CASCADE ({}) failed: {}'.format(_dev.upper(), exc))

        _save_records(results, partial_path)

    print('\n--- Duration scaling ---')
    for dur in durations:
        print('  duration={}s...'.format(dur))
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=fixed_cells, fs=fs, duration=dur, tau=tau
            )
        except MemoryError:
            print('  Skipping duration={}: MemoryError'.format(dur))
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * fixed_cells * n_frames) / t_my
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'OMSI',
                                'Duration': dur, 'Time': t_my, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print('    OMSI: {:.1f}s'.format(t_my))
            except Exception as exc:
                print('    OMSI failed: {}'.format(exc))

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * fixed_cells * n_frames) / t_trad
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'CaImAn MCMC',
                                'Duration': dur, 'Time': t_trad, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print('    CaImAn MCMC: {:.1f}s'.format(t_trad))
            except Exception as exc:
                print('    CaImAn MCMC failed: {}'.format(exc))

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, fixed_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'OASIS',
                                'Duration': dur, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print('    OASIS: {:.1f}s'.format(t_oasis))
            except Exception as exc:
                print('    OASIS failed: {}'.format(exc))

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_dur_{dur}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Duration_Scaling', 'Model': _model,
                                    'Duration': dur, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'N_Cells': fixed_cells, 'Frames': n_frames})
                    print('    CASCADE ({}): {:.1f}s'.format(_dev.upper(), t_cascade))
                except Exception as exc:
                    print('    CASCADE ({}) failed: {}'.format(_dev.upper(), exc))

        _save_records(results, partial_path)

    if run_matlab and matlab_records is not None:
        print('\nInjecting {} precomputed CaImAn MCMC (scalability) records...'.format(len(matlab_records)))
        results.extend(matlab_records)
        _save_records(results, partial_path)

    return


def benchmark_params(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True, matlab_records=None, experiments=('tau', 'fs')):
    """
    Benchmark accuracy across calcium decay time constants and frame rates.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_mine : bool, optional
        Whether to run OMSI.
    run_cascade : bool, optional
        Whether to run CASCADE.
    matlab_records : list of dict or None, optional
        Precomputed CaImAn MCMC records to inject instead of running MATLAB.
    experiments : tuple of str, optional
        Which sweeps to run: 'tau' and/or 'fs'. When only 'fs' is run, existing
        Tau_Sensitivity rows in the partial file are left untouched.
    """
    n_cells  = 50
    duration = 300
    tau_values = [0.2, 0.5, 0.8, 1.2, 2.0]
    fixed_fs   = 30.0
    fs_values  = [7.5, 8.7, 10, 14, 20, 24, 30, 39, 50, 71, 100]
    fixed_tau  = 1.2

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_params_partial.npz')
    save_exp = None if 'tau' in experiments else 'Fs_Sensitivity'

    if 'tau' in experiments:
        print('\n--- Tau sensitivity ---')
    for tau in (tau_values if 'tau' in experiments else []):
        print('  tau={}s...'.format(tau))
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fixed_fs, duration=duration, tau=tau
            )
            true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fixed_fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'OMSI', tau, fixed_fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fixed_fs),
                                    sweeps=np.median(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print('    OMSI: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fixed_fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'CaImAn MCMC', tau, fixed_fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fixed_fs),
                                    sweeps=np.median(sweeps),
                                    n_cells=n_cells, duration=duration))
                print('    CaImAn MCMC: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fixed_fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'OASIS', tau, fixed_fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fixed_fs),
                                    n_cells=n_cells, duration=duration))
                print('    OASIS: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fixed_fs, data_dir, f'bench_tau_{tau}_{_dev}', device=_dev)
                    results.append(_row('Tau_Sensitivity', _model, tau, fixed_fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fixed_fs),
                                        n_cells=n_cells, duration=duration))
                    print('    CASCADE ({}): F1={:.3f} ± {:.3f}'.format(_dev.upper(), results[-1]['F1'], results[-1]['F1_mad']))

        except Exception as exc:
            print('  Failed for tau={}: {}'.format(tau, exc))
        _save_records(results, partial_path)

    if 'fs' in experiments:
        print('\n--- Frame-rate sensitivity ---')
    for fs in (fs_values if 'fs' in experiments else []):
        print('  fs={}Hz...'.format(fs))
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=duration, tau=fixed_tau
            )
            true_events = [helpers.make_event_ground_truth(s, fixed_tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'OMSI', fixed_tau, fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=np.median(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print('    OMSI: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fs, tau=fixed_tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'CaImAn MCMC', fixed_tau, fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=np.median(sweeps),
                                    n_cells=n_cells, duration=duration))
                print('    CaImAn MCMC: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, fixed_tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'OASIS', fixed_tau, fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fs),
                                    n_cells=n_cells, duration=duration))
                print('    OASIS: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_fs_{fs}_{_dev}', device=_dev)
                    results.append(_row('Fs_Sensitivity', _model, fixed_tau, fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fs),
                                        n_cells=n_cells, duration=duration))
                    print('    CASCADE ({}): F1={:.3f} ± {:.3f}'.format(_dev.upper(), results[-1]['F1'], results[-1]['F1_mad']))

        except Exception as exc:
            print('  Failed for fs={}: {}'.format(fs, exc))
        _save_records(results, partial_path, experiment=save_exp)

    if run_matlab and matlab_records is not None:
        print('\nInjecting {} precomputed CaImAn MCMC (params) records...'.format(len(matlab_records)))
        results.extend(matlab_records)
        _save_records(results, partial_path, experiment=save_exp)

    return


def benchmark_noise_sensitivity(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                                  run_cascade=True, matlab_records=None, cells_only=False):
    """
    Benchmark accuracy across a range of SNR levels.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_mine : bool, optional
        Whether to run OMSI.
    run_cascade : bool, optional
        Whether to run CASCADE.
    matlab_records : list of dict or None, optional
        Precomputed CaImAn MCMC records to inject instead of running MATLAB.
    cells_only : bool, optional
        If True, only save per-cell results without updating the main table.
    """
    n_cells    = 50
    duration   = 300
    fs         = 30.0
    tau        = 1.2
    snr_levels = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]

    results      = []
    cell_records = []
    partial_path = os.path.join(data_dir, 'benchmark_noise_sensitivity_partial.npz')
    cells_path   = os.path.join(data_dir, 'benchmark_noise_sensitivity_cells.npz')

    print('Generating fixed cell population (n_cells={}, duration={}s)...'.format(n_cells, duration))
    _, true_spikes, clean_traces, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau, snr=1e6
    )
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

    peak_signals = np.array([
        np.percentile(clean_traces[i], 99) - np.percentile(clean_traces[i], 1)
        for i in range(n_cells)
    ])
    peak_signals = np.maximum(peak_signals, 1e-9)

    def _append_cell_rows(model_name, snr_val, pred_spk):
        """Record per-cell precision and recall for a given model and SNR level."""
        p, r, _   = OMSI.compute_accuracy_strict(true_spikes, pred_spk, tolerance=0.1)
        pw, rw, _ = helpers.compute_accuracy_window(true_spikes, pred_spk)
        for i in range(n_cells):
            cell_records.append({
                'Model':            model_name,
                'SNR':              float(snr_val),
                'Precision':        float(p[i]),
                'Recall':           float(r[i]),
                'Precision_window': float(pw[i]),
                'Recall_window':    float(rw[i]),
            })

    for snr_val in snr_levels:
        print('  SNR={}...'.format(snr_val))
        try:
            sigmas = peak_signals / snr_val
            dff = clean_traces + np.random.normal(0, sigmas[:, None],
                                                   size=clean_traces.shape)

            base = {'Experiment': 'Noise_Sensitivity', 'SNR': snr_val,
                    'N_Cells': n_cells, 'Duration': duration}

            def km(pred_spk):
                """Shorthand for computing all accuracy metrics against ground-truth spikes."""
                return _metrics(true_spikes, pred_spk, true_events, fs)

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append({**base, 'Model': 'OMSI', 'Time': t_my,
                                 **km(res['optim_spikes'])})
                _append_cell_rows('OMSI', snr_val, res['optim_spikes'])
                print('    OMSI: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append({**base, 'Model': 'CaImAn MCMC', 'Time': t_trad,
                                 **km(trad_spikes)})
                _append_cell_rows('CaImAn MCMC', snr_val, trad_spikes)
                print('    CaImAn MCMC: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({**base, 'Model': 'OASIS', 'Time': t_oasis,
                                 **km(oas_spk)})
                _append_cell_rows('OASIS', snr_val, oas_spk)
                print('    OASIS: F1={:.3f} ± {:.3f}'.format(results[-1]['F1'], results[-1]['F1_mad']))

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_noise_snr{snr_val}_{_dev}', device=_dev)
                    results.append({**base, 'Model': _model, 'Time': t_cascade,
                                     **km(cascade_spikes)})
                    _append_cell_rows(_model, snr_val, cascade_spikes)
                    print('    CASCADE ({}): F1={:.3f} ± {:.3f}'.format(_dev.upper(), results[-1]['F1'], results[-1]['F1_mad']))

        except Exception as exc:
            print('  Failed for SNR={}: {}'.format(snr_val, exc))
        if not cells_only:
            _save_records(results, partial_path)

    if not cells_only and run_matlab and matlab_records is not None:
        print('\nInjecting {} precomputed CaImAn MCMC (noise) records...'.format(len(matlab_records)))
        results.extend(matlab_records)
        _save_records(results, partial_path)

    if cell_records:
        _save_records(cell_records, cells_path)
        print('  Saved {} cell-level rows: {}'.format(len(cell_records), cells_path))

    return


def benchmark_firing_rate_sensitivity(data_dir, run_oasis=True, run_matlab=True,
                                      run_mine=True, run_cascade=True, matlab_records=None):
    """
    Benchmark accuracy as a function of per-cell firing rate.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_mine : bool, optional
        Whether to run OMSI.
    run_cascade : bool, optional
        Whether to run CASCADE.
    matlab_records : list of dict or None, optional
        Precomputed CaImAn MCMC records to inject instead of running MATLAB.
    """
    n_cells  = 250
    duration = 300
    fs       = 30.0
    tau      = 1.2

    print('Generating synthetic data (n_cells={}, duration={}s)...'.format(n_cells, duration))
    dff, true_spikes, _, _, firing_rates, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]
    npz_spikes  = {'true': true_spikes}
    npz_calcium = {}

    all_results  = []
    partial_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')

    def per_cell(model, i, pred_spk_i, time_i):
        """
        Compute per-cell accuracy metrics and return a result record dict.

        Parameters
        ----------
        model : str
            Algorithm name.
        i : int
            Cell index.
        pred_spk_i : ndarray
            Predicted spike times in seconds for cell i.
        time_i : float
            Inference time for cell i in seconds.

        Returns
        -------
        dict
            Per-cell accuracy record.
        """
        prec,   rec,   f1   = OMSI.compute_accuracy_strict([true_spikes[i]], [pred_spk_i])
        prec_w, rec_w, f1_w = helpers.compute_accuracy_window([true_spikes[i]], [pred_spk_i])
        prec_e, rec_e, f1_e = helpers.compute_accuracy_window([true_events[i]], [pred_spk_i])
        cosmic = helpers.compute_cosmic([true_spikes[i]], [pred_spk_i], fs)
        return {
            'model': model, 'cell_id': i, 'firing_rate': float(firing_rates[i]),
            'precision': prec[0], 'recall': rec[0], 'f1': f1[0],
            'precision_window': prec_w[0], 'recall_window': rec_w[0], 'f1_window': f1_w[0],
            'precision_event': prec_e[0], 'recall_event': rec_e[0], 'f1_event': f1_e[0],
            'cosmic': cosmic[0], 'time': time_i,
        }

    if run_mine:
        print('\nRunning OMSI...')
        try:
            t0  = time.time()
            res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                   benchmark=True)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('OMSI', i, res['optim_spikes'][i],
                                             res['optim_times_per_cell'][i]))
            npz_spikes['my_method'] = res['optim_spikes']
            print('  Finished in {:.1f}s'.format(total_time))
        except Exception as exc:
            print('  OMSI failed: {}'.format(exc))
        _save_records(all_results, partial_path)

    if run_matlab:
        if matlab_records is not None:
            print('\nInjecting {} precomputed CaImAn MCMC (firing rate) records...'.format(len(matlab_records)))
            all_results.extend(matlab_records)
            _save_records(all_results, partial_path)
        else:
            print('\nRunning CaImAn MCMC...')
            trad_spikes_all = []
            for i in range(n_cells):
                print('  Processing cell {}/{}...'.format(i+1, n_cells), end='\r')
                try:
                    t0 = time.time()
                    trad_spk, _, _, _ = run_matlab_pnevMCMC(
                        dff[i:i+1], fs=fs, tau=tau, n_sweeps='auto')
                    time_taken = time.time() - t0
                    trad_spikes_all.append(trad_spk[0])
                    all_results.append(per_cell('CaImAn MCMC', i, trad_spk[0], time_taken))
                except Exception as exc:
                    print('\n  CaImAn MCMC failed on cell {}: {}'.format(i, exc))
                    trad_spikes_all.append(np.array([]))
            print('\n  Finished.')
            npz_spikes['trad_mcmc'] = trad_spikes_all
            _save_records(all_results, partial_path)

    if run_oasis:
        print('\nRunning OASIS...')
        try:
            t0 = time.time()
            oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('OASIS', i, oas_spk[i], np.nan))
            npz_spikes['oasis']  = oas_spk
            npz_calcium['oasis'] = np.array(oas_cal)
            print('  Finished in {:.1f}s'.format(total_time))
        except Exception as exc:
            print('  OASIS failed: {}'.format(exc))
        _save_records(all_results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print('\nRunning CASCADE (subprocess, {})...'.format(_dev.upper()))
            try:
                _, cascade_spikes, t_cascade = _run_cascade_inference(
                    dff, fs, data_dir, f'bench_firerate_cascade_{_dev}', device=_dev)
                for i in range(n_cells):
                    all_results.append(per_cell(_model, i, cascade_spikes[i], np.nan))
                npz_spikes[f'cascade_{_dev}'] = cascade_spikes
                print('  CASCADE ({}) finished in {:.1f}s'.format(_dev.upper(), t_cascade))
            except Exception as exc:
                print('  CASCADE ({}) failed: {}'.format(_dev.upper(), exc))
            _save_records(all_results, partial_path)

    if all_results:
        npz_save = {'dff': dff, 'fs': fs, 'tau': tau,
                    'firing_rates': np.array(firing_rates)}
        for k, v in npz_spikes.items():
            npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
        for k, v in npz_calcium.items():
            npz_save[f'calcium_{k}'] = v
        np.savez(os.path.join(data_dir, 'firing_rate_sensitivity_traces.npz'), **npz_save)

    return

_CASCADE_SR_SEED     = 77
_CASCADE_SR_N_CELLS  = 50
_CASCADE_SR_DURATION = 300
_CASCADE_SR_TAU      = 1.2


def benchmark_cascade_sample_rate(data_dir, run_cascade=True):
    """
    Compare CASCADE accuracy at 7.5 Hz vs 30 Hz sampling rates.

    Parameters
    ----------
    data_dir : str
        Directory for result files.
    run_cascade : bool, optional
        Whether to run CASCADE inference (skips if False, uses NaN placeholders).
    """
    from simulation_helpers import generate_synthetic_data

    out_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    rng = np.random.default_rng(_CASCADE_SR_SEED)

    results = {}
    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        print('  {} Hz...'.format(fs))
        np.random.seed(int(rng.integers(0, 2**31)))
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=_CASCADE_SR_N_CELLS, fs=fs,
            duration=_CASCADE_SR_DURATION, tau=_CASCADE_SR_TAU)
        true_events = [helpers.make_event_ground_truth(s, _CASCADE_SR_TAU)
                       for s in true_spikes]

        ts_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_true_spikes.npz')
        np.savez(ts_path, true_spikes=np.array(true_spikes, dtype=object),
                 tau=_CASCADE_SR_TAU)

        if not run_cascade:
            results[f'fb_{suffix}']     = np.full(_CASCADE_SR_N_CELLS, np.nan)
            results[f'cosmic_{suffix}'] = np.full(_CASCADE_SR_N_CELLS, np.nan)
            continue

        cascade_spikes = None
        for dev in ('gpu', 'cpu'):
            out_file = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_{dev}_output.npz')
            if os.path.exists(out_file):
                try:
                    _d = np.load(out_file, allow_pickle=True)
                    cascade_spikes = list(_d['cascade_spikes'])
                    print('    Loaded existing CASCADE ({}) output.'.format(dev.upper()))
                    break
                except Exception:
                    pass
            try:
                _, cascade_spikes, _ = _run_cascade_inference(
                    dff, fs, data_dir, f'cascade_samplerate_{fs}hz_{dev}', device=dev)
                print('    CASCADE ({}) done.'.format(dev.upper()))
                break
            except Exception as exc:
                print('    CASCADE {} failed: {}'.format(dev.upper(), exc))

        if cascade_spikes is None:
            results[f'fb_{suffix}']     = np.full(_CASCADE_SR_N_CELLS, np.nan)
            results[f'cosmic_{suffix}'] = np.full(_CASCADE_SR_N_CELLS, np.nan)
            continue

        prec, rec, _ = OMSI.compute_accuracy_strict(true_spikes, cascade_spikes,
                                                      tolerance=0.1)
        b2    = BETA ** 2
        denom = b2 * prec + rec
        fb    = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        cosmic = helpers.compute_cosmic(true_spikes, cascade_spikes, fs)

        results[f'fb_{suffix}']     = fb
        results[f'cosmic_{suffix}'] = cosmic
        print('    Fβ={:.3f} ± {:.3f}  CosMIC={:.3f} ± {:.3f}'.format(
            np.nanmedian(fb), _mad(fb), np.nanmedian(cosmic), _mad(cosmic)))

    np.savez(out_path, **results)
    print('  Saved: {}'.format(out_path))


def _median_mad(vals):

    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return med, mad


def _fmt_median_mad(vals, width=14):

    med, mad = _median_mad(vals)
    if np.isnan(med):
        return 'n/a'.ljust(width)
    return '{:.3f} +/- {:.3f}'.format(med, mad).ljust(width)


def print_summary(data_dir=_DEFAULT_DATA_DIR):

    prec_col = 'Precision' if USE_STRICT_ACCURACY else 'Precision_window'
    rec_col  = 'Recall'    if USE_STRICT_ACCURACY else 'Recall_window'
    fb_col   = 'F1'        if USE_STRICT_ACCURACY else 'F1_window'
    models_order = ['fMCSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU']

    header = '{:<14} {:<12} {:>8}  {:<14} {:<14} {:<14} {:<14} {:<12}'.format(
        'Experiment', 'Model', 'Var', 'Fbeta', 'Precision', 'Recall', 'CosMIC', 'Time (s)')

    def _fmt_time(vals):
        """Format a time value/array: plain if a single measurement, median +/- MAD if several."""
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return 'n/a'.ljust(12)
        if len(v) == 1:
            return '{:.3g}'.format(v[0]).ljust(12)
        return _fmt_median_mad(v, width=12)

    def _print_rows(rows):
        """Print one summary line per (Model, variable) using per-cell arrays."""
        print(header)
        print('-' * len(header))
        for exp, var_label, var_val, model, fb, p, r, cos, t in rows:
            print('{:<14} {:<12} {:>8}  {} {} {} {} {}'.format(
                exp, model, var_label.format(var_val),
                _fmt_median_mad(fb), _fmt_median_mad(p),
                _fmt_median_mad(r), _fmt_median_mad(cos), _fmt_time(t)))

    def _rows_from_row_table(path, exp_name, var_col, var_fmt='{:g}'):
        """
        Build (exp, var_label, var_val, model, fb, p, r, cos, time) rows from
        a table written via _row (PerCell column present when re-run since
        the per-cell change; older/injected rows fall back to the single
        mean). Time is always the row's single stored wall-clock value.
        """
        out = []
        if not os.path.exists(path):
            return out
        tbl = _load_records(path)
        if _tbl_len(tbl) == 0 or 'Model' not in tbl:
            return out
        sub = _tbl_filter(tbl, 'Experiment', exp_name) if 'Experiment' in tbl else tbl
        if _tbl_len(sub) == 0:
            return out
        sub = _tbl_sort(sub, var_col)
        per_cell = _per_cell_metrics(sub)
        for model in models_order:
            mmask = sub['Model'] == model
            if mmask.sum() == 0:
                continue
            idx = np.where(mmask)[0]
            for i in idx:
                pc = per_cell[i] if per_cell is not None else {}
                if pc:
                    fb, p, r, cos = pc[fb_col], pc[prec_col], pc[rec_col], pc['COSMIC']
                else:
                    fb  = [sub[fb_col][i]]
                    p   = [sub[prec_col][i]]
                    r   = [sub[rec_col][i]]
                    cos = [sub['COSMIC'][i]]
                t = [sub['Time'][i]] if 'Time' in sub else [np.nan]
                out.append((exp_name, var_fmt, sub[var_col][i], model, fb, p, r, cos, t))
        return out

    print('\n=== Sweeps (accuracy at each MCMC sweep count; OASIS/CASCADE are single baselines) ===')
    rows = _rows_from_row_table(os.path.join(data_dir, 'benchmark_sweeps_partial.npz'),
                                'Sweeps', 'Sweeps', 'sweeps={:g}')
    _print_rows(rows) if rows else print('  No data. Run --mode test first.')

    print('\n=== Tau sensitivity ===')
    rows = _rows_from_row_table(os.path.join(data_dir, 'benchmark_params_partial.npz'),
                                'Tau_Sensitivity', 'Tau', 'tau={:g}s')
    _print_rows(rows) if rows else print('  No data. Run --mode test first.')

    print('\n=== Frame-rate sensitivity ===')
    rows = _rows_from_row_table(os.path.join(data_dir, 'benchmark_params_partial.npz'),
                                'Fs_Sensitivity', 'Fs', 'fs={:g}Hz')
    _print_rows(rows) if rows else print('  No data. Run --mode test first.')

    print('\n=== Noise sensitivity ===')
    print('  (CosMIC has no per-cell record for this benchmark; shown as the stored mean, no spread)')
    cells_path = os.path.join(data_dir, 'benchmark_noise_sensitivity_cells.npz')
    agg_path   = os.path.join(data_dir, 'benchmark_noise_sensitivity_partial.npz')
    if os.path.exists(cells_path):
        cells = _load_records(cells_path)
        agg   = _load_records(agg_path) if os.path.exists(agg_path) else {}
        rows = []
        for model in models_order:
            mmask = cells['Model'] == model
            if mmask.sum() == 0:
                continue
            for snr_val in sorted(set(cells['SNR'][mmask].astype(float)), reverse=True):
                smask = mmask & (cells['SNR'].astype(float) == snr_val)
                p_c = cells[prec_col][smask].astype(float)
                r_c = cells[rec_col][smask].astype(float)
                fb_c = _fbeta(p_c, r_c)
                cos = [np.nan]
                t = [np.nan]
                if agg and 'Model' in agg:
                    amask = (agg['Model'] == model) & (agg['SNR'].astype(float) == snr_val)
                    if amask.sum() > 0:
                        if 'COSMIC' in agg:
                            cos = [agg['COSMIC'][amask][0]]
                        if 'Time' in agg:
                            t = [agg['Time'][amask][0]]
                rows.append(('Noise_Sensitivity', 'SNR={:g}', snr_val, model, fb_c, p_c, r_c, cos, t))
        _print_rows(rows) if rows else print('  No data. Run --mode test first.')
    else:
        print('  No data. Run --mode test (or --mode noise-cells) first.')

    print('\n=== Firing-rate sensitivity (median +/- MAD pooled across all cells) ===')
    print('  (Time is per-cell here, unlike the other sections above -- OASIS/CASCADE')
    print('   only time the whole batch, so their per-cell Time is n/a)')
    fr_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')
    if os.path.exists(fr_path):
        tbl = _load_records(fr_path)
        if _tbl_len(tbl) > 0 and 'model' in tbl:
            fr_prec = 'precision' if USE_STRICT_ACCURACY else 'precision_window'
            fr_rec  = 'recall'    if USE_STRICT_ACCURACY else 'recall_window'
            fr_fb   = 'f1'        if USE_STRICT_ACCURACY else 'f1_window'
            rows = []
            for model in models_order:
                mmask = tbl['model'] == model
                if mmask.sum() == 0:
                    continue
                rows.append(('Firing_Rate', '{}', '', model,
                            tbl[fr_fb][mmask].astype(float), tbl[fr_prec][mmask].astype(float),
                            tbl[fr_rec][mmask].astype(float), tbl['cosmic'][mmask].astype(float),
                            tbl['time'][mmask].astype(float)))
            _print_rows(rows) if rows else print('  No data. Run --mode test first.')
        else:
            print('  No data. Run --mode test first.')
    else:
        print('  No data. Run --mode test first.')

    print('\n=== CASCADE: 7.5 Hz vs 30 Hz (per-cell Fbeta and CosMIC; no Precision/Recall or Time stored) ===')
    cr_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    if os.path.exists(cr_path):
        d = np.load(cr_path, allow_pickle=True)
        hdr = '{:<14} {:>8}  {:<14} {:<14}'.format('Experiment', 'Var', 'Fbeta', 'CosMIC')
        print(hdr)
        print('-' * len(hdr))
        for suffix, label in [('7', '7.5 Hz'), ('30', '30 Hz')]:
            if f'fb_{suffix}' in d.files:
                print('{:<14} {:>8}  {} {}'.format(
                    'Cascade_SR', label,
                    _fmt_median_mad(d[f'fb_{suffix}']), _fmt_median_mad(d[f'cosmic_{suffix}'])))
    else:
        print('  No data. Run --mode cascade-samplerate first.')

    print('\n=== Compute time: cell-count scaling (fixed 300s recordings; single run per point) ===')
    scale_path = os.path.join(data_dir, 'benchmark_scalability_partial.npz')
    if os.path.exists(scale_path):
        scale_tbl = _load_records(scale_path)
    else:
        scale_tbl = {}
        print('  No data. Run --mode test first.')
    if scale_tbl and 'Model' in scale_tbl:
        thdr = '{:<12} {:>10}  {:<12}'.format('Model', 'Var', 'Time (s)')
        cell_sub = _tbl_filter(scale_tbl, 'Experiment', 'Cell_Scaling')
        if _tbl_len(cell_sub) > 0:
            print(thdr)
            print('-' * len(thdr))
            for model in models_order:
                mmask = cell_sub['Model'] == model
                if mmask.sum() == 0:
                    continue
                sub = _tbl_sort({k: v[mmask] for k, v in cell_sub.items()}, 'N_Cells')
                for i in range(_tbl_len(sub)):
                    print('{:<12} {:>10}  {}'.format(
                        model, 'n={:g}'.format(sub['N_Cells'][i]), _fmt_time([sub['Time'][i]])))
        else:
            print('  No cell-count scaling data.')

        print('\n=== Compute time: recording-duration scaling (fixed 100 cells; single run per point) ===')
        dur_sub = _tbl_filter(scale_tbl, 'Experiment', 'Duration_Scaling')
        if _tbl_len(dur_sub) > 0:
            print(thdr)
            print('-' * len(thdr))
            for model in models_order:
                mmask = dur_sub['Model'] == model
                if mmask.sum() == 0:
                    continue
                sub = _tbl_sort({k: v[mmask] for k, v in dur_sub.items()}, 'Duration')
                for i in range(_tbl_len(sub)):
                    print('{:<12} {:>10}  {}'.format(
                        model, 'dur={:g}s'.format(sub['Duration'][i]), _fmt_time([sub['Time'][i]])))
        else:
            print('  No duration-scaling data.')

    print('\nNote: MAD is unscaled (median(|x - median(x)|)); multiply by 1.4826 for a')
    print('normal-consistent estimate comparable to standard deviation.')


def run_test(data_dir=_DEFAULT_DATA_DIR, run_fmcsi=True, run_matlab=True,
             run_oasis=True, run_cascade=True):

    Parameters
    ----------
    data_dir : str, optional
        Directory for result files.
    run_omsi : bool, optional
        Whether to run OMSI.
    run_matlab : bool, optional
        Whether to run CaImAn MCMC via MATLAB.
    run_oasis : bool, optional
        Whether to run OASIS.
    run_cascade : bool, optional
        Whether to run CASCADE.
    """
    os.makedirs(data_dir, exist_ok=True)

    ext = None
    if run_matlab:
        print('Loading pre-computed caiman MCMC data from:\n  {}'.format(_MATLAB_PRECOMPUTED_DIR))
        ext = _load_external_matlab_data()
        total = sum(len(ext[k]) for k in ('sweeps', 'scalability', 'params', 'noise_sensitivity', 'firing_rate'))
        print('  Loaded {} CaImAn MCMC records across all benchmarks.'.format(total))

    kw_shared = dict(run_oasis=run_oasis, run_mine=run_omsi, run_cascade=run_cascade,
                     run_matlab=run_matlab)

    print('=== Sweeps benchmark ===')
    benchmark_sweeps(data_dir, **kw_shared,
                     matlab_records=(ext['sweeps'] or None) if ext else None)
    print('\n=== Scalability benchmark ===')
    benchmark_scalability(data_dir, **kw_shared,
                          matlab_records=(ext['scalability'] or None) if ext else None)
    print('\n=== Parameter sensitivity benchmark ===')
    benchmark_params(data_dir, **kw_shared,
                     matlab_records=(ext['params'] or None) if ext else None)
    print('\n=== Noise sensitivity benchmark ===')
    benchmark_noise_sensitivity(data_dir, **kw_shared,
                                matlab_records=(ext['noise_sensitivity'] or None) if ext else None)
    print('\n=== Firing-rate sensitivity benchmark ===')
    benchmark_firing_rate_sensitivity(data_dir, **kw_shared,
                                      matlab_records=(ext['firing_rate'] or None) if ext else None)
    print('\n=== CASCADE 7.5 Hz vs 30 Hz comparison ===')
    benchmark_cascade_sample_rate(data_dir, run_cascade=run_cascade)
    print('\nTest mode complete.')


def _fbeta(prec, rec):

    p  = np.asarray(prec, dtype=float)
    r  = np.asarray(rec,  dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _fit_scaling(x, y):

    if len(x) < 3:
        return np.nan, np.nan, 'N/A'
    x, y = np.array(x, float), np.array(y, float)
    def r2(y_pred):
        """Compute R-squared of y_pred against the outer y."""
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 0.0 if ss_tot < 1e-9 else 1 - ss_res / ss_tot
    try:
        r2_lin  = r2(np.poly1d(np.polyfit(x, y, 1))(x))
    except Exception:
        r2_lin  = -np.inf
    try:
        r2_poly = r2(np.poly1d(np.polyfit(x, y, 2))(x))
    except Exception:
        r2_poly = -np.inf
    conclusion = 'Linear' if r2_lin >= r2_poly - 0.02 else 'Polynomial'
    return r2_lin, r2_poly, conclusion


def _set_three_ticks_x(ax):
    """
    Set three evenly spaced x-axis ticks based on plotted data range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to modify in place.
    """
    all_x = [v for line in ax.get_lines() for v in line.get_xdata()]
    if len(all_x) < 2:
        return
    lo, hi = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    if lo == hi:
        return
    ax.set_xticks([lo, (lo + hi) / 2, hi])
    ax.set_xticklabels([f'{v:.3g}' for v in [lo, (lo + hi) / 2, hi]])


def _filter_cascade_shared_x(cascade_sub, full_tbl, xcol):
    """
    Restrict CASCADE rows to x-values shared with non-CASCADE methods.

    Parameters
    ----------
    cascade_sub : dict
        Subset of the table containing only CASCADE rows.
    full_tbl : dict
        Full benchmark table including all methods.
    xcol : str
        Name of the x-axis column (e.g. 'N_Cells', 'Tau').

    Returns
    -------
    dict
        Filtered CASCADE rows.
    """
    non_cascade = np.array([not str(m).startswith('CASCADE') for m in full_tbl['Model']], dtype=bool)
    vals = full_tbl[xcol][non_cascade].astype(float)
    other_x = set(vals[~np.isnan(vals)].tolist())
    mask = np.isin(cascade_sub[xcol].astype(float), list(other_x))
    return {k: v[mask] for k, v in cascade_sub.items()}


_CASCADE_CMP_COLOR_7P5 = 'tab:red'
_CASCADE_CMP_COLOR_30  = 'tab:cyan'


def _rebuild_cascade_sample_rate_data(data_dir):
    """
    Recompute the CASCADE sample-rate comparison file from saved outputs.

    Loads previously saved true spikes and CASCADE output files from data_dir
    and writes cascade_7p5_vs_30hz_data.npz.

    Parameters
    ----------
    data_dir : str
        Directory containing saved output files.
    """
    out_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    results  = {}

    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        ts_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_true_spikes.npz')
        if not os.path.exists(ts_path):
            print('  No saved true_spikes for {} Hz -- re-run --mode test to regenerate.'.format(fs))
            return

        try:
            td          = np.load(ts_path, allow_pickle=True)
            true_spikes = list(td['true_spikes'])
            tau         = float(td['tau']) if 'tau' in td else _CASCADE_SR_TAU
        except Exception as exc:
            print('  Could not load {}: {}'.format(ts_path, exc))
            return

        cascade_spikes = None
        for dev in ('gpu', 'cpu'):
            out_file = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_{dev}_output.npz')
            if os.path.exists(out_file):
                try:
                    d = np.load(out_file, allow_pickle=True)
                    cascade_spikes = list(d['cascade_spikes'])
                    break
                except Exception:
                    pass
        if cascade_spikes is None:
            print('  No CASCADE output found for {} Hz -- re-run --mode test.'.format(fs))
            return

        prec, rec, _ = OMSI.compute_accuracy_strict(true_spikes, cascade_spikes,
                                                      tolerance=0.1)
        b2    = BETA ** 2
        denom = b2 * prec + rec
        fb    = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        cosmic = helpers.compute_cosmic(true_spikes, cascade_spikes, fs)
        results[f'fb_{suffix}']     = fb
        results[f'cosmic_{suffix}'] = cosmic
        print('  Rebuilt {} Hz: Fβ={:.3f} ± {:.3f}  CosMIC={:.3f} ± {:.3f}'.format(
            fs, np.nanmedian(fb), _mad(fb), np.nanmedian(cosmic), _mad(cosmic)))

    if len(results) == 4:
        np.savez(out_path, **results)
        print('  Saved rebuilt data: {}'.format(out_path))


def _plot_cascade_comparison(ax, data_dir):
    """
    Plot violin comparison of CASCADE accuracy at 7.5 Hz vs 30 Hz.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot into.
    data_dir : str
        Directory containing cascade_7p5_vs_30hz_data.npz.
    """
    from matplotlib.patches import Patch

    npz_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    _needs_rebuild = not os.path.exists(npz_path)
    if not _needs_rebuild:
        _d = np.load(npz_path)
        if all(not np.any(np.isfinite(_d[k])) for k in _d.files):
            _needs_rebuild = True
    if _needs_rebuild:
        print('  Attempting to rebuild cascade sample-rate data from output files...')
        _rebuild_cascade_sample_rate_data(data_dir)
    if not os.path.exists(npz_path):
        ax.text(0.5, 0.5, 'No data\n(run --mode test)',
                transform=ax.transAxes, ha='center', va='center', fontsize=7)
        return

    data     = np.load(npz_path)
    fb_7     = data['fb_7'];     fb_30     = data['fb_30']
    cosmic_7 = data['cosmic_7']; cosmic_30 = data['cosmic_30']

    all_pos      = [1, 2, 3.15, 4.15]
    all_datasets = [
        (fb_7[np.isfinite(fb_7)],          _CASCADE_CMP_COLOR_7P5),
        (fb_30[np.isfinite(fb_30)],         _CASCADE_CMP_COLOR_30),
        (cosmic_7[np.isfinite(cosmic_7)],   _CASCADE_CMP_COLOR_7P5),
        (cosmic_30[np.isfinite(cosmic_30)], _CASCADE_CMP_COLOR_30),
    ]
    pos      = [p for p, (d, _) in zip(all_pos, all_datasets) if len(d) > 0]
    datasets = [(d, c) for d, c in all_datasets if len(d) > 0]
    if not datasets:
        ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes,
                ha='center', va='center', fontsize=7)
        return
    parts = ax.violinplot([d for d, _ in datasets], positions=pos,
                          showmedians=True, widths=0.65)
    for pc, (_, col) in zip(parts['bodies'], datasets):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[partname].set_color('k'); parts[partname].set_linewidth(0.8)

    ax.set_xticks([(pos[0] + pos[1]) / 2, (pos[2] + pos[3]) / 2])
    ax.set_xticklabels([r'$F_\beta$', 'CosMIC'])
    ax.legend(handles=[
        Patch(facecolor=_CASCADE_CMP_COLOR_7P5, alpha=0.75, label='7.5 Hz'),
        Patch(facecolor=_CASCADE_CMP_COLOR_30,  alpha=0.75, label='30 Hz'),
    ], loc='upper right', handlelength=1.0, handleheight=0.8,
       borderpad=0.4, labelspacing=0.2, frameon=False)
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.1)


def _running_median_mad(x, y, x_out, bandwidth):
    """
    Compute running median and median absolute deviation over a sliding window.

    Parameters
    ----------
    x : ndarray
        Independent variable values.
    y : ndarray
        Dependent variable values.
    x_out : ndarray
        Center points at which to evaluate the running statistic.
    bandwidth : float
        Half-width of the sliding window.

    Returns
    -------
    medians : ndarray
        Running median at each point in x_out (NaN where fewer than 5 points).
    mads : ndarray
        Running median absolute deviation at each point in x_out.
    """
    medians = np.full(len(x_out), np.nan)
    mads    = np.full(len(x_out), np.nan)
    for i, xc in enumerate(x_out):
        mask = (x >= xc - bandwidth) & (x <= xc + bandwidth)
        if mask.sum() >= 5:
            vals        = y[mask]
            medians[i]  = np.median(vals)
            mads[i]     = _mad(vals)
    return medians, mads


def _load_benchmark_tables(data_dir):
    """
    Load and merge the scaling/sensitivity benchmark tables used by figure 2.

    Parameters
    ----------
    data_dir : str
        Directory containing benchmark result .npz files.

    Returns
    -------
    combined : dict
        Columnar table of all benchmark rows, including external CaImAn MCMC records.
    dur_cascade_tbl : dict
        CASCADE duration-scaling rows from the alternate directory (empty if unavailable).
    """
    partial_files = [
        'benchmark_sweeps_partial.npz',
        'benchmark_scalability_partial.npz',
        'benchmark_params_partial.npz',
        'benchmark_noise_sensitivity_partial.npz',
    ]

    all_records = []
    for fname in partial_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            try:
                tbl = _load_records(fpath)
                all_records.append(tbl)
                print('Loaded {}  ({} rows)'.format(fpath, _tbl_len(tbl)))
            except Exception as exc:
                print('Error reading {}: {}'.format(fpath, exc))

    if not all_records:
        raise RuntimeError(f'No partial benchmark files found in {data_dir}. '
                           'Run --mode test first.')

    combined = _tbl_concat(all_records)

    ext = _load_external_matlab_data()
    ext_benchmark_keys = ['sweeps', 'scalability', 'params', 'noise_sensitivity']
    ext_records = []
    # Locally re-run (Experiment, Model) pairs take precedence over the precomputed ones.
    local_pairs = set(zip(combined['Experiment'], combined['Model']))
    for key in ext_benchmark_keys:
        ext_records.extend(r for r in ext[key]
                           if (r.get('Experiment'), r.get('Model')) not in local_pairs)
    if ext_records:
        ext_tbl = _records_to_tbl(ext_records)
        combined = _tbl_concat([combined, ext_tbl])
        print('Injected {} CaImAn MCMC records from external files.'.format(len(ext_records)))

    extra_gpu_path = os.path.join(data_dir, 'benchmark_scalability_cascade_gpu_extra.npz')
    if os.path.exists(extra_gpu_path):
        try:
            extra_gpu_tbl = _load_records(extra_gpu_path)
            keep_rows = []
            for i in range(_tbl_len(extra_gpu_tbl)):
                exp   = str(extra_gpu_tbl['Experiment'][i])
                model = str(extra_gpu_tbl['Model'][i])
                if model != 'CASCADE_GPU' or exp not in ('Cell_Scaling', 'Duration_Scaling'):
                    continue  # this file should only ever hold these, but don't trust it blindly
                xcol = 'N_Cells' if exp == 'Cell_Scaling' else 'Duration'
                xval = float(extra_gpu_tbl[xcol][i])
                already_real = (
                    (combined.get('Model', np.array([])) == model) &
                    (combined.get('Experiment', np.array([])) == exp) &
                    (combined.get(xcol, np.array([])).astype(float) == xval)
                ) if combined else np.array([], dtype=bool)
                if np.any(already_real):
                    continue  # the real benchmark already has this point -- prefer it
                keep_rows.append({k: extra_gpu_tbl[k][i] for k in extra_gpu_tbl})
            if keep_rows:
                combined = _tbl_concat([combined, _records_to_tbl(keep_rows)])
                print('Injected {} CASCADE (GPU) point(s) from {}.'.format(
                    len(keep_rows), extra_gpu_path))
        except Exception as exc:
            print('Warning: could not load {}: {}'.format(extra_gpu_path, exc))

    dur_cascade_tbl = {}
    for _ext in ('.npz', '.json'):
        _alt_sc_path = os.path.join(_CASCADE_DURATION_ALT_DIR,
                                    f'benchmark_scalability_partial{_ext}')
        if not os.path.exists(_alt_sc_path):
            continue
        try:
            if _ext == '.json':
                import json as _json
                with open(_alt_sc_path) as _f:
                    _raw = _json.load(_f)
                _rows = [r for r in _raw
                         if str(r.get('Model', '')).startswith('CASCADE')
                         and r.get('Experiment') == 'Duration_Scaling']

                for _r in _rows:
                    if _r['Model'] == 'CASCADE':
                        _r['Model'] = 'CASCADE_GPU'
                if _rows:
                    dur_cascade_tbl = _records_to_tbl(_rows)
            else:
                _alt = _load_records(_alt_sc_path)
                _mask = np.array([
                    str(m).startswith('CASCADE') and str(e) == 'Duration_Scaling'
                    for m, e in zip(_alt.get('Model', []), _alt.get('Experiment', []))
                ], dtype=bool)
                if _mask.sum() > 0:
                    dur_cascade_tbl = {k: v[_mask] for k, v in _alt.items()}
            if dur_cascade_tbl:
                print('Loaded {} CASCADE duration rows from alt dir.'.format(_tbl_len(dur_cascade_tbl)))
                break
        except Exception as _exc:
            print('Warning: could not load alt CASCADE duration data: {}'.format(_exc))

    return combined, dur_cascade_tbl


def _load_noise_cells(data_dir):
    """
    Load the cell-level noise sensitivity table, if present.

    Parameters
    ----------
    data_dir : str
        Directory containing benchmark_noise_sensitivity_cells.npz.

    Returns
    -------
    dict or None
        Columnar table of per-cell noise rows, or None if unavailable.
    """
    _cells_path = os.path.join(data_dir, 'benchmark_noise_sensitivity_cells.npz')
    _noise_cells = None
    if os.path.exists(_cells_path):
        try:
            _noise_cells = _load_records(_cells_path)
            print('Loaded {} cell-level noise rows.'.format(_tbl_len(_noise_cells)))
        except Exception as _exc:
            print('Warning: could not load noise cells file: {}'.format(_exc))
    return _noise_cells


def plot_figure(data_dir=_DEFAULT_DATA_DIR):
    """
    Load benchmark results and render figure 2 panels A and B.

    Panel A shows scaling and sensitivity benchmarks (sweeps, cells, duration,
    tau, noise). Panel B shows frame-rate sensitivity and CASCADE comparison.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing benchmark result .npz files.
    """
    combined, dur_cascade_tbl = _load_benchmark_tables(data_dir)

    fr_tbl = {}
    fr_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')
    if os.path.exists(fr_path):
        try:
            fr_tbl = _load_records(fr_path)
            print('Loaded {}  ({} rows)'.format(fr_path, _tbl_len(fr_tbl)))
        except Exception as exc:
            print('Error reading {}: {}'.format(fr_path, exc))
    fr_ext_records = _load_external_matlab_data()['firing_rate']
    if fr_ext_records:
        fr_ext_tbl = _records_to_tbl(fr_ext_records)
        fr_tbl = _tbl_concat([fr_tbl, fr_ext_tbl])
        print('Injected {} CaImAn MCMC firing-rate records from external files.'.format(len(fr_ext_records)))

    f1_col   = 'F1'        if USE_STRICT_ACCURACY else 'F1_window'
    prec_col = 'Precision' if USE_STRICT_ACCURACY else 'Precision_window'
    rec_col  = 'Recall'    if USE_STRICT_ACCURACY else 'Recall_window'

    scaling_stats = []

    _legend_labels = {
        'OMSI': 'OMSI', 'CaImAn MCMC': 'CaImAn MCMC', 'OASIS': 'OASIS',
        'CASCADE_GPU': 'CASCADE (GPU)', 'CASCADE_CPU': 'CASCADE (CPU)',
    }
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS[m], marker='.', linestyle='-',
                   label=_legend_labels[m])
        for m in ['OMSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU']
    ]

    _legend_labels1 = {
        'OMSI': 'OMSI', 'CaImAn MCMC': 'CaImAn MCMC', 'OASIS': 'OASIS',
        'CASCADE_GPU': 'CASCADE',
    }
    legend_handles1 = [
        plt.Line2D([0], [0], color=COLORS[m], marker='.', linestyle='-',
                   label=_legend_labels1[m])
        for m in ['OMSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_GPU']
    ]


    mosaic_A = [
        ['sweeps',   'sweeps',   'sweeps',   'sweeps',
         'cells',    'cells',    'cells',    'cells',
         'duration', 'duration', 'duration', 'duration'],
        ['tau_p',    'tau_p',    'tau_p',    'tau_r',
         'tau_r',    'tau_r',    'noise_p',  'noise_p',
         'noise_p',  'noise_r',  'noise_r',  'noise_r'],
    ]

    figA, axA = plt.subplot_mosaic(mosaic_A, figsize=(7, 3.5), dpi=300,
                                    gridspec_kw={'height_ratios': [3, 2]})

    for model in ['OMSI', 'CaImAn MCMC']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Sweeps'), 'Sweeps')
        if _tbl_len(subset) > 0:
            axA['sweeps'].plot(subset['Sweeps'], subset['Time'] / 60.0,
                               '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Sweeps'], subset['Time'])
            scaling_stats.append({'Experiment': 'Sweeps', 'Model': model,
                                   'Variable': 'Sweeps', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axA['sweeps'].set_xlabel('# sweeps')
    axA['sweeps'].set_ylabel('compute time (min)')
    axA['sweeps'].set_yscale('log')

    for model in ['CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'OMSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if _tbl_len(subset) > 0:
            axA['cells'].plot(subset['N_Cells'], subset['Time'] / 60.,
                              '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['N_Cells'], subset['Time'])
            scaling_stats.append({'Experiment': 'Cell_Scaling', 'Model': model,
                                   'Variable': 'N_Cells', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axA['cells'].set_xlabel('# cells')
    axA['cells'].set_ylabel('compute time (min)')
    axA['cells'].set_yscale('log')

    for model in ['CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'OMSI']:
        if model.startswith('CASCADE') and dur_cascade_tbl:
            src = dur_cascade_tbl
        else:
            src = combined
        m_rows = _tbl_filter(src, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Duration_Scaling'), 'Duration')
        if _tbl_len(subset) > 0:
            axA['duration'].plot(subset['Duration'] / 60., subset['Time'] / 60.,
                                 '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Duration'], subset['Time'])
            scaling_stats.append({'Experiment': 'Duration_Scaling', 'Model': model,
                                   'Variable': 'Duration', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})

    _extrap = {'CaImAn MCMC': 496.0, 'CASCADE_CPU': 5.0}
    for model, t_extrap in _extrap.items():
        color = COLORS.get(model, 'k')

        src = dur_cascade_tbl if model.startswith('CASCADE') and dur_cascade_tbl else combined
        m_rows = _tbl_filter(src, 'Model', model)
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Duration_Scaling'), 'Duration')
        if _tbl_len(subset) > 0:
            last_dur  = float(subset['Duration'][-1]) / 60.
            if last_dur >= 120.:
                continue  # real data already reaches the extrapolation target
            last_time = float(subset['Time'][-1]) / 60.
            axA['duration'].plot([last_dur, 120.], [last_time, t_extrap],
                                 '-', color=color)
        axA['duration'].plot(120., t_extrap, '.', color=color)

    axA['duration'].set_xlabel('recording duration (min)')
    axA['duration'].set_ylabel('compute time (min)')
    axA['duration'].set_yscale('log')
    axA['duration'].set_xticks([0, 60, 120])
    axA['duration'].set_xticklabels(['0', '60', '120'])

    for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'OMSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Tau_Sensitivity'), 'Tau')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Tau_Sensitivity'), 'Tau')
        if _tbl_len(subset) > 0:
            axA['tau_p'].plot(subset['Tau'][:-1], subset[prec_col][:-1], '.-',
                              color=COLORS.get(model, 'k'))
            axA['tau_r'].plot(subset['Tau'][:-1], subset[rec_col][:-1],  '.-',
                              color=COLORS.get(model, 'k'))

    for ax_key, xlabel, ylabel in [
        ('tau_p', r'$\tau$ (s)', 'Precision'),
        ('tau_r', r'$\tau$ (s)', 'Recall'),
    ]:
        axA[ax_key].set_xlabel(xlabel)
        axA[ax_key].set_ylabel(ylabel)
        axA[ax_key].set_ylim(0.45, 1.05)
        _set_three_ticks_x(axA[ax_key])

    _noise_cells = _load_noise_cells(data_dir)

    _prec_col_cell = 'Precision_window' if not USE_STRICT_ACCURACY else 'Precision'
    _rec_col_cell  = 'Recall_window'    if not USE_STRICT_ACCURACY else 'Recall'

    if _noise_cells is not None and _tbl_len(_noise_cells) > 0:
        _all_snr = _noise_cells['SNR'].astype(float)
        _snr_levels_sorted = np.sort(np.unique(_all_snr))[::-1]

        for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'OMSI']:
            _mask_m = _noise_cells['Model'] == model
            if _mask_m.sum() == 0:
                continue
            color = COLORS.get(model, 'k')
            snr_vals, med_p, mad_p, med_r, mad_r = [], [], [], [], []
            for snr_val in _snr_levels_sorted:
                _mask_sn = _mask_m & (_all_snr == snr_val)
                if _mask_sn.sum() < 2:
                    continue
                _py = _noise_cells[_prec_col_cell].astype(float)[_mask_sn]
                _ry = _noise_cells[_rec_col_cell].astype(float)[_mask_sn]
                snr_vals.append(snr_val)
                med_p.append(np.nanmedian(_py))
                mad_p.append(_mad(_py))
                med_r.append(np.nanmedian(_ry))
                mad_r.append(_mad(_ry))
            if len(snr_vals) >= 2:
                sv   = np.array(snr_vals)
                mp_  = np.array(med_p); sp_ = np.array(mad_p)
                mr_  = np.array(med_r); sr_ = np.array(mad_r)
                axA['noise_p'].plot(sv, mp_, '.-', color=color)
                axA['noise_p'].fill_between(sv, mp_ - sp_, mp_ + sp_,
                                             color=color, alpha=0.25, linewidth=0)
                axA['noise_r'].plot(sv, mr_, '.-', color=color)
                axA['noise_r'].fill_between(sv, mr_ - sr_, mr_ + sr_,
                                             color=color, alpha=0.25, linewidth=0)
    else:
        for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'OMSI']:
            m_rows = _tbl_filter(combined, 'Model', model)
            if _tbl_len(m_rows) == 0:
                continue
            subset = _tbl_filter(m_rows, 'Experiment', 'Noise_Sensitivity')
            if _tbl_len(subset) == 0 or 'SNR' not in subset:
                continue
            subset = _tbl_sort(subset, 'SNR')
            if _tbl_len(subset) > 0:
                axA['noise_p'].plot(subset['SNR'], subset[prec_col], '.-',
                                    color=COLORS.get(model, 'k'))
                axA['noise_r'].plot(subset['SNR'], subset[rec_col], '.-',
                                    color=COLORS.get(model, 'k'))

    for ax_key, xlabel, ylabel in [
        ('noise_p', 'SNR', 'Precision'),
        ('noise_r', 'SNR', 'Recall'),
    ]:
        axA[ax_key].set_xlabel(xlabel)
        axA[ax_key].set_ylabel(ylabel)
        axA[ax_key].set_ylim(-0.05, 1.05)
        axA[ax_key].set_xscale('log')

    figA.legend(handles=legend_handles, loc='upper center', ncol=5,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    print('\nScaling statistics:')
    print('{:<22} {:<12} {:<10} {:<8} {:<8} {:<12}'.format(
        'Experiment', 'Model', 'Variable', 'Lin R^2', 'Poly R^2', 'Fit'))
    print('-' * 80)
    for s in scaling_stats:
        print('{:<22} {:<12} {:<10} {:<8.4f} {:<8.4f} {:<12}'.format(
            s['Experiment'], s['Model'], s['Variable'],
            s['Lin_R2'], s['Poly_R2'], s['Conclusion']))

    figA.subplots_adjust(wspace=20., hspace=0.55, top=0.88)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure2A.{sfx}')
        figA.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved: {}'.format(out))
    plt.close(figA)


    mosaic_B = [
        ['fs_p',  'fs_r'     ],
        ['fs_fb', 'fs_cosmic'],
        ['casc',  'casc'     ],
    ]
    figB, axB = plt.subplot_mosaic(
        mosaic_B, figsize=(3.25, 4.5), dpi=300,
        gridspec_kw={'height_ratios': [2, 2, 3]},
    )

    for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'OMSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset_fs = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Fs_Sensitivity'), 'Fs')
        subset_fs = {k: v[np.array(subset_fs['Fs'], dtype=float) != 100.0]
                     for k, v in subset_fs.items()}
        if model.startswith('CASCADE'):
            subset_fs = _filter_cascade_shared_x(
                subset_fs, _tbl_filter(combined, 'Experiment', 'Fs_Sensitivity'), 'Fs')
        if _tbl_len(subset_fs) > 0:
            fb_fs = _fbeta(subset_fs[prec_col], subset_fs[rec_col])
            axB['fs_p'].plot(subset_fs['Fs'], subset_fs[prec_col], '.-',
                             color=COLORS.get(model, 'k'))
            axB['fs_r'].plot(subset_fs['Fs'], subset_fs[rec_col],  '.-',
                             color=COLORS.get(model, 'k'))
            axB['fs_fb'].plot(subset_fs['Fs'], fb_fs, '.-',
                              color=COLORS.get(model, 'k'))
            axB['fs_cosmic'].plot(subset_fs['Fs'], subset_fs['COSMIC'], '.-',
                                  color=COLORS.get(model, 'k'))

    axB['fs_p'].set_ylim([0.45, 1.05])
    axB['fs_r'].set_ylim([0.45, 1.05])
    axB['fs_fb'].set_ylim([0.45, 1.05])
    axB['fs_cosmic'].set_ylim(-0.05, 1.05)

    for ax_key, xlabel, ylabel in [
        ('fs_p',      'sample rate (Hz)', 'Precision'),
        ('fs_r',      'sample rate (Hz)', 'Recall'),
        ('fs_fb',     'sample rate (Hz)', r'$F_\beta$'),
        ('fs_cosmic', 'sample rate (Hz)', 'CosMIC'),
    ]:
        axB[ax_key].set_xlabel(xlabel)
        axB[ax_key].set_ylabel(ylabel)
        _set_three_ticks_x(axB[ax_key])

    _plot_cascade_comparison(axB['casc'], data_dir)

    figB.legend(handles=legend_handles1, loc='upper center', ncol=5,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)
    figB.tight_layout()

    figA.subplots_adjust(top=0.88)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure2B.{sfx}')
        figB.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved: {}'.format(out))
    plt.close(figB)


_PREC_COL = 'Precision' if USE_STRICT_ACCURACY else 'Precision_window'
_REC_COL  = 'Recall'    if USE_STRICT_ACCURACY else 'Recall_window'
_FBETA_COL = 'Fbeta'    if USE_STRICT_ACCURACY else 'Fbeta_window'


def _print_experiment(tbl, experiment, xcol, cols, models, x_scale=1.0, x_label=None,
                      exclude_x=(), drop_last=False, full_tbl=None, print_median=False,
                      show_mad=False):
    """
    Print one row per (model, x) for a benchmark experiment.

    Parameters
    ----------
    tbl : dict
        Columnar benchmark table.
    experiment : str
        Experiment name to filter on (e.g. 'Tau_Sensitivity').
    xcol : str
        Column used as the independent variable.
    cols : list of (str, str, str, float)
        (column name or 'Fbeta', header label, format spec, divisor) for each value to print.
    models : list of str
        Models to print, in order.
    x_scale : float, optional
        Divisor applied to x values before printing.
    x_label : str, optional
        Header for the x column; defaults to xcol.
    exclude_x : tuple of float, optional
        x values to skip, matching the figure.
    drop_last : bool, optional
        Drop the largest x value per model, matching the figure.
    full_tbl : dict, optional
        Table used to restrict CASCADE rows to x values shared with other methods.
    print_median : bool, optional
        After each model's rows, print the median ± MAD of each value across all x.
    show_mad : bool, optional
        Append the across-cell MAD (the '<col>_mad' column) to each value when
        available. 'Fbeta' then uses the median of per-cell F-beta rather than
        F-beta of the median precision and recall.
    """
    width  = 15 if (print_median or show_mad) else 12
    header = '  {:<14} {:>10}'.format('Model', x_label or xcol)
    for _, label, _, _ in cols:
        header += ' {:>{w}}'.format(label, w=width)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for model in models:
        subset = _tbl_sort(_tbl_filter(_tbl_filter(tbl, 'Model', model),
                                       'Experiment', experiment), xcol)
        if _tbl_len(subset) == 0:
            continue
        if exclude_x:
            keep = ~np.isin(np.array(subset[xcol], dtype=float), exclude_x)
            subset = {k: v[keep] for k, v in subset.items()}
        if model.startswith('CASCADE') and full_tbl is not None:
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(full_tbl, 'Experiment', experiment), xcol)
        if drop_last:
            subset = {k: v[:-1] for k, v in subset.items()}
        vals = np.full((_tbl_len(subset), len(cols)), np.nan)
        for i in range(_tbl_len(subset)):
            line = '  {:<14} {:>10.4g}'.format(model, float(subset[xcol][i]) / x_scale)
            for j, (col, _, fmt, scale) in enumerate(cols):
                if col == 'Fbeta' and show_mad and _FBETA_COL in subset \
                        and np.isfinite(float(subset[_FBETA_COL][i])):
                    col = _FBETA_COL
                if col == 'Fbeta':
                    val = float(_fbeta(subset[_PREC_COL][i], subset[_REC_COL][i]))
                else:
                    val = float(subset[col][i])
                vals[i, j] = val / scale
                cell = format(vals[i, j], fmt)
                mad_col = col + '_mad'
                if show_mad and mad_col in subset and np.isfinite(float(subset[mad_col][i])):
                    cell += ' ± ' + format(float(subset[mad_col][i]) / scale, fmt)
                line += ' {:>{w}}'.format(cell, w=width)
            print(line)
        if print_median and _tbl_len(subset) > 0:
            line = '  {:<14} {:>10}'.format(model, 'median')
            for j, (_, _, fmt, _) in enumerate(cols):
                cell = '{} ± {}'.format(format(float(np.nanmedian(vals[:, j])), fmt),
                                        format(float(_mad(vals[:, j])), fmt))
                line += ' {:>{w}}'.format(cell, w=width)
            print(line)


def print_stats(data_dir=_DEFAULT_DATA_DIR):
    """
    Print the values plotted in figure 2 without rendering the figure.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing benchmark result .npz files.
    """
    combined, dur_cascade_tbl = _load_benchmark_tables(data_dir)
    noise_cells = _load_noise_cells(data_dir)

    all_models = ['OMSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU']
    time_cols  = [('Time', 'Time (min)', '.3f', 60.0)]
    acc_cols   = [(_PREC_COL, 'Precision', '.3f', 1.0), (_REC_COL, 'Recall', '.3f', 1.0)]

    print('\n' + '=' * 72)
    print('FIGURE 2 STATISTICS')
    print('=' * 72)

    print('\n--- Compute time vs # sweeps ---')
    _print_experiment(combined, 'Sweeps', 'Sweeps', time_cols,
                      ['OMSI', 'CaImAn MCMC'], x_label='Sweeps')

    print('\n--- Compute time vs # cells ---')
    _print_experiment(combined, 'Cell_Scaling', 'N_Cells', time_cols,
                      all_models, x_label='Cells', full_tbl=combined)

    print('\n--- Compute time vs recording duration ---')
    # Match the figure: CASCADE duration rows come from the alternate directory when available.
    dur_tbl = combined
    if dur_cascade_tbl:
        not_cascade = np.array([not str(m).startswith('CASCADE') for m in combined['Model']], dtype=bool)
        dur_tbl = _tbl_concat([{k: v[not_cascade] for k, v in combined.items()}, dur_cascade_tbl])
    _print_experiment(dur_tbl, 'Duration_Scaling', 'Duration', time_cols,
                      all_models, x_scale=60.0, x_label='Dur (min)')

    print('\n--- Scaling fits (compute time) ---')
    print('  {:<18} {:<14} {:>8} {:>9}  {}'.format('Experiment', 'Model', 'Lin R^2', 'Poly R^2', 'Fit'))
    for experiment, xcol, models, src in [
        ('Sweeps',           'Sweeps',   ['OMSI', 'CaImAn MCMC'], combined),
        ('Cell_Scaling',     'N_Cells',  all_models,               combined),
        ('Duration_Scaling', 'Duration', all_models,               dur_tbl),
    ]:
        for model in models:
            subset = _tbl_sort(_tbl_filter(_tbl_filter(src, 'Model', model),
                                           'Experiment', experiment), xcol)
            if experiment == 'Cell_Scaling' and model.startswith('CASCADE') and _tbl_len(subset):
                subset = _filter_cascade_shared_x(
                    subset, _tbl_filter(combined, 'Experiment', experiment), xcol)
            if _tbl_len(subset) == 0:
                continue
            r2l, r2p, c = _fit_scaling(subset[xcol], subset['Time'])
            print('  {:<18} {:<14} {:>8.4f} {:>9.4f}  {}'.format(experiment, model, r2l, r2p, c))

    print('\n--- Accuracy vs tau ---')
    _print_experiment(combined, 'Tau_Sensitivity', 'Tau', acc_cols, all_models,
                      x_label='Tau (s)', drop_last=True, full_tbl=combined)

    print('\n--- Accuracy vs SNR (median ± MAD across cells) ---')
    if noise_cells is not None and _tbl_len(noise_cells) > 0:
        snr_all = noise_cells['SNR'].astype(float)
        print('  {:<14} {:>8} {:>18} {:>18}'.format('Model', 'SNR', 'Precision', 'Recall'))
        for model in all_models:
            mask_m = noise_cells['Model'] == model
            for snr in np.sort(np.unique(snr_all[mask_m])):
                mask = mask_m & (snr_all == snr)
                if mask.sum() < 2:
                    continue
                p = noise_cells[_PREC_COL].astype(float)[mask]
                r = noise_cells[_REC_COL].astype(float)[mask]
                print('  {:<14} {:>8.4g} {:>18} {:>18}'.format(
                    model, snr,
                    '{:.3f} ± {:.3f}'.format(np.nanmedian(p), _mad(p)),
                    '{:.3f} ± {:.3f}'.format(np.nanmedian(r), _mad(r))))
    else:
        _print_experiment(combined, 'Noise_Sensitivity', 'SNR', acc_cols, all_models)

    print('\n--- Accuracy vs sample rate (median ± MAD across cells where available; '
          'last row per model: median ± MAD across Fs) ---')
    _print_experiment(combined, 'Fs_Sensitivity', 'Fs',
                      acc_cols + [('Fbeta', 'F_beta', '.3f', 1.0), ('COSMIC', 'CosMIC', '.3f', 1.0)],
                      all_models, x_label='Fs (Hz)', exclude_x=(100.0,), full_tbl=combined,
                      print_median=True, show_mad=True)

    print('\n--- CASCADE 7.5 Hz vs 30 Hz (median ± MAD across cells) ---')
    npz_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        for key, label in [('fb', 'F_beta'), ('cosmic', 'CosMIC')]:
            for suffix, fs_label in [('7', '7.5 Hz'), ('30', '30 Hz')]:
                v = d[f'{key}_{suffix}']
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                print('  {:<8} {:<8} {:.3f} ± {:.3f}  (n={})'.format(
                    label, fs_label, np.median(v), _mad(v), len(v)))
    else:
        print('  No data at {} (run --mode cascade-samplerate).'.format(npz_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Figure 2: scaling and sensitivity benchmarks'
    )
    parser.add_argument('--mode', required=True,
                        choices=['test', 'plot', 'print', 'noise-cells', 'cascade-samplerate',
                                 'fs-sensitivity'],
                        help='"test" runs all benchmarks; "plot" generates the figure; '
                             '"print" prints the plotted values without rendering; '
                             '"noise-cells" runs only the noise sensitivity benchmark and writes '
                             'benchmark_noise_sensitivity_cells.npz without touching other result files; '
                             '"cascade-samplerate" runs only the CASCADE 7.5Hz-vs-30Hz comparison '
                             'and writes cascade_7p5_vs_30hz_data.npz without touching other result files; '
                             '"fs-sensitivity" re-runs only the sample-rate sweep and replaces the '
                             'Fs_Sensitivity rows of benchmark_params_partial.npz (tau rows are kept)')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument('--no-omsi',   action='store_true', help='Skip OMSI')
    parser.add_argument('--no-matlab',  action='store_true', help='Skip MATLAB')
    parser.add_argument('--no-oasis',   action='store_true', help='Skip OASIS')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    args = parser.parse_args()

    with no_power_throttling(verbose=True):
        if args.mode == 'test':
            run_test(
                data_dir    = args.data_dir,
                run_omsi   = not args.no_omsi,
                run_matlab  = not args.no_matlab,
                run_oasis   = not args.no_oasis,
                run_cascade = not args.no_cascade,
            )
        elif args.mode == 'noise-cells':
            os.makedirs(args.data_dir, exist_ok=True)
            print('=== Noise sensitivity cell-level benchmark (cells_only) ===')
            benchmark_noise_sensitivity(
                args.data_dir,
                run_oasis   = not args.no_oasis,
                run_matlab  = not args.no_matlab,
                run_mine    = not args.no_omsi,
                run_cascade = not args.no_cascade,
                cells_only  = True,
            )
        elif args.mode == 'cascade-samplerate':
            os.makedirs(args.data_dir, exist_ok=True)
            print('=== CASCADE 7.5 Hz vs 30 Hz comparison ===')
            benchmark_cascade_sample_rate(args.data_dir, run_cascade=not args.no_cascade)
        elif args.mode == 'fs-sensitivity':
            os.makedirs(args.data_dir, exist_ok=True)
            print('=== Frame-rate sensitivity benchmark (fs only) ===')
            benchmark_params(
                args.data_dir,
                run_oasis   = not args.no_oasis,
                run_matlab  = not args.no_matlab,
                run_mine    = not args.no_omsi,
                run_cascade = not args.no_cascade,
                experiments = ('fs',),
            )
        elif args.mode == 'print':
            print_stats(data_dir=args.data_dir)
        else:
            plot_figure(data_dir=args.data_dir)
