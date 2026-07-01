# -*- coding: utf-8 -*-
"""
figures/characterize_optimizations.py

Benchmarks fMCSI optimizer settings and initialization strategies via parameter sweeps on synthetic data.

Functions
---------
_init_tau
    Estimate AR time constants from a fluorescence trace.
_default_T_supp
    Compute default support length for the exponential filter basis.
_build_T_supp_grid
    Build a geometric grid of T_supp values to sweep.
run_T_supp_sweep
    Run sweep over T_supp values and save results.
plot_T_supp_sweep
    Plot saved T_supp sweep results.
_fbeta
    Compute F-beta score from precision and recall.
_sp_peaks
    Detect spike peaks from a continuous spike signal.
_nnls_init
    Compute NNLS-based initialization for spike deconvolution.
_foopsi_init
    Compute FOOPSI-based initialization for spike deconvolution.
run_init_comparison
    Run NNLS vs. FOOPSI init comparison and save results.
plot_init_comparison
    Plot saved init comparison results.
_make_foopsi_init
    Build a FOOPSI-initialized fMCSI sample dict.
run_fmcsi_init_comparison
    Run fMCSI with NNLS vs. FOOPSI init and save results.
plot_fmcsi_init_comparison
    Plot saved fMCSI init comparison results.
plot_combined_init
    Plot combined init comparison figure.
_build_tol_grid
    Build a geometric grid of tolerance values to sweep.
_run_tol_sweep
    Run a tolerance parameter sweep and collect results.
_save_tol_sweep
    Save tolerance sweep results to .npz.
run_conv_tol_sweep
    Run convergence tolerance sweep and save results.
run_burn_tol_sweep
    Run burn-in tolerance sweep and save results.
_plot_tol_sweep
    Plot a saved tolerance sweep .npz file.
plot_conv_tol_sweep
    Plot saved convergence tolerance sweep.
plot_burn_tol_sweep
    Plot saved burn-in tolerance sweep.
plot_combined_opt
    Plot combined optimization parameter sweep figure.
_dff_snr
    Estimate SNR of a dF/F trace.
run_snr_filter_sweep
    Run SNR filter sweep and save per-cell accuracy.
plot_snr_filter_sweep
    Plot saved SNR filter sweep results.
run_snr_threshold_sweep
    Run SNR threshold sweep across synthetic populations.
plot_snr_threshold_sweep
    Plot saved SNR threshold sweep results.
_snr_get_sensor
    Map a dataset name to a calcium sensor label.
print_snr_stats
    Print SNR statistics by sensor for figure4 datasets.


DMM, March 2026
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import lfilter as _lfilter, find_peaks as _find_peaks
from scipy.optimize import minimize as _minimize

import OMSI
import OMSI.helpers as helpers
from OMSI.sampler import _build_ef_nb
from OMSI.get_init_sample import (
    get_init_sample,
    _get_sn, _estimate_time_constants, _ar_kernel, _block_nnls_deconv,
)
from simulation_helpers import generate_synthetic_data

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'opt'
)

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

np.random.seed(7)

_N_CELLS  = 200
_DURATION = 2400.
_FS       = 30.0
_TAU      = 1.2
_COLOR    = '#4C72B0'


def _init_tau(Y_cell, fs, p=2):
    """Estimate AR time constants from an observed fluorescence trace.

    Parameters
    ----------
    Y_cell : array_like
        Single-cell dF/F trace.
    fs : float
        Sampling rate in Hz.
    p : int, optional
        AR model order. Default is 2.

    Returns
    -------
    tau : ndarray
        Time constants in frames.
    gr : ndarray
        Corresponding AR roots, clipped to (1e-10, 0.998).
    diff_gr : float
        Difference between the two AR roots.
    """
    params = {'f': fs, 'p': p, 'defg': [0.6, 0.95]}
    try:
        SAM = get_init_sample(Y_cell, params)
        g   = np.atleast_1d(SAM['g']).flatten()
        gr  = np.sort(np.real(np.roots(np.concatenate(([1.0], -g)))))
        gr  = np.clip(gr, 1e-10, 0.998)
        tau = -1.0 / np.log(gr)
        if p == 1:
            tau[0] = np.inf
        return tau, gr, float(gr[1] - gr[0])
    except Exception:
        gr  = np.array([0.6, 0.95])
        tau = -1.0 / np.log(gr)
        return tau, gr, float(gr[1] - gr[0])


def _default_T_supp(tau, diff_gr, T, p=2, prec=1e-2):
    """Compute the default support length for the exponential filter basis.

    Parameters
    ----------
    tau : ndarray
        AR time constants in frames.
    diff_gr : float
        Difference between AR roots.
    T : int
        Total number of frames.
    p : int, optional
        AR model order. Default is 2.
    prec : float, optional
        Precision threshold for truncating the filter. Default is 1e-2.

    Returns
    -------
    int
        Number of frames in the default support.
    """
    t_arr = np.arange(T + 1, dtype=np.float64)
    _, ef_d, _, _, _ = _build_ef_nb(tau, diff_gr, t_arr, T, p, prec)

    return len(ef_d)


def _build_T_supp_grid(default_supp, T, n_shorter=8, n_longer=8):
    """Build a geometric grid of T_supp values spanning below and above the default.

    Parameters
    ----------
    default_supp : int
        Default support length to center the grid around.
    T : int
        Maximum frame count (full-length reference point).
    n_shorter : int, optional
        Number of grid points below the default. Default is 8.
    n_longer : int, optional
        Number of grid points above the default. Default is 8.

    Returns
    -------
    list of int
        Sorted unique T_supp values including default, shorter, longer, and T.
    """
    shorter = np.round(
        np.geomspace(0.01, default_supp - 1, n_shorter)
    ).astype(int)
    longer = np.round(
        np.geomspace(default_supp + 1, T - 1, n_longer)
    ).astype(int)
    grid = np.unique(np.concatenate([shorter, [default_supp], longer, [T]]))
    return grid.tolist()


def run_T_supp_sweep(data_dir):
    """Run a sweep over T_supp values, benchmark fMCSI on a synthetic population, and save results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'T_supp_sweep.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s)...'.format(
              _N_CELLS, _DURATION, _FS, _TAU))
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_DURATION, tau=_TAU
    )
    true_events = [helpers.make_event_ground_truth(s, _TAU) for s in true_spikes]
    n_frames = dff.shape[1]

    tau_rep, gr_rep, diff_gr_rep = _init_tau(dff[0], _FS)
    default_supp = _default_T_supp(tau_rep, diff_gr_rep, n_frames)
    print('  Tau (frames): {:.2f}, {:.2f}  '
          'gr: {:.4f}, {:.4f}'.format(
              tau_rep[0], tau_rep[1], gr_rep[0], gr_rep[1]))
    print('  Default T_supp (prec=1e-2): {} / {} frames'.format(
        default_supp, n_frames))

    T_supp_grid = _build_T_supp_grid(default_supp, n_frames)
    print('  Sweep ({} values): {}'.format(len(T_supp_grid), T_supp_grid))

    rows = []
    for ts in T_supp_grid:
        label = 'T' if ts == n_frames else str(ts)
        print('\n  T_supp={} ...'.format(ts))
        params = {
            'f':         _FS,
            'p':         2,
            'auto_stop': True,
            'upd_gam':   0,
            'T_supp':    ts,
        }
        try:
            t0  = time.time()
            res = OMSI.deconv(dff, params=params, benchmark=True)
            elapsed = time.time() - t0

            per_cell_t = res['optim_times_per_cell']
            nsweeps    = res['optim_nsamples']
            pred       = res['optim_spikes']

            prec_s, rec_s, _ = helpers.compute_accuracy_strict(true_spikes, pred)
            _,      _,    f1_e = helpers.compute_accuracy_window(true_events, pred)
            cosmic     = helpers.compute_cosmic(true_spikes, pred, _FS)
            fb = np.array([_fbeta(float(prec_s[i]), float(rec_s[i]))
                            for i in range(len(prec_s))])

            rows.append({
                'T_supp':          ts,
                'is_default':      ts == default_supp,
                'is_full':         ts == n_frames,
                'total_time':      elapsed,
                'mean_time':       float(np.mean(per_cell_t)),
                'std_time':        float(np.std(per_cell_t, ddof=1)),
                'mean_nsweeps':    float(np.mean(nsweeps)),
                'std_nsweeps':     float(np.std(nsweeps, ddof=1)),
                'mean_f1_window':  float(np.mean(fb)),
                'std_f1_window':   float(np.std(fb, ddof=1)),
                'mean_f1_event':   float(np.mean(f1_e)),
                'mean_cosmic':     float(np.mean(cosmic)),
                'std_cosmic':      float(np.std(cosmic, ddof=1)),
            })
            print('    Total={:.1f}s  '
                  'mean_cell={:.3f}s  '
                  'mean_sweeps={:.1f}  '
                  'F_beta={:.3f}  '
                  'CosMIC={:.3f}'.format(
                      elapsed, np.mean(per_cell_t), np.mean(nsweeps),
                      np.mean(fb), np.mean(cosmic)))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))

    if not rows:
        print('No results collected.')
        return

    np.savez(
        out_path,
        T_supp       = np.array([r['T_supp']          for r in rows]),
        mean_time    = np.array([r['mean_time']        for r in rows]),
        std_time     = np.array([r['std_time']         for r in rows]),
        mean_nsweeps = np.array([r['mean_nsweeps']     for r in rows]),
        std_nsweeps  = np.array([r['std_nsweeps']      for r in rows]),
        mean_f1      = np.array([r['mean_f1_window']   for r in rows]),
        std_f1       = np.array([r['std_f1_window']    for r in rows]),
        mean_cosmic  = np.array([r['mean_cosmic']      for r in rows]),
        std_cosmic   = np.array([r['std_cosmic']       for r in rows]),
        default_supp = np.array([default_supp]),
        n_frames     = np.array([n_frames]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_T_supp_sweep(data_dir):
    """Plot time-per-cell and F_beta vs. T_supp from a saved sweep .npz file.

    Parameters
    ----------
    data_dir : str
        Directory containing the T_supp_sweep.npz file.
    """
    out_path = os.path.join(data_dir, 'T_supp_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode test first.')

    d            = np.load(out_path)
    T_supp       = d['T_supp'].astype(float)
    mt           = d['mean_time']
    st           = d['std_time']
    mf1          = d['mean_f1']
    sf1          = d['std_f1']
    mcos         = d['mean_cosmic']
    scos         = d['std_cosmic']
    default_supp = int(d['default_supp'][0])
    n_frames     = int(d['n_frames'][0])

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)

    for ax, y, yerr, ylabel in [
        (axes[0], mt,  st,  'time per cell (sec)'),
        (axes[1], mf1, sf1, '$F_\\beta$'),
    ]:
        mask = np.arange(len(T_supp))[1:-1]
        mask = np.hstack([mask[0:4], mask[5], mask[7:]]).astype(int)
        print(mask)
        ax.plot(T_supp[mask] * (1.0 / _FS), y[mask], '.-', color=_COLOR, zorder=3)
        if ylabel == 'time per cell (sec)':
            ax.fill_between(T_supp[mask] * (1.0 / _FS), y[mask] - yerr[mask], y[mask] + yerr[mask],
                            color=_COLOR, alpha=0.25, linewidth=0)
        ax.axvline(default_supp * (1.0 / _FS), color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6, label='default')
        ax.set_xlabel('$T_{supp}$ (sec)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_ylim([0, 500])

    axes[1].set_ylim(0, 0.51)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'T_supp_sweep.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


_BETA = 0.5

def _fbeta(precision, recall, beta=_BETA):
    """Compute the F-beta score from precision and recall.

    Parameters
    ----------
    precision : float
        Precision value in [0, 1].
    recall : float
        Recall value in [0, 1].
    beta : float, optional
        Beta weight. Default is _BETA (0.5).

    Returns
    -------
    float
        F-beta score, or 0.0 if denominator is zero.
    """
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

_NNLS_COLOR   = '#4C72B0'
_FOOPSI_COLOR = 'tab:red'
_DFF_ALPHA    = 0.35
_INIT_DURATION = 300.0
_TRACE_WINDOW  = 10.0


def _sp_peaks(sp, fs, thresh_frac=0.15, min_gap_s=0.05):
    """Detect spike peak indices from a continuous spike amplitude signal.

    Parameters
    ----------
    sp : array_like or None
        Continuous spike signal.
    fs : float
        Sampling rate in Hz.
    thresh_frac : float, optional
        Fraction of max amplitude used as peak threshold. Default is 0.15.
    min_gap_s : float, optional
        Minimum gap between peaks in seconds. Default is 0.05.

    Returns
    -------
    ndarray
        Array of peak frame indices as floats.
    """
    if sp is None or np.max(sp) < 1e-12:
        return np.array([], dtype=float)
    thresh = thresh_frac * np.max(sp)
    min_gap = max(1, int(min_gap_s * fs))
    peaks, _ = _find_peaks(sp, height=thresh, distance=min_gap)
    return peaks.astype(float)


def _nnls_init(Y_cell, fs, p=2):
    """Compute NNLS-based spike and calcium initialization for a single cell.

    Parameters
    ----------
    Y_cell : array_like
        Single-cell dF/F trace.
    fs : float
        Sampling rate in Hz.
    p : int, optional
        AR model order. Default is 2.

    Returns
    -------
    sp : ndarray
        Estimated spike amplitudes per frame.
    calcium : ndarray
        Reconstructed calcium trace.
    """
    sn = _get_sn(Y_cell, [0.25, 0.5])
    g  = _estimate_time_constants(Y_cell, p, sn)
    h  = _ar_kernel(g, len(Y_cell))
    sp = _block_nnls_deconv(Y_cell, h, len(Y_cell))
    calcium = _lfilter([1.0], np.concatenate(([1.0], -g)), sp)
    return sp, calcium


def _foopsi_init(Y_cell, fs, tau=_TAU):
    """Compute FOOPSI-based spike and calcium initialization for a single cell.

    Parameters
    ----------
    Y_cell : array_like
        Single-cell dF/F trace.
    fs : float
        Sampling rate in Hz.
    tau : float, optional
        Calcium decay time constant in seconds. Default is _TAU.

    Returns
    -------
    s : ndarray
        Non-negative spike signal estimated via L-BFGS-B.
    calcium : ndarray
        Reconstructed calcium trace from the spike signal.
    """
    T   = len(Y_cell)
    sn  = _get_sn(Y_cell, [0.25, 0.5])
    g   = np.exp(-1.0 / (fs * tau))
    lam = sn

    def _fwd(s):
        """Apply forward AR filter."""
        return _lfilter([1.0], [1.0, -g], s)

    def _adj(v):
        """Apply adjoint AR filter."""
        return _lfilter([1.0], [1.0, -g], v[::-1])[::-1]

    def _obj(s):
        """Evaluate L1-penalized least-squares objective and gradient."""
        c   = _fwd(s)
        res = c - Y_cell
        f   = 0.5 * float(np.dot(res, res)) + lam * float(s.sum())
        grad = _adj(res) + lam
        return f, grad

    result = _minimize(
        _obj, np.zeros(T), method='L-BFGS-B', jac=True,
        bounds=[(0.0, None)] * T,
        options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6},
    )
    s = np.maximum(result.x, 0.0)
    return s, _fwd(s)


def run_init_comparison(data_dir):
    """Run NNLS and FOOPSI initializations on a synthetic population and save correlation results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'init_comparison.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s)...'.format(
              _N_CELLS, _INIT_DURATION, _FS, _TAU))
    dff, true_spikes, clean_traces, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_INIT_DURATION, tau=_TAU
    )

    r_nnls_true   = np.zeros(_N_CELLS)
    r_foopsi_true = np.zeros(_N_CELLS)
    r_cross       = np.zeros(_N_CELLS)
    nnls_times    = np.zeros(_N_CELLS)
    foopsi_times  = np.zeros(_N_CELLS)
    firing_rates  = np.array([len(s) / _INIT_DURATION for s in true_spikes])

    nnls_calcium_store   = []
    foopsi_calcium_store = []
    nnls_sp_store        = []
    foopsi_sp_store      = []

    print('Running NNLS and FOOPSI init on each cell...')
    for i in range(_N_CELLS):
        t0 = time.perf_counter(); sp_n, ca_n = _nnls_init(dff[i], _FS);   nnls_times[i]   = time.perf_counter() - t0
        t0 = time.perf_counter(); sp_f, ca_f = _foopsi_init(dff[i], _FS); foopsi_times[i] = time.perf_counter() - t0

        nnls_calcium_store.append(ca_n)
        foopsi_calcium_store.append(ca_f)
        nnls_sp_store.append(sp_n)
        foopsi_sp_store.append(sp_f)

        true_ca = clean_traces[i]
        r_nnls_true[i]   = float(np.corrcoef(ca_n,   true_ca)[0, 1])
        r_foopsi_true[i] = float(np.corrcoef(ca_f,   true_ca)[0, 1])
        r_cross[i]       = float(np.corrcoef(ca_n,   ca_f)[0, 1])

        if (i + 1) % 10 == 0:
            print('  {}/{}  '
                  'nnls={:.1f}ms  '
                  'foopsi={:.1f}ms  '
                  'r_cross={:.3f}'.format(
                      i + 1, _N_CELLS,
                      np.mean(nnls_times[:i + 1]) * 1e3,
                      np.mean(foopsi_times[:i + 1]) * 1e3,
                      np.mean(r_cross[:i + 1])))

    np.savez(
        out_path,
        dff              = dff,
        clean_traces     = clean_traces,
        nnls_calcium     = np.array(nnls_calcium_store),
        foopsi_calcium   = np.array(foopsi_calcium_store),
        nnls_sp          = np.array(nnls_sp_store, dtype=np.float32),
        foopsi_sp        = np.array(foopsi_sp_store, dtype=np.float32),
        true_spikes      = np.array(true_spikes, dtype=object),
        r_nnls_true      = r_nnls_true,
        r_foopsi_true    = r_foopsi_true,
        r_cross          = r_cross,
        nnls_times       = nnls_times,
        foopsi_times     = foopsi_times,
        firing_rates     = firing_rates,
        n_frames         = np.array([dff.shape[1]]),
        fs               = np.array([_FS]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_init_comparison(data_dir):
    """Plot example traces and timing scatter from a saved init comparison .npz file.

    Parameters
    ----------
    data_dir : str
        Directory containing the init_comparison.npz file.
    """
    out_path = os.path.join(data_dir, 'init_comparison.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode init-test first.')

    d = np.load(out_path, allow_pickle=True)
    dff          = d['dff']
    clean_traces = d['clean_traces']
    nnls_sp      = d['nnls_sp'].astype(np.float64)
    foopsi_sp    = d['foopsi_sp'].astype(np.float64)
    true_spikes  = d['true_spikes']
    r_cross      = d['r_cross']
    nnls_times   = d['nnls_times']
    foopsi_times = d['foopsi_times']
    firing_rates = d['firing_rates']
    fs           = float(d['fs'][0])
    n_cells      = dff.shape[0]

    example_cells = [49, 0, 43]
    win_offsets   = [0, 0, 0]
    win_frames    = int(_TRACE_WINDOW * fs)

    fig = plt.figure(figsize=(7, 4), dpi=300)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.38)
    gs_left  = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0], hspace=0.42)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], hspace=0.55)
    axd = {
        't0':      fig.add_subplot(gs_left[0]),
        't1':      fig.add_subplot(gs_left[1]),
        't2':      fig.add_subplot(gs_left[2]),
        'time':    fig.add_subplot(gs_right[0]),
        'r_cross': fig.add_subplot(gs_right[1]),
    }

    def _norm(x):
        """Normalize array to [0, 1]."""
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo + 1e-12)

    for row_idx, (ax_key, cell_idx, win_start) in enumerate(
            zip(['t0', 't1', 't2'], example_cells, win_offsets)):
        ax = axd[ax_key]
        s  = win_start
        e  = s + win_frames
        t_ax = np.arange(win_frames) / fs

        gt   = clean_traces[cell_idx, s:e]
        spn  = nnls_sp[cell_idx, s:e]
        spf  = foopsi_sp[cell_idx, s:e]
        t_start_s = s / fs
        t_end_s   = t_start_s + _TRACE_WINDOW
        sp_true = true_spikes[cell_idx]
        sp_true = sp_true[(sp_true >= t_start_s) & (sp_true < t_end_s)] - t_start_s

        gt_lo, gt_hi = gt.min(), gt.max()
        def _scale_gt(x):
            """Scale ground-truth trace to [0, 1] using its own min/max."""
            return (x - gt_lo) / (gt_hi - gt_lo + 1e-12)

        def _scale_sp(x):
            """Scale spike signal to [0, 1] by its maximum."""
            hi = np.max(x) if np.max(x) > 1e-12 else 1.0
            return x / hi

        ax.plot(t_ax, _scale_gt(gt), color='k', alpha=0.25, lw=1.0,
                label='ground truth', zorder=2)
        ax.plot(t_ax, _scale_sp(spf), color=_FOOPSI_COLOR, lw=0.8,
                label='FOOPSI', zorder=3, alpha=0.5)
        ax.plot(t_ax, _scale_sp(spn), color=_NNLS_COLOR,   lw=0.8,
                label='NNLS',   zorder=3, alpha=0.5)


        ax.eventplot(sp_true, lineoffsets=1.18, linelengths=0.18,
                     colors='k', linewidths=0.8)

        ax.set_xlim(0, _TRACE_WINDOW)
        ax.set_ylim(-0.05, 1.42)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlabel('time (sec)' if row_idx == 2 else '')

        ax.text(0.01, 0.97, str(row_idx + 1),
                transform=ax.transAxes, fontsize=5.5, fontweight='bold',
                va='top', ha='left', color='k')
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax_t = axd['time']

    bins = np.linspace(0, 200, 30)

    ax_t.scatter(foopsi_times * 1e3, nnls_times * 1e3, color='k', s=1)

    ax_t.set_xlabel('FOOPSI time per cell (msec)')
    ax_t.set_ylabel('NNLS time per cell (msec)')

    ax_t.axis('equal')
    ax_t.set_xlim(bottom=0)
    ax_t.set_ylim(bottom=0)

    ax_h = axd['r_cross']
    r_cross_valid = r_cross[np.isfinite(r_cross)]
    ax_h.hist(r_cross_valid, bins=20, color='#555555', edgecolor='white', linewidth=0.3)

    ax_h.set_xlabel('correlation')
    ax_h.set_ylabel('cells')
    ax_h.set_xlim([0,1])

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'init_comparison.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def _make_foopsi_init(Y_cell, fs, tau=_TAU, p=2):
    """Build an fMCSI-compatible sample dict initialized from the FOOPSI spike estimate.

    Parameters
    ----------
    Y_cell : array_like
        Single-cell dF/F trace.
    fs : float
        Sampling rate in Hz.
    tau : float, optional
        Calcium decay time constant in seconds. Default is _TAU.
    p : int, optional
        AR model order. Default is 2.

    Returns
    -------
    dict
        fMCSI sample dict with FOOPSI-derived spiketimes_, lam_, and C_in fields.
    """
    init_params = {'f': fs, 'p': p, 'defg': [0.6, 0.95]}
    SAM = dict(get_init_sample(Y_cell, init_params))

    sp_signal, ca_f = _foopsi_init(Y_cell, fs, tau)

    T      = len(Y_cell)
    sp_max = float(np.max(sp_signal)) if sp_signal.size > 0 else 0.0
    if sp_max > 0:
        indices = np.where(sp_signal > 0.15 * sp_max)[0]
    else:
        indices = np.array([], dtype=int)

    spiketimes_ = indices.astype(float) + np.random.rand(len(indices)) - 0.5
    oob = spiketimes_ >= T
    spiketimes_[oob] = 2.0 * T - spiketimes_[oob]

    SAM['spiketimes_'] = spiketimes_
    SAM['lam_']        = len(spiketimes_) / float(T)
    SAM['C_in']        = float(max(ca_f[0] - SAM['b_'], 0.0))
    return SAM


def run_fmcsi_init_comparison(data_dir):
    """Run fMCSI with NNLS and FOOPSI inits on a synthetic population and save accuracy results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'fmcsi_init_comparison.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s)...'.format(
              _N_CELLS, _INIT_DURATION, _FS, _TAU))
    dff, true_spikes, clean_traces, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_INIT_DURATION, tau=_TAU
    )
    firing_rates = np.array([len(s) / _INIT_DURATION for s in true_spikes])

    print('Pre-computing FOOPSI inits...')
    foopsi_inits = [_make_foopsi_init(dff[i], _FS) for i in range(_N_CELLS)]
    print('  Done ({} inits).'.format(_N_CELLS))

    base_params = {'f': _FS, 'p': 2, 'auto_stop': True}

    print('Running fMCSI with NNLS init (full population)...')
    p_n = dict(base_params, init=None)
    r_n = OMSI.deconv(dff, params=p_n, true_spikes=true_spikes, benchmark=True)
    nnls_times    = r_n['optim_times_per_cell']
    nnls_nsamples = r_n['optim_nsamples']
    nnls_prob     = r_n['optim_prob']
    if r_n['optim_precision'] is not None:
        nnls_fb = np.array([
            _fbeta(float(r_n['optim_precision'][i]), float(r_n['optim_recall'][i]))
            for i in range(_N_CELLS)
        ])
    else:
        nnls_fb = np.full(_N_CELLS, np.nan)

    print('Running fMCSI with FOOPSI init (full population)...')
    p_f = dict(base_params, init=foopsi_inits)
    r_f = OMSI.deconv(dff, params=p_f, true_spikes=true_spikes, benchmark=True)
    foopsi_times    = r_f['optim_times_per_cell']
    foopsi_nsamples = r_f['optim_nsamples']
    foopsi_prob     = r_f['optim_prob']
    if r_f['optim_precision'] is not None:
        foopsi_fb = np.array([
            _fbeta(float(r_f['optim_precision'][i]), float(r_f['optim_recall'][i]))
            for i in range(_N_CELLS)
        ])
    else:
        foopsi_fb = np.full(_N_CELLS, np.nan)

    print('NNLS   — mean/cell: {:.2f}s  '
          'mean samples: {}  '
          'mean Fb: {:.3f}'.format(
              np.mean(nnls_times), int(np.mean(nnls_nsamples)), np.nanmean(nnls_fb)))
    print('FOOPSI — mean/cell: {:.2f}s  '
          'mean samples: {}  '
          'mean Fb: {:.3f}'.format(
              np.mean(foopsi_times), int(np.mean(foopsi_nsamples)), np.nanmean(foopsi_fb)))

    np.savez(
        out_path,
        dff              = dff,
        clean_traces     = clean_traces,
        nnls_prob        = nnls_prob.astype(np.float32),
        foopsi_prob      = foopsi_prob.astype(np.float32),
        nnls_times       = nnls_times,
        foopsi_times     = foopsi_times,
        nnls_fb          = nnls_fb,
        foopsi_fb        = foopsi_fb,
        nnls_nsamples    = nnls_nsamples,
        foopsi_nsamples  = foopsi_nsamples,
        firing_rates     = firing_rates,
        true_spikes      = np.array(true_spikes, dtype=object),
        n_frames         = np.array([dff.shape[1]]),
        fs               = np.array([_FS]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_fmcsi_init_comparison(data_dir):
    """Plot example traces, timing histograms, and F_beta histograms from a saved fMCSI init comparison.

    Parameters
    ----------
    data_dir : str
        Directory containing the fmcsi_init_comparison.npz file.
    """
    out_path = os.path.join(data_dir, 'fmcsi_init_comparison.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode conv-test first.')

    _FOOPSI_COLOR = 'tab:red'

    d = np.load(out_path, allow_pickle=True)
    dff            = d['dff']
    clean_traces   = d['clean_traces']
    nnls_prob   = d['nnls_prob'].astype(np.float64)
    foopsi_prob = d['foopsi_prob'].astype(np.float64)
    nnls_times     = d['nnls_times']
    foopsi_times   = d['foopsi_times']
    nnls_fb        = d['nnls_fb']
    foopsi_fb      = d['foopsi_fb']
    nnls_nsamples  = d['nnls_nsamples']
    foopsi_nsamples = d['foopsi_nsamples']
    firing_rates   = d['firing_rates']
    true_spikes    = d['true_spikes']
    fs             = float(d['fs'][0])
    n_cells        = dff.shape[0]

    example_cells = [49, 0, 43]
    win_frames    = int(_TRACE_WINDOW * fs)

    fig = plt.figure(figsize=(6.5, 4), dpi=300)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.38)
    gs_left  = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0], hspace=0.42)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], hspace=0.55)
    axd = {
        't0':      fig.add_subplot(gs_left[0]),
        't1':      fig.add_subplot(gs_left[1]),
        't2':      fig.add_subplot(gs_left[2]),
        'time':    fig.add_subplot(gs_right[0]),
        'f1':      fig.add_subplot(gs_right[1]),
    }

    for row_idx, (ax_key, cell_idx) in enumerate(
            zip(['t0', 't1', 't2'], example_cells)):
        ax  = axd[ax_key]
        T   = min(win_frames, dff.shape[1])
        t_ax = np.arange(T) / fs

        gt  = clean_traces[cell_idx, :T]
        pn  = nnls_prob[cell_idx, :T]
        pf  = foopsi_prob[cell_idx, :T]
        sp_true = true_spikes[cell_idx]
        sp_true = sp_true[sp_true < _TRACE_WINDOW]

        gt_lo, gt_hi = gt.min(), gt.max()
        def _scale_gt(x):
            """Scale ground-truth trace to [0, 1] using its own min/max."""
            return (x - gt_lo) / (gt_hi - gt_lo + 1e-12)

        def _scale_prob(x):
            """Scale probability signal to [0, 1] by its maximum."""
            hi = np.max(x) if np.max(x) > 1e-12 else 1.0
            return x / hi

        ax.plot(t_ax, _scale_gt(gt),    color='k', alpha=0.25, lw=1.0,
                label='ground truth', zorder=2)
        ax.plot(t_ax, _scale_prob(pf),  color=_FOOPSI_COLOR, lw=0.8,
                label='FOOPSI-initialized', zorder=3, alpha=0.5)
        ax.plot(t_ax, _scale_prob(pn),  color=_NNLS_COLOR,   lw=0.8,
                label='NNLS-initialized',   zorder=3, alpha=0.5)
        ax.eventplot(sp_true, lineoffsets=1.18, linelengths=0.18,
                     colors='k', linewidths=0.8)

        ax.set_xlim(0, _TRACE_WINDOW)
        ax.set_ylim(-0.05, 1.42)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlabel('time (sec)' if row_idx == 2 else '')

        ax.text(0.01, 0.97, str(row_idx + 1),
                transform=ax.transAxes, fontsize=5.5, fontweight='bold',
                va='top', ha='left', color='k')
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax_t = axd['time']
    bins = np.linspace(0, max(nnls_times.max(), foopsi_times.max()) * 1.05, 12)
    ax_t.hist(foopsi_times, bins=bins, color=_FOOPSI_COLOR,
              alpha=0.6, label='FOOPSI-initialized', edgecolor='none')
    ax_t.hist(nnls_times,   bins=bins, color=_NNLS_COLOR,
              alpha=0.6, label='NNLS-initialized',   edgecolor='none')

    ax_t.set_xlabel('time per cell (sec)')
    ax_t.set_ylabel('cells')

    ax_f = axd['f1']
    valid_n = nnls_fb[np.isfinite(nnls_fb) * (nnls_fb>0)]
    valid_f = foopsi_fb[np.isfinite(foopsi_fb) * (foopsi_fb>0)]
    rbins = np.linspace(0, 1, 12)
    ax_f.hist(valid_f, bins=rbins, color=_FOOPSI_COLOR, alpha=0.6,
              label='FOOPSI-initialized', edgecolor='none')
    ax_f.hist(valid_n, bins=rbins, color=_NNLS_COLOR,   alpha=0.6,
              label='NNLS-initialized',   edgecolor='none')

    print('Average $F_\\beta$ (β=0.5) — NNLS init: {:.3f}  FOOPSI init: {:.3f}'.format(
        np.nanmean(valid_n), np.nanmean(valid_f)))
    print('Average time per cell (sec) — NNLS init: {:.3f}  FOOPSI init: {:.3f}'.format(
        np.mean(nnls_times), np.mean(foopsi_times)))
    ax_f.set_xlabel('$F_\\beta$')
    ax_f.set_ylabel('cells')

    ax_f.legend(frameon=False, fontsize=6, loc='upper left', reverse=True)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'fmcsi_init_comparison.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def plot_combined_init(data_dir):
    """Plot a combined figure merging raw init traces, timing, and fMCSI accuracy comparisons.

    Parameters
    ----------
    data_dir : str
        Directory containing init_comparison.npz and fmcsi_init_comparison.npz.
    """
    init_path = os.path.join(data_dir, 'init_comparison.npz')
    conv_path = os.path.join(data_dir, 'fmcsi_init_comparison.npz')
    for p in (init_path, conv_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'No data at {p}. Run the corresponding test mode first.')

    d1 = np.load(init_path, allow_pickle=True)
    clean_traces      = d1['clean_traces']
    nnls_sp           = d1['nnls_sp'].astype(np.float64)
    foopsi_sp         = d1['foopsi_sp'].astype(np.float64)
    true_spikes       = d1['true_spikes']
    r_cross           = d1['r_cross']
    nnls_times_init   = d1['nnls_times']
    foopsi_times_init = d1['foopsi_times']
    fs                = float(d1['fs'][0])

    d2 = np.load(conv_path, allow_pickle=True)
    nnls_times_conv   = d2['nnls_times']
    foopsi_times_conv = d2['foopsi_times']
    nnls_fb           = d2['nnls_fb']
    foopsi_fb         = d2['foopsi_fb']

    example_cells = [49, 0, 43]
    win_offsets   = [0, 0, 0]
    win_frames    = int(_TRACE_WINDOW * fs)

    fig = plt.figure(figsize=(8.25, 4), dpi=300)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs_outer  = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2], wspace=0.44)
    gs_left   = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0], hspace=0.42)
    gs_middle = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], hspace=0.55)
    gs_right  = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[2], hspace=0.55)
    axd = {
        't0':        fig.add_subplot(gs_left[0]),
        't1':        fig.add_subplot(gs_left[1]),
        't2':        fig.add_subplot(gs_left[2]),
        'time_init': fig.add_subplot(gs_middle[0]),
        'r_cross':   fig.add_subplot(gs_middle[1]),
        'time_conv': fig.add_subplot(gs_right[0]),
        'f1':        fig.add_subplot(gs_right[1]),
    }

    for row_idx, (ax_key, cell_idx, win_start) in enumerate(
            zip(['t0', 't1', 't2'], example_cells, win_offsets)):
        ax = axd[ax_key]
        s, e = win_start, win_start + win_frames
        t_ax = np.arange(win_frames) / fs

        gt   = clean_traces[cell_idx, s:e]
        spn  = nnls_sp[cell_idx, s:e]
        spf  = foopsi_sp[cell_idx, s:e]
        t_start_s = s / fs
        sp_true = true_spikes[cell_idx]
        sp_true = sp_true[(sp_true >= t_start_s) & (sp_true < t_start_s + _TRACE_WINDOW)] - t_start_s

        gt_lo, gt_hi = gt.min(), gt.max()
        def _scale_gt(x): return (x - gt_lo) / (gt_hi - gt_lo + 1e-12)
        def _scale_sp(x):
            """Scale spike signal to [0, 1] by its maximum."""
            hi = np.max(x) if np.max(x) > 1e-12 else 1.0
            return x / hi

        ax.plot(t_ax, _scale_gt(gt),  color='k', alpha=0.25, lw=1.0,
                label='ground truth', zorder=2)
        ax.plot(t_ax, _scale_sp(spn), color=_NNLS_COLOR,   lw=0.8,
                label='NNLS',   zorder=3)
        ax.plot(t_ax, _scale_sp(spf), color=_FOOPSI_COLOR, lw=0.8,
                label='FOOPSI', zorder=3)
        ax.eventplot(sp_true, lineoffsets=1.18, linelengths=0.18,
                     colors='k', linewidths=0.8)
        ax.set_xlim(0, _TRACE_WINDOW)
        ax.set_ylim(-0.05, 1.42)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_xlabel('time (sec)' if row_idx == 2 else '')
        ax.text(0.01, 0.97, str(row_idx + 1),
                transform=ax.transAxes, fontsize=5.5, fontweight='bold',
                va='top', ha='left', color='k')
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax_ti = axd['time_init']
    ax_ti.scatter(
        foopsi_times_init * 1e3,
        nnls_times_init * 1e3,
        color='k',
        s=1
    )
    ax_ti.set_xlabel('FOOPSI init. time (msec)')
    ax_ti.set_ylabel('NNLS init. time (msec)')
    ax_ti.plot([0,175],[0,175], color='tab:cyan', alpha=0.5)
    ax_ti.set_xlim([0,175])
    ax_ti.set_ylim([0,175])

    ax_rc = axd['r_cross']
    r_cross_valid = r_cross[np.isfinite(r_cross)]
    ax_rc.hist(r_cross_valid, bins=np.linspace(0,1,12), color='k', linewidth=0.3)
    ax_rc.set_xlabel('correlation')
    ax_rc.set_ylabel('cells')
    ax_rc.set_xlim([0, 1])
    ax_rc.set_ylim([0,75])
    ax_rc.set_yticks([0,25,50,75])

    ax_tc = axd['time_conv']
    ax_tc.scatter(
        foopsi_times_conv,
        nnls_times_conv,
        color='k',
        s=1
    )
    ax_tc.set_xlim([0,6.5])
    ax_tc.set_ylim([0,6.5])
    ax_tc.plot([0,7],[0,7], color='tab:cyan', alpha=0.5)
    ax_tc.set_xlabel('FOOPSI time per cell (sec)')
    ax_tc.set_ylabel('NNLS time per cell (sec)')


    ax_fb = axd['f1']
    valid_n = nnls_fb[np.isfinite(nnls_fb) & (nnls_fb > 0)]
    valid_f = foopsi_fb[np.isfinite(foopsi_fb) & (foopsi_fb > 0)]

    ax_fb.scatter(
        valid_f,
        valid_n,
        color='k',
        s=1
    )
    ax_fb.set_xlim([0,1.05])
    ax_fb.set_ylim([0,1.05])
    ax_fb.plot([0,1.05],[0,1.05], color='tab:cyan', alpha=0.5)
    ax_fb.set_xlabel('FOOPSI F$_\\beta$')
    ax_fb.set_ylabel('NNLS F$_\\beta$')
    ax_fb.set_yticks([0,0.25,0.5,0.75,1])
    ax_fb.set_xticks([0,0.25,0.5,0.75,1])

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'combined_init_comparison.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))

    plt.close(fig)


_DEFAULT_CONV_TOL = 10 ** -1.5  # Was 0.00067; weakened, see combined_opt sweep.
_DEFAULT_BURN_TOL = 1e-4        # Was 0.005; moved into burn-in trough.

# Min_sweeps=300 (the default) gates how soon the post-burn-in convergence
# check can fire, which floors total sweep count regardless of how loose
# conv_tol/burn_tol are set. Lower it per-sweep so loosening the tolerance
# being tested can actually shorten the chain enough to reveal a quality drop.
_CONV_TEST_MIN_SWEEPS = 50
_BURN_TEST_MIN_SWEEPS = 50

# B (initial samples before burn-in is even checked) and win (the trailing
# window used to test amplitude stability) are normally 75/100, which floors
# burn-in completion at sweep ~175-200 no matter how loose burn_tol is.
# Shrunk here -- test-only, not the production default -- so a loose enough
# burn_tol/conv_tol can actually let the chain stop while still under-mixed,
# which is what's needed to see accuracy fall off.
_TEST_B           = 5
_TEST_WIN         = 5
_TEST_CHECK_EVERY = 5

# Default generate_synthetic_data() population is too easy to show a sweep
# effect (median SNR ~50, almost all cells far above the SNR=2.0 production
# gate) -- a chain barely past burn-in already lands on the right answer
# regardless of conv_tol/burn_tol. Override snr per-cell here, test-only, to
# resemble a real post-filter population: most cells sit just above the gate,
# with a shrinking tail of better-quality cells, rather than a flat box.
_TEST_SNR_FLOOR = 2.0   # Matches the production skip_snr gate.
_TEST_SNR_SCALE = 2.0   # Exponential decay scale above the floor.
_TEST_SNR_MAX   = 20.0  # Clips the rare long tail.

# Test-only cell count and session length for the conv_tol/burn_tol sweeps --
# 100 cells over 20 min (vs. the default _N_CELLS=200 / _DURATION=2400s used
# elsewhere) matches a typical real recording length and keeps these sweeps
# fast to re-run.
_TEST_N_CELLS  = 100
_TEST_DURATION = 1200.0

# None of these sweeps override max_sweeps, so they all run against the
# sampler's default cap -- used to draw a reference line on sweep-count
# panels marking "ran out the clock" vs. genuine convergence.
_MAX_SWEEPS_DEFAULT = 2000

def _build_tol_grid(default_val, lower_mult, upper_mult, n_below=4, n_above=5):
    """Build a geometric grid of tolerance values spanning a specified multiplier range.

    Parameters
    ----------
    default_val : float
        Center value for the grid.
    lower_mult : float
        Multiplier applied to default_val for the lower bound.
    upper_mult : float
        Multiplier applied to default_val for the upper bound.
    n_below : int, optional
        Number of points below the default. Default is 4.
    n_above : int, optional
        Number of points above the default. Default is 5.

    Returns
    -------
    list of float
        Grid of n_below + n_above + 1 values.
    """
    # Geomspace from default_val*lower_mult to default_val*upper_mult.
    # Multipliers are picked per-parameter from a direct measurement of
    # mean_sweeps vs. the tested value (see callers) -- outside that band
    # the chain just runs to max_sweeps regardless of the tolerance, so
    # testing there can never move F_beta.
    n_total = n_below + n_above + 1
    return np.geomspace(default_val * lower_mult, default_val * upper_mult, n_total).tolist()


def _run_tol_sweep(dff, true_spikes, param_name, grid, fs, min_sweeps):
    """Run fMCSI across a grid of tolerance values and collect accuracy/timing rows.

    Parameters
    ----------
    dff : ndarray
        Fluorescence data array, shape (n_cells, T).
    true_spikes : list of ndarray
        Ground-truth spike times per cell in seconds.
    param_name : str
        Name of the fMCSI parameter to sweep (e.g. 'conv_tol').
    grid : list of float
        Values to test for param_name.
    fs : float
        Sampling rate in Hz.
    min_sweeps : int
        Minimum sweep count passed to fMCSI.

    Returns
    -------
    list of dict
        One dict per grid point with val, mean_time, std_time, mean_nsweeps,
        std_nsweeps, mean_f1, and std_f1 fields.
    """
    rows = []
    for val in grid:
        print('\n  {}={:.2e} ...'.format(param_name, val))
        params = {
            'f': fs, 'p': 2, 'auto_stop': True, 'upd_gam': 0,
            'min_sweeps': min_sweeps,
            'B': _TEST_B, 'win': _TEST_WIN, 'check_every': _TEST_CHECK_EVERY,
            'conv_tol': _DEFAULT_CONV_TOL,
            'burn_tol': _DEFAULT_BURN_TOL,
        }
        params[param_name] = val
        try:
            res        = OMSI.deconv(dff, params=params, benchmark=True)
            per_cell_t = res['optim_times_per_cell']
            nsweeps    = res['optim_nsamples']
            pred       = res['optim_spikes']
            prec_s, rec_s, _ = helpers.compute_accuracy_strict(true_spikes, pred)
            cosmic     = helpers.compute_cosmic(true_spikes, pred, fs)
            fb = np.array([_fbeta(float(prec_s[i]), float(rec_s[i]))
                            for i in range(len(prec_s))])
            rows.append({
                'val':          val,
                'mean_time':    float(np.mean(per_cell_t)),
                'std_time':     float(np.std(per_cell_t, ddof=1)),
                'mean_nsweeps': float(np.mean(nsweeps)),
                'std_nsweeps':  float(np.std(nsweeps, ddof=1)),
                'mean_f1':      float(np.mean(fb)),
                'std_f1':       float(np.std(fb, ddof=1)),
            })
            print('    Mean_cell={:.3f}s  '
                  'mean_sweeps={:.1f}  '
                  'F_beta={:.3f}  CosMIC={:.3f}'.format(
                      np.mean(per_cell_t), np.mean(nsweeps),
                      np.mean(fb), np.mean(cosmic)))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))
    return rows


def _save_tol_sweep(out_path, rows, default_val):
    """Save tolerance sweep result rows to a .npz file.

    Parameters
    ----------
    out_path : str
        Output file path.
    rows : list of dict
        Result rows from _run_tol_sweep.
    default_val : float
        Default parameter value to store as a reference.
    """
    np.savez(
        out_path,
        tol          = np.array([r['val']          for r in rows]),
        mean_time    = np.array([r['mean_time']     for r in rows]),
        std_time     = np.array([r['std_time']      for r in rows]),
        mean_nsweeps = np.array([r['mean_nsweeps']  for r in rows]),
        std_nsweeps  = np.array([r['std_nsweeps']   for r in rows]),
        mean_f1      = np.array([r['mean_f1']       for r in rows]),
        std_f1       = np.array([r['std_f1']        for r in rows]),
        default_val  = np.array([default_val]),
    )
    print('\nSaved to {}.'.format(out_path))


def run_conv_tol_sweep(data_dir):
    """Run a convergence tolerance sweep on a synthetic low-SNR population and save results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'conv_tol_sweep.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s, '
          'snr~floor={}+exp({}))...'.format(
              _TEST_N_CELLS, _TEST_DURATION, _FS, _TAU,
              _TEST_SNR_FLOOR, _TEST_SNR_SCALE))
    snr = np.clip(_TEST_SNR_FLOOR + np.random.exponential(_TEST_SNR_SCALE, size=_TEST_N_CELLS),
                  _TEST_SNR_FLOOR, _TEST_SNR_MAX)
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_TEST_N_CELLS, fs=_FS, duration=_TEST_DURATION, tau=_TAU, snr=snr)

    # Measured mean_sweeps vs. conv_tol on this population: pinned at
    # max_sweeps=2000 for everything from default_val/1000 up through
    # ~default_val*10; the real decline runs from default_val itself out to
    # default_val*1000. Range narrowed to where the curve actually moves.
    grid = _build_tol_grid(_DEFAULT_CONV_TOL, lower_mult=1.0, upper_mult=1000.0)
    print('  Sweep ({} conv_tol values): {}'.format(
        len(grid), ['{:.2e}'.format(v) for v in grid]))

    rows = _run_tol_sweep(dff, true_spikes, 'conv_tol', grid, _FS, _CONV_TEST_MIN_SWEEPS)
    if rows:
        _save_tol_sweep(out_path, rows, _DEFAULT_CONV_TOL)
    else:
        print('No results collected.')


def run_burn_tol_sweep(data_dir):
    """Run a burn-in tolerance sweep on a synthetic low-SNR population and save results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'burn_tol_sweep.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s, '
          'snr~floor={}+exp({}))...'.format(
              _TEST_N_CELLS, _TEST_DURATION, _FS, _TAU,
              _TEST_SNR_FLOOR, _TEST_SNR_SCALE))
    snr = np.clip(_TEST_SNR_FLOOR + np.random.exponential(_TEST_SNR_SCALE, size=_TEST_N_CELLS),
                  _TEST_SNR_FLOOR, _TEST_SNR_MAX)
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_TEST_N_CELLS, fs=_FS, duration=_TEST_DURATION, tau=_TAU, snr=snr)

    # Measured mean_sweeps vs. burn_tol on this population: pinned at
    # max_sweeps=2000 below ~default_val/500 (burn-in itself never
    # completes) and above ~default_val*4 (burn-in completes too early,
    # leaving real pre-convergence drift that conv_tol then never
    # satisfies). The non-flat region sits between those two ends.
    grid = _build_tol_grid(_DEFAULT_BURN_TOL, lower_mult=0.002, upper_mult=4.0)
    print('  Sweep ({} burn_tol values): {}'.format(
        len(grid), ['{:.2e}'.format(v) for v in grid]))

    rows = _run_tol_sweep(dff, true_spikes, 'burn_tol', grid, _FS, _BURN_TEST_MIN_SWEEPS)
    if rows:
        _save_tol_sweep(out_path, rows, _DEFAULT_BURN_TOL)
    else:
        print('No results collected.')


def _plot_tol_sweep(data_dir, npz_name, xlabel, fig_stem):
    """Load a tolerance sweep .npz file and plot time-per-cell and F_beta vs. tolerance.

    Parameters
    ----------
    data_dir : str
        Directory containing the .npz file.
    npz_name : str
        Filename of the .npz to load.
    xlabel : str
        X-axis label for the plots.
    fig_stem : str
        Stem for the output figure filenames.
    """
    out_path = os.path.join(data_dir, npz_name)
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}.')

    d           = np.load(out_path)
    tols        = d['tol'].astype(float)
    mt          = d['mean_time']
    st          = d['std_time']
    mf1         = d['mean_f1']
    sf1         = d['std_f1']
    default_val = float(d['default_val'][0])

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)
    for ax, y, yerr, ylabel in [
        (axes[0], mt,  st,  'time per cell (sec)'),
        (axes[1], mf1, sf1, '$F_\\beta$'),
    ]:
        ax.plot(tols, y, '.-', color=_COLOR, zorder=3)
        if ylabel == 'time per cell (sec)':
            ax.fill_between(tols, y - yerr, y + yerr,
                            color=_COLOR, alpha=0.25, linewidth=0)
        ax.axvline(default_val, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)

    axes[1].set_ylim(0, 0.51)
    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, '{}.{}'.format(fig_stem, sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def plot_conv_tol_sweep(data_dir):
    """Plot the saved convergence tolerance sweep results.

    Parameters
    ----------
    data_dir : str
        Directory containing the conv_tol_sweep.npz file.
    """
    _plot_tol_sweep(data_dir, 'conv_tol_sweep.npz',
                    'convergence threshold', 'conv_tol_sweep')


def plot_burn_tol_sweep(data_dir):
    """Plot the saved burn-in tolerance sweep results.

    Parameters
    ----------
    data_dir : str
        Directory containing the burn_tol_sweep.npz file.
    """
    _plot_tol_sweep(data_dir, 'burn_tol_sweep.npz',
                    'burn-in completion threshold', 'burn_tol_sweep')


def plot_combined_opt(data_dir):
    """Plot a 4x3 combined figure of all optimization parameter sweeps.

    Parameters
    ----------
    data_dir : str
        Directory containing T_supp_sweep.npz, conv_tol_sweep.npz,
        burn_tol_sweep.npz, and snr_threshold_sweep.npz.
    """
    t_supp_path   = os.path.join(data_dir, 'T_supp_sweep.npz')
    conv_tol_path = os.path.join(data_dir, 'conv_tol_sweep.npz')
    burn_tol_path = os.path.join(data_dir, 'burn_tol_sweep.npz')
    snr_path      = os.path.join(data_dir, 'snr_threshold_sweep.npz')
    for p in (t_supp_path, conv_tol_path, burn_tol_path, snr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'No data at {p}.')

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 9.0), dpi=300)

    def _sweeps_panel(ax, x, mns, sns, xlabel):
        """Plot mean sweep count with std shading on ax."""
        ax.fill_between(x, mns - sns, mns + sns,
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, mns, '.-', color=_COLOR, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('sweep count')
        ax.set_xscale('log')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, _MAX_SWEEPS_DEFAULT)

    d            = np.load(t_supp_path)
    T_supp       = d['T_supp'].astype(float)
    mt           = d['mean_time']
    st           = d['std_time']
    mf1          = d['mean_f1']
    sf1          = d['std_f1']
    mns          = d['mean_nsweeps']
    sns          = d['std_nsweeps']
    default_supp = int(d['default_supp'][0])

    mask = np.arange(len(T_supp))[1:-1]
    mask = np.hstack([mask[0:4], mask[5], mask[7:]]).astype(int)

    for ax, y, yerr, ylabel in [
        (axes[0, 0], mt,  st,  'time per cell (sec)'),
        (axes[0, 1], mf1, sf1, '$F_\\beta$'),
    ]:
        ax.fill_between(T_supp[mask] * (1.0 / _FS),
                        y[mask] - yerr[mask], y[mask] + yerr[mask],
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(T_supp[mask] * (1.0 / _FS), y[mask], '.-', color=_COLOR, zorder=3)
        ax.axvline(default_supp * (1.0 / _FS), color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('$T_{supp}$ (sec)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xlim(T_supp[mask].min() * (1.0 / _FS), T_supp[mask].max() * (1.0 / _FS))
        ax.set_ylim(bottom=0)

    axes[0, 0].set_ylim(0, 500)
    axes[0, 1].set_ylim(0, 1)

    _sweeps_panel(axes[0, 2], T_supp[mask] * (1.0 / _FS), mns[mask], sns[mask],
                  '$T_{supp}$ (sec)')
    axes[0, 2].axvline(default_supp * (1.0 / _FS), color='k', linestyle='--',
                       linewidth=0.8, alpha=0.6)

    d           = np.load(conv_tol_path)
    tols        = d['tol'].astype(float)
    mt          = d['mean_time']
    st          = d['std_time']
    mf1         = d['mean_f1']
    sf1         = d['std_f1']
    mns         = d['mean_nsweeps']
    sns         = d['std_nsweeps']
    default_val = _DEFAULT_CONV_TOL  # Updated default; npz still reflects old value.

    for ax, y, yerr, ylabel in [
        (axes[1, 0], mt,  st,  'time per cell (sec)'),
        (axes[1, 1], mf1, sf1, '$F_\\beta$'),
    ]:
        ax.fill_between(tols, y - yerr, y + yerr,
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(tols, y, '.-', color=_COLOR, zorder=3)
        ax.axvline(default_val, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('convergence threshold')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xlim(tols.min(), tols.max())
        ax.set_ylim(bottom=0)

    axes[1, 1].set_ylim(0, 1)

    _sweeps_panel(axes[1, 2], tols, mns, sns, 'convergence threshold')
    axes[1, 2].axvline(default_val, color='k', linestyle='--',
                       linewidth=0.8, alpha=0.6)

    d           = np.load(burn_tol_path)
    tols        = d['tol'].astype(float)
    mt          = d['mean_time']
    st          = d['std_time']
    mf1         = d['mean_f1']
    sf1         = d['std_f1']
    mns         = d['mean_nsweeps']
    sns         = d['std_nsweeps']
    default_val = _DEFAULT_BURN_TOL  # Updated default; npz still reflects old value.

    for ax, y, yerr, ylabel in [
        (axes[2, 0], mt,  st,  'time per cell (sec)'),
        (axes[2, 1], mf1, sf1, '$F_\\beta$'),
    ]:
        ax.fill_between(tols, y - yerr, y + yerr,
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(tols, y, '.-', color=_COLOR, zorder=3)
        ax.axvline(default_val, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('burn-in completion threshold')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xlim(tols.min(), tols.max())
        ax.set_ylim(bottom=0)

    axes[2, 1].set_ylim(0, 1)

    _sweeps_panel(axes[2, 2], tols, mns, sns, 'burn-in completion threshold')
    axes[2, 2].axvline(default_val, color='k', linestyle='--',
                       linewidth=0.8, alpha=0.6)

    d           = np.load(snr_path)
    snr_levels  = d['snr_levels'].astype(float)
    mean_fb     = d['mean_fb'].astype(float)
    std_fb      = d['std_fb'].astype(float)
    mean_cosmic = d['mean_cosmic'].astype(float)
    std_cosmic  = d['std_cosmic'].astype(float)
    mean_ns     = d['mean_nsweeps'].astype(float)
    std_ns      = d['std_nsweeps'].astype(float)
    threshold   = float(d['threshold'][0])

    for ax, mean, std, ylabel in [
        (axes[3, 0], mean_fb,     std_fb,     '$F_\\beta$'),
        (axes[3, 1], mean_cosmic, std_cosmic, 'CosMIC'),
    ]:
        valid = np.isfinite(mean) & np.isfinite(std)
        x, y, ye = snr_levels[valid], mean[valid], std[valid]
        ax.fill_between(x, np.clip(y - ye, 0, 1), np.clip(y + ye, 0, 1),
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, y, '.-', color=_COLOR, zorder=3)
        ax.axvline(threshold, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1.)

    valid_ns = np.isfinite(mean_ns) & np.isfinite(std_ns)
    ax = axes[3, 2]
    ax.fill_between(snr_levels[valid_ns],
                    mean_ns[valid_ns] - std_ns[valid_ns],
                    mean_ns[valid_ns] + std_ns[valid_ns],
                    color=_COLOR, alpha=0.25, linewidth=0)
    ax.plot(snr_levels[valid_ns], mean_ns[valid_ns], '.-', color=_COLOR, zorder=3)
    ax.axvline(threshold, color='k', linestyle='--',
               linewidth=0.8, alpha=0.6)
    ax.set_xlabel('SNR')
    ax.set_ylabel('sweep count')
    ax.set_xlim(snr_levels[valid_ns].min(), snr_levels[valid_ns].max())
    ax.set_ylim(0, _MAX_SWEEPS_DEFAULT)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'combined_opt.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def _dff_snr(fluo):
    """Estimate the signal-to-noise ratio of a dF/F trace.

    Parameters
    ----------
    fluo : array_like
        Fluorescence trace.

    Returns
    -------
    float
        SNR estimated as (99th percentile - 8th percentile) / MAD-based noise.
    """
    f      = np.asarray(fluo, dtype=np.float64)
    sn_mad = float(np.median(np.abs(np.diff(f)))) / 0.6745 if len(f) > 1 else 1e-4
    peak   = float(np.percentile(f, 99))
    base   = float(np.percentile(f,  8))
    return (peak - base) / (sn_mad + 1e-9)


def run_snr_filter_sweep(data_dir):
    """Run fMCSI without SNR pre-filtering and save per-cell accuracy vs. SNR.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'snr_filter_sweep.npz')

    print('Generating synthetic population '
          '(n={}, T={}s, fs={}Hz, tau={}s)...'.format(
              _N_CELLS, _DURATION, _FS, _TAU))
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_DURATION, tau=_TAU
    )

    snr = np.array([_dff_snr(dff[i]) for i in range(_N_CELLS)])

    params = {
        'f':         _FS,
        'p':         2,
        'auto_stop': True,
        'upd_gam':   0,
        'conv_tol':  _DEFAULT_CONV_TOL,
        'burn_tol':  _DEFAULT_BURN_TOL,
        'skip_snr':  True,
    }

    print('Running OMSI on {} synthetic cells (no SNR pre-filter)...'.format(_N_CELLS))
    res = OMSI.deconv(dff, params=params, benchmark=True)
    pred = res['optim_spikes']

    prec_s, rec_s, _ = helpers.compute_accuracy_strict(true_spikes, pred)
    cosmic           = helpers.compute_cosmic(true_spikes, pred, _FS)
    fbeta = np.array([_fbeta(float(prec_s[i]), float(rec_s[i]))
                       for i in range(len(prec_s))])

    np.savez(
        out_path,
        snr       = snr,
        fbeta     = fbeta,
        cosmic    = cosmic,
        threshold = np.array([_SNR_THRESHOLD]),
    )
    print('\nSaved {} cells to {}.'.format(_N_CELLS, out_path))


def plot_snr_filter_sweep(data_dir):
    """Plot per-cell F_beta and CosMIC vs. SNR from a saved SNR filter sweep.

    Parameters
    ----------
    data_dir : str
        Directory containing the snr_filter_sweep.npz file.
    """
    out_path = os.path.join(data_dir, 'snr_filter_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode snr-filter-test first.')

    d         = np.load(out_path)
    snr       = d['snr'].astype(float)
    fbeta     = d['fbeta'].astype(float)
    cosmic    = d['cosmic'].astype(float)
    threshold = float(d['threshold'][0])

    n_below_thresh = int(np.sum(snr < threshold))

    snr_max = 15.0
    bins = np.unique(np.concatenate([
        np.linspace(0.0, threshold, 6),
        np.linspace(threshold, snr_max, 13)[1:],
    ]))
    in_range  = snr <= snr_max
    snr       = snr[in_range]
    fbeta     = fbeta[in_range]
    cosmic    = cosmic[in_range]
    bin_ids   = np.clip(np.digitize(snr, bins) - 1, 0, len(bins) - 2)

    n_bins      = len(bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    f1_mean  = np.full(n_bins, np.nan)
    f1_std   = np.full(n_bins, np.nan)
    cos_mean = np.full(n_bins, np.nan)
    cos_std  = np.full(n_bins, np.nan)
    n_bin    = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() < 2:
            continue
        n_bin[b]    = mask.sum()
        f1_mean[b]  = np.nanmean(fbeta[mask])
        f1_std[b]   = np.nanstd(fbeta[mask])
        cos_mean[b] = np.nanmean(cosmic[mask])
        cos_std[b]  = np.nanstd(cosmic[mask])

    valid = n_bin >= 2
    x = bin_centers[valid]

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)
    fig.suptitle('{} cells below SNR threshold (with spikes)'.format(n_below_thresh),
                 fontsize=6, y=1.02)
    for ax, mean, std, ylabel in [
        (axes[0], f1_mean[valid],  f1_std[valid],  '$F_\\beta$'),
        (axes[1], cos_mean[valid], cos_std[valid], 'CosMIC'),
    ]:
        ax.fill_between(x,
                        np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, mean, '.-', color=_COLOR, zorder=3)
        ax.axvline(threshold, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, snr_max)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'snr_filter_sweep.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


_SNR_THRESHOLD      = 2.0
_SNR_N_CELLS        = 50
_SNR_DURATION       = 120.0
_SNR_N_LEVELS       = 18


def run_snr_threshold_sweep(data_dir):
    """Run fMCSI across a range of fixed SNR levels and save accuracy results.

    Parameters
    ----------
    data_dir : str
        Directory where the output .npz file is written.
    """
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'snr_threshold_sweep.npz')

    snr_levels = np.geomspace(0.3, 3.0 * _SNR_THRESHOLD, _SNR_N_LEVELS)
    print('SNR sweep: {} levels from {:.2f} to {:.2f}  '
          '(threshold={})'.format(
              _SNR_N_LEVELS, snr_levels[0], snr_levels[-1], _SNR_THRESHOLD))
    print('  {} cells x {}s each'.format(_SNR_N_CELLS, _SNR_DURATION))

    mean_fb      = np.full(_SNR_N_LEVELS, np.nan)
    std_fb       = np.full(_SNR_N_LEVELS, np.nan)
    mean_cosmic  = np.full(_SNR_N_LEVELS, np.nan)
    std_cosmic   = np.full(_SNR_N_LEVELS, np.nan)
    mean_nsweeps = np.full(_SNR_N_LEVELS, np.nan)
    std_nsweeps  = np.full(_SNR_N_LEVELS, np.nan)

    params = {
        'f':         _FS,
        'p':         2,
        'auto_stop': True,
        'upd_gam':   0,
        'conv_tol':  _DEFAULT_CONV_TOL,
        'burn_tol':  _DEFAULT_BURN_TOL,
    }

    for k, snr_val in enumerate(snr_levels):
        print('\n  [{}/{}] SNR={:.3f} ...'.format(k + 1, _SNR_N_LEVELS, snr_val))
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=_SNR_N_CELLS, fs=_FS, duration=_SNR_DURATION,
            tau=_TAU, snr=float(snr_val),
        )
        try:
            res  = OMSI.deconv(dff, params=params, true_spikes=true_spikes, benchmark=True)
            pred = res['optim_spikes']

            cosmic_v   = helpers.compute_cosmic(true_spikes, pred, _FS)

            mean_cosmic[k]  = float(np.mean(cosmic_v))
            std_cosmic[k]   = float(np.std(cosmic_v, ddof=1))
            mean_nsweeps[k] = float(np.mean(res['optim_nsamples']))
            std_nsweeps[k]  = float(np.std(res['optim_nsamples'], ddof=1))

            if res['optim_precision'] is not None:
                fb_arr = np.array([
                    _fbeta(float(res['optim_precision'][i]),
                           float(res['optim_recall'][i]))
                    for i in range(_SNR_N_CELLS)
                ])
                mean_fb[k] = float(np.nanmean(fb_arr))
                std_fb[k]  = float(np.nanstd(fb_arr))

            print('    F_beta={:.3f}  CosMIC={:.3f}  '
                  'mean_sweeps={:.1f}'.format(
                      mean_fb[k], mean_cosmic[k], mean_nsweeps[k]))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))

    np.savez(
        out_path,
        snr_levels   = snr_levels,
        mean_fb      = mean_fb,
        std_fb       = std_fb,
        mean_cosmic  = mean_cosmic,
        std_cosmic   = std_cosmic,
        mean_nsweeps = mean_nsweeps,
        std_nsweeps  = std_nsweeps,
        threshold    = np.array([_SNR_THRESHOLD]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_snr_threshold_sweep(data_dir):
    """Plot F_beta and CosMIC vs. SNR level from a saved SNR threshold sweep.

    Parameters
    ----------
    data_dir : str
        Directory containing the snr_threshold_sweep.npz file.
    """
    out_path = os.path.join(data_dir, 'snr_threshold_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f'No data at {out_path}. Run --mode snr-thresh-test first.')

    d           = np.load(out_path)
    snr_levels  = d['snr_levels'].astype(float)
    mean_fb     = d['mean_fb'].astype(float)
    std_fb      = d['std_fb'].astype(float)
    mean_cosmic = d['mean_cosmic'].astype(float)
    std_cosmic  = d['std_cosmic'].astype(float)
    threshold   = float(d['threshold'][0])

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)
    for ax, mean, std, ylabel in [
        (axes[0], mean_fb,     std_fb,     '$F_\\beta$'),
        (axes[1], mean_cosmic, std_cosmic, 'CosMIC'),
    ]:
        valid = np.isfinite(mean) & np.isfinite(std)
        x, y, ye = snr_levels[valid], mean[valid], std[valid]
        ax.fill_between(x,
                        np.clip(y - ye, 0, 1),
                        np.clip(y + ye, 0, 1),
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, y, '.-', color=_COLOR, zorder=3)
        ax.axvline(threshold, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6, label='threshold={}'.format(threshold))
        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'snr_threshold_sweep.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


_SNR_SENSOR_ORDER = [
    'GCaMP6f', 'GCaMP6s', 'GCaMP8f', 'GCaMP8m',
    'GCaMP5k', 'OGB1', 'jGECO', 'XCaMP', 'R-CaMP', 'jRCaMP',
]
_SNR_EXCLUDED = {'Other', 'Cal520'}


def _snr_get_sensor(ds_name):
    """Map a dataset name to a canonical calcium sensor label.

    Parameters
    ----------
    ds_name : str
        Dataset filename or identifier string.

    Returns
    -------
    str
        Sensor label (e.g. 'GCaMP6f', 'OGB1') or 'Other' if unrecognized.
    """
    s = ds_name.lower()
    for keyword, label in [
        ('gcaMP8s', 'GCaMP8s'), ('gcaMP8m', 'GCaMP8m'), ('gcaMP8f', 'GCaMP8f'),
        ('gcaMP7f', 'GCaMP7f'), ('gcaMP6s', 'GCaMP6s'), ('gcaMP6f', 'GCaMP6f'),
        ('gcaMP5k', 'GCaMP5k'), ('jgeco',   'jGECO'),   ('xcaMP',   'XCaMP'),
        ('jrcamp',  'jRCaMP'),  ('rcamp',   'R-CaMP'),  ('ogb',     'OGB1'),
        ('cal520',  'Cal520'),
    ]:
        if keyword.lower() in s:
            return label
    return 'Other'


def print_snr_stats(fig4_data_dir):
    """Print a table of SNR statistics by sensor for the figure4 ground-truth datasets.

    Parameters
    ----------
    fig4_data_dir : str
        Root directory containing the ground_truth_traces_fmcsi/ subdirectory.
    """
    traces_dir = os.path.join(fig4_data_dir, 'ground_truth_traces_fmcsi')
    if not os.path.isdir(traces_dir):
        print('Traces directory not found: {}'.format(traces_dir))
        return

    snr_by_sensor = {}
    for fname in sorted(os.listdir(traces_dir)):
        if not fname.endswith('_traces.npz'):
            continue
        ds_name = fname.replace('_traces.npz', '')
        sensor  = _snr_get_sensor(ds_name)
        if sensor in _SNR_EXCLUDED:
            continue
        try:
            npz     = np.load(os.path.join(traces_dir, fname), allow_pickle=False)
            n_cells = int(npz['n_cells'])
        except Exception as exc:
            print('  Warning: {}: {}'.format(fname, exc))
            continue
        for i in range(n_cells):
            trace = npz['dff_{}'.format(i)].astype(np.float64)
            noise = _get_sn(trace, [0.25, 0.5])
            b     = float(np.percentile(trace, 8))
            peak  = float(np.percentile(trace, 99))
            snr   = (peak - b) / (noise + 1e-9)
            snr_by_sensor.setdefault(sensor, []).append(snr)

    print('SNR statistics by sensor (figure4 datasets):')
    print('  {:<12}  {:>5}  {:>10}  {:>10}'.format('Sensor', 'n', 'mean SNR', 'std SNR'))
    print('  {}  {}  {}  {}'.format('-' * 12, '-' * 5, '-' * 10, '-' * 10))
    for sensor in _SNR_SENSOR_ORDER:
        if sensor not in snr_by_sensor:
            continue
        vals = np.array(snr_by_sensor[sensor])
        print('  {:<12}  {:>5}  {:>10.2f}  {:>10.2f}'.format(
            sensor, len(vals), np.mean(vals), np.std(vals)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='T_supp sensitivity and init comparison benchmarks for fMCSI'
    )
    parser.add_argument(
        '--mode', required=True,
        choices=[
            'test',
            'plot',
            'init-test',
            'init-plot',
            'conv-test',
            'conv-plot',
            'combined-plot',
            'tol-conv-test',
            'tol-conv-plot',
            'tol-burn-test',
            'tol-burn-plot',
            'combined-opt-plot',
            'snr-filter-test',
            'snr-filter-plot',
            'snr-thresh-test',
            'snr-thresh-plot',
            'snr-stats',
        ],
    )
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument(
        '--fig4-data-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig4'),
        help='figure4 data directory (required for snr-stats mode)',
    )
    args = parser.parse_args()

    if args.mode == 'test':
        run_T_supp_sweep(args.data_dir)
    elif args.mode == 'plot':
        plot_T_supp_sweep(args.data_dir)
    elif args.mode == 'init-test':
        run_init_comparison(args.data_dir)
    elif args.mode == 'init-plot':
        plot_init_comparison(args.data_dir)
    elif args.mode == 'conv-test':
        run_fmcsi_init_comparison(args.data_dir)
    elif args.mode == 'conv-plot':
        plot_fmcsi_init_comparison(args.data_dir)
    elif args.mode == 'combined-plot':  ### THIS ONE
        plot_combined_init(args.data_dir)
    elif args.mode == 'tol-conv-test':
        run_conv_tol_sweep(args.data_dir)
    elif args.mode == 'tol-conv-plot':
        plot_conv_tol_sweep(args.data_dir)
    elif args.mode == 'tol-burn-test':
        run_burn_tol_sweep(args.data_dir)
    elif args.mode == 'tol-burn-plot':
        plot_burn_tol_sweep(args.data_dir)
    elif args.mode == 'combined-opt-plot': ### AND THIS ONE
        plot_combined_opt(args.data_dir)
    elif args.mode == 'snr-filter-test':
        run_snr_filter_sweep(args.data_dir)
    elif args.mode == 'snr-filter-plot':
        plot_snr_filter_sweep(args.data_dir)
    elif args.mode == 'snr-thresh-test':
        run_snr_threshold_sweep(args.data_dir)
    elif args.mode == 'snr-thresh-plot':
        plot_snr_threshold_sweep(args.data_dir)
    elif args.mode == 'snr-stats':
        print_snr_stats(args.fig4_data_dir)
