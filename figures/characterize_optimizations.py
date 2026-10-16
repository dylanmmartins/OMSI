# -*- coding: utf-8 -*-
"""
figures/characterize_optimizations.py

Benchmarks OMSI optimizer settings and initialization strategies via parameter sweeps on synthetic data.

DMM, March 2026
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.signal import lfilter as _lfilter, find_peaks as _find_peaks
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
from scipy.optimize import minimize as _minimize

import OMSI
import OMSI.helpers as helpers
from OMSI.sampler import _build_ef_nb
from OMSI.get_init_sample import (
    get_init_sample,
    _get_sn, _estimate_time_constants, _ar_kernel, _block_nnls_deconv,
    _otsu_1d, _init_score, _amplitude_given_spikes,
)
from simulation_helpers import generate_synthetic_data
from OMSI._win_perf import no_power_throttling

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
_FB_COLOR     = '#1AA7C4'
_COSMIC_COLOR = '#9C2F8F'


def _init_tau(Y_cell, fs, p=2):

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

    t_arr = np.arange(T + 1, dtype=np.float64)
    _, ef_d, _, _, _ = _build_ef_nb(tau, diff_gr, t_arr, T, p, prec)

    return len(ef_d)


def _build_T_supp_grid(default_supp, T, n_shorter=8, n_longer=8):

    shorter = np.round(
        np.geomspace(0.01, default_supp - 1, n_shorter)
    ).astype(int)
    longer = np.round(
        np.geomspace(default_supp + 1, T - 1, n_longer)
    ).astype(int)
    grid = np.unique(np.concatenate([shorter, [default_supp], longer, [T]]))
    return grid.tolist()


def run_T_supp_sweep(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'T_supp_sweep.npz')

    print('Generating figure-1 population '
          '(n={}, T={}s, fs={}Hz, tau={}s)...'.format(
              _N_CELLS, _DURATION, _FS, _TAU))
    dff, true_spikes, _ = _fig1_population(_N_CELLS, _DURATION)
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
                'med_time':        float(np.median(per_cell_t)),
                'mad_time':        float(_mad(per_cell_t)),
                'med_nsweeps':     float(np.median(nsweeps)),
                'mad_nsweeps':     float(_mad(nsweeps)),
                'nsweeps_cells':   np.asarray(nsweeps, dtype=float),
                'med_f1_window':   float(np.nanmedian(fb)),
                'mad_f1_window':   float(_mad(fb)),
                'med_f1_event':    float(np.nanmedian(f1_e)),
                'med_cosmic':      float(np.nanmedian(cosmic)),
                'mad_cosmic':      float(_mad(cosmic)),
            })
            r = rows[-1]
            print('    Total={:.1f}s  '
                  'cell={:.3f} ± {:.3f}s  '
                  'sweeps={:.1f} ± {:.1f}  '
                  'F_beta={:.3f} ± {:.3f}  '
                  'CosMIC={:.3f} ± {:.3f}'.format(
                      elapsed, r['med_time'], r['mad_time'],
                      r['med_nsweeps'], r['mad_nsweeps'],
                      r['med_f1_window'], r['mad_f1_window'],
                      r['med_cosmic'], r['mad_cosmic']))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))

    if not rows:
        print('No results collected.')
        return

    np.savez(
        out_path,
        T_supp       = np.array([r['T_supp']          for r in rows]),
        med_time    = np.array([r['med_time']        for r in rows]),
        mad_time     = np.array([r['mad_time']         for r in rows]),
        med_nsweeps = np.array([r['med_nsweeps']     for r in rows]),
        mad_nsweeps  = np.array([r['mad_nsweeps']      for r in rows]),
        nsweeps_cells = np.stack([r['nsweeps_cells'] for r in rows]),
        med_f1      = np.array([r['med_f1_window']   for r in rows]),
        mad_f1       = np.array([r['mad_f1_window']    for r in rows]),
        med_cosmic  = np.array([r['med_cosmic']      for r in rows]),
        mad_cosmic   = np.array([r['mad_cosmic']       for r in rows]),
        default_supp = np.array([default_supp]),
        n_frames     = np.array([n_frames]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_T_supp_sweep(data_dir):

    out_path = os.path.join(data_dir, 'T_supp_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode test first.')

    d            = np.load(out_path)
    T_supp       = d['T_supp'].astype(float)
    mt           = d['med_time']
    st           = d['mad_time']
    mf1          = d['med_f1']
    sf1          = d['mad_f1']
    mcos         = d['med_cosmic']
    scos         = d['mad_cosmic']
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

    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

def _mad(x, axis=None):

    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)

_NNLS_COLOR   = '#4C72B0'
_FOOPSI_COLOR = 'tab:red'
_DFF_ALPHA    = 0.35
_INIT_DURATION = 300.0
_TRACE_WINDOW  = 10.0

def _sp_peaks(sp, fs, thresh_frac=0.15, min_gap_s=0.05):
    if sp is None or np.max(sp) < 1e-12:
        return np.array([], dtype=float)
    thresh = thresh_frac * np.max(sp)
    min_gap = max(1, int(min_gap_s * fs))
    peaks, _ = _find_peaks(sp, height=thresh, distance=min_gap)
    return peaks.astype(float)


def _nnls_init(Y_cell, fs, p=2):

    sn = _get_sn(Y_cell, [0.25, 0.5])
    g  = _estimate_time_constants(Y_cell, p, sn)
    h  = _ar_kernel(g, len(Y_cell))
    sp = _block_nnls_deconv(Y_cell, h, len(Y_cell))
    calcium = _lfilter([1.0], np.concatenate(([1.0], -g)), sp)
    return sp, calcium


def _foopsi_init(Y_cell, fs, tau=_TAU):

    T   = len(Y_cell)
    sn  = _get_sn(Y_cell, [0.25, 0.5])
    g   = np.exp(-1.0 / (fs * tau))
    lam = sn

    def _fwd(s):
        return _lfilter([1.0], [1.0, -g], s)

    def _adj(v):
        return _lfilter([1.0], [1.0, -g], v[::-1])[::-1]

    def _obj(s):
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

_SPIKE_CORR_SMOOTH = 0.100


def _spike_corrs(sp_a, sp_b, fs, sigma_s=_SPIKE_CORR_SMOOTH):

    out = np.full(len(sp_a), np.nan)
    for i, (a, b) in enumerate(zip(sp_a, sp_b)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if sigma_s > 0:
            a = _gaussian_filter1d(a, sigma_s * fs)
            b = _gaussian_filter1d(b, sigma_s * fs)
        if a.std() > 0 and b.std() > 0:
            out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def run_init_comparison(data_dir):

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
    r_cross_ca    = np.zeros(_N_CELLS)
    nnls_times    = np.zeros(_N_CELLS)
    foopsi_times  = np.zeros(_N_CELLS)
    firing_rates  = np.array([len(s) / _INIT_DURATION for s in true_spikes])

    nnls_calcium_store   = []
    foopsi_calcium_store = []
    nnls_sp_store        = []
    foopsi_sp_store      = []

    _nnls_init(dff[0], _FS)
    _foopsi_init(dff[0], _FS)

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
        r_cross[i]       = float(_spike_corrs([sp_n], [sp_f], _FS)[0])
        r_cross_ca[i]    = float(np.corrcoef(ca_n,   ca_f)[0, 1])

        if (i + 1) % 10 == 0:
            print('  {}/{}  '
                  'nnls={:.1f} ± {:.1f}ms  '
                  'foopsi={:.1f} ± {:.1f}ms  '
                  'spike corr={:.3f} ± {:.3f}'.format(
                      i + 1, _N_CELLS,
                      np.median(nnls_times[:i + 1]) * 1e3,   _mad(nnls_times[:i + 1]) * 1e3,
                      np.median(foopsi_times[:i + 1]) * 1e3, _mad(foopsi_times[:i + 1]) * 1e3,
                      np.nanmedian(r_cross[:i + 1]),         _mad(r_cross[:i + 1])))

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
        r_cross_calcium  = r_cross_ca,
        spike_corr_smooth_s = np.array([_SPIKE_CORR_SMOOTH]),
        nnls_times       = nnls_times,
        foopsi_times     = foopsi_times,
        firing_rates     = firing_rates,
        n_frames         = np.array([dff.shape[1]]),
        fs               = np.array([_FS]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_init_comparison(data_dir):

    out_path = os.path.join(data_dir, 'init_comparison.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode init-test first.')

    d = np.load(out_path, allow_pickle=True)
    dff          = d['dff']
    clean_traces = d['clean_traces']
    nnls_sp      = d['nnls_sp'].astype(np.float64)
    foopsi_sp    = d['foopsi_sp'].astype(np.float64)
    true_spikes  = d['true_spikes']
    r_cross      = _spike_corrs(d['nnls_sp'], d['foopsi_sp'], float(d['fs'][0]))
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
            return (x - gt_lo) / (gt_hi - gt_lo + 1e-12)

        def _scale_sp(x):
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

    ax_h.set_xlabel('spike correlation')
    ax_h.set_ylabel('cells')
    ax_h.set_xlim([0,1])

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'init_comparison.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def _make_foopsi_init(Y_cell, fs, tau=_TAU, p=2):

    init_params = {'f': fs, 'p': p, 'defg': [0.6, 0.95]}
    SAM = dict(get_init_sample(Y_cell, init_params))

    sp, _ = _foopsi_init(Y_cell, fs, tau)

    Y = np.asarray(Y_cell, dtype=np.float64)
    T = len(Y)
    g, sn = np.atleast_1d(SAM['g']).astype(float), float(SAM['sg'])
    roots_abs = np.abs(np.roots(np.concatenate([[1.0], -g])))
    g_decay = min(float(np.max(roots_abs)) if len(roots_abs) else float(np.max(g)), 0.9999)
    ge = g_decay ** np.arange(T)
    tau_frames = max(1.0, -1.0 / np.log(max(g_decay, 1e-6)))
    h = _ar_kernel(g, min(T, max(50, int(np.ceil(5 * tau_frames)))))

    sp_max = float(np.max(sp)) if len(sp) else 0.0
    indices = np.where(sp > 0.15 * sp_max)[0] if sp_max > 0 else np.array([], dtype=int)
    if len(indices):
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        masses = np.array([sp[grp].sum() for grp in groups])
        centers = np.array([np.sum(grp * sp[grp]) / sp[grp].sum() for grp in groups])
    else:
        masses, centers = np.array([]), np.array([])

    if len(masses) >= 4:
        keep = masses > _otsu_1d(masses)
        if 0 < keep.sum() < len(masses):
            if _init_score(Y, centers[keep], h, sn, T) < _init_score(Y, centers, h, sn, T):
                masses, centers = masses[keep], centers[keep]

    spiketimes_ = np.abs(centers)
    oob = spiketimes_ >= T
    spiketimes_[oob] = 2.0 * T - spiketimes_[oob]
    SAM['spiketimes_'] = spiketimes_
    SAM['lam_'] = len(spiketimes_) / float(T)

    A = _amplitude_given_spikes(Y, centers, h, ge, T) if len(centers) else None
    if A is None and len(masses):
        A = float(np.percentile(masses, 25))
    if A is None:
        A = sn
    if len(g) == 2 and g[0] ** 2 + 4 * g[1] > 0:
        A = A / np.sqrt(g[0] ** 2 + 4 * g[1])
    SAM['A_'] = A
    return SAM


def run_omsi_init_comparison(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'omsi_init_comparison.npz')

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

    print('Running OMSI with NNLS init (full population)...')
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

    print('Running OMSI with FOOPSI init (full population)...')
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

    for tag, t_, ns_, fb_ in [('NNLS  ', nnls_times,   nnls_nsamples,   nnls_fb),
                              ('FOOPSI', foopsi_times, foopsi_nsamples, foopsi_fb)]:
        print('{} -- time/cell: {:.2f} ± {:.2f}s  '
              'samples: {:.0f} ± {:.0f}  '
              'Fb: {:.3f} ± {:.3f}'.format(
                  tag, np.median(t_), _mad(t_), np.median(ns_), _mad(ns_),
                  np.nanmedian(fb_), _mad(fb_)))

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


def plot_omsi_init_comparison(data_dir):

    out_path = os.path.join(data_dir, 'omsi_init_comparison.npz')
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
            return (x - gt_lo) / (gt_hi - gt_lo + 1e-12)

        def _scale_prob(x):
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

    print('Median $F_\\beta$ (β=0.5) -- NNLS init: {:.3f} ± {:.3f}  FOOPSI init: {:.3f} ± {:.3f}'.format(
        np.nanmedian(valid_n), _mad(valid_n), np.nanmedian(valid_f), _mad(valid_f)))
    print('Median time per cell (sec) -- NNLS init: {:.3f} ± {:.3f}  FOOPSI init: {:.3f} ± {:.3f}'.format(
        np.median(nnls_times), _mad(nnls_times), np.median(foopsi_times), _mad(foopsi_times)))
    ax_f.set_xlabel('$F_\\beta$')
    ax_f.set_ylabel('cells')

    ax_f.legend(frameon=False, fontsize=6, loc='upper left', reverse=True)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'omsi_init_comparison.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def plot_combined_init(data_dir):

    init_path = os.path.join(data_dir, 'init_comparison.npz')
    conv_path = os.path.join(data_dir, 'omsi_init_comparison.npz')
    for p in (init_path, conv_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'No data at {p}. Run the corresponding test mode first.')

    d1 = np.load(init_path, allow_pickle=True)
    clean_traces      = d1['clean_traces']
    nnls_sp           = d1['nnls_sp'].astype(np.float64)
    foopsi_sp         = d1['foopsi_sp'].astype(np.float64)
    true_spikes       = d1['true_spikes']
    r_cross           = _spike_corrs(d1['nnls_sp'], d1['foopsi_sp'], float(d1['fs'][0]))
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
    ax_ti.plot([0,300],[0,300], color='tab:cyan', alpha=0.5)
    ax_ti.set_xlim([0,300])
    ax_ti.set_ylim([0,300])

    ax_rc = axd['r_cross']
    r_cross_valid = r_cross[np.isfinite(r_cross)]
    ax_rc.hist(r_cross_valid, bins=np.linspace(0,1,12), color='k', linewidth=0.3)
    ax_rc.set_xlabel('spike correlation')
    ax_rc.set_ylabel('cells')
    ax_rc.set_xlim([0, 1])
    print('NNLS vs FOOPSI spike correlation ({:.0f} ms smoothing): median {:.3f} ± {:.3f} (n={})'.format(
        _SPIKE_CORR_SMOOTH * 1000, np.median(r_cross_valid), _mad(r_cross_valid), len(r_cross_valid)))

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


_DEFAULT_CONV_TOL = 0.05
_DEFAULT_BURN_TOL = 1e-3

_CONV_TEST_MIN_SWEEPS = 50
_BURN_TEST_MIN_SWEEPS = 50

_TEST_B           = 5
_TEST_WIN         = 5
_TEST_CHECK_EVERY = 5

_TEST_N_CELLS  = 100
_TEST_DURATION = 1200.0

_FIG1_SNR_RANGE       = (2.0, 20.0)
_FIG1_AMP_CV          = 0.1
_FIG1_TAU_CV          = 0.05
_FIG1_DRIFT_SD        = 0.2
_FIG1_DRIFT_TIMESCALE = 60.0


def _fig1_population(n_cells, duration, snr=None):

    if snr is None:
        lo, hi = _FIG1_SNR_RANGE
        snr = np.exp(np.random.uniform(np.log(lo), np.log(hi), n_cells))
    else:
        snr = np.full(n_cells, float(snr))
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=_FS, duration=duration, tau=_TAU, snr=snr,
        amp_cv=_FIG1_AMP_CV, tau_cv=_FIG1_TAU_CV, drift_sd=_FIG1_DRIFT_SD,
        drift_timescale=_FIG1_DRIFT_TIMESCALE)
    return dff, true_spikes, snr

_MAX_SWEEPS_DEFAULT = 1000

def _build_tol_grid(default_val, lower_mult, upper_mult, n_below=4, n_above=5):

    n_total = n_below + n_above + 1
    return np.geomspace(default_val * lower_mult, default_val * upper_mult, n_total).tolist()


def _run_tol_sweep(dff, true_spikes, param_name, grid, fs, min_sweeps):

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
                'med_time':     float(np.median(per_cell_t)),
                'mad_time':     float(_mad(per_cell_t)),
                'med_nsweeps':  float(np.median(nsweeps)),
                'mad_nsweeps':  float(_mad(nsweeps)),
                'med_f1':       float(np.nanmedian(fb)),
                'mad_f1':       float(_mad(fb)),
                'med_cosmic':   float(np.nanmedian(cosmic)),
                'mad_cosmic':   float(_mad(cosmic)),
            })
            r = rows[-1]
            print('    cell={:.3f} ± {:.3f}s  '
                  'sweeps={:.1f} ± {:.1f}  '
                  'F_beta={:.3f} ± {:.3f}  CosMIC={:.3f} ± {:.3f}'.format(
                      r['med_time'], r['mad_time'],
                      r['med_nsweeps'], r['mad_nsweeps'],
                      r['med_f1'], r['mad_f1'],
                      np.nanmedian(cosmic), _mad(cosmic)))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))
    return rows


def _save_tol_sweep(out_path, rows, default_val):

    np.savez(
        out_path,
        tol          = np.array([r['val']          for r in rows]),
        med_time    = np.array([r['med_time']     for r in rows]),
        mad_time     = np.array([r['mad_time']      for r in rows]),
        med_nsweeps = np.array([r['med_nsweeps']  for r in rows]),
        mad_nsweeps  = np.array([r['mad_nsweeps']   for r in rows]),
        med_f1      = np.array([r['med_f1']       for r in rows]),
        mad_f1       = np.array([r['mad_f1']        for r in rows]),
        med_cosmic  = np.array([r['med_cosmic']   for r in rows]),
        mad_cosmic   = np.array([r['mad_cosmic']    for r in rows]),
        default_val  = np.array([default_val]),
    )
    print('\nSaved to {}.'.format(out_path))


def run_conv_tol_sweep(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'conv_tol_sweep.npz')

    print('Generating figure-1 population (n={}, T={}s, fs={}Hz, tau={}s)...'.format(
        _TEST_N_CELLS, _TEST_DURATION, _FS, _TAU))
    dff, true_spikes, _ = _fig1_population(_TEST_N_CELLS, _TEST_DURATION)

    grid = _build_tol_grid(_DEFAULT_CONV_TOL, lower_mult=0.001, upper_mult=10.0)
    print('  Sweep ({} conv_tol values): {}'.format(
        len(grid), ['{:.2e}'.format(v) for v in grid]))

    rows = _run_tol_sweep(dff, true_spikes, 'conv_tol', grid, _FS, _CONV_TEST_MIN_SWEEPS)
    if rows:
        _save_tol_sweep(out_path, rows, _DEFAULT_CONV_TOL)
    else:
        print('No results collected.')


def run_burn_tol_sweep(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'burn_tol_sweep.npz')

    print('Generating figure-1 population (n={}, T={}s, fs={}Hz, tau={}s)...'.format(
        _TEST_N_CELLS, _TEST_DURATION, _FS, _TAU))
    dff, true_spikes, _ = _fig1_population(_TEST_N_CELLS, _TEST_DURATION)

    grid = _build_tol_grid(_DEFAULT_BURN_TOL, lower_mult=0.002, upper_mult=4.0)
    print('  Sweep ({} burn_tol values): {}'.format(
        len(grid), ['{:.2e}'.format(v) for v in grid]))

    rows = _run_tol_sweep(dff, true_spikes, 'burn_tol', grid, _FS, _BURN_TEST_MIN_SWEEPS)
    if rows:
        _save_tol_sweep(out_path, rows, _DEFAULT_BURN_TOL)
    else:
        print('No results collected.')


def _plot_tol_sweep(data_dir, npz_name, xlabel, fig_stem):

    out_path = os.path.join(data_dir, npz_name)
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}.')

    d           = np.load(out_path)
    tols        = d['tol'].astype(float)
    mt          = d['med_time']
    st          = d['mad_time']
    mf1         = d['med_f1']
    sf1         = d['mad_f1']
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

    _plot_tol_sweep(data_dir, 'conv_tol_sweep.npz',
                    'convergence threshold', 'conv_tol_sweep')


def plot_burn_tol_sweep(data_dir):

    _plot_tol_sweep(data_dir, 'burn_tol_sweep.npz',
                    'burn-in completion threshold', 'burn_tol_sweep')


def plot_combined_opt(data_dir):

    t_supp_path   = os.path.join(data_dir, 'T_supp_sweep.npz')
    conv_tol_path = os.path.join(data_dir, 'conv_tol_sweep.npz')
    burn_tol_path = os.path.join(data_dir, 'burn_tol_sweep.npz')
    snr_path      = os.path.join(data_dir, 'snr_threshold_sweep.npz')
    for p in (t_supp_path, conv_tol_path, burn_tol_path, snr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'No data at {p}.')

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 9.0), dpi=300)

    def _sweeps_panel(ax, x, mns, sns, xlabel):

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
    mt           = d['med_time']
    st           = d['mad_time']
    mf1          = d['med_f1']
    sf1          = d['mad_f1']
    mns          = d['med_nsweeps']
    sns          = d['mad_nsweeps']
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
    mt          = d['med_time']
    st          = d['mad_time']
    mf1         = d['med_f1']
    sf1         = d['mad_f1']
    mns         = d['med_nsweeps']
    sns         = d['mad_nsweeps']
    default_val = _DEFAULT_CONV_TOL

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
    mt          = d['med_time']
    st          = d['mad_time']
    mf1         = d['med_f1']
    sf1         = d['mad_f1']
    mns         = d['med_nsweeps']
    sns         = d['mad_nsweeps']
    default_val = _DEFAULT_BURN_TOL

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
    med_fb     = d['med_fb'].astype(float)
    mad_fb      = d['mad_fb'].astype(float)
    med_cosmic = d['med_cosmic'].astype(float)
    mad_cosmic  = d['mad_cosmic'].astype(float)
    med_ns     = d['med_nsweeps'].astype(float)
    mad_ns      = d['mad_nsweeps'].astype(float)
    threshold   = float(d['threshold'][0])

    for ax, med, mad, ylabel in [
        (axes[3, 0], med_fb,     mad_fb,     '$F_\\beta$'),
        (axes[3, 1], med_cosmic, mad_cosmic, 'CosMIC'),
    ]:
        valid = np.isfinite(med) & np.isfinite(mad)
        x, y, ye = snr_levels[valid], med[valid], mad[valid]
        ax.fill_between(x, np.clip(y - ye, 0, 1), np.clip(y + ye, 0, 1),
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, y, '.-', color=_COLOR, zorder=3)
        ax.axvline(threshold, color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1.)

    valid_ns = np.isfinite(med_ns) & np.isfinite(mad_ns)
    ax = axes[3, 2]
    ax.fill_between(snr_levels[valid_ns],
                    med_ns[valid_ns] - mad_ns[valid_ns],
                    med_ns[valid_ns] + mad_ns[valid_ns],
                    color=_COLOR, alpha=0.25, linewidth=0)
    ax.plot(snr_levels[valid_ns], med_ns[valid_ns], '.-', color=_COLOR, zorder=3)
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


_ADD_MOVE_DURATION = 120.0
_ADD_MOVE_N_CELLS  = 50
_ADD_MOVE_GRID     = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def run_add_move_sweep(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'add_move_sweep.npz')

    n_frames    = int(_ADD_MOVE_DURATION * _FS)
    default_val = int(np.ceil(n_frames / 500))
    grid        = _ADD_MOVE_GRID
    print('add_move sweep: {} (default={}, T={} frames, {} cells)'.format(
        grid, default_val, n_frames, _ADD_MOVE_N_CELLS))

    dff, true_spikes, _ = _fig1_population(_ADD_MOVE_N_CELLS, _ADD_MOVE_DURATION)

    n = len(grid)
    med_fb, mad_fb = np.full(n, np.nan), np.full(n, np.nan)
    med_cosmic, mad_cosmic = np.full(n, np.nan), np.full(n, np.nan)
    med_time, med_nsweeps = np.full(n, np.nan), np.full(n, np.nan)

    for k, val in enumerate(grid):
        print('\n  [{}/{}] add_move={} ...'.format(k + 1, n, val))
        params = {
            'f': _FS, 'p': 2, 'auto_stop': True, 'upd_gam': 0,
            'conv_tol': _DEFAULT_CONV_TOL, 'burn_tol': _DEFAULT_BURN_TOL,
            'add_move': int(val),
        }
        try:
            res = OMSI.deconv(dff, params=params, benchmark=True)
            pred = res['optim_spikes']
            prec_s, rec_s, _ = helpers.compute_accuracy_strict(true_spikes, pred)
            fb = np.array([_fbeta(float(prec_s[i]), float(rec_s[i]))
                           for i in range(len(prec_s))])
            cosmic = helpers.compute_cosmic(true_spikes, pred, _FS)
            med_fb[k], mad_fb[k] = float(np.nanmedian(fb)), float(_mad(fb))
            med_cosmic[k], mad_cosmic[k] = float(np.nanmedian(cosmic)), float(_mad(cosmic))
            med_time[k] = float(np.median(res['optim_times_per_cell']))
            med_nsweeps[k] = float(np.median(res['optim_nsamples']))
            print('    F_beta={:.3f} ± {:.3f}  CosMIC={:.3f} ± {:.3f}  '
                  'cell={:.3f}s  sweeps={:.1f}'.format(
                      med_fb[k], mad_fb[k], med_cosmic[k], mad_cosmic[k],
                      med_time[k], med_nsweeps[k]))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))

    np.savez(
        out_path,
        add_move=np.array(grid), med_fb=med_fb, mad_fb=mad_fb,
        med_cosmic=med_cosmic, mad_cosmic=mad_cosmic,
        med_time=med_time, med_nsweeps=med_nsweeps,
        default_val=np.array([default_val]),
    )
    print('\nSaved to {}.'.format(out_path))


def _band(ax, x, y, e, color, label=None, clip01=False):

    lo, hi = y - e, y + e
    if clip01:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, y, '-o', ms=3, color=color, zorder=3, label=label)


def _mean_boot(v, n_boot=1000, seed=0):

    rng = np.random.RandomState(seed)
    mean, lo, hi = (np.full(len(v), np.nan) for _ in range(3))
    for i, row in enumerate(np.asarray(v, dtype=float)):
        row = row[np.isfinite(row)]
        if len(row) == 0:
            continue
        boots = row[rng.randint(0, len(row), (n_boot, len(row)))].mean(axis=1)
        mean[i] = row.mean()
        lo[i], hi[i] = np.percentile(boots, [2.5, 97.5])
    return mean, lo, hi


def _sweeps_mean(ax, x, d, name, idx=None, color='k'):

    if 'nsweeps_cells' not in d.files:
        raise KeyError('{} has no per-cell sweep counts -- rerun its test mode.'.format(name))
    cells = d['nsweeps_cells'] if idx is None else d['nsweeps_cells'][idx]
    m, lo, hi = _mean_boot(cells)
    ok = np.isfinite(m)
    ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.2, linewidth=0)
    ax.plot(x[ok], m[ok], '-o', ms=3, color=color, zorder=3)
    ax.set_ylabel('sweep count (mean)')
    ax.set_ylim(0, max(_MAX_SWEEPS_DEFAULT, float(np.nanmax(hi[ok])) * 1.05))


def _default_line(ax, x):

    ax.axvline(x, color='k', linestyle='--', linewidth=0.8, alpha=0.6)


_AM_FB_COLOR     = 'c'
_AM_COSMIC_COLOR = 'r'
_AM_LINE_COLOR   = 'k'


def plot_combined_opt_add_move(data_dir):

    from matplotlib.lines import Line2D

    names = ('add_move_sweep', 'add_move_duration', 'T_supp_sweep', 'conv_tol_sweep',
             'burn_tol_sweep', 'snr_threshold_sweep')
    paths = {n: os.path.join(data_dir, n + '.npz') for n in names}
    for pth in paths.values():
        if not os.path.exists(pth):
            raise FileNotFoundError(f'No data at {pth}.')

    fig = plt.figure(figsize=(9.6, 7.2), dpi=300)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.6, wspace=0.45)
    panel = iter([(r, c) for r in range(3) for c in range(4)])

    def _next_ax():
        return fig.add_subplot(gs[next(panel)])

    def _accuracy(ax, x, fb, fb_e, cos=None, cos_e=None):
        ok = np.isfinite(fb) & np.isfinite(fb_e)
        _band(ax, x[ok], fb[ok], fb_e[ok], _AM_FB_COLOR, label='$F_\\beta$', clip01=True)
        if cos is not None:
            ok = np.isfinite(cos) & np.isfinite(cos_e)
            _band(ax, x[ok], cos[ok], cos_e[ok], _AM_COSMIC_COLOR, label='CosMIC', clip01=True)
        ax.set_ylabel('accuracy')
        ax.set_ylim(0, 1)

    def _cosmic_or_none(d, name):
        if 'med_cosmic' in d.files:
            return d['med_cosmic'], d['mad_cosmic']
        print('  {} has no CosMIC -- rerun its test mode to add it; showing F_beta only.'.format(name))
        return None, None

    d = np.load(paths['T_supp_sweep'])
    idx = np.arange(len(d['T_supp']))[1:-1]
    idx = np.hstack([idx[0:4], idx[5], idx[7:]]).astype(int)
    xs = d['T_supp'].astype(float)[idx] / _FS
    cos, cos_e = _cosmic_or_none(d, 'T_supp_sweep.npz')
    ax_acc = _next_ax()
    _accuracy(ax_acc, xs, d['med_f1'][idx], d['mad_f1'][idx],
              None if cos is None else cos[idx], None if cos is None else cos_e[idx])
    ax = _next_ax()
    _band(ax, xs, d['med_time'][idx], d['mad_time'][idx], _AM_LINE_COLOR)
    ax.set_ylim(0, 250)
    ax.set_ylabel('time per cell (sec)')
    _sweeps_mean(_next_ax(), xs, d, 'T_supp_sweep.npz', idx, color=_AM_LINE_COLOR)
    for ax in fig.axes[-3:]:
        _default_line(ax, int(d['default_supp'][0]) / _FS)
        ax.set_xscale('log')
        ax.set_xlim(xs.min(), xs.max())
        ax.set_xlabel('$T_{supp}$ (sec)')

    d = np.load(paths['add_move_sweep'])
    x, default_am = d['add_move'].astype(float), float(d['default_val'][0])
    ax_a = _next_ax()
    _accuracy(ax_a, x, d['med_fb'], d['mad_fb'], d['med_cosmic'], d['mad_cosmic'])
    ax_t = _next_ax()
    ax_t.plot(x, d['med_time'], '-o', ms=3, color=_AM_LINE_COLOR)
    ax_t.set_ylabel('time per cell (sec)')
    ax_t.set_ylim(bottom=0)
    for a in (ax_a, ax_t):
        a.set_xlim(x.min(), x.max())

    dd = np.load(paths['add_move_duration'])
    grid, secs = dd['grid'].astype(float), dd['frames'] / _FS
    conv, lo_c, hi_c, _ = _add_move_conv(dd['ns'])
    colors = plt.cm.jet(np.linspace(0, 1, len(secs)))
    ax_s = _next_ax()
    for i in range(len(secs)):
        ax_s.fill_between(grid, lo_c[i], hi_c[i], color=colors[i], alpha=0.15, linewidth=0)
        ax_s.plot(grid, conv[i], '-o', ms=3, color=colors[i], lw=0.9,
                  label='{:g} s'.format(round(secs[i])))
    ax_s.set_xlim(grid.min(), grid.max())
    ax_s.set_ylim(bottom=0)
    ax_s.set_ylabel('sweeps to converge')
    ax_s.legend(frameon=False, fontsize=5, title='duration', title_fontsize=5,
                handlelength=1.2, loc='upper right')
    for a in (ax_a, ax_t, ax_s):
        _default_line(a, default_am)
        a.set_xscale('log', base=2)
        a.set_xlabel('add/remove proposals per sweep')

    for name, default, xlabel in [
            ('conv_tol_sweep', _DEFAULT_CONV_TOL, 'convergence threshold'),
            ('burn_tol_sweep', _DEFAULT_BURN_TOL, 'burn-in threshold')]:
        d = np.load(paths[name])
        xt = d['tol'].astype(float)
        cos, cos_e = _cosmic_or_none(d, name + '.npz')
        ax = _next_ax()
        _accuracy(ax, xt, d['med_f1'], d['mad_f1'], cos, cos_e)
        ax_t = _next_ax()
        _band(ax_t, xt, d['med_time'], d['mad_time'], _AM_LINE_COLOR)
        ax_t.set_ylabel('time per cell (sec)')
        ax_t.set_ylim(bottom=0)
        for a in (ax, ax_t):
            _default_line(a, default)
            a.set_xscale('log')
            a.set_xlim(xt.min(), xt.max())
            a.set_xlabel(xlabel)

    d = np.load(paths['snr_threshold_sweep'])
    snr, thr = d['snr_levels'].astype(float), float(d['threshold'][0])
    ax = _next_ax()
    _accuracy(ax, snr, d['med_fb'], d['mad_fb'], d['med_cosmic'], d['mad_cosmic'])
    ax_n = _next_ax()
    _sweeps_mean(ax_n, snr, d, 'snr_threshold_sweep.npz', color=_AM_LINE_COLOR)
    for a in (ax, ax_n):
        _default_line(a, thr)
        a.set_xlim(snr.min(), snr.max())
        a.set_xlabel('SNR')

    handles = [Line2D([], [], color=_AM_FB_COLOR, label='$F_\\beta$'),
               Line2D([], [], color=_AM_COSMIC_COLOR, label='CosMIC'),
               Line2D([], [], color='k', ls='--', lw=0.8, alpha=0.6, label='default')]
    ax_acc.legend(handles=handles, loc='lower right', frameon=False, fontsize=6,
                  handlelength=1.5)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'combined_opt_add_move.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def plot_add_move_sweep(data_dir):

    plot_combined_opt_add_move(data_dir)


def run_combined_opt_add_move(data_dir):

    for name, run in [('add/remove proposals', run_add_move_sweep),
                      ('add/remove x duration', run_add_move_duration_sweep),
                      ('T_supp', run_T_supp_sweep),
                      ('convergence threshold', run_conv_tol_sweep),
                      ('burn-in threshold', run_burn_tol_sweep),
                      ('SNR threshold', run_snr_threshold_sweep)]:
        print('\n=== {} sweep ==='.format(name))
        t0 = time.time()
        run(data_dir)
        print('=== {} sweep done in {:.1f} min ==='.format(name, (time.time() - t0) / 60))
    plot_combined_opt_add_move(data_dir)


def _pm(m, e, fmt='{:.3f}'):
    return (fmt + ' ± ' + fmt).format(m, e)


def _nearest(x, v):
    x = np.asarray(x, dtype=float)
    if np.all(x > 0) and v > 0:
        return int(np.argmin(np.abs(np.log(x) - np.log(v))))
    return int(np.argmin(np.abs(x - v)))


def _p_fmt(res):
    return 'n/a' if not np.isfinite(res['p']) else '{:.2g}'.format(res['p'])


def _sweep_rows(rows, panel, x, x_fmt, default, fb, fb_e, cos, cos_e, t, t_e):

    i0 = _nearest(x, default)
    for i in range(len(x)):
        note = 'default' if i == i0 and np.isclose(x[i], default, rtol=1e-6) else (
            'nearest to default {}'.format(x_fmt.format(default)) if i == i0 else '')
        rows.append((panel, x_fmt.format(x[i]),
                     _pm(fb[i], fb_e[i]) if np.isfinite(fb[i]) else 'n/a',
                     _pm(cos[i], cos_e[i]) if cos is not None and np.isfinite(cos[i]) else 'n/a',
                     (_pm(t[i], t_e[i], '{:.2f}') if t_e is not None else '{:.2f}'.format(t[i]))
                     if np.isfinite(t[i]) else 'n/a',
                     note))
    return i0


def _print_table(header, rows):

    widths = [max(len(str(r[k])) for r in [header] + rows) for k in range(len(header))]
    line = '  '.join('{:<' + str(w) + '}' for w in widths)
    print(line.format(*header))
    print('  '.join('-' * w for w in widths))
    for r in rows:
        print(line.format(*[str(v) for v in r]))


def print_results(data_dir):

    from stats_helpers import signed_rank

    def _load(name):
        p = os.path.join(data_dir, name + '.npz')
        if not os.path.exists(p):
            print('[{} not found -- run its test mode.]'.format(p))
            return None
        return np.load(p, allow_pickle=True)

    print('\ncombined_init_comparison (median ± MAD across cells; paired Wilcoxon '
          'signed-rank, NNLS vs FOOPSI)\n')
    rows = []
    d1, d2 = _load('init_comparison'), _load('omsi_init_comparison')
    if d1 is not None:
        fs = float(d1['fs'][0])
        n, T = d1['nnls_sp'].shape
        rows.append(('population', '{} cells, {:.0f} s, {:g} Hz, tau {:g} s'.format(
            n, T / fs, fs, _TAU), '', '', '', ''))
        tn, tf = d1['nnls_times'] * 1e3, d1['foopsi_times'] * 1e3
        res = signed_rank(tn, tf)
        rows.append(('init time (ms/cell)', '', _pm(np.median(tn), _mad(tn), '{:.1f}'),
                     _pm(np.median(tf), _mad(tf), '{:.1f}'), _p_fmt(res),
                     'FOOPSI/NNLS {:.1f}x; NNLS faster on {:.0f}% of cells'.format(
                         np.median(tf) / np.median(tn), 100 * np.mean(tn < tf))))
        r = _spike_corrs(d1['nnls_sp'], d1['foopsi_sp'], fs)
        r = r[np.isfinite(r)]
        rows.append(('NNLS-FOOPSI spike corr.', '{:.0f} ms smoothing'.format(
                         _SPIKE_CORR_SMOOTH * 1e3),
                     _pm(np.median(r), _mad(r)), '', '',
                     'n={}; IQR {:.3f}-{:.3f}; {:.0f}% > 0.8'.format(
                         len(r), *np.percentile(r, [25, 75]), 100 * np.mean(r > 0.8))))
    if d2 is not None:
        for label, a, b, fmt, scale in [
                ('OMSI time (s/cell)', d2['nnls_times'], d2['foopsi_times'], '{:.2f}', 1),
                ('OMSI posterior samples', d2['nnls_nsamples'], d2['foopsi_nsamples'], '{:.0f}', 1),
                ('OMSI F_beta', d2['nnls_fb'], d2['foopsi_fb'], '{:.3f}', 1)]:
            a, b = np.asarray(a, float) * scale, np.asarray(b, float) * scale
            res = signed_rank(a, b)
            rows.append((label, '', _pm(np.nanmedian(a), _mad(a), fmt),
                         _pm(np.nanmedian(b), _mad(b), fmt), _p_fmt(res),
                         'n={}; median paired diff {}'.format(
                             res['n'], ('{:+' + fmt[2:]).format(res['med_diff']))))
        fn, ff = d2['nnls_fb'], d2['foopsi_fb']
        rows.append(('F_beta = 0 (not in scatter)', '', str(int(np.sum(fn <= 0))),
                     str(int(np.sum(ff <= 0))), '', 'cells'))
    _print_table(('quantity', 'setting', 'NNLS', 'FOOPSI', 'p', 'note'), rows)

    print('\n\ncombined_opt_add_move (F_beta, CosMIC, time: median ± MAD across cells)\n')
    rows = []
    sweep_rows = []

    d = _load('T_supp_sweep')
    if d is not None:
        idx = np.arange(len(d['T_supp']))[1:-1]
        idx = np.hstack([idx[0:4], idx[5], idx[7:]]).astype(int)
        x = d['T_supp'].astype(float)[idx] / _FS
        i0 = _sweep_rows(rows, 'T_supp (s)', x, '{:.2f}', int(d['default_supp'][0]) / _FS,
                         d['med_f1'][idx], d['mad_f1'][idx],
                         d['med_cosmic'][idx] if 'med_cosmic' in d.files else None,
                         d['mad_cosmic'][idx] if 'mad_cosmic' in d.files else None,
                         d['med_time'][idx], d['mad_time'][idx])
        m, lo, hi = _mean_boot(d['nsweeps_cells'][idx])
        for i in range(len(x)):
            sweep_rows.append(('T_supp (s)', '{:.2f}'.format(x[i]),
                               '{:.0f} [{:.0f}-{:.0f}]'.format(m[i], lo[i], hi[i]),
                               'default' if i == i0 else ''))
        rows.append(('', '', '', '', '', ''))
        print('T_supp: {} cells, {:.0f} s each.'.format(
            d['nsweeps_cells'].shape[1], int(d['n_frames'][0]) / _FS))

    d = _load('add_move_sweep')
    if d is not None:
        _sweep_rows(rows, 'add/remove per sweep', d['add_move'].astype(float), '{:g}',
                    float(d['default_val'][0]), d['med_fb'], d['mad_fb'],
                    d['med_cosmic'], d['mad_cosmic'], d['med_time'], None)
        rows.append(('', '', '', '', '', ''))
        print('add/remove: {} cells, {:.0f} s each (time column is median only).'.format(
            _ADD_MOVE_N_CELLS, _ADD_MOVE_DURATION))

    for name, default, panel in [('conv_tol_sweep', _DEFAULT_CONV_TOL, 'convergence threshold'),
                                 ('burn_tol_sweep', _DEFAULT_BURN_TOL, 'burn-in threshold')]:
        d = _load(name)
        if d is None:
            continue
        _sweep_rows(rows, panel, d['tol'].astype(float), '{:.2g}', default,
                    d['med_f1'], d['mad_f1'],
                    d['med_cosmic'] if 'med_cosmic' in d.files else None,
                    d['mad_cosmic'] if 'mad_cosmic' in d.files else None,
                    d['med_time'], d['mad_time'])
        rows.append(('', '', '', '', '', ''))
    print('Thresholds: {} cells, {:.0f} s each, min_sweeps {}.'.format(
        _TEST_N_CELLS, _TEST_DURATION, _CONV_TEST_MIN_SWEEPS))

    d = _load('snr_threshold_sweep')
    if d is not None:
        snr, thr = d['snr_levels'].astype(float), float(d['threshold'][0])
        m, lo, hi = _mean_boot(d['nsweeps_cells'])
        for i in range(len(snr)):
            rows.append(('SNR', '{:.2f}'.format(snr[i]), _pm(d['med_fb'][i], d['mad_fb'][i]),
                         _pm(d['med_cosmic'][i], d['mad_cosmic'][i]), '',
                         'simulated SNR below threshold {:g}'.format(thr) if snr[i] < thr else ''))
            sweep_rows.append(('SNR', '{:.2f}'.format(snr[i]),
                               '{:.0f} [{:.0f}-{:.0f}]'.format(m[i], lo[i], hi[i]),
                               'simulated SNR below threshold' if snr[i] < thr else ''))
        print('SNR: {} cells per level, {:.0f} s each.'.format(
            d['nsweeps_cells'].shape[1], _SNR_DURATION))
    print()
    _print_table(('panel', 'value', 'F_beta', 'CosMIC', 'time/cell (s)', 'note'), rows)

    print('\nSweep counts (mean [95% bootstrap CI] across cells)\n')
    _print_table(('panel', 'value', 'sweeps', 'note'), sweep_rows)

    d = _load('add_move_duration')
    if d is not None:
        frames, grid = d['frames'], d['grid'].astype(float)
        conv, lo_c, hi_c, _ = _add_move_conv(d['ns'])
        print('\nSweeps to converge (population; [IQR] from bootstrapping cells; '
              '* = default ceil(T/500)). {} cells, {} sweeps per chain, threshold {:.0f}%.\n'.format(
                  d['ns'].shape[2], _ADM_SWEEPS, 100 * _ADM_CONV_THR))
        rows = []
        for i, f in enumerate(frames):
            rows.append(['{:g} s'.format(round(f / _FS))] + [
                '{:.0f} [{:.0f}-{:.0f}]{}'.format(conv[i, a], lo_c[i, a], hi_c[i, a],
                                                  '*' if grid[a] == np.ceil(f / 500) else '')
                for a in range(len(grid))])
        _print_table(['duration'] + ['{:g}'.format(g) for g in grid], rows)


_ADM_FRAMES    = [1000, 2000, 4000, 8000, 16000]
_ADM_GRID      = [1, 2, 4, 8, 16, 32, 64, 128]
_ADM_N_CELLS   = 100
_ADM_SEED      = 7
_ADM_SWEEPS    = 2000
_ADM_CONV_THR  = 0.10
_ADM_SMOOTH    = 10
_ADM_FLOOR     = 25
_ADM_PLATEAU_SLACK = 2.0


def _rel_err_curves(ns, ref):

    k = _ADM_SMOOTH
    c = np.cumsum(np.insert(np.asarray(ns, dtype=float), 0, 0.0, axis=-1), axis=-1)
    sm = (c[..., k:] - c[..., :-k]) / k
    return np.abs(sm - ref[..., None]) / np.maximum(ref[..., None], 1.0)


def _pop_conv_sweeps(err):

    med = np.median(err, axis=0)
    bad = np.where(med > _ADM_CONV_THR)[0]
    return 0 if len(bad) == 0 else int(bad[-1] + _ADM_SMOOTH)


def _add_move_dur_task(args):

    from OMSI.sampler import cont_ca_sampler
    y, add_move, seed = args
    params = {
        'f': _FS, 'p': 2, 'upd_gam': 0, 'auto_stop': False,
        'B': 0, 'Nsamples': _ADM_SWEEPS, 'add_move': int(add_move),
        'return_full': True, 'seed': int(seed),
    }

    np.random.seed(int(seed))
    t0 = time.time()
    res = cont_ca_sampler(y, params)
    elapsed = time.time() - t0
    ns = np.asarray(res['chain']['ns'], dtype=np.int32)
    out = np.full(_ADM_SWEEPS, -1, dtype=np.int32)
    out[:min(len(ns), _ADM_SWEEPS)] = ns[:_ADM_SWEEPS]
    return out, elapsed


def run_add_move_duration_sweep(data_dir, n_workers=None):

    from multiprocessing import Pool
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'add_move_duration.npz')

    nT, nA, nC = len(_ADM_FRAMES), len(_ADM_GRID), _ADM_N_CELLS
    ns_all = np.full((nT, nA, nC, _ADM_SWEEPS), -1, dtype=np.int32)
    times = np.full((nT, nA, nC), np.nan)

    np.random.seed(_ADM_SEED)
    dff_full, _, snr = _fig1_population(nC, (max(_ADM_FRAMES) + 0.5) / _FS)
    assert dff_full.shape[1] == max(_ADM_FRAMES), dff_full.shape

    with Pool(n_workers) as pool:
        for i, n_frames in enumerate(_ADM_FRAMES):
            dff = dff_full[:, :n_frames]

            tasks = [(dff[c], am, _ADM_SEED + c)
                     for am in _ADM_GRID for c in range(nC)]
            print('\n[{}/{}] T={} frames (default add_move={}): {} chains'.format(
                i + 1, nT, n_frames, int(np.ceil(n_frames / 500)), len(tasks)))
            t0 = time.time()
            for j, (ns, el) in enumerate(pool.imap(_add_move_dur_task, tasks)):
                a, c = divmod(j, nC)
                ns_all[i, a, c] = ns
                times[i, a, c] = el
            print('  done in {:.0f}s'.format(time.time() - t0))
            np.savez(out_path, frames=np.array(_ADM_FRAMES[:i + 1]),
                     grid=np.array(_ADM_GRID), ns=ns_all[:i + 1], times=times[:i + 1],
                     snr=snr)
    print('\nSaved to {}.'.format(out_path))


def _add_move_conv(ns, n_boot=200):

    nT, nA, nC, nS = ns.shape
    rng = np.random.RandomState(0)

    tail = ns[..., -nS // 4:]
    ref = np.median(np.median(tail, axis=-1), axis=1)

    err = _rel_err_curves(ns, ref[:, None, :])
    conv = np.array([[_pop_conv_sweeps(err[i, a]) for a in range(nA)] for i in range(nT)])

    boot_idx = rng.randint(0, nC, (n_boot, nC))
    conv_b = np.array([[[_pop_conv_sweeps(err[i, a][idx]) for idx in boot_idx]
                        for a in range(nA)] for i in range(nT)])
    lo, hi = np.percentile(conv_b, 25, axis=-1), np.percentile(conv_b, 75, axis=-1)
    return conv, lo, hi, conv_b


def plot_add_move_duration(data_dir):

    path = os.path.join(data_dir, 'add_move_duration.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No data at {path}. Run --mode add-move-dur-test first.')
    d = np.load(path)
    frames, grid, times = d['frames'], d['grid'], d['times']
    nT, nA, nC, nS = d['ns'].shape
    conv, lo_c, hi_c, conv_b = _add_move_conv(d['ns'])
    n_boot = conv_b.shape[-1]

    cmap = plt.get_cmap('viridis')
    colors = [cmap(v) for v in np.linspace(0.05, 0.9, nT)]
    default = np.ceil(frames / 500).astype(int)
    a_def = np.array([int(np.where(grid == v)[0][0]) for v in default])
    a_fixed = int(np.where(grid == 8)[0][0])
    x = frames / _FS

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes

    for i in range(nT):
        ax_a.fill_between(grid, lo_c[i], hi_c[i], color=colors[i], alpha=0.15, linewidth=0)
        ax_a.plot(grid, conv[i], '-o', ms=3, color=colors[i],
                  label='{:g} s'.format(round(x[i])))
        ax_a.plot(grid[a_def[i]], conv[i, a_def[i]], '*', color=colors[i], mec='k',
                  mew=0.5, ms=8, zorder=4)
    ax_a.set_xscale('log', base=2)
    ax_a.set_xlabel('add/remove proposals per sweep')
    ax_a.set_ylabel('sweeps to converge')
    ax_a.legend(frameon=False, fontsize=5, title='duration', title_fontsize=5)

    idx_sc = (np.arange(nT), a_def)
    for series, series_lo, series_hi, col, lab in [
            (conv[:, a_fixed], lo_c[:, a_fixed], hi_c[:, a_fixed], '#C44E52', 'fixed (8)'),
            (conv[idx_sc], lo_c[idx_sc], hi_c[idx_sc], _COLOR, '$\\lceil T/500\\rceil$')]:
        ax_b.fill_between(x, series_lo, series_hi, color=col, alpha=0.2, linewidth=0)
        ax_b.plot(x, series, '.-', color=col, label=lab)
    ax_b.set_xscale('log', base=2)
    ax_b.set_xlabel('recording duration (s)')
    ax_b.set_ylabel('sweeps to converge')
    ax_b.set_ylim(bottom=0)
    ax_b.legend(frameon=False, fontsize=6, title='add/remove', title_fontsize=6)

    t_sweep = times / nS
    for v, col, lab in [(t_sweep[:, a_fixed], '#C44E52', 'fixed (8)'),
                        (t_sweep[idx_sc], _COLOR, '$\\lceil T/500\\rceil$')]:
        ax_c.plot(x, np.median(v, axis=-1) * 1e3, '.-', color=col, label=lab)
    ax_c.set_xscale('log', base=2)
    ax_c.set_yscale('log')
    ax_c.set_xlabel('recording duration (s)')
    ax_c.set_ylabel('time per sweep (ms)')

    def smallest(c):
        thr = _ADM_PLATEAU_SLACK * max(c.min(), _ADM_FLOOR)
        return grid[int(np.argmax(c <= thr))]

    best = np.array([smallest(conv[i]) for i in range(nT)], dtype=float)
    boots = np.array([[smallest(conv_b[i, :, b]) for b in range(n_boot)] for i in range(nT)],
                     dtype=float)
    lo_b, hi_b = np.percentile(boots, 25, axis=1), np.percentile(boots, 75, axis=1)
    ax_d.errorbar(x, best, yerr=[best - lo_b, hi_b - best], fmt='o', color='k', ms=3,
                  lw=0.8, capsize=2, label='smallest reaching plateau')
    xx = np.array([x.min(), x.max()])
    ax_d.plot(xx, xx * _FS / 500, '--', color=_COLOR, lw=0.9, label='$T/500$')
    ax_d.set_xscale('log', base=2)
    ax_d.set_yscale('log', base=2)
    ax_d.set_xlabel('recording duration (s)')
    ax_d.set_ylabel('minimal add/remove per sweep')
    ax_d.legend(frameon=False, fontsize=6)

    for ax, letter in zip(axes, 'abcd'):
        ax.text(-0.25, 1.05, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'add_move_duration.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)



def _dff_snr(fluo):

    f      = np.asarray(fluo, dtype=np.float64)
    sn_mad = float(np.median(np.abs(np.diff(f)))) / 0.6745 if len(f) > 1 else 1e-4
    peak   = float(np.percentile(f, 99))
    base   = float(np.percentile(f,  8))
    return (peak - base) / (sn_mad + 1e-9)


def run_snr_filter_sweep(data_dir):

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

    out_path = os.path.join(data_dir, 'snr_filter_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'No data at {out_path}. Run --mode snr-filter-test first.')

    d        = np.load(out_path)
    snr       = d['snr'].astype(float)
    fbeta    = d['fbeta'].astype(float)
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
    f1_med   = np.full(n_bins, np.nan)
    f1_mad   = np.full(n_bins, np.nan)
    cos_med  = np.full(n_bins, np.nan)
    cos_mad  = np.full(n_bins, np.nan)
    n_bin    = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() < 2:
            continue
        n_bin[b]    = mask.sum()
        f1_med[b]   = np.nanmedian(fbeta[mask])
        f1_mad[b]   = _mad(fbeta[mask])
        cos_med[b]  = np.nanmedian(cosmic[mask])
        cos_mad[b]  = _mad(cosmic[mask])

    valid = n_bin >= 2
    x = bin_centers[valid]

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)
    fig.suptitle('{} cells below SNR threshold (with spikes)'.format(n_below_thresh),
                 fontsize=6, y=1.02)
    for ax, med, mad, ylabel in [
        (axes[0], f1_med[valid],  f1_mad[valid],  '$F_\\beta$'),
        (axes[1], cos_med[valid], cos_mad[valid], 'CosMIC'),
    ]:
        ax.fill_between(x,
                        np.clip(med - mad, 0, 1),
                        np.clip(med + mad, 0, 1),
                        color=_COLOR, alpha=0.25, linewidth=0)
        ax.plot(x, med, '.-', color=_COLOR, zorder=3)
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

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'snr_threshold_sweep.npz')

    snr_levels = np.geomspace(0.3, 3.0 * _SNR_THRESHOLD, _SNR_N_LEVELS)
    print('SNR sweep: {} levels from {:.2f} to {:.2f}  '
          '(threshold={})'.format(
              _SNR_N_LEVELS, snr_levels[0], snr_levels[-1], _SNR_THRESHOLD))
    print('  {} cells x {}s each'.format(_SNR_N_CELLS, _SNR_DURATION))

    med_fb      = np.full(_SNR_N_LEVELS, np.nan)
    mad_fb       = np.full(_SNR_N_LEVELS, np.nan)
    med_cosmic  = np.full(_SNR_N_LEVELS, np.nan)
    mad_cosmic   = np.full(_SNR_N_LEVELS, np.nan)
    med_nsweeps = np.full(_SNR_N_LEVELS, np.nan)
    mad_nsweeps  = np.full(_SNR_N_LEVELS, np.nan)

    params = {
        'f':         _FS,
        'p':         2,
        'auto_stop': True,
        'upd_gam':   0,
        'conv_tol':  _DEFAULT_CONV_TOL,
        'burn_tol':  _DEFAULT_BURN_TOL,
    }

    nsweeps_cells = np.full((_SNR_N_LEVELS, _SNR_N_CELLS), np.nan)
    for k, snr_val in enumerate(snr_levels):
        print('\n  [{}/{}] SNR={:.3f} ...'.format(k + 1, _SNR_N_LEVELS, snr_val))
        dff, true_spikes, _ = _fig1_population(_SNR_N_CELLS, _SNR_DURATION, snr=snr_val)
        try:
            res  = OMSI.deconv(dff, params=params, true_spikes=true_spikes, benchmark=True)
            pred = res['optim_spikes']

            cosmic_v   = helpers.compute_cosmic(true_spikes, pred, _FS)

            med_cosmic[k]  = float(np.nanmedian(cosmic_v))
            mad_cosmic[k]  = float(_mad(cosmic_v))
            med_nsweeps[k] = float(np.median(res['optim_nsamples']))
            mad_nsweeps[k] = float(_mad(res['optim_nsamples']))
            nsweeps_cells[k] = np.asarray(res['optim_nsamples'], dtype=float)

            if res['optim_precision'] is not None:
                fb_arr = np.array([
                    _fbeta(float(res['optim_precision'][i]),
                           float(res['optim_recall'][i]))
                    for i in range(_SNR_N_CELLS)
                ])
                med_fb[k] = float(np.nanmedian(fb_arr))
                mad_fb[k] = float(_mad(fb_arr))

            print('    F_beta={:.3f} ± {:.3f}  CosMIC={:.3f} ± {:.3f}  '
                  'sweeps={:.1f} ± {:.1f}'.format(
                      med_fb[k], mad_fb[k], med_cosmic[k], mad_cosmic[k],
                      med_nsweeps[k], mad_nsweeps[k]))
        except Exception as exc:
            print('    FAILED: {}'.format(exc))

    np.savez(
        out_path,
        snr_levels   = snr_levels,
        med_fb      = med_fb,
        mad_fb       = mad_fb,
        med_cosmic  = med_cosmic,
        mad_cosmic   = mad_cosmic,
        med_nsweeps = med_nsweeps,
        mad_nsweeps  = mad_nsweeps,
        nsweeps_cells = nsweeps_cells,
        threshold    = np.array([_SNR_THRESHOLD]),
    )
    print('\nSaved to {}.'.format(out_path))


def plot_snr_threshold_sweep(data_dir):

    out_path = os.path.join(data_dir, 'snr_threshold_sweep.npz')
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f'No data at {out_path}. Run --mode snr-thresh-test first.')

    d           = np.load(out_path)
    snr_levels  = d['snr_levels'].astype(float)
    med_fb     = d['med_fb'].astype(float)
    mad_fb      = d['mad_fb'].astype(float)
    med_cosmic = d['med_cosmic'].astype(float)
    mad_cosmic  = d['mad_cosmic'].astype(float)
    threshold   = float(d['threshold'][0])

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)
    for ax, med, mad, ylabel in [
        (axes[0], med_fb,     mad_fb,     '$F_\\beta$'),
        (axes[1], med_cosmic, mad_cosmic, 'CosMIC'),
    ]:
        valid = np.isfinite(med) & np.isfinite(mad)
        x, y, ye = snr_levels[valid], med[valid], mad[valid]
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

    traces_dir = os.path.join(fig4_data_dir, 'ground_truth_traces_omsi')
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
    print('  {:<12}  {:>5}  {:>20}'.format('Sensor', 'n', 'SNR (med ± MAD)'))
    print('  {}  {}  {}'.format('-' * 12, '-' * 5, '-' * 20))
    for sensor in _SNR_SENSOR_ORDER:
        if sensor not in snr_by_sensor:
            continue
        vals = np.array(snr_by_sensor[sensor])
        print('  {:<12}  {:>5}  {:>20}'.format(
            sensor, len(vals), '{:.2f} ± {:.2f}'.format(np.median(vals), _mad(vals))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='T_supp sensitivity and init comparison benchmarks for OMSI'
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
            'add-move-test',
            'add-move-plot',
            'add-move-dur-test',
            'add-move-dur-plot',
            'combined-opt-add-move-test',
            'print',
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
    
    with no_power_throttling(verbose=True):
        if args.mode == 'test':
            run_T_supp_sweep(args.data_dir)
        elif args.mode == 'plot':
            plot_T_supp_sweep(args.data_dir)
        elif args.mode == 'init-test':
            run_init_comparison(args.data_dir)
        elif args.mode == 'init-plot':
            plot_init_comparison(args.data_dir)
        elif args.mode == 'conv-test':
            run_omsi_init_comparison(args.data_dir)
        elif args.mode == 'conv-plot':
            plot_omsi_init_comparison(args.data_dir)
        elif args.mode == 'combined-plot':
            plot_combined_init(args.data_dir)
        elif args.mode == 'tol-conv-test':
            run_conv_tol_sweep(args.data_dir)
        elif args.mode == 'tol-conv-plot':
            plot_conv_tol_sweep(args.data_dir)
        elif args.mode == 'tol-burn-test':
            run_burn_tol_sweep(args.data_dir)
        elif args.mode == 'tol-burn-plot':
            plot_burn_tol_sweep(args.data_dir)
        elif args.mode == 'combined-opt-plot':
            plot_combined_opt(args.data_dir)
        elif args.mode == 'snr-filter-test':
            run_snr_filter_sweep(args.data_dir)
        elif args.mode == 'snr-filter-plot':
            plot_snr_filter_sweep(args.data_dir)
        elif args.mode == 'snr-thresh-test':
            run_snr_threshold_sweep(args.data_dir)
        elif args.mode == 'snr-thresh-plot':
            plot_snr_threshold_sweep(args.data_dir)
        elif args.mode == 'add-move-test':
            run_add_move_sweep(args.data_dir)
        elif args.mode == 'add-move-plot':
            plot_add_move_sweep(args.data_dir)
        elif args.mode == 'add-move-dur-test':
            run_add_move_duration_sweep(args.data_dir)
        elif args.mode == 'add-move-dur-plot':
            plot_add_move_duration(args.data_dir)
        elif args.mode == 'combined-opt-add-move-test':
            run_combined_opt_add_move(args.data_dir)
        elif args.mode == 'snr-stats':
            print_snr_stats(args.fig4_data_dir)
        elif args.mode == 'print':
            print_results(args.data_dir)
