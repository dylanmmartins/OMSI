
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
mpl.rcParams['font.size']    = 7

np.random.seed(7)

_N_CELLS  = 200
_DURATION = 2400.
_FS       = 30.0
_TAU      = 1.2
_COLOR    = '#4C72B0'


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

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_DURATION, tau=_TAU
    )
    true_events = [helpers.make_event_ground_truth(s, _TAU) for s in true_spikes]
    n_frames = dff.shape[1]

    tau_rep, gr_rep, diff_gr_rep = _init_tau(dff[0], _FS)
    default_supp = _default_T_supp(tau_rep, diff_gr_rep, n_frames)
    print(f'  tau (frames): {tau_rep[0]:.2f}, {tau_rep[1]:.2f}  '
          f'gr: {gr_rep[0]:.4f}, {gr_rep[1]:.4f}')
    print(f'  Default T_supp (prec=1e-2): {default_supp} / {n_frames} frames')

    T_supp_grid = _build_T_supp_grid(default_supp, n_frames)
    print(f'  Sweep ({len(T_supp_grid)} values): {T_supp_grid}')

    rows = []
    for ts in T_supp_grid:
        label = 'T' if ts == n_frames else str(ts)
        print(f'\n  T_supp={ts} ...')
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
            pred       = res['optim_spikes']

            _, _, f1_w = helpers.compute_accuracy_window(true_spikes, pred)
            _, _, f1_e = helpers.compute_accuracy_window(true_events, pred)
            cosmic     = helpers.compute_cosmic(true_spikes, pred, _FS)

            rows.append({
                'T_supp':          ts,
                'is_default':      ts == default_supp,
                'is_full':         ts == n_frames,
                'total_time':      elapsed,
                'mean_time':       float(np.mean(per_cell_t)),
                'std_time':        float(np.std(per_cell_t, ddof=1)),
                'mean_f1_window':  float(np.mean(f1_w)),
                'std_f1_window':   float(np.std(f1_w, ddof=1)),
                'mean_f1_event':   float(np.mean(f1_e)),
                'mean_cosmic':     float(np.mean(cosmic)),
                'std_cosmic':      float(np.std(cosmic, ddof=1)),
            })
            print(f'    total={elapsed:.1f}s  '
                  f'mean_cell={np.mean(per_cell_t):.3f}s  '
                  f'F1={np.mean(f1_w):.3f}  '
                  f'CosMIC={np.mean(cosmic):.3f}')
        except Exception as exc:
            print(f'    FAILED: {exc}')

    if not rows:
        print('No results collected.')
        return

    np.savez(
        out_path,
        T_supp       = np.array([r['T_supp']          for r in rows]),
        mean_time    = np.array([r['mean_time']        for r in rows]),
        std_time     = np.array([r['std_time']         for r in rows]),
        mean_f1      = np.array([r['mean_f1_window']   for r in rows]),
        std_f1       = np.array([r['std_f1_window']    for r in rows]),
        mean_cosmic  = np.array([r['mean_cosmic']      for r in rows]),
        std_cosmic   = np.array([r['std_cosmic']       for r in rows]),
        default_supp = np.array([default_supp]),
        n_frames     = np.array([n_frames]),
    )
    print(f'\nSaved -> {out_path}')


def plot_T_supp_sweep(data_dir):

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

    # print('T_shuff shape is ', np.shape(T_supp))
    # print(T_supp)
    # print(T_supp * (1.0 / _FS))

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), dpi=300)

    for ax, y, yerr, ylabel in [
        (axes[0], mt,  st,  'time per cell (sec)'),
        (axes[1], mf1, sf1, '$F_\\beta$'),
    ]:
        mask = np.arange(len(T_supp))[1:-1]
        mask = np.hstack([mask[0:4], mask[5], mask[7:]]).astype(int)
        print(mask)
        # ax.plot(T_supp[:-1] * (1.0 / _FS), y[:-1], '.-', color=_COLOR, zorder=3)
        ax.plot(T_supp[mask] * (1.0 / _FS), y[mask], '.-', color=_COLOR, zorder=3)
        if ylabel == 'time per cell (sec)':
            ax.fill_between(T_supp[mask] * (1.0 / _FS), y[mask] - yerr[mask], y[mask] + yerr[mask],
                            color=_COLOR, alpha=0.25, linewidth=0)
        ax.axvline(default_supp * (1.0 / _FS), color='k', linestyle='--',
                   linewidth=0.8, alpha=0.6, label='default')
        # ax.axhline(0, color='k', linestyle='--',
        #            linewidth=0.8, alpha=0.6, label='T')
        # print('default T_supp:',default_supp * (1.0 / _FS), 'sec', f'({default_supp} frames)')
        # print('shortest window tested is' f'{T_supp[0] * (1.0 / _FS):.2f} sec ({T_supp[0]} frames)')
        ax.set_xlabel('$T_{supp}$ (sec)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_ylim([0, 100])

    axes[1].set_ylim(0, 0.51)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'T_supp_sweep.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


_BETA = 0.5

def _fbeta(precision, recall, beta=_BETA):
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

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


def run_init_comparison(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'init_comparison.npz')

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_INIT_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
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
            print(f'  {i+1}/{_N_CELLS}  '
                  f'nnls={np.mean(nnls_times[:i+1])*1e3:.1f}ms  '
                  f'foopsi={np.mean(foopsi_times[:i+1])*1e3:.1f}ms  '
                  f'r_cross={np.mean(r_cross[:i+1]):.3f}')

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
    print(f'\nSaved -> {out_path}')


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
            ax.legend(frameon=False, fontsize=6, loc='upper left')#,

    ax_t = axd['time']

    bins = np.linspace(0, 200, 30)
    # ax_t.hist(foopsi_times * 1e3, bins=bins, color=_FOOPSI_COLOR,
    #           alpha=0.6, label='FOOPSI', edgecolor='none')
    # ax_t.hist(nnls_times   * 1e3, bins=bins, color=_NNLS_COLOR,
    #           alpha=0.6, label='NNLS',   edgecolor='none')

    ax_t.scatter(foopsi_times * 1e3, nnls_times * 1e3, color='k', s=1)

    ax_t.set_xlabel('FOOPSI time per cell (msec)')
    ax_t.set_ylabel('NNLS time per cell (msec)')

    # ax_t.set_xlim([0,200])
    # ax_t.legend(frameon=False, fontsize=6, reverse=True, loc='upper right')
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
        out = os.path.join(data_dir, f'init_comparison.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def _make_foopsi_init(Y_cell, fs, tau=_TAU, p=2):

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

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'fmcsi_init_comparison.npz')

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_INIT_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
    dff, true_spikes, clean_traces, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_INIT_DURATION, tau=_TAU
    )
    firing_rates = np.array([len(s) / _INIT_DURATION for s in true_spikes])

    print('Pre-computing FOOPSI inits...')
    foopsi_inits = [_make_foopsi_init(dff[i], _FS) for i in range(_N_CELLS)]
    print(f'  Done ({_N_CELLS} inits).')

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

    print(f'NNLS   — mean/cell: {np.mean(nnls_times):.2f}s  '
          f'mean samples: {int(np.mean(nnls_nsamples))}  '
          f'mean Fb: {np.nanmean(nnls_fb):.3f}')
    print(f'FOOPSI — mean/cell: {np.mean(foopsi_times):.2f}s  '
          f'mean samples: {int(np.mean(foopsi_nsamples))}  '
          f'mean Fb: {np.nanmean(foopsi_fb):.3f}')

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
    print(f'\nSaved -> {out_path}')


def plot_fmcsi_init_comparison(data_dir):

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
            ax.legend(frameon=False, fontsize=6, loc='upper left')#,

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

    print('average $F_\\beta$ (β=0.5) — NNLS init: {:.3f}  FOOPSI init: {:.3f}'.format(
        np.nanmean(valid_n), np.nanmean(valid_f)))
    print('average time per cell (sec) — NNLS init: {:.3f}  FOOPSI init: {:.3f}'.format(
        np.mean(nnls_times), np.mean(foopsi_times)))
    ax_f.set_xlabel('$F_\\beta$')
    ax_f.set_ylabel('cells')

    ax_f.legend(frameon=False, fontsize=6, loc='upper left', reverse=True)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'fmcsi_init_comparison.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def plot_combined_init(data_dir):
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
    # bins_init = np.linspace(0, 200, 30)
    # ax_ti.hist(foopsi_times_init * 1e3, bins=bins_init, color=_FOOPSI_COLOR,
    #            alpha=0.6, label='FOOPSI', edgecolor='none')
    # ax_ti.hist(nnls_times_init   * 1e3, bins=bins_init, color=_NNLS_COLOR,
    #            alpha=0.6, label='NNLS',   edgecolor='none')
    # ax_ti.set_xlabel('init time (msec)')
    # ax_ti.set_ylabel('cells')
    # ax_ti.set_xlim([0, 200])
    # ax_ti.legend(frameon=False, fontsize=6, reverse=True, loc='upper right')
    ax_ti.scatter(
        foopsi_times_init * 1e3,
        nnls_times_init * 1e3,
        color='k',
        s=1
    )
    # ax_ti.axis('equal')
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
    # parts_tc = ax_tc.violinplot(
    #     [nnls_times_conv, foopsi_times_conv], positions=[1, 2],
    #     showmedians=True, widths=0.65, showextrema=False
    # )
    # for pc, col in zip(parts_tc['bodies'], [_NNLS_COLOR, _FOOPSI_COLOR]):
    #     pc.set_facecolor(col); pc.set_alpha(0.75)
    # parts_tc['cmedians'].set_color('k'); parts_tc['cmedians'].set_linewidth(0.8)
    # ax_tc.set_xticks([1, 2])
    # ax_tc.set_xticklabels(['NNLS\ninit', 'FOOPSI\ninit'], fontsize=6)
    # # ax_tc.set_xlabel('fMCSI time per cell (sec)')
    # ax_tc.set_ylabel('time per cell (sec)')

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
    # parts_fb = ax_fb.violinplot(
    #     [valid_n, valid_f], positions=[1, 2],
    #     showmedians=True, widths=0.65, showextrema=False
    # )
    # for pc, col in zip(parts_fb['bodies'], [_NNLS_COLOR, _FOOPSI_COLOR]):
    #     pc.set_facecolor(col); pc.set_alpha(0.75)
    # parts_fb['cmedians'].set_color('k'); parts_fb['cmedians'].set_linewidth(0.8)
    # ax_fb.set_xticks([1, 2])
    # ax_fb.set_xticklabels(['NNLS\ninit', 'FOOPSI\ninit'], fontsize=6)
    # # ax_fb.set_xlabel('$F_\\beta$ (β=0.5)')
    # ax_fb.set_ylabel('$F_\\beta$')
    # ax_fb.set_ylim(0, 1.05)

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
        out = os.path.join(data_dir, f'combined_init_comparison.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')

    plt.close(fig)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='T_supp sensitivity and init comparison benchmarks for fMCSI'
    )
    parser.add_argument(
        '--mode', required=True,
        choices=['test', 'plot', 'init-test', 'init-plot', 'conv-test', 'conv-plot', 'combined-plot'],
    )
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
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
    else:
        plot_combined_init(args.data_dir)