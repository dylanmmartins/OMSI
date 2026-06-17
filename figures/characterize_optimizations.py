
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
                'mean_f1_window':  float(np.mean(fb)),
                'std_f1_window':   float(np.std(fb, ddof=1)),
                'mean_f1_event':   float(np.mean(f1_e)),
                'mean_cosmic':     float(np.mean(cosmic)),
                'std_cosmic':      float(np.std(cosmic, ddof=1)),
            })
            print(f'    total={elapsed:.1f}s  '
                  f'mean_cell={np.mean(per_cell_t):.3f}s  '
                  f'F_beta={np.mean(fb):.3f}  '
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
        ax.set_ylim([0, 500])

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


_DEFAULT_CONV_TOL = 0.00067 # was 0.001; tightened 1.5x
_DEFAULT_BURN_TOL = 0.005   # original default

# min_sweeps=300 (the default) gates how soon the post-burn-in convergence
# check can fire, which floors total sweep count regardless of how loose
# conv_tol/burn_tol are set. lower it per-sweep so loosening the tolerance
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

def _build_tol_grid(default_val, n_below=7, n_above=7):
    # single geomspace symmetric in log around default_val: with an odd total
    # point count its exact middle point is sqrt(a*b) = default_val, so it's
    # included for free without bunching extra points (0.9x/1.1x) next to it.
    n_total = n_below + n_above + 1
    return np.geomspace(default_val / 1000.0, default_val * 1000.0, n_total).tolist()


def _run_tol_sweep(dff, true_spikes, param_name, grid, fs, min_sweeps):
    rows = []
    for val in grid:
        print(f'\n  {param_name}={val:.2e} ...')
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
            pred       = res['optim_spikes']
            prec_s, rec_s, _ = helpers.compute_accuracy_strict(true_spikes, pred)
            cosmic     = helpers.compute_cosmic(true_spikes, pred, fs)
            fb = np.array([_fbeta(float(prec_s[i]), float(rec_s[i]))
                            for i in range(len(prec_s))])
            rows.append({
                'val':       val,
                'mean_time': float(np.mean(per_cell_t)),
                'std_time':  float(np.std(per_cell_t, ddof=1)),
                'mean_f1':   float(np.mean(fb)),
                'std_f1':    float(np.std(fb, ddof=1)),
            })
            print(f'    mean_cell={np.mean(per_cell_t):.3f}s  '
                  f'F_beta={np.mean(fb):.3f}  CosMIC={np.mean(cosmic):.3f}')
        except Exception as exc:
            print(f'    FAILED: {exc}')
    return rows


def _save_tol_sweep(out_path, rows, default_val):
    np.savez(
        out_path,
        tol         = np.array([r['val']       for r in rows]),
        mean_time   = np.array([r['mean_time'] for r in rows]),
        std_time    = np.array([r['std_time']  for r in rows]),
        mean_f1     = np.array([r['mean_f1']   for r in rows]),
        std_f1      = np.array([r['std_f1']    for r in rows]),
        default_val = np.array([default_val]),
    )
    print(f'\nSaved -> {out_path}')


def run_conv_tol_sweep(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'conv_tol_sweep.npz')

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_DURATION, tau=_TAU)

    grid = _build_tol_grid(_DEFAULT_CONV_TOL)
    print(f'  Sweep ({len(grid)} conv_tol values): {[f"{v:.2e}" for v in grid]}')

    rows = _run_tol_sweep(dff, true_spikes, 'conv_tol', grid, _FS, _CONV_TEST_MIN_SWEEPS)
    if rows:
        _save_tol_sweep(out_path, rows, _DEFAULT_CONV_TOL)
    else:
        print('No results collected.')


def run_burn_tol_sweep(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'burn_tol_sweep.npz')

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=_N_CELLS, fs=_FS, duration=_DURATION, tau=_TAU)

    grid = _build_tol_grid(_DEFAULT_BURN_TOL)
    print(f'  Sweep ({len(grid)} burn_tol values): {[f"{v:.2e}" for v in grid]}')

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
        out = os.path.join(data_dir, f'{fig_stem}.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
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

    fig, axes = plt.subplots(4, 2, figsize=(4.8, 9.0), dpi=300)

    d            = np.load(t_supp_path)
    T_supp       = d['T_supp'].astype(float)
    mt           = d['mean_time']
    st           = d['std_time']
    mf1          = d['mean_f1']
    sf1          = d['std_f1']
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
        ax.set_ylim(bottom=0)

    axes[0, 1].set_ylim(0, 1)

    d           = np.load(conv_tol_path)
    tols        = d['tol'].astype(float)
    mt          = d['mean_time']
    st          = d['std_time']
    mf1         = d['mean_f1']
    sf1         = d['std_f1']
    default_val = float(d['default_val'][0])

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
        ax.set_ylim(bottom=0)

    axes[1, 1].set_ylim(0, 1)

    d           = np.load(burn_tol_path)
    tols        = d['tol'].astype(float)
    mt          = d['mean_time']
    st          = d['std_time']
    mf1         = d['mean_f1']
    sf1         = d['std_f1']
    default_val = float(d['default_val'][0])

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
        ax.set_ylim(bottom=0)

    axes[2, 1].set_ylim(0, 1)

    d           = np.load(snr_path)
    snr_levels  = d['snr_levels'].astype(float)
    mean_fb     = d['mean_fb'].astype(float)
    std_fb      = d['std_fb'].astype(float)
    mean_cosmic = d['mean_cosmic'].astype(float)
    std_cosmic  = d['std_cosmic'].astype(float)
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
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'combined_opt.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
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

    print(f'Generating synthetic population '
          f'(n={_N_CELLS}, T={_DURATION}s, fs={_FS}Hz, tau={_TAU}s)...')
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

    print(f'Running OMSI on {_N_CELLS} synthetic cells (no SNR pre-filter)...')
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
    print(f'\nSaved {_N_CELLS} cells -> {out_path}')


def plot_snr_filter_sweep(data_dir):
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
    fig.suptitle(f'{n_below_thresh} cells below SNR threshold (with spikes)',
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
        out = os.path.join(data_dir, f'snr_filter_sweep.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


_SNR_THRESHOLD      = 2.0
_SNR_N_CELLS        = 50
_SNR_DURATION       = 120.0
_SNR_N_LEVELS       = 18


def run_snr_threshold_sweep(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'snr_threshold_sweep.npz')

    snr_levels = np.geomspace(0.3, 3.0 * _SNR_THRESHOLD, _SNR_N_LEVELS)
    print(f'SNR sweep: {_SNR_N_LEVELS} levels from {snr_levels[0]:.2f} to '
          f'{snr_levels[-1]:.2f}  (threshold={_SNR_THRESHOLD})')
    print(f'  {_SNR_N_CELLS} cells x {_SNR_DURATION}s each')

    mean_fb    = np.full(_SNR_N_LEVELS, np.nan)
    std_fb     = np.full(_SNR_N_LEVELS, np.nan)
    mean_cosmic = np.full(_SNR_N_LEVELS, np.nan)
    std_cosmic  = np.full(_SNR_N_LEVELS, np.nan)

    params = {
        'f':         _FS,
        'p':         2,
        'auto_stop': True,
        'upd_gam':   0,
        'conv_tol':  _DEFAULT_CONV_TOL,
        'burn_tol':  _DEFAULT_BURN_TOL,
    }

    for k, snr_val in enumerate(snr_levels):
        print(f'\n  [{k+1}/{_SNR_N_LEVELS}] SNR={snr_val:.3f} ...')
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=_SNR_N_CELLS, fs=_FS, duration=_SNR_DURATION,
            tau=_TAU, snr=float(snr_val),
        )
        try:
            res  = OMSI.deconv(dff, params=params, true_spikes=true_spikes, benchmark=True)
            pred = res['optim_spikes']

            cosmic_v   = helpers.compute_cosmic(true_spikes, pred, _FS)

            mean_cosmic[k] = float(np.mean(cosmic_v))
            std_cosmic[k]  = float(np.std(cosmic_v, ddof=1))

            if res['optim_precision'] is not None:
                fb_arr = np.array([
                    _fbeta(float(res['optim_precision'][i]),
                           float(res['optim_recall'][i]))
                    for i in range(_SNR_N_CELLS)
                ])
                mean_fb[k] = float(np.nanmean(fb_arr))
                std_fb[k]  = float(np.nanstd(fb_arr))

            print(f'    F_beta={mean_fb[k]:.3f}  CosMIC={mean_cosmic[k]:.3f}')
        except Exception as exc:
            print(f'    FAILED: {exc}')

    np.savez(
        out_path,
        snr_levels  = snr_levels,
        mean_fb     = mean_fb,
        std_fb      = std_fb,
        mean_cosmic = mean_cosmic,
        std_cosmic  = std_cosmic,
        threshold   = np.array([_SNR_THRESHOLD]),
    )
    print(f'\nSaved -> {out_path}')


def plot_snr_threshold_sweep(data_dir):
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
                   linewidth=0.8, alpha=0.6, label=f'threshold={threshold}')
        ax.set_xlabel('SNR')
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'snr_threshold_sweep.{sfx}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
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

    traces_dir = os.path.join(fig4_data_dir, 'ground_truth_traces_fmcsi')
    if not os.path.isdir(traces_dir):
        print(f'Traces directory not found: {traces_dir}')
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
            print(f'  Warning: {fname}: {exc}')
            continue
        for i in range(n_cells):
            trace = npz[f'dff_{i}'].astype(np.float64)
            noise = _get_sn(trace, [0.25, 0.5])
            b     = float(np.percentile(trace, 8))
            peak  = float(np.percentile(trace, 99))
            snr   = (peak - b) / (noise + 1e-9)
            snr_by_sensor.setdefault(sensor, []).append(snr)

    print('SNR statistics by sensor (figure4 datasets):')
    print(f'  {"Sensor":<12}  {"n":>5}  {"mean SNR":>10}  {"std SNR":>10}')
    print(f'  {"-"*12}  {"-"*5}  {"-"*10}  {"-"*10}')
    for sensor in _SNR_SENSOR_ORDER:
        if sensor not in snr_by_sensor:
            continue
        vals = np.array(snr_by_sensor[sensor])
        print(f'  {sensor:<12}  {len(vals):>5}  {np.mean(vals):>10.2f}  {np.std(vals):>10.2f}')


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
    elif args.mode == 'snr-stats':
        print_snr_stats(args.fig4_data_dir)

