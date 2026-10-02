# -*- coding: utf-8 -*-
"""
figures/auto_stop_quality.py

DMM, October 2026
"""

import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import glob
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import OMSI.helpers as helpers
from OMSI.sampler import cont_ca_sampler
from OMSI.get_init_sample import get_init_sample
from OMSI.deconv import spikes_from_samples
from OMSI._win_perf import no_power_throttling

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'auto_stop_quality')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

BETA = 0.5

COND = {'fs': 30.0, 'tau': 1.2, 'snr': 6.0}

MAX_RATE = 0.6
COND_COLOR = '#C0182B'

N_INIT_CELLS = 5

INIT_LABELS = ['nnls', 'empty', 'foopsi', 'dense', 'nnls_jit']
INIT_NAMES = {
    'nnls':     'NNLS (default)',
    'empty':    'no spikes',
    'foopsi':   'FOOPSI',
    'dense':    'extra spikes',
    'nnls_jit': 'NNLS, jittered',
}

INIT_STYLES = {
    'nnls':     dict(color='#C0182B', ls='-'),
    'empty':    dict(color='#EDA100', ls='-'),
    'foopsi':   dict(color='#1BAF7A', ls='-'),
    'dense':    dict(color='#5A0F2E', ls='-'),
    'nnls_jit': dict(color='#E87BA4', ls='-'),
}

JITTER_FRAMES = 2.0
AUTO_B = 75
SWEEP_CHECKPOINTS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
RESULT_VERSION = 4

def _score(ref_sp, pred_sp, fs):
    ref_sp, pred_sp = np.asarray(ref_sp), np.asarray(pred_sp)
    p, r, _ = helpers.compute_accuracy_window([ref_sp], [pred_sp])
    p, r = float(p[0]), float(r[0])
    b2 = BETA ** 2
    denom = b2 * p + r
    fb = (1 + b2) * p * r / denom if denom > 0 else 0.0

    return fb, float(helpers.compute_cosmic([ref_sp], [pred_sp], fs)[0])


def _log_idx(n, n_pts=400):
    idx = np.unique(np.geomspace(1, n, n_pts).astype(int) - 1)
    return idx[idx < n]


def _decade_ticks(n_max):
    return [10 ** k for k in range(int(np.floor(np.log10(n_max))) + 1)]


def _sci(n):
    k = int(np.floor(np.log10(n)))
    m = n / 10 ** k
    if np.isclose(m, 1.0):
        return '$10^{{{}}}$'.format(k)
    return '${:g}{{\\times}}10^{{{}}}$'.format(m, k)


def _make_inits(y, params, seeds):

    T = len(y)
    np.random.seed(seeds[0])
    base = get_init_sample(y, dict(params))

    inits, labels = [], []
    for k, seed in enumerate(seeds):
        np.random.seed(seed)
        sam = dict(base)
        label = INIT_LABELS[k] if k < len(INIT_LABELS) else 'nnls_jit'
        if label == 'nnls':
            sp = base['spiketimes_']
        elif label == 'empty':
            sp = np.empty(0)
        elif label == 'foopsi':
            sp = get_init_sample(y, dict(params, init_method='foopsi'))['spiketimes_']
        elif label == 'dense':
            n_extra = max(len(base['spiketimes_']), int(0.02 * T))
            sp = np.concatenate([base['spiketimes_'], np.random.rand(n_extra) * T])
        else:
            sp = base['spiketimes_'] + JITTER_FRAMES * np.random.randn(len(base['spiketimes_']))
            sp = np.abs(sp)
            sp = np.where(sp > T, 2.0 * T - sp, sp)
        sam['spiketimes_'] = np.sort(np.asarray(sp, dtype=np.float64))
        inits.append(sam)
        labels.append(label)
    return inits, labels


def _auto_stop_point(Am, ns, p):

    B = int(p.get('B', AUTO_B))
    max_sweeps  = int(p.get('max_sweeps', 1000))
    min_sweeps  = int(p.get('min_sweeps', 300))
    burn_tol    = float(p.get('burn_tol', 1e-3))
    conv_tol    = float(p.get('conv_tol', 0.05))
    check_every = int(p.get('check_every', 50))
    win         = int(p.get('win', 100))

    n_total = min(max_sweeps, len(ns))
    B_final = B
    burn_in_done = False
    for i in range(check_every, n_total, check_every):
        if not burn_in_done:
            if i >= B + win:
                recent = Am[i - win:i]
                m1 = np.mean(recent[:win // 2])
                m2 = np.mean(recent[win // 2:])
                if m2 > 1e-9 and abs(m1 - m2) < burn_tol * m2:
                    burn_in_done = True
                    B_final = i
        else:
            n_samp = i - B_final
            if n_samp >= min_sweeps:
                cur = ns[B_final:i]
                mid = n_samp // 2
                m1 = np.mean(cur[:mid])
                m2 = np.mean(cur[mid:])
                denom = m2 if m2 > 1e-9 else 1.0
                if abs(m1 - m2) < conv_tol * denom:
                    return B_final, i + 1, False
    return B_final, max_sweeps, True


def _run_chain(y, params, seed, true_sp, n_sweeps, init=None):

    T = len(y)
    fs = float(params['f'])
    p = dict(params)
    p.update({'seed': int(seed), 'return_full': True, 'auto_stop': False,
              'B': AUTO_B, 'Nsamples': int(n_sweeps) - AUTO_B})
    if init is not None:
        p['init'] = init

    np.random.seed(seed)
    t0 = time.time()
    S = cont_ca_sampler(y, p)
    elapsed = time.time() - t0
    if 'chain' not in S:
        return None
    ch = S['chain']
    ss = ch['ss']
    ns = np.asarray(ch['ns'], dtype=float)
    N = len(ss)

    init_sp = np.clip(np.asarray(p['init']['spiketimes_'], dtype=float) - 1.0, 0, T - 1) / fs

    grid = [c for c in SWEEP_CHECKPOINTS if c < N] + [N]
    fb = [_score(true_sp, init_sp, fs)[0]]
    for n in grid:
        _, sp = spikes_from_samples(ss[n // 2:n], T, fs, lag_s=0.0)
        fb.append(_score(true_sp, sp, fs)[0])

    b0, stop, hit_max = _auto_stop_point(np.asarray(ch['Am'], dtype=float), ns, p)
    _, sp_auto = spikes_from_samples(ss[b0:stop], T, fs, lag_s=0.0)
    _, sp_long = spikes_from_samples(ss[N // 2:N], T, fs, lag_s=0.0)
    fb_auto, cos_auto = _score(true_sp, sp_auto, fs)
    fb_long, cos_long = _score(true_sp, sp_long, fs)

    tidx = _log_idx(N)
    return {
        'time': elapsed,
        'p_switched': bool(p.get('p', 2) != params.get('p', 2)),
        'trace_idx': tidx,
        'trace_ns': ns[tidx].astype(np.float32),
        'ns_init': len(init_sp),
        'sweep_grid': np.array([0] + grid),
        'sweep_fbeta': np.array(fb),
        'B_final': int(b0), 'stop_idx': int(stop), 'hit_max': bool(hit_max),
        'ns_at_stop': float(ns[stop - 1]),
        'ns_mean_auto': float(ns[b0:stop].mean()),
        'ns_mean_long': float(ns[N // 2:].mean()),
        'fbeta_auto': fb_auto, 'fbeta_long': fb_long,
        'cosmic_auto': cos_auto, 'cosmic_long': cos_long,
    }


def _run_cell(task):

    y = np.asarray(task['dff'], dtype=np.float64)
    true_sp = np.asarray(task['true_spikes'], dtype=float)
    params = task['params']
    seeds = task['seeds']
    t_cell = time.time()

    result = {
        'cell_idx': task['cell_idx'], 'fs': float(params['f']), 'n_frames': len(y),
        'n_true': len(true_sp), 'n_sweeps': task['n_sweeps'], 'skipped': False,
        'version': RESULT_VERSION,
    }

    inits, labels = _make_inits(y, params, seeds)
    chains = {}
    for k, (init, label) in enumerate(zip(inits, labels)):
        r = _run_chain(y, params, seeds[k], true_sp, task['n_sweeps'], init=init)
        if r is None:
            result['skipped'] = True
            break
        chains[label] = r
    result['chains'] = chains
    result['cell_time'] = time.time() - t_cell
    np.save(task['out_path'], np.array(result, dtype=object), allow_pickle=True)

    if result['skipped']:
        return 'Cell {}: skipped by SNR gate.'.format(task['cell_idx'])
    c = chains['nnls']
    return 'Cell {}: {} chains, auto stop {}, F-beta {:.3f} auto vs {:.3f} long ({:.0f}s).'.format(
        task['cell_idx'], len(chains), c['stop_idx'], c['fbeta_auto'], c['fbeta_long'],
        result['cell_time'])


def run_test(data_dir=_DEFAULT_DATA_DIR, n_cells=10, n_init_cells=N_INIT_CELLS,
             duration=120.0, n_sweeps=10000, workers=None, seed=0, overwrite=False):

    from simulation_helpers import generate_synthetic_data

    cell_dir = os.path.join(data_dir, 'cells')
    os.makedirs(cell_dir, exist_ok=True)

    print('Generating {} non-bursty cells at SNR {:g}, rate <= {:g} Hz...'.format(
        n_cells, COND['snr'], MAX_RATE))
    np.random.seed(seed)
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=COND['fs'], duration=duration, tau=COND['tau'], snr=COND['snr'],
        bursty_frac=0.0, max_rate=MAX_RATE)
    # Same params as the figure 2 OMSI runs.
    params = {'f': COND['fs'], 'p': 2}

    todo = []
    for i in range(n_cells):
        out_path = os.path.join(cell_dir, 'cell_{:03d}.npy'.format(i))
        n_starts = len(INIT_LABELS) if i < n_init_cells else 1
        if os.path.exists(out_path) and not overwrite:
            # Reuse only if saved with same chain settings.
            prev = np.load(out_path, allow_pickle=True).item()
            same = (prev.get('n_sweeps') == n_sweeps
                    and len(prev.get('chains', {})) == n_starts)
            if prev.get('version') == RESULT_VERSION and (prev.get('skipped') or same):
                continue
        rng = np.random.default_rng([seed, i])
        todo.append({
            'cell_idx': i, 'dff': dff[i], 'true_spikes': true_spikes[i], 'params': params,
            'seeds': [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n_starts)],
            'n_sweeps': n_sweeps, 'out_path': out_path,
        })

    print('{} cells total, {} to run ({} already done).'.format(
        n_cells, len(todo), n_cells - len(todo)))
    if not todo:
        return

    # Multi-start cells first so the slow ones start early.
    todo.sort(key=lambda t: -len(t['seeds']))
    workers = workers or os.cpu_count()
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(_run_cell, t): t for t in todo}
        for n_done, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                msg = fut.result()
            except Exception as exc:
                msg = 'Cell {} failed: {}'.format(t['cell_idx'], exc)
            print('  [{}/{}] {}'.format(n_done, len(todo), msg), flush=True)
    print('Finished in {:.1f} min. Results in {}.'.format((time.time() - t0) / 60, cell_dir))


def _load_results(data_dir):

    paths = sorted(glob.glob(os.path.join(data_dir, 'cells', 'cell_*.npy')))
    res = [np.load(p, allow_pickle=True).item() for p in paths]
    n_skip = sum(r['skipped'] for r in res)
    if n_skip:
        print('{} of {} cells skipped by SNR gate.'.format(n_skip, len(res)))
    return [r for r in res if not r['skipped']]


def _plot_init_cell(ax, r):

    for label in INIT_LABELS:
        c = r['chains'].get(label)
        if c is None:
            continue
        st = INIT_STYLES[label]
        ax.plot(c['trace_idx'] + 1, c['trace_ns'], lw=1.0, label=INIT_NAMES[label], **st)
        ax.plot(c['stop_idx'], c['ns_at_stop'], marker='v', ms=3.5, mfc=st['color'],
                mec='k', mew=0.4, ls='none', zorder=4)
    ax.axhline(r['n_true'], color='k', lw=0.7, ls=':', zorder=0)
    ax.text(0.97, 0.97, 'N={}'.format(r['n_true']), transform=ax.transAxes,
            ha='right', va='top', fontsize=5.5)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.8, r['n_sweeps'] * 1.3)
    ticks = _decade_ticks(r['n_sweeps'])
    ax.set_xticks(ticks)
    ax.set_xticklabels(['$10^{{{}}}$'.format(int(np.log10(t))) for t in ticks])
    ax.minorticks_off()
    ax.set_title('cell {}'.format(r['cell_idx'] + 1), fontsize=6.5)


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    results = _load_results(data_dir)
    if not results:
        print('No results in {}.'.format(data_dir))
        return
    multi = [r for r in results if len(r['chains']) > 1][:N_INIT_CELLS]
    n_sweeps = max(r['n_sweeps'] for r in results)

    fig = plt.figure(figsize=(7.0, 4.8), dpi=300)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.55,
                           left=0.08, right=0.98, top=0.86, bottom=0.10)

    n_top = max(len(multi), 1)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n_top, subplot_spec=gs[0], wspace=0.35)
    axes_a = []
    for i, r in enumerate(multi):
        ax = fig.add_subplot(gs_top[i])
        _plot_init_cell(ax, r)
        ax.set_xlabel('Sweep', labelpad=1)
        if i == 0:
            ax.set_ylabel('Spike count')
        axes_a.append(ax)
    if axes_a:
        handles, labels = axes_a[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], marker='v', ms=3.5, mfc='0.6', mec='k',
                                  mew=0.4, ls='none'))
        labels.append('auto-stop point')
        handles.append(plt.Line2D([], [], color='k', lw=0.7, ls=':'))
        labels.append('true count')
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
                   fontsize=6, bbox_to_anchor=(0.53, 0.995), handlelength=2.5,
                   columnspacing=1.2)

    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.6,
                                              width_ratios=[0.5, 1.0, 1.0, 0.5])

    for col, (key, name) in enumerate([('fbeta', '$F_\\beta$'), ('cosmic', 'CosMIC')], 1):
        ax = fig.add_subplot(gs_bot[col])
        va = np.array([r['chains']['nnls'][key + '_auto'] for r in results])
        vl = np.array([r['chains']['nnls'][key + '_long'] for r in results])
        ax.plot([0, 1.0], [0, 1.0], color='0.5', lw=0.6, ls='--', zorder=0)
        ax.scatter(va, vl, s=12, color=COND_COLOR, linewidths=0, zorder=3)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal')
        ax.set_xlabel('{}, auto-stop'.format(name))
        ax.set_ylabel('{}, {} sweeps'.format(name, _sci(n_sweeps)))

    for ext in ('png', 'svg'):
        path = os.path.join(data_dir, 'auto_stop_quality.{}'.format(ext))
        fig.savefig(path, dpi=300)
        print('Saved figure to {}.'.format(path))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    results = _load_results(data_dir)
    if not results:
        print('No results in {}.'.format(data_dir))
        return

    print('NNLS-start chains, {} cells:'.format(len(results)))
    print('  {:>4}  {:>6}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}'.format(
        'cell', 'n_true', 'stop', 'ns_auto', 'ns_long', 'Fb_auto', 'Fb_long', 'diff'))
    for r in results:
        c = r['chains']['nnls']
        print('  {:>4}  {:>6}  {:>5}{}  {:>7.1f}  {:>7.1f}  {:>7.3f}  {:>7.3f}  {:>+7.3f}'.format(
            r['cell_idx'], r['n_true'], c['stop_idx'], '*' if c['hit_max'] else ' ',
            c['ns_mean_auto'], c['ns_mean_long'], c['fbeta_auto'], c['fbeta_long'],
            c['fbeta_long'] - c['fbeta_auto']))
    print('  (* = hit max_sweeps)')
    for key, name in [('fbeta', 'F-beta'), ('cosmic', 'CosMIC')]:
        va = np.array([r['chains']['nnls'][key + '_auto'] for r in results])
        vl = np.array([r['chains']['nnls'][key + '_long'] for r in results])
        print('Median {}: {:.3f} auto-stop, {:.3f} long -- median diff {:+.3f}, '
              'range {:+.3f} to {:+.3f}.'.format(name, np.median(va), np.median(vl),
                                                 np.median(vl - va), (vl - va).min(),
                                                 (vl - va).max()))

    multi = [r for r in results if len(r['chains']) > 1]
    if multi:
        print('\nMulti-start cells, mean spike count over kept window '
              '(auto-stop / long), true count in brackets:')
        for r in multi:
            parts = ['{} {:.0f}/{:.0f}'.format(l, r['chains'][l]['ns_mean_auto'],
                                               r['chains'][l]['ns_mean_long'])
                     for l in INIT_LABELS if l in r['chains']]
            print('  cell {} [{}]: {}'.format(r['cell_idx'], r['n_true'], ', '.join(parts)))


def main():

    parser = argparse.ArgumentParser(
        description='F-beta at OMSI auto-stop vs long chains, and start-point convergence')
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'print'],
                        help='"test" runs chains and saves per-cell results; '
                             '"plot" generates the figure; '
                             '"print" prints summary statistics')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument('--n-cells', type=int, default=10, help='Cells in total')
    parser.add_argument('--n-init-cells', type=int, default=N_INIT_CELLS,
                        help='Cells that also get the four non-default starts')
    parser.add_argument('--duration', type=float, default=120., help='Duration (s)')
    parser.add_argument('--n-sweeps', type=int, default=10000, help='Chain length')
    parser.add_argument('--workers', type=int, default=None, help='Worker processes')
    parser.add_argument('--seed', type=int, default=0, help='Base seed')
    parser.add_argument('--overwrite', action='store_true', help='Rerun cells with saved results')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(data_dir=args.data_dir, n_cells=args.n_cells,
                 n_init_cells=args.n_init_cells, duration=args.duration,
                 n_sweeps=args.n_sweeps, workers=args.workers, seed=args.seed,
                 overwrite=args.overwrite)
    elif args.mode == 'print':
        print_stats(data_dir=args.data_dir)
    else:
        plot_figure(data_dir=args.data_dir)


if __name__ == '__main__':
    with no_power_throttling(verbose=True):
        main()
