# -*- coding: utf-8 -*-
"""
figures/caiman_convergence.py

DMM, October 2026
"""


import os
import argparse
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from OMSI.get_init_sample import get_init_sample
from convergence_benchmark import (
    _DEFAULT_DATA_DIR, GROUP_ORDER, GROUP_NAMES, SCALAR_KEYS, RHAT_THR, ESS_THR,
    SYNTH_CONDITIONS, METHOD_COLORS,
    RESULT_VERSION as OMSI_VERSION, rhat_rank, ess_bulk, mean_acf, _loglik,
    _load_results, _by_group, _strip,
)
from run_pnev_MCMC import run_matlab_pnevMCMC

N_DEFAULT = 7
OTHER_STARTS = ['empty', 'nnls', 'dense']

FIX_KW = {'fix_rise_bug': True, 'init_thr': 0.70}

RESULT_VERSION = 1


def _out_dir(data_dir, fixed, pilot=False):
    name = 'caiman_fixed' if fixed else 'caiman'
    return os.path.join(data_dir, name + '_pilot' if pilot else name)


def _omsi_cells(data_dir):
    res = [r for r in _load_results(data_dir) if not r['skipped']]
    if not res:
        raise FileNotFoundError('No OMSI results in {}. Run convergence_benchmark.py '
                                '--mode test first.'.format(data_dir))
    old = [r for r in res if r.get('version') != OMSI_VERSION]
    if old:
        raise RuntimeError('{} OMSI result files are from an older version -- rerun '
                           'convergence_benchmark.py --mode test.'.format(len(old)))
    return res


def _starts(y, fs, seed):
    T = len(y)
    np.random.seed(seed)
    nnls = np.asarray(get_init_sample(y, {'f': fs, 'p': 2})['spiketimes_'], dtype=float)
    n_extra = max(len(nnls), int(0.02 * T))
    dense = np.concatenate([nnls, 1.0 + np.random.rand(n_extra) * (T - 1)])
    return [np.empty(0), np.sort(nnls), np.sort(dense)]


def _traces(chains, fs, n_valid):
    out = {k: [] for k in SCALAR_KEYS}
    for c in chains:
        sg = np.sqrt(c['sn2'])
        ll = _loglik(c['sse'], n_valid, sg)
        out['ns'].append(c['ns'])
        out['Am'].append(c['Am'])
        out['Cb'].append(c['Cb'])
        out['sg'].append(sg)
        out['tau_decay'].append(c['tau'][:, 1] / fs)
        out['loglik'].append(ll)
        out['logpost'].append(ll + c['ns'] * np.log(np.maximum(c['lam'], 1e-300)))
    return {k: np.stack(v) for k, v in out.items()}


def _diagnostics(tr, dflt):
    def _max(d):
        finite = [v for v in d.values() if not np.isnan(v)]
        return max(finite) if finite else np.nan

    out = {
        'rhat':             {k: rhat_rank(v) for k, v in tr.items()},
        'rhat_default':     {k: rhat_rank(v[dflt]) for k, v in tr.items()},
        'ess_bulk':         {k: ess_bulk(v) for k, v in tr.items()},
        'ess_bulk_default': {k: ess_bulk(v[dflt]) for k, v in tr.items()},
        'acf_ns_default':   mean_acf(tr['ns'][dflt], 300),
    }
    out['rhat_max'] = _max(out['rhat'])
    out['rhat_default_max'] = _max(out['rhat_default'])
    return out


def _run_cell(task):
    r = task['omsi']
    y = np.asarray(r['dff'], dtype=np.float64)
    fs = float(r['fs'])
    N = task['n_sweeps']
    burn = int(round(task['warmup_frac'] * N))
    seed = task['seed']

    labels = ['caiman'] * N_DEFAULT + OTHER_STARTS
    starts = [None] * N_DEFAULT + task['starts']
    seeds = [seed + k for k in range(len(labels))]

    t0 = time.time()
    chains = run_matlab_pnevMCMC(
        np.tile(y, (len(labels), 1)), fs=fs, n_sweeps=N - burn, burn_in=burn,
        seeds=seeds, start_spikes=starts, return_chain=True, work_dir=task['work_dir'],
        verbose=False, **(FIX_KW if task['fixed'] else {}))
    elapsed = time.time() - t0

    ok = [i for i, c in enumerate(chains) if c is not None]
    dflt = [j for j, i in enumerate(ok) if labels[i] == 'caiman']
    result = {
        'group': r['group'], 'cell_idx': r['cell_idx'], 'fs': fs, 'n_frames': len(y),
        'n_sweeps': N, 'warmup': burn, 'fixed': task['fixed'], 'version': RESULT_VERSION,
        'labels': [labels[i] for i in ok], 'n_failed': len(labels) - len(ok),
        'session_time': elapsed,
        'chain_time': [chains[i]['time'] for i in ok],
    }
    if len(dflt) < 2:
        result['skipped'] = True
        np.save(task['out_path'], np.array(result, dtype=object), allow_pickle=True)
        return '{} cell {}: only {} default chains finished -- skipped.'.format(
            r['group'], r['cell_idx'], len(dflt))

    tr = _traces([chains[i] for i in ok], fs, int(np.sum(np.isfinite(y))))
    result['skipped'] = False
    result.update(_diagnostics(tr, dflt))
    result['default_idx'] = dflt
    result['chain_ns'] = [float(v) for v in tr['ns'].mean(axis=1)]
    np.save(task['out_path'], np.array(result, dtype=object), allow_pickle=True)
    return '{} cell {}: max R-hat {:.3f} (default starts {:.3f}), ESS(ns) {:.0f} ({:.0f}s).'.format(
        r['group'], r['cell_idx'], result['rhat_max'], result['rhat_default_max'],
        result['ess_bulk_default']['ns'], elapsed)


def run_test(data_dir=_DEFAULT_DATA_DIR, fixed=False, n_sweeps=None, warmup_frac=0.5,
             workers=6, seed=0, overwrite=False, max_cells=None, pilot=False):
    out_dir = _out_dir(data_dir, fixed, pilot)
    cell_dir = os.path.join(out_dir, 'cells')
    os.makedirs(cell_dir, exist_ok=True)

    tasks = []
    for r in _omsi_cells(data_dir):
        if max_cells is not None and r['cell_idx'] >= max_cells:
            continue
        name = '{}_{:03d}'.format(r['group'], r['cell_idx'])
        out_path = os.path.join(cell_dir, name + '.npy')
        if os.path.exists(out_path) and not overwrite:
            prev = np.load(out_path, allow_pickle=True).item()
            if prev.get('version') == RESULT_VERSION:
                continue
        cell_seed = seed + 1000 * GROUP_ORDER.index(r['group']) + 20 * r['cell_idx']
        tasks.append({
            'omsi': r, 'n_sweeps': int(n_sweeps or r['n_sweeps']),
            'warmup_frac': warmup_frac, 'seed': cell_seed,
            'starts': _starts(np.asarray(r['dff'], dtype=np.float64), float(r['fs']),
                              cell_seed),
            'fixed': fixed, 'out_path': out_path,
            'work_dir': os.path.join(out_dir, 'work', name),
        })

    print('{} cells to run ({}), {} MATLAB sessions at a time.'.format(
        len(tasks), 'CaImAn, fixed' if fixed else 'CaImAn', workers))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_cell, t): t for t in tasks}
        for k, f in enumerate(as_completed(futs), 1):
            t = futs[f]['omsi']
            try:
                msg = f.result()
            except Exception as exc:
                msg = '{} cell {}: FAILED ({}).'.format(t['group'], t['cell_idx'], exc)
            print('[{}/{}, {:.0f} min] {}'.format(k, len(tasks), (time.time() - t0) / 60, msg))


def _load(data_dir, fixed, pilot=False):
    out = {}
    for p in sorted(glob.glob(os.path.join(_out_dir(data_dir, fixed, pilot), 'cells', '*.npy'))):
        r = np.load(p, allow_pickle=True).item()
        if r.get('version') == RESULT_VERSION and not r['skipped']:
            out[(r['group'], r['cell_idx'])] = r
    return out


def _shared_max(d, keys):
    v = [d[k] for k in keys if k in d and not np.isnan(d[k])]
    return max(v) if v else np.nan


def _series(omsi, cai, keys):
    both = [r for r in omsi if (r['group'], r['cell_idx']) in cai]
    pair = lambda r: cai[(r['group'], r['cell_idx'])]
    return {
        'omsi_rhat':     _by_group(both, lambda r: _shared_max(r['rhat_default'], keys)),
        'omsi_rhat_all': _by_group(both, lambda r: _shared_max(r['rhat'], keys)),
        'cai_rhat':      _by_group(both, lambda r: _shared_max(pair(r)['rhat_default'], keys)),
        'cai_rhat_all':  _by_group(both, lambda r: _shared_max(pair(r)['rhat'], keys)),
        'omsi_ess_ns':   _by_group(both, lambda r: r['ess_bulk_default']['ns']),
        'omsi_ess_ll':   _by_group(both, lambda r: r['ess_bulk_default']['loglik']),
        'cai_ess_ns':    _by_group(both, lambda r: pair(r)['ess_bulk_default']['ns']),
        'cai_ess_ll':    _by_group(both, lambda r: pair(r)['ess_bulk_default']['loglik']),
    }


def _keys(cai):
    return [k for k in SCALAR_KEYS
            if any(not np.isnan(r['rhat_default'][k]) for r in cai.values())]


def plot_figure(data_dir=_DEFAULT_DATA_DIR, fixed=False, pilot=False):
    omsi = [r for r in _omsi_cells(data_dir) if r['group'] in SYNTH_CONDITIONS]
    cai = {k: v for k, v in _load(data_dir, fixed, pilot).items() if k[0] in SYNTH_CONDITIONS}
    if not cai:
        raise FileNotFoundError('No CaImAn results in {}. Run --mode test first.'.format(
            _out_dir(data_dir, fixed, pilot)))
    keys = _keys(cai)
    s = _series(omsi, cai, keys)
    name = 'CaImAn, fixed' if fixed else 'CaImAn'
    o_kw = {'color': METHOD_COLORS['OMSI']}
    c_kw = {'color': METHOD_COLORS['CaImAn']}

    fig, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(6.5, 4.6), dpi=300,
                                     gridspec_kw={'hspace': 0.8})

    _strip(ax_r, [
        ('OMSI, own starts',      s['omsi_rhat'],     dict(o_kw, filled=False)),
        ('OMSI, all starts',      s['omsi_rhat_all'], o_kw),
        (name + ', own starts',   s['cai_rhat'],      dict(c_kw, filled=False)),
        (name + ', all starts',   s['cai_rhat_all'],  c_kw),
    ], 'max split $\\hat{R}$', hline=RHAT_THR, log=True)
    ax_r.set_ylim(0.7, 2e3)
    ax_r.set_yticks([1, 10, 100, 1000])
    ax_r.set_yticklabels(['1', '10', '100', '$\\geq$1000'])
    ax_r.minorticks_off()
    ax_r.legend(frameon=False, fontsize=5.5, handletextpad=0.2, loc='upper left',
                bbox_to_anchor=(1.0, 1.0))

    _strip(ax_e, [
        ('OMSI, spike count',     s['omsi_ess_ns'], dict(o_kw, filled=False)),
        ('OMSI, log-lik.',        s['omsi_ess_ll'], o_kw),
        (name + ', spike count',  s['cai_ess_ns'],  dict(c_kw, filled=False)),
        (name + ', log-lik.',     s['cai_ess_ll'],  c_kw),
    ], 'bulk ESS', hline=ESS_THR, log=True)
    ax_e.minorticks_off()
    ax_e.legend(frameon=False, fontsize=5.5, handletextpad=0.2, loc='upper left',
                bbox_to_anchor=(1.0, 1.0))

    stem = 'caiman_fixed_convergence' if fixed else 'caiman_convergence'
    fig_dir = _out_dir(data_dir, fixed, pilot) if pilot else data_dir
    for ext in ('png', 'svg'):
        out = os.path.join(fig_dir, '{}.{}'.format(stem, ext))
        fig.savefig(out, bbox_inches='tight')
        print('Saved {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR, fixed=False, pilot=False):
    omsi = _omsi_cells(data_dir)
    cai = _load(data_dir, fixed, pilot)
    if not cai:
        raise FileNotFoundError('No CaImAn results in {}.'.format(
            _out_dir(data_dir, fixed, pilot)))
    keys = _keys(cai)
    s = _series(omsi, cai, keys)
    name = 'CaImAn, fixed' if fixed else 'CaImAn'
    n_sw = sorted({r['n_sweeps'] for r in cai.values()})
    print('{} vs OMSI, {} sweeps per chain (first half warmup); max R-hat over: {}.'.format(
        name, '/'.join(str(v) for v in n_sw), ', '.join(keys)))
    print('Medians over cells both methods ran; [cells < 1.01 R-hat or >= 400 ESS].\n')

    def _m(d, g, thr, below):
        v = d.get(g, np.array([]))
        v = v[np.isfinite(v)]
        if not len(v):
            return '{:>16}'.format('n/a')
        n_ok = int(np.sum(v < thr)) if below else int(np.sum(v >= thr))
        return '{:>9.3g} [{:>2}/{:<2}]'.format(np.median(v), n_ok, len(v))

    rows = [
        ('max R-hat, own starts',  'omsi_rhat',   'cai_rhat',   RHAT_THR, True),
        ('max R-hat, all starts',  'omsi_rhat_all', 'cai_rhat_all', RHAT_THR, True),
        ('bulk ESS, spike count',  'omsi_ess_ns', 'cai_ess_ns', ESS_THR, False),
        ('bulk ESS, log-lik.',     'omsi_ess_ll', 'cai_ess_ll', ESS_THR, False),
    ]
    for g in GROUP_ORDER:
        if g not in s['cai_rhat']:
            continue
        print('--- {} ({} cells) ---'.format(GROUP_NAMES[g], len(s['cai_rhat'][g])))
        print('  {:<24} {:>16} {:>16}'.format('', 'OMSI', name))
        for label, ko, kc, thr, below in rows:
            print('  {:<24} {} {}'.format(label, _m(s[ko], g, thr, below),
                                          _m(s[kc], g, thr, below)))
        t = [np.median(r['chain_time']) for (gg, _), r in cai.items() if gg == g]
        print('  {:<24} {:>16.1f}'.format('CaImAn s per chain', np.median(t)))
    n_fail = sum(r['n_failed'] for r in cai.values())
    if n_fail:
        print('\n{} CaImAn chains failed and were left out.'.format(n_fail))


def main():
    parser = argparse.ArgumentParser(
        description='Convergence diagnostics for CaImAn MCMC on the OMSI benchmark cells')
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'print'])
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='convergence_benchmark.py data directory')
    parser.add_argument('--fixed', action='store_true',
                        help='CaImAn with the rise-time fix and init threshold')
    parser.add_argument('--n-sweeps', type=int, default=None,
                        help='Chain length incl. warmup (default: match OMSI)')
    parser.add_argument('--warmup-frac', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=6, help='Concurrent MATLAB sessions')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true', help='Rerun cells with saved results')
    parser.add_argument('--pilot', action='store_true',
                        help='2 cells per condition, 1000 sweeps, kept in a separate '
                             'pilot folder')
    args = parser.parse_args()

    if args.mode == 'test':
        kw = dict(fixed=args.fixed, n_sweeps=args.n_sweeps, warmup_frac=args.warmup_frac,
                  workers=args.workers, seed=args.seed, overwrite=args.overwrite)
        if args.pilot:
            kw.update(n_sweeps=args.n_sweeps or 1000, max_cells=2)
        run_test(data_dir=args.data_dir, pilot=args.pilot, **kw)
    elif args.mode == 'plot':
        plot_figure(args.data_dir, fixed=args.fixed, pilot=args.pilot)
    else:
        print_stats(args.data_dir, fixed=args.fixed, pilot=args.pilot)


if __name__ == '__main__':
    main()
