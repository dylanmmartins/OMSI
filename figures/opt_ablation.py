# -*- coding: utf-8 -*-
"""
figures/opt_ablation.py

DMM, September 2026
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

_OMSI_COLOR = '#4C72B0'

_SHORT_LABELS = {
    'i':      '(i) no local support window',
    'ii':     '(ii) add/remove moves T/100',
    'iii':    '(iii) FOOPSI initialization',
    'iv':     '(iv) fixed 750 sweeps',
    'v':      '(v) O(T*K) reconstruction',
    'vi':     '(vi) no kernel-energy cache',
    'vii':    '(vii) no SNR screen',
    'x_par':  '(x) no parallelism',
    'x_jit':  '(x) no Numba',
}

CODE_LEVEL = ['v', 'vi']

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'opt_ablation')
_POP_SEED = 20260926

_BASE_PARAMS = {'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1}

CONDITIONS = {
    'full':  ('OMSI (all optimizations)', {}, {}),
    'i':     ('(i) no local support window (full-length kernel, prec=0)',
              {'prec': 0.0}, {}),
    'ii':    ('(ii) add/remove moves T/100 (CaImAn default) instead of T/500',
              {'add_move': 'T/100'}, {}),
    'iii':   ('(iii) FOOPSI initialization instead of NNLS',
              {'init_method': 'foopsi'}, {}),
    'iv':    ('(iv) fixed 250 burn-in + 500 samples instead of adaptive stopping',
              {'auto_stop': False, 'B': 250, 'Nsamples': 500}, {}),
    'v':     ('(v) per-spike O(T*K) calcium reconstruction instead of O(T) accumulators',
              {'ablate_accumulator': True}, {}),
    'vi':    ('(vi) kernel energy recomputed on every proposal instead of cached',
              {'ablate_energy_cache': True}, {}),
    'vii':   ('(vii) no SNR screen',
              {'skip_snr': True}, {}),
    'x_par': ('(x) no parallelism (Ray limited to 1 CPU)',
              {}, {'ray_cpus': 1}),
    'x_jit': ('(x) no Numba compilation (NUMBA_DISABLE_JIT=1)',
              {}, {'env': {'NUMBA_DISABLE_JIT': '1'}}),
    'caiman': ('CaImAn MCMC (MATLAB, cells in a serial for-loop, 250 + 500 sweeps)',
               {}, {'matlab': True}),
}

NOT_ABLATED = [
    ('(viii) AR(1) kernel and lambda re-estimation for fast indicators',
     'only triggers for fast indicators at high frame rates; inactive at tau=1.2 s, 30 Hz'),
    ('(ix) fixed time constants for tau < 0.6 s', 'inactive at tau=1.2 s'),
    ('(x) float32 and array reuse', 'built into the code; needs source changes'),
]


def _make_population(data_dir, n_cells):
    path = os.path.join(data_dir, 'population.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as d:
            pop = {k: d[k] for k in d.files}
        if len(pop['noisy']) == n_cells:
            return pop
        print('Existing population has {} cells, not {}; regenerating.'.format(
            len(pop['noisy']), n_cells))

    import figure1 as f1
    from simulation_helpers import generate_synthetic_data

    np.random.seed(_POP_SEED)
    sim_snr = f1.draw_sim_snr(n_cells)
    noisy, true_spikes, _, _, _, _, prm = generate_synthetic_data(
        n_cells=n_cells, fs=f1.FS, duration=f1.DURATION, tau=f1.TAU, measured_snr=sim_snr,
        amp_cv=f1.AMP_CV, tau_cv=f1.TAU_CV, drift_sd=f1.DRIFT_SD,
        drift_timescale=f1.DRIFT_TIMESCALE, return_params=True)
    sim_snr = prm['measured_snr']
    ts = np.empty(n_cells, dtype=object)
    for i, s in enumerate(true_spikes):
        ts[i] = np.asarray(s, dtype=float)
    pop = {'noisy': noisy.astype(np.float32), 'true_spikes': ts, 'fs': np.float64(f1.FS),
           'tau': np.float64(f1.TAU), 'sim_snr': sim_snr, 'sim_tau': prm['tau']}
    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, **pop)
    print('Saved population ({} cells, {:.0f} s at {:g} Hz) to {}.'.format(
        n_cells, noisy.shape[1] / f1.FS, f1.FS, path))
    return pop


def _warm_up(fs):
    import OMSI
    from simulation_helpers import generate_synthetic_data
    state = np.random.get_state()
    y, ts, *_ = generate_synthetic_data(n_cells=1, fs=fs, duration=30, tau=1.2, snr=np.array([8.0]))
    OMSI.deconv(y, dict(_BASE_PARAMS, f=fs), true_spikes=ts, benchmark=True)
    np.random.set_state(state)


def _limit_ray_cpus(n_cpus):
    import ray
    orig_init = ray.init

    def _init(*args, **kwargs):
        kwargs['num_cpus'] = n_cpus
        return orig_init(*args, **kwargs)

    ray.init = _init


def _run_condition(key, data_dir, n_cells):
    label, overrides, extra = CONDITIONS[key]
    pop = _make_population(data_dir, n_cells)
    noisy, true_spikes = pop['noisy'], list(pop['true_spikes'])
    fs, tau = float(pop['fs']), float(pop['tau'])
    n_frames = noisy.shape[1]
    print('Condition {}: {}'.format(key, label))

    if extra.get('matlab'):
        from run_pnev_MCMC import run_matlab_pnevMCMC
        t0 = time.time()
        spikes, _, _, sweeps, cell_times = run_matlab_pnevMCMC(
            noisy.astype(np.float64), fs=fs, tau=tau, n_sweeps=500,
            true_spikes=true_spikes, return_cell_times=True)
        wall = time.time() - t0
        if np.all(np.asarray(sweeps) == 0):
            raise RuntimeError('MATLAB returned no result; nothing saved.')
        n_sweeps = np.full(n_cells, 500)
    else:
        import OMSI
        if extra.get('ray_cpus'):
            _limit_ray_cpus(extra['ray_cpus'])
        params = dict(_BASE_PARAMS, f=fs)
        for k, v in overrides.items():
            params[k] = int(np.ceil(n_frames / 100)) if v == 'T/100' else v
        if os.environ.get('NUMBA_DISABLE_JIT') != '1':
            _warm_up(fs)
        t0 = time.time()
        res = OMSI.deconv(noisy, params, true_spikes=true_spikes, benchmark=True)
        wall = time.time() - t0
        spikes = list(res['optim_spikes'])
        cell_times = np.asarray(res['optim_times_per_cell'], dtype=float)
        n_sweeps = np.asarray(res['optim_nsamples'])

    sp = np.empty(n_cells, dtype=object)
    for i, s in enumerate(spikes):
        sp[i] = np.asarray(s, dtype=float).ravel()
    out = os.path.join(data_dir, 'cond_{}.npz'.format(key))
    np.savez(out, spikes=sp, cell_times=cell_times, n_sweeps=n_sweeps,
             wall=np.float64(wall), label=label)
    print('  wall {:.1f} s; median {:.2f} s/cell; saved {}.'.format(
        wall, np.nanmedian(cell_times), out))


def _scores(true_spikes, spikes, fs):
    import OMSI.helpers as helpers
    import figure1 as f1
    prec, rec, _ = helpers.compute_accuracy_window(true_spikes, spikes)
    return f1._fbeta(prec, rec), helpers.compute_cosmic(true_spikes, spikes, fs)


def _load_results(data_dir):
    with np.load(os.path.join(data_dir, 'population.npz'), allow_pickle=True) as d:
        true_spikes, fs = list(d['true_spikes']), float(d['fs'])

    results = {}
    for key in CONDITIONS:
        path = os.path.join(data_dir, 'cond_{}.npz'.format(key))
        if not os.path.exists(path):
            continue
        with np.load(path, allow_pickle=True) as d:
            r = {k: d[k] for k in d.files}
        r['fb'], r['cos'] = _scores(true_spikes, list(r['spikes']), fs)
        results[key] = r
    if 'full' not in results:
        raise RuntimeError('No results for the full OMSI condition; run --mode run first.')
    return results


def _summarize(data_dir):
    from stats_helpers import signed_rank, fmt_test

    results = _load_results(data_dir)
    ref = results['full']

    rows = []
    for key, (label, _, _) in CONDITIONS.items():
        if key not in results:
            rows.append({'condition': label, 'note': 'not run'})
            continue
        r = results[key]
        ct = np.asarray(r['cell_times'], float)
        rows.append({
            'condition':          label,
            'wall_s':             float(r['wall']),
            'wall_vs_full':       float(r['wall']) / float(ref['wall']),
            'cpu_s_per_cell':     float(np.nanmedian(ct)),
            'cpu_vs_full':        float(np.nansum(ct) / np.nansum(ref['cell_times'])),
            'samples_per_cell':   float(np.nanmedian(r['n_sweeps'])),
            'fbeta':              float(np.nanmedian(r['fb'])),
            'cosmic':             float(np.nanmedian(r['cos'])),
            'fbeta_p':            '' if key == 'full' else fmt_test(signed_rank(ref['fb'], r['fb'])),
            'cosmic_p':           '' if key == 'full' else fmt_test(signed_rank(ref['cos'], r['cos'])),
            'note':               '',
        })
    for label, reason in NOT_ABLATED:
        rows.append({'condition': label, 'note': 'not ablated: ' + reason})
    return rows


def _print_table(rows, data_dir):
    cols = [('condition', 'Condition', '{}'), ('wall_s', 'Wall (s)', '{:.1f}'),
            ('wall_vs_full', 'Wall / full', '{:.2f}x'),
            ('cpu_s_per_cell', 'Median s/cell', '{:.2f}'),
            ('cpu_vs_full', 'Total cell time / full', '{:.2f}x'),
            ('samples_per_cell', 'Median posterior samples', '{:.0f}'),
            ('fbeta', 'Median F_beta', '{:.3f}'), ('fbeta_p', 'F_beta vs full', '{}'),
            ('cosmic', 'Median CosMIC', '{:.3f}'), ('cosmic_p', 'CosMIC vs full', '{}'),
            ('note', 'Note', '{}')]
    lines = ['| ' + ' | '.join(h for _, h, _ in cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for r in rows:
        cells = [fmt.format(r[k]) if k in r and r[k] != '' else '' for k, _, fmt in cols]
        lines.append('| ' + ' | '.join(cells) + ' |')
    md = '\n'.join(lines)
    print('\n' + md)
    print('\nWall: total time for all cells, including Ray start-up. s/cell: time '
          'inside each cell\'s sampler (for CaImAn, measured in MATLAB). Posterior samples: '
          'sweeps kept after burn-in (OMSI does not report its burn-in length).\n'
          'Ratios above 1 mean the ablated version is slower. p-values: two-sided '
          'Wilcoxon signed-rank, paired by cell, uncorrected.')
    with open(os.path.join(data_dir, 'opt_ablation_table.md'), 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    with open(os.path.join(data_dir, 'opt_ablation_table.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _, _ in cols], extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Saved table to {}.'.format(os.path.join(data_dir, 'opt_ablation_table.md')))


def _plot_figure(data_dir):
    from matplotlib.markers import MarkerStyle
    from stats_helpers import signed_rank, _stars

    results = _load_results(data_dir)
    ref = results['full']
    keys = [k for k in CONDITIONS if k not in ('full', 'caiman') and k in results]
    if not keys:
        print('No ablation conditions saved yet; no figure made.')
        return
    y = np.arange(len(keys))[::-1].astype(float)
    no_violin = {'x_par'}
    diamond = MarkerStyle('D', joinstyle='miter')

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 0.3 * len(keys) + 1.0), dpi=300,
                            sharey=True, gridspec_kw={'width_ratios': [1.3, 1, 1]})

    def _violins(ax, data):
        rows = [(np.asarray(d, float), yi) for d, yi, k in zip(data, y, keys)
                if k not in no_violin]
        rows = [(d[np.isfinite(d)], yi) for d, yi in rows]
        rows = [(d, yi) for d, yi in rows if len(d) > 1]
        if not rows:
            return
        parts = ax.violinplot([d for d, _ in rows], positions=[yi for _, yi in rows],
                              vert=False, showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(_OMSI_COLOR)
            pc.set_edgecolor('none')
            pc.set_alpha(0.75)
        for name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            parts[name].set_color('k')
            parts[name].set_linewidth(0.8)

    ax = axs[0]
    ratios = [np.log10(np.asarray(results[k]['cell_times'], float) /
                       np.asarray(ref['cell_times'], float)) for k in keys]
    _violins(ax, ratios)
    wall = [np.log10(float(results[k]['wall']) / float(ref['wall'])) for k in keys]
    ax.scatter(wall, y, marker=diamond, s=16, color='k', linewidths=0, zorder=3,
               label='total wall time')
    ax.axvline(0.0, color='0.6', lw=0.6, ls='--')
    vals = np.concatenate([r[np.isfinite(r)] for r in ratios] + [np.array(wall)])
    lo, hi = np.floor(vals.min() * 2) / 2, np.ceil(vals.max() * 2) / 2
    ticks = [t for t in (0.1, 1, 10, 100, 1000) if lo - 0.1 <= np.log10(t) <= hi + 0.1]
    minor = [np.log10(m * t) for t in (0.1, 1, 10, 100) for m in range(2, 10)
             if lo - 0.1 <= np.log10(m * t) <= hi + 0.1]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels(['{:g}x'.format(t) for t in ticks])
    ax.set_xticks(minor, minor=True)
    ax.set_xlim(lo - 0.1, hi + 0.1)
    ax.set_xlabel('fold slowdown vs full OMSI')
    ax.legend(loc='upper right', frameon=False, fontsize=6, handletextpad=0.2)

    for ax, key, label in [(axs[1], 'fb', r'$\Delta F_\beta$ vs full OMSI'),
                           (axs[2], 'cos', r'$\Delta$CosMIC vs full OMSI')]:
        diffs = [np.asarray(results[k][key], float) - np.asarray(ref[key], float) for k in keys]
        _violins(ax, diffs)
        ax.axvline(0.0, color='0.6', lw=0.6, ls='--')
        shown = [d for d, k in zip(diffs, keys) if k not in no_violin]
        lim = max(np.nanmax(np.abs(np.concatenate(shown))), 0.02) * 1.1
        ax.set_xlim(-lim, lim)
        for yi, k in zip(y, keys):
            if k in no_violin:
                continue
            stars = _stars(signed_rank(results[k][key], ref[key])['p'])
            if stars and stars != 'n.s.':
                ax.text(lim * 0.98, yi, stars, ha='right', va='center', fontsize=6)
        ax.set_xlabel(label)

    axs[0].set_yticks(y)
    axs[0].set_yticklabels([_SHORT_LABELS.get(k, k) for k in keys])
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(data_dir, 'opt_ablation.{}'.format(ext)), bbox_inches='tight')
    plt.close(fig)
    print('Saved figure to {}.'.format(os.path.join(data_dir, 'opt_ablation.png')))


def main():

    parser = argparse.ArgumentParser(description='Ablation of OMSI optimizations (i)-(x)')
    parser.add_argument('--mode', required=True, choices=['run', 'table', 'worker'],
                        help='run: run every missing condition, then make the table and figure; '
                             'table: rebuild the table and figure from saved results')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR)
    parser.add_argument('--n-cells', type=int, default=100)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=None,
                        help='Conditions to run (default: all)')
    parser.add_argument('--no-matlab', action='store_true', help='Skip the CaImAn reference row')
    parser.add_argument('--force', action='store_true', help='Re-run conditions already saved')
    parser.add_argument('--code-level-only', action='store_true',
                        help='Run only the code-level ablations (v) and (vi) (plus the full-OMSI '
                             'baseline if it is not saved yet) and save their data; no table '
                             'or figure')
    parser.add_argument('--condition', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    if args.mode == 'worker':
        _run_condition(args.condition, args.data_dir, args.n_cells)
        return

    if args.mode == 'run':
        _make_population(args.data_dir, args.n_cells)
        if args.code_level_only:
            keys = list(CODE_LEVEL)
            if not os.path.exists(os.path.join(args.data_dir, 'cond_full.npz')):
                keys = ['full'] + keys
        else:
            keys = args.conditions or list(CONDITIONS)
        if args.no_matlab:
            keys = [k for k in keys if k != 'caiman']
        for key in keys:
            out = os.path.join(args.data_dir, 'cond_{}.npz'.format(key))
            if os.path.exists(out) and not args.force:
                print('Condition {} already saved; skipping (use --force to re-run).'.format(key))
                continue
            env = dict(os.environ)
            env.update(CONDITIONS[key][2].get('env', {}))
            cmd = [sys.executable, os.path.abspath(__file__), '--mode', 'worker',
                   '--condition', key, '--data-dir', args.data_dir,
                   '--n-cells', str(args.n_cells)]
            if subprocess.run(cmd, env=env).returncode != 0:
                print('  ERROR: condition {} failed; continuing with the others.'.format(key))
        if args.code_level_only:
            print('Saved code-level ablation data to {} (no table or figure).'.format(args.data_dir))
            return

    _print_table(_summarize(args.data_dir), args.data_dir)
    _plot_figure(args.data_dir)


if __name__ == '__main__':
    main()
