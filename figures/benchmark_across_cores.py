# -*- coding: utf-8 -*-
"""
figures/benchmark_across_cores.py

Scaling of OMSI wall-clock time with the number of CPU cores.

DMM, October 2026
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

import OMSI
from OMSI._win_perf import no_power_throttling

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, 'data', 'cores')
_FIG1_NPZ = os.path.join(_HERE, 'data', 'fig1', 'fixed_benchmark_OMSI.npz')
_RESULTS_NAME = 'benchmark_across_cores.json'

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 7

COLOR = '#4C72B0'

def _load_population(fig1_npz=_FIG1_NPZ):

    if os.path.isfile(fig1_npz):
        print('Loading figure 1 population from {}.'.format(fig1_npz))
        d = np.load(fig1_npz, allow_pickle=True)
        return d['noisy_traces'], list(d['true_spikes']), float(d['f'])

    print('No figure 1 results at {}; regenerating its population.'.format(fig1_npz))
    import figure1
    from simulation_helpers import generate_synthetic_data
    sim_snr = figure1.draw_sim_snr(figure1.N_CELLS)
    noisy, true_spikes, _, _, _, _, _ = generate_synthetic_data(
        n_cells=figure1.N_CELLS, fs=figure1.FS, duration=figure1.DURATION,
        tau=figure1.TAU, measured_snr=sim_snr, amp_cv=figure1.AMP_CV,
        tau_cv=figure1.TAU_CV, drift_sd=figure1.DRIFT_SD,
        drift_timescale=figure1.DRIFT_TIMESCALE, return_params=True
    )
    return noisy, list(true_spikes), float(figure1.FS)


def _default_core_counts():

    n = os.cpu_count() or 1
    counts = [2 ** k for k in range(int(np.log2(n)) + 1)]
    if counts[-1] != n:
        counts.append(n)
    return counts


def _load_results(path):

    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'runs': []}


def _save_results(results, path):

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def run_benchmark(data_dir=_DEFAULT_DATA_DIR, core_counts=None, n_cells=None,
                  overwrite=False):

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _RESULTS_NAME)

    noisy, true_spikes, fs = _load_population()
    if n_cells is not None:
        noisy, true_spikes = noisy[:n_cells], true_spikes[:n_cells]
    n, n_frames = noisy.shape

    params = {
        'f':        fs,
        'p':        2,
        'Nsamples': 200,
        'B':        75,
        'marg':     0,
        'upd_gam':  1,
    }

    results = _load_results(path)
    if results['runs'] and (results.get('n_cells') != n or results.get('n_frames') != n_frames):
        raise ValueError('{} holds results for {} cells x {} frames, not {} x {}. '
                         'Use another --data-dir.'.format(
                             path, results.get('n_cells'), results.get('n_frames'), n, n_frames))
    results.update({
        'hostname':      platform.node(),
        'processor':     platform.processor(),
        'logical_cores': os.cpu_count(),
        'n_cells':       n,
        'n_frames':      n_frames,
        'fs':            fs,
    })
    try:
        import psutil
        results['physical_cores'] = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    done = {r['num_cpus'] for r in results['runs']}
    for k in core_counts or _default_core_counts():
        if k in done and not overwrite:
            print('\n{} cores already in {}; skipping.'.format(k, path))
            continue

        print('\nRunning OMSI on {} cells with {} cores...'.format(n, k))
        t0 = time.perf_counter()
        res = OMSI.deconv(noisy, params, true_spikes=true_spikes, benchmark=True,
                          num_cpus=k)
        wall = time.perf_counter() - t0
        cpu_s = float(np.sum(res['optim_times_per_cell']))
        f1 = float(np.nanmedian(res['optim_F1']))
        print('  {} cores: {:.1f}s wall, {:.1f}s summed per-cell, median F1 {:.3f}'.format(
            k, wall, cpu_s, f1))

        results['runs'] = [r for r in results['runs'] if r['num_cpus'] != k]
        results['runs'].append({
            'num_cpus':    k,
            'wall_s':      wall,
            'cell_time_s': cpu_s,
            'median_F1':   f1,
            'timestamp':   datetime.now().isoformat(timespec='seconds'),
        })
        results['runs'].sort(key=lambda r: r['num_cpus'])
        _save_results(results, path)

    print('\nSaved results to {}.'.format(path))


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    path = os.path.join(data_dir, _RESULTS_NAME)
    results = _load_results(path)
    if not results['runs']:
        print('No results in {} -- run with --mode run first.'.format(path))
        return

    cores = np.array([r['num_cpus'] for r in results['runs']], dtype=float)
    wall = np.array([r['wall_s'] for r in results['runs']]) / 60.0

    fig, ax = plt.subplots(figsize=(2.4, 2.2), dpi=300)

    ideal = wall[0] * cores[0] / cores
    ax.plot(cores, ideal, '--', color='#999999', lw=1, label='ideal scaling')
    ax.plot(cores, wall, '-o', color=COLOR, lw=1.5, ms=4, label='OMSI')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.FixedLocator(cores))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: '{:g}'.format(v)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('available CPU cores')
    ax.set_ylabel('wall-clock time (min)')

    fig.tight_layout()
    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, 'benchmark_across_cores.{}'.format(ext))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved figure to {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    path = os.path.join(data_dir, _RESULTS_NAME)
    results = _load_results(path)
    if not results['runs']:
        print('No results in {} -- run with --mode run first.'.format(path))
        return

    runs = results['runs']
    ref = runs[0]

    print('\n' + '=' * 105)
    print('OMSI WALL-CLOCK TIME VS. CPU CORES')
    print('=' * 105)
    print('Machine: {}  ({}; {} logical / {} physical cores)'.format(
        results.get('hostname', '?'), results.get('processor', '?'),
        results.get('logical_cores', '?'), results.get('physical_cores', '?')))
    print('Population: {} cells x {} frames at {:g} Hz ({:.0f} min)'.format(
        results['n_cells'], results['n_frames'], results['fs'],
        results['n_frames'] / results['fs'] / 60.0))

    print('\n{:>6}  {:>10}  {:>10}  {:>9}  {:>11}  {:>16}  {:>10}  {:>19}'.format(
        'Cores', 'Wall (s)', 'Wall (min)', 'Speed-up', 'Efficiency', 'Per-cell sum (s)',
        'Median F1', 'Run at'))
    print('-' * 105)
    for r in runs:
        speedup = ref['wall_s'] / r['wall_s']
        efficiency = speedup * ref['num_cpus'] / r['num_cpus']
        print('{:>6d}  {:>10.1f}  {:>10.2f}  {:>8.2f}x  {:>10.1f}%  {:>16.1f}  {:>10.3f}  {:>19}'.format(
            r['num_cpus'], r['wall_s'], r['wall_s'] / 60.0, speedup, 100.0 * efficiency,
            r['cell_time_s'], r['median_F1'], r.get('timestamp', '')))
    print('  Speed-up and efficiency are relative to the {}-core run.'.format(ref['num_cpus']))


def main():

    parser = argparse.ArgumentParser(
        description='OMSI wall-clock time vs. number of CPU cores on figure 1\'s population.'
    )
    parser.add_argument('--mode', default='run', choices=['run', 'plot', 'print', 'all'],
                        help='Run the benchmark, draw the figure, print the timings, '
                             'or all three.')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for the results JSON and figure.')
    parser.add_argument('--cores', default='',
                        help='Comma-separated core counts. Defaults to powers of two '
                             'up to every logical core.')
    parser.add_argument('--n-cells', type=int, default=None,
                        help='Use only the first N cells (smoke test; use a separate --data-dir).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun core counts already in the results JSON.')
    args = parser.parse_args()

    cores = [int(c) for c in args.cores.split(',') if c.strip()] or None

    if args.mode in ('run', 'all'):
        with no_power_throttling(verbose=True):
            run_benchmark(args.data_dir, core_counts=cores, n_cells=args.n_cells,
                          overwrite=args.overwrite)
    if args.mode in ('plot', 'all'):
        plot_figure(args.data_dir)
    if args.mode in ('print', 'all'):
        print_stats(args.data_dir)


if __name__ == '__main__':
    main()
