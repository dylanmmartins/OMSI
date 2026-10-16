# -*- coding: utf-8 -*-
"""
figures/snr_test.py

DMM, September 2026
"""

import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from simulation_helpers import generate_synthetic_data

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'snr_test'
)

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

GATE_THRESHOLD = 2.0

SNR_LEVELS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

N_CELLS  = 50
DURATION = 300.0
FS       = 30.0
TAU      = 1.2

COLOR_MAIN   = '#4C72B0'
COLOR_FLOOR  = '#8172B3'
COLOR_GATE   = '#C44E52'
COLOR_REAL   = '#55A868'

np.random.seed(3)


def gate_statistic(y):

    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan

    sn_mad = float(np.median(np.abs(np.diff(y)))) / 0.6745
    peak   = float(np.percentile(y, 99))
    base   = float(np.percentile(y, 8))

    return (peak - base) / (sn_mad + 1e-9)


def _white_noise_floor(n_frames=20000, n_reps=200, seed=0):

    rng = np.random.default_rng(seed)
    return np.array([
        gate_statistic(rng.standard_normal(n_frames)) for _ in range(n_reps)
    ])


def _synthetic_sweep():

    print('Generating clean population (n_cells={}, duration={}s)...'.format(
        N_CELLS, DURATION))
    _, _, clean, _, _, _ = generate_synthetic_data(
        n_cells=N_CELLS, fs=FS, duration=DURATION, tau=TAU, snr=1e6
    )

    peak = np.array([
        np.percentile(clean[i], 99) - np.percentile(clean[i], 1)
        for i in range(N_CELLS)
    ])
    peak = np.maximum(peak, 1e-9)

    snr_grid, stats = [], []
    for snr_val in SNR_LEVELS:
        sigmas = peak / snr_val
        dff = clean + np.random.normal(0, sigmas[:, None], size=clean.shape)
        for i in range(N_CELLS):
            snr_grid.append(snr_val)
            stats.append(gate_statistic(dff[i]))
        print('  SNR={:6.2f}  median gate stat {:.2f}'.format(
            snr_val, float(np.median(stats[-N_CELLS:]))))

    return np.array(snr_grid), np.array(stats)


def _load_fig3_traces(fig3_dir):

    traces = []
    for path in sorted(glob.glob(os.path.join(fig3_dir, '*_traces.npz'))):
        try:
            z = np.load(path, allow_pickle=True)
        except Exception as exc:
            print('  Skipping {}: {}'.format(os.path.basename(path), exc))
            continue
        if 'dff' not in z:
            continue
        dff = z['dff']
        if dff.dtype == object:
            traces.extend(list(dff))
        else:
            traces.extend([dff[i] for i in range(dff.shape[0])])
    return traces


def _load_fig4_traces(fig4_trace_dir):

    traces, labels, n_true = [], [], []
    for path in sorted(glob.glob(os.path.join(fig4_trace_dir, '*_traces.npz'))):
        name = os.path.basename(path).replace('_traces.npz', '')
        try:
            z = np.load(path, allow_pickle=True)
        except Exception as exc:
            print('  Skipping {}: {}'.format(name, exc))
            continue
        n_cells = int(z['n_cells']) if 'n_cells' in z else 0
        for i in range(n_cells):
            key = 'dff_{}'.format(i)
            if key in z:
                traces.append(z[key])
                labels.append(name)
                spk_key = 'true_spikes_{}'.format(i)
                n_true.append(len(z[spk_key]) if spk_key in z else 0)
    return traces, labels, n_true


def _real_trace_stats(figures_dir):

    fig3_dir = os.path.join(figures_dir, 'data', 'fig3')
    fig4_dir = os.path.join(figures_dir, 'data', 'fig4', 'ground_truth_traces_omsi')

    print('Scoring Allen traces...')
    allen = np.array([gate_statistic(t) for t in _load_fig3_traces(fig3_dir)])

    print('Scoring ground-truth traces...')
    gt_traces, gt_labels, gt_n_true = _load_fig4_traces(fig4_dir)
    gt = np.array([gate_statistic(t) for t in gt_traces])

    print('  Allen: {} traces.  Ground truth: {} traces.'.format(len(allen), len(gt)))

    gt_ok = np.isfinite(gt) if len(gt) else np.zeros(0, dtype=bool)

    return {
        'allen': allen[np.isfinite(allen)] if len(allen) else np.array([]),
        'groundtruth': gt[gt_ok] if len(gt) else np.array([]),
        'groundtruth_labels': np.array(gt_labels, dtype=object)[gt_ok]
                              if len(gt) else np.array([], dtype=object),
        'groundtruth_n_true': np.array(gt_n_true, dtype=float)[gt_ok]
                              if len(gt) else np.array([]),
    }


def _degenerate_cases(n_frames=20000, seed=7):

    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)

    cases = [
        ('constant trace',        np.ones(n_frames)),
        ('constant + 1 outlier',  np.r_[np.ones(n_frames - 1), 5.0]),
        ('Nyquist alternation',   np.where(t % 2, 1.0, -1.0)),
        ('half alternation',      0.5 * rng.standard_normal(n_frames)
                                  + 1.5 * np.where(t % 2, 1.0, -1.0)),
        ('uniform noise',         rng.random(n_frames)),
        ('white Gaussian noise',  rng.standard_normal(n_frames)),
        ('Laplace noise',         rng.laplace(0.0, 1.0, n_frames)),
        ('heavy tail (t, df=2)',  rng.standard_t(2, n_frames)),
        ('smoothed noise (sd=2)', np.convolve(
            rng.standard_normal(n_frames),
            np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
            / np.sum(np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)), 'same')),
    ]

    names  = [c[0] for c in cases]
    values = np.array([gate_statistic(c[1]) for c in cases])
    return names, values


def run_test(data_dir=_DEFAULT_DATA_DIR):

    os.makedirs(data_dir, exist_ok=True)
    figures_dir = os.path.dirname(os.path.abspath(__file__))

    floor = _white_noise_floor()
    print('White-noise floor: {:.3f} (5-95 pct {:.3f}-{:.3f}).'.format(
        float(np.median(floor)), *np.percentile(floor, [5, 95])))

    snr_grid, snr_stats = _synthetic_sweep()
    real = _real_trace_stats(figures_dir)
    case_names, case_values = _degenerate_cases()

    out_path = os.path.join(data_dir, 'snr_test_results.npz')
    np.savez(
        out_path,
        floor=floor,
        snr_grid=snr_grid,
        snr_stats=snr_stats,
        allen=real['allen'],
        groundtruth=real['groundtruth'],
        groundtruth_labels=real['groundtruth_labels'],
        groundtruth_n_true=real['groundtruth_n_true'],
        case_names=np.array(case_names, dtype=object),
        case_values=case_values,
        threshold=GATE_THRESHOLD,
    )
    print('Saved results to {}.'.format(out_path))


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    res_path = os.path.join(data_dir, 'snr_test_results.npz')
    if not os.path.exists(res_path):
        print('No results at {} -- run with --mode test first.'.format(res_path))
        return

    z = np.load(res_path, allow_pickle=True)
    floor_med = float(np.median(z['floor']))
    thr = float(z['threshold'])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), dpi=300)
    ax_a, ax_b, ax_c = axes

    snr_grid, stats = z['snr_grid'], z['snr_stats']
    levels = np.unique(snr_grid)
    med = np.array([np.median(stats[snr_grid == s]) for s in levels])
    lo  = np.array([np.percentile(stats[snr_grid == s], 5) for s in levels])
    hi  = np.array([np.percentile(stats[snr_grid == s], 95) for s in levels])

    ax_a.fill_between(levels, lo, hi, color=COLOR_MAIN, alpha=0.25, lw=0)
    ax_a.plot(levels, med, 'o-', color=COLOR_MAIN, ms=2.5, lw=1.0,
              label='simulated cells')
    ax_a.axhline(floor_med, color=COLOR_FLOOR, ls='--', lw=0.8,
                 label='white-noise floor ({:.2f})'.format(floor_med))
    ax_a.axhline(thr, color=COLOR_GATE, ls='-', lw=0.8,
                 label='skip gate ({:.1f})'.format(thr))
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel('simulated SNR')
    ax_a.set_ylabel('gate statistic')
    ax_a.set_title('a', loc='left', fontweight='bold')
    ax_a.legend(frameon=False, fontsize=5, loc='upper left')

    allen, gt = z['allen'], z['groundtruth']
    pooled = np.concatenate([a for a in (allen, gt) if len(a)])
    if len(pooled):
        bins = np.logspace(np.log10(max(pooled.min() * 0.8, 1e-2)),
                           np.log10(pooled.max() * 1.2), 40)
        if len(gt):
            ax_b.hist(gt, bins=bins, color=COLOR_REAL, alpha=0.75,
                      label='ground truth (n={})'.format(len(gt)), edgecolor='none')
        if len(allen):
            ax_b.hist(allen, bins=bins, color=COLOR_MAIN, alpha=0.6,
                      label='Allen (n={})'.format(len(allen)), edgecolor='none')
        ax_b.axvline(floor_med, color=COLOR_FLOOR, ls='--', lw=0.8)
        ax_b.axvline(thr, color=COLOR_GATE, ls='-', lw=0.8)
        ax_b.set_xscale('log')

        ax_b.annotate('lowest trace\n{:.2f}'.format(float(pooled.min())),
                      xy=(float(pooled.min()), ax_b.get_ylim()[1] * 0.04),
                      xytext=(float(pooled.min()) * 1.6, ax_b.get_ylim()[1] * 0.42),
                      fontsize=5, ha='left', va='center',
                      arrowprops=dict(arrowstyle='->', lw=0.5,
                                      shrinkA=0, shrinkB=1))
    ax_b.set_xlabel('gate statistic')
    ax_b.set_ylabel('traces')
    ax_b.set_title('b', loc='left', fontweight='bold')
    ax_b.legend(frameon=False, fontsize=5, loc='upper right')

    names  = list(z['case_names'])
    values = z['case_values']
    order  = np.argsort(values)
    ypos   = np.arange(len(names))
    colors = [COLOR_GATE if values[i] < thr else COLOR_MAIN for i in order]

    plotted = np.maximum(values[order], 1e-2)
    ax_c.barh(ypos, plotted, color=colors, height=0.7)
    ax_c.axvline(thr, color=COLOR_GATE, ls='-', lw=0.8)
    ax_c.axvline(floor_med, color=COLOR_FLOOR, ls='--', lw=0.8)
    for y, val, shown in zip(ypos, values[order], plotted):
        ax_c.text(shown * 1.15, y, '{:.2f}'.format(val), fontsize=4.5,
                  va='center', ha='left')
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([names[i] for i in order], fontsize=5)
    ax_c.set_xscale('log')
    ax_c.set_xlim(right=plotted.max() * 4)
    ax_c.set_xlabel('gate statistic')
    ax_c.set_title('c', loc='left', fontweight='bold')

    fig.tight_layout()
    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, 'snr_test.{}'.format(ext))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved figure to {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    res_path = os.path.join(data_dir, 'snr_test_results.npz')
    if not os.path.exists(res_path):
        print('No results at {} -- run with --mode test first.'.format(res_path))
        return

    z = np.load(res_path, allow_pickle=True)
    floor = z['floor']
    thr   = float(z['threshold'])

    print('\nWhite-noise floor of the gate statistic')
    print('  median {:.3f}   5-95 pct {:.3f} to {:.3f}'.format(
        float(np.median(floor)), *np.percentile(floor, [5, 95])))
    print('  Gate threshold is {:.1f}, i.e. {:.2f} below the noise floor.'.format(
        thr, float(np.median(floor)) - thr))

    print('\nSimulated SNR sweep')
    snr_grid, stats = z['snr_grid'], z['snr_stats']
    print('  {:>10s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}'.format(
        'SNR', 'min', 'median', 'max', 'gated'))
    for s in np.unique(snr_grid):
        v = stats[snr_grid == s]
        print('  {:10.2f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8d}'.format(
            s, v.min(), np.median(v), v.max(), int((v < thr).sum())))

    print('\nReal traces')
    for name, key in [('Allen', 'allen'), ('Ground truth', 'groundtruth')]:
        v = z[key]
        if not len(v):
            print('  {:14s} no traces found on disk.'.format(name))
            continue
        print('  {:14s} n={:4d}  min {:.2f}  median {:.2f}  max {:.2f}  '
              'excluded at {:.1f}: {}'.format(
                  name, len(v), v.min(), np.median(v), v.max(), thr,
                  int((v < thr).sum())))

    pooled = np.concatenate([z['allen'], z['groundtruth']])
    gt, gt_n = z['groundtruth'], z['groundtruth_n_true']
    have_spikes = len(gt) and len(gt_n) == len(gt) and gt_n.sum() > 0

    if len(pooled):
        print('\nExclusion vs threshold, pooled real traces (n={})'.format(len(pooled)))
        print('  {:>10s}  {:>9s}  {:>8s}  {:>14s}'.format(
            'threshold', 'excluded', 'percent', 'true spikes'))
        for t in (2.0, 2.64, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            n_ex = int((pooled < t).sum())
            if have_spikes:
                spk = '{:13.1f}%'.format(
                    100.0 * gt_n[gt < t].sum() / gt_n.sum())
            else:
                spk = '{:>14s}'.format('n/a')
            print('  {:10.2f}  {:9d}  {:7.1f}%  {}'.format(
                t, n_ex, 100.0 * n_ex / len(pooled), spk))
        if have_spikes:
            print('  True-spike column: share of ground-truth spikes living in '
                  'excluded cells.')

    print('\nWhat reaches the gate')
    names, values = list(z['case_names']), z['case_values']
    for i in np.argsort(values):
        flag = 'GATED' if values[i] < thr else ''
        print('  {:24s} {:12.2f}  {}'.format(names[i], values[i], flag))
    print('')


def main():

    parser = argparse.ArgumentParser(
        description='Supplementary panel: low-SNR skip gate is unreachable by noise.'
    )
    parser.add_argument('--mode', default='plot',
                        choices=['test', 'plot', 'print', 'all'],
                        help='Run analyses, draw figure, print stats, or all.')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for results and figure output.')
    args = parser.parse_args()

    if args.mode in ('test', 'all'):
        run_test(args.data_dir)
    if args.mode in ('plot', 'all'):
        plot_figure(args.data_dir)
    if args.mode in ('print', 'all'):
        print_stats(args.data_dir)


if __name__ == '__main__':
    main()
