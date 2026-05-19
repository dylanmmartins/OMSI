#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.signal import find_peaks

import fMCSI
from fMCSI.helpers import compute_cosmic
from simulation_helpers import generate_synthetic_data

mpl.rcParams['axes.spines.top']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

np.random.seed(42)

FS       = 30.0
DURATION = 600
TAU      = 1.2
N_CELLS  = 100
BETA     = 0.5

COLORS = {
    'fMCSI':   '#4C72B0',
    'MATLAB':  '#DD8452',
    'OASIS':   '#55A868',
    'CASCADE': '#8172B3',
}

OASIS_THRESHOLDS   = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
CASCADE_THRESHOLDS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'figS1')

#   'threshold' : return every frame where s > thresh * sigma (default, matches figure1–4)
#   'peaks'     : find local maxima above thresh * sigma with minimum inter-peak distance
OASIS_SPIKE_DETECTION = 'peaks'

_NPZ_NAME = 'figS1_threshold_sweep.npz'


def _run_cascade_inference(dff, fs, n_cells, data_dir, prefix='figS1_cascade'):
    script      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')

    if os.path.exists(output_path):
        print(f'  Reusing existing CASCADE output: {output_path}')
    else:
        np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
        print(f'  Calling CASCADE subprocess (n_cells={n_cells}, fs={fs})...')
        subprocess.run(
            ['conda', 'run', '-n', 'cascade', 'python', script,
             '--mode', 'inference',
             '--input',  input_path,
             '--output', output_path],
            check=True,
        )
    result = np.load(output_path, allow_pickle=True)
    cascade_spikes = list(result['cascade_spikes'])
    return result['cascade_probs'], cascade_spikes, float(result['cascade_time'])


def _oasis_spikes(dff, fs, tau, n_cells):
    from oasis.functions import deconvolve
    diff   = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    s_list = []
    for i in range(n_cells):
        g = np.exp(-1 / (fs * tau))
        _, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        s_list.append(s)
    return s_list, sigmas


def _probs_to_spikes(probs, fs, height=0.2):

    min_dist = max(1, int(0.05 * fs))
    peaks, _ = find_peaks(probs, height=height, distance=min_dist)
    return peaks / fs


def _fbeta(precision, recall):
    p, r = np.asarray(precision, float), np.asarray(recall, float)
    b2   = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _compute_metrics(true_spikes, spikes_list):
    prec, rec, _ = fMCSI.helpers.compute_accuracy_window(true_spikes, spikes_list)
    fb           = _fbeta(prec, rec)
    cosmic       = compute_cosmic(true_spikes, spikes_list, FS)
    return prec, rec, fb, cosmic


def run_test(data_dir='.', run_oasis=True, run_cascade=True):

    os.makedirs(data_dir, exist_ok=True)

    print('Generating synthetic calcium traces...')
    noisy, true_spikes, clean, timestamps, firing_rates, kurtosis = generate_synthetic_data(
        n_cells=N_CELLS, fs=FS, duration=DURATION, tau=TAU,
        target_kurtosis_range=(0.0, 25.0),
    )

    save = {
        'true_spikes':        np.array(true_spikes, dtype=object),
        'oasis_thresholds':   OASIS_THRESHOLDS,
        'cascade_thresholds': CASCADE_THRESHOLDS,
        'oasis_spike_detection': OASIS_SPIKE_DETECTION,
    }

    if run_oasis:
        print('\nRunning OASIS deconvolution...')
        t0 = time.time()
        s_list, sigmas = _oasis_spikes(noisy, FS, TAU, N_CELLS)
        print(f'  deconvolution took {time.time()-t0:.1f}s')

        n_thresh = len(OASIS_THRESHOLDS)
        oasis_precision = np.zeros((n_thresh, N_CELLS))
        oasis_recall    = np.zeros((n_thresh, N_CELLS))
        oasis_fbeta     = np.zeros((n_thresh, N_CELLS))
        oasis_cosmic    = np.zeros((n_thresh, N_CELLS))

        print('  sweeping thresholds:')
        for ti, thresh in enumerate(OASIS_THRESHOLDS):
            if OASIS_SPIKE_DETECTION == 'threshold':
                spikes = [np.where(s > thresh * sigmas[i])[0] / FS
                          for i, s in enumerate(s_list)]
            else:
                min_dist = max(1, int(0.05 * FS))
                spikes = [np.array(find_peaks(s, height=thresh * sigmas[i], distance=min_dist)[0]) / FS
                          for i, s in enumerate(s_list)]
            prec, rec, fb, cosmic = _compute_metrics(true_spikes, spikes)
            oasis_precision[ti] = prec
            oasis_recall[ti]    = rec
            oasis_fbeta[ti]     = fb
            oasis_cosmic[ti]    = cosmic
            print(f'    thresh={thresh:.3f}  |  '
                  f'P={np.nanmean(prec):.3f}  R={np.nanmean(rec):.3f}  '
                  f'Fb={np.nanmean(fb):.3f}  CosMIC={np.nanmean(cosmic):.3f}')

        save.update({
            'oasis_precision': oasis_precision,
            'oasis_recall':    oasis_recall,
            'oasis_fbeta':     oasis_fbeta,
            'oasis_cosmic':    oasis_cosmic,
        })

    if run_cascade:
        print('\nRunning CASCADE...')
        cascade_probs, _, cascade_elapsed = _run_cascade_inference(
            noisy, FS, N_CELLS, data_dir
        )
        print(f'  CASCADE inference took {cascade_elapsed:.1f}s')

        n_thresh = len(CASCADE_THRESHOLDS)
        cascade_precision = np.zeros((n_thresh, N_CELLS))
        cascade_recall    = np.zeros((n_thresh, N_CELLS))
        cascade_fbeta     = np.zeros((n_thresh, N_CELLS))
        cascade_cosmic    = np.zeros((n_thresh, N_CELLS))

        print('  sweeping thresholds:')
        for ti, thresh in enumerate(CASCADE_THRESHOLDS):
            spikes = [_probs_to_spikes(cascade_probs[i], FS, height=thresh)
                      for i in range(N_CELLS)]
            prec, rec, fb, cosmic = _compute_metrics(true_spikes, spikes)
            cascade_precision[ti] = prec
            cascade_recall[ti]    = rec
            cascade_fbeta[ti]     = fb
            cascade_cosmic[ti]    = cosmic
            print(f'    thresh={thresh:.3f}  |  '
                  f'P={np.nanmean(prec):.3f}  R={np.nanmean(rec):.3f}  '
                  f'Fb={np.nanmean(fb):.3f}  CosMIC={np.nanmean(cosmic):.3f}')

        save.update({
            'cascade_precision': cascade_precision,
            'cascade_recall':    cascade_recall,
            'cascade_fbeta':     cascade_fbeta,
            'cascade_cosmic':    cascade_cosmic,
        })

    out = os.path.join(data_dir, _NPZ_NAME)
    np.savez(out, **save)
    print(f'\nSaved -> {out}')
    print('Test mode complete.')


def plot_figure(data_dir='.'):

    path = os.path.join(data_dir, _NPZ_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Results not found at {path}. Run --mode test first.'
        )
    res = np.load(path, allow_pickle=True)

    has_oasis   = 'oasis_precision'   in res
    has_cascade = 'cascade_precision' in res

    oasis_thresholds   = res['oasis_thresholds']   if 'oasis_thresholds'   in res else res['thresholds']
    cascade_thresholds = res['cascade_thresholds'] if 'cascade_thresholds' in res else res['thresholds']

    metrics = [
        ('precision', 'Precision'),
        ('recall',    'Recall'),
        ('fbeta',     r'$F_\beta$ score'),
        ('cosmic',    'CosMIC score'),
    ]

    rows = []
    if has_oasis:
        rows.append(('OASIS',   'oasis',   oasis_thresholds,   'Detection threshold'))
    if has_cascade:
        rows.append(('CASCADE', 'cascade', cascade_thresholds, 'Detection threshold'))

    n_rows = len(rows)
    if n_rows == 0:
        raise RuntimeError('No results found in NPZ file.')

    fig = plt.figure(figsize=(7.0, 1.85 * n_rows + 0.4), dpi=300)
    gs  = gridspec.GridSpec(n_rows, 4, figure=fig,
                            hspace=0.65, wspace=0.38,
                            left=0.10, right=0.98,
                            top=0.88,  bottom=0.18)

    for row_idx, (method, prefix, thresholds, xlabel) in enumerate(rows):
        color = COLORS[method]

        for col, (metric_key, ylabel) in enumerate(metrics):
            ax   = fig.add_subplot(gs[row_idx, col])
            vals = res[f'{prefix}_{metric_key}']
            mean = np.nanmean(vals, axis=1)
            std  = np.nanstd(vals,  axis=1)

            ax.fill_between(thresholds,
                            np.clip(mean - std, 0, 1),
                            np.clip(mean + std, 0, 1),
                            color=color, alpha=0.18, zorder=1)
            ax.plot(thresholds, mean, '.-',
                    color=color, lw=1.2, ms=4, zorder=3)
            if method == 'OASIS' and OASIS_SPIKE_DETECTION == 'peaks':
                ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.55, zorder=2)
            else:
                ax.axvline(0.5, color='k', lw=0.8, ls='--', alpha=0.55, zorder=2)

            ax.set_ylim([0, 1.08])
            ax.set_xlim([thresholds[0], thresholds[-1]])
            ax.tick_params(axis='both', labelsize=6)
            ax.set_xlabel(xlabel, fontsize=6)

            if row_idx == 0:
                ax.set_title(ylabel, fontsize=7, pad=3)

            if col > 0:
                ax.yaxis.set_ticklabels([])
            else:
                ax.set_ylabel(ylabel, fontsize=6)
                ax.text(-0.38, 0.5, method,
                        transform=ax.transAxes,
                        rotation=90, va='center', ha='center',
                        color='k', fontsize=7, fontweight='bold')

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, f'figureS1.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    path = os.path.join(data_dir, _NPZ_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Results not found at {path}. Run --mode test first.')
    res = np.load(path, allow_pickle=True)

    oasis_thresholds   = res['oasis_thresholds']   if 'oasis_thresholds'   in res else res['thresholds']
    cascade_thresholds = res['cascade_thresholds'] if 'cascade_thresholds' in res else res['thresholds']

    rows = []
    if 'oasis_fbeta' in res:
        rows.append(('OASIS',   'oasis',   oasis_thresholds))
    if 'cascade_fbeta' in res:
        rows.append(('CASCADE', 'cascade', cascade_thresholds))

    print('\n' + '='*72)
    print('FIGURE S1 STATISTICS — maximum performance vs detection threshold')
    print('='*72)

    for method, prefix, thresholds in rows:
        print(f'\n--- {method} ---')
        for metric_key, metric_label in [('fbeta', 'F_beta'), ('cosmic', 'CosMIC')]:
            arr  = res[f'{prefix}_{metric_key}']           # shape (n_thresh, n_cells)
            mean = np.nanmean(arr, axis=1)                 # shape (n_thresh,)
            std  = np.nanstd(arr,  axis=1)
            best_idx = int(np.argmax(mean))
            print(f'  Max mean {metric_label:>8}: {mean[best_idx]:.3f}  '
                  f'(std={std[best_idx]:.3f})  '
                  f'at threshold={thresholds[best_idx]:.3f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Figure S1: threshold sensitivity for OASIS and CASCADE'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'print'],
                        help='"test" runs inference and writes results; '
                             '"plot" loads results and generates the figure')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files (default: .)')
    parser.add_argument('--no-oasis',   action='store_true', help='Skip OASIS')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(
            data_dir    = args.data_dir,
            run_oasis   = not args.no_oasis,
            run_cascade = not args.no_cascade,
        )
    elif args.mode == 'plot':
        plot_figure(data_dir=args.data_dir)
    else:
        print_stats(data_dir=args.data_dir)
