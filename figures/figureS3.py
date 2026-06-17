#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Patch

import OMSI
import OMSI.helpers as helpers

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'figS3')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

BETA  = 0.5
COLOR_7P5 = 'tab:red'
COLOR_30  = 'tab:cyan'

_CASCADE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'run_cascade_subprocess.py')

_CASCADE_MODELS = {
    7.5:  'Global_EXC_7.5Hz_smoothing200ms',
    30.0: 'Global_EXC_30Hz_smoothing50ms_causalkernel',
}

_NPZ_NAME = 'cascade_7p5_vs_30hz_data.npz'


def _run_cascade_inference(dff, fs, data_dir, prefix, device='gpu'):

    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')
    model_name  = _CASCADE_MODELS.get(fs, 'Global_EXC_30Hz_smoothing50ms_causalkernel')

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python', _CASCADE_SCRIPT,
         '--mode', 'inference',
         '--input',  input_path,
         '--output', output_path,
         '--model',  model_name,
         '--device', device],
        check=True,
    )
    result = np.load(output_path, allow_pickle=True)
    return list(result['cascade_spikes']), float(result['cascade_time'])


def _fbeta(prec, rec):
    p  = np.asarray(prec, dtype=float)
    r  = np.asarray(rec,  dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def run_test(data_dir=_DEFAULT_DATA_DIR, run_cascade=True):

    from simulation_helpers import generate_synthetic_data

    os.makedirs(data_dir, exist_ok=True)

    n_cells  = 50
    duration = 300
    tau      = 1.2
    out_path = os.path.join(data_dir, _NPZ_NAME)

    results = {}
    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        print(f'\n=== {fs} Hz ===')
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=n_cells, fs=fs, duration=duration, tau=tau)

        if not run_cascade:
            results[f'fb_{suffix}']     = np.full(n_cells, np.nan)
            results[f'cosmic_{suffix}'] = np.full(n_cells, np.nan)
            continue

        cascade_spikes = None
        for dev in ('gpu', 'cpu'):
            try:
                cascade_spikes, elapsed = _run_cascade_inference(
                    dff, fs, data_dir, f'cascade_samplerate_{fs}hz', device=dev)
                print(f'  CASCADE ({dev.upper()}) finished in {elapsed:.1f}s')
                break
            except Exception as exc:
                print(f'  CASCADE {dev.upper()} failed: {exc}')

        if cascade_spikes is None:
            results[f'fb_{suffix}']     = np.full(n_cells, np.nan)
            results[f'cosmic_{suffix}'] = np.full(n_cells, np.nan)
            continue

        prec, rec, _ = OMSI.compute_accuracy_strict(true_spikes, cascade_spikes,
                                                      tolerance=0.1)
        fb     = _fbeta(prec, rec)
        cosmic = helpers.compute_cosmic(true_spikes, cascade_spikes, fs)
        results[f'fb_{suffix}']     = fb
        results[f'cosmic_{suffix}'] = cosmic
        print(f'  mean F_beta={np.nanmean(fb):.3f}  CosMIC={np.nanmean(cosmic):.3f}')

    np.savez(out_path, **results)
    print(f'\nSaved -> {out_path}')
    print('Test mode complete.')


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    os.makedirs(data_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, _NPZ_NAME)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'Results not found at {npz_path}. Run --mode test first.')

    data     = np.load(npz_path)
    fb_7     = data['fb_7'];     fb_30     = data['fb_30']
    cosmic_7 = data['cosmic_7']; cosmic_30 = data['cosmic_30']

    fig = plt.figure(figsize=(3, 2.5), dpi=200)
    gs  = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.95,
                            top=0.88, bottom=0.18)
    ax = fig.add_subplot(gs[0])

    pos      = [1, 2, 3.3, 4.3]
    datasets = [
        (fb_7[np.isfinite(fb_7)],          COLOR_7P5),
        (fb_30[np.isfinite(fb_30)],         COLOR_30),
        (cosmic_7[np.isfinite(cosmic_7)],   COLOR_7P5),
        (cosmic_30[np.isfinite(cosmic_30)], COLOR_30),
    ]

    parts = ax.violinplot([d for d, _ in datasets], positions=pos,
                          showmedians=True, widths=0.65)
    for pc, (_, col) in zip(parts['bodies'], datasets):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[partname].set_color('k')
        parts[partname].set_linewidth(0.8)

    ax.set_xticks(pos)
    ax.set_xticklabels([r'$F_\beta$', r'$F_\beta$', 'CosMIC', 'CosMIC'], fontsize=6)
    ax.legend(handles=[
        Patch(facecolor=COLOR_7P5, alpha=0.75, label='7.5 Hz'),
        Patch(facecolor=COLOR_30,  alpha=0.75, label='30 Hz'),
    ], loc='upper right', handlelength=1.0, handleheight=0.8,
       borderpad=0.4, labelspacing=0.2, frameon=False, fontsize=6)
    ax.set_ylabel('Score', fontsize=7)
    ax.set_ylim(0, 1.15)

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, f'figureS3.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def main():
    
    parser = argparse.ArgumentParser(
        description='Figure S3 — CASCADE performance at 7.5 Hz vs 30 Hz'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot'],
                        help='"test" runs inference and writes NPZ; '
                             '"plot" loads NPZ and generates the figure')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(data_dir=args.data_dir, run_cascade=not args.no_cascade)
    else:
        plot_figure(data_dir=args.data_dir)


if __name__ == '__main__':
    main()
