# -*- coding: utf-8 -*-
"""
Fixed Benchmark on Simulated Data

To run inference:
    $ python figure1.py --mode test --data-dir /path/to/results

To create figure:
    $ python figure1.py --mode plot --data-dir /path/to/results

Written DMM, March 2026
"""

import argparse
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl
from scipy.signal import find_peaks
from oasis.functions import deconvolve as oasis_deconv

import OMSI
from run_pnev_MCMC import run_matlab_pnevMCMC
from simulation_helpers import generate_synthetic_data

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig1')

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7

np.random.seed(3)

FS       = 30.0
DURATION = 60 * 20
TAU      = 1.2
N_CELLS  = 500
BETA     = 0.5
USE_STRICT_ACCURACY = True  # Hungarian one-to-one matching (compute_accuracy_strict)

COLORS = {
    'fMCSI':      '#4C72B0',
    'MATLAB':      '#DD8452',
    'OASIS':       '#55A868',
    'CASCADE_GPU': '#8172B3',
    'CASCADE_CPU': '#B39DDB',
}

_NPZ_NAMES = {
    'fMCSI':      'fixed_benchmark_fMCSI.npz',
    'MATLAB':      'fixed_benchmark_MATLAB.npz',
    'OASIS':       'fixed_benchmark_OASIS.npz',
    'CASCADE_GPU': 'fixed_benchmark_CASCADE_GPU.npz',
    'CASCADE_CPU': 'fixed_benchmark_CASCADE_CPU.npz',
}

#   'threshold' : return every frame where s > height * sigma (default)
#   'peaks'     : find local maxima above height * sigma with minimum inter-peak distance
OASIS_SPIKE_DETECTION = 'peaks'


def _oasis_spikes_from_s(s, sigma, fs, height=1.0):
    thresh = height * sigma
    if OASIS_SPIKE_DETECTION == 'peaks':
        min_dist = max(1, int(0.05 * fs))
        peaks, _ = find_peaks(s, height=thresh, distance=min_dist)
        return peaks / fs
    return np.where(s > thresh)[0] / fs


def _run_cascade_inference(dff, fs, n_cells, data_dir, prefix='fig1_cascade', device='gpu'):

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))

    print(f"Calling CASCADE subprocess (n_cells={n_cells}, fs={fs}, device={device})...")
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python', script,
         '--mode', 'inference',
         '--input', input_path,
         '--output', output_path,
         '--device', device],
        check=True
    )

    result = np.load(output_path, allow_pickle=True)
    cascade_probs  = result['cascade_probs']
    cascade_spikes = list(result['cascade_spikes'])
    cascade_time   = float(result['cascade_time'])
    return cascade_probs, cascade_spikes, cascade_time


def _fbeta(precision, recall):

    p  = np.asarray(precision, dtype=float)
    r  = np.asarray(recall,    dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def run_test(data_dir=_DEFAULT_DATA_DIR, run_fmcsi=True, run_matlab=True,
             run_oasis=True, run_cascade=True):

    os.makedirs(data_dir, exist_ok=True)

    print('Generating synthetic spikes and calcium traces...')
    noisy, true_spikes, clean, timestamps, firing_rates, kurtosis = generate_synthetic_data(
        n_cells=N_CELLS, fs=FS, duration=DURATION, tau=TAU,
        target_kurtosis_range=(0.0, 25.0)
    )
    timestamps = np.arange(noisy.shape[1]) / FS

    params = {
        'f':        FS,
        'p':        2,
        'Nsamples': 200,
        'B':        75,
        'marg':     0,
        'upd_gam':  1,
    }

    shared = {
        'true_spikes':   np.array(true_spikes, dtype=object),
        'clean_traces':  clean,
        'noisy_traces':  noisy,
        'f':             FS,
        'duration':      DURATION,
        'tau':           TAU,
        'n_cells':       N_CELLS,
        'time':          timestamps,
        'firing_rates':  firing_rates,
        'kurtosis':      kurtosis,
    }

    if run_fmcsi:

        print('\nRunning fMCSI...')
        t0 = time.time()
        optim_dict = OMSI.deconv(noisy, params, true_spikes=true_spikes, benchmark=True)
        optim_time = time.time() - t0
        print(f'  fMCSI took {optim_time:.1f}s ({optim_time/N_CELLS:.3f}s/cell)')
        print(f'  P={np.nanmean(optim_dict["optim_precision"]):.3f}  '
              f'R={np.nanmean(optim_dict["optim_recall"]):.3f}')
        save = {**shared, **optim_dict, 'optim_time': optim_time}
        np.savez(os.path.join(data_dir, _NPZ_NAMES['fMCSI']), **save)

    if run_matlab:

        print('\nRunning MATLAB...')
        t0 = time.time()
        trad_spikes, trad_traces, trad_probs, _ = run_matlab_pnevMCMC(
            noisy, fs=FS, tau=TAU, n_sweeps=500, true_spikes=true_spikes
        )
        matlab_time = time.time() - t0
        trad_prec, trad_rec, trad_F1 = OMSI.compute_accuracy_strict(true_spikes, trad_spikes)
        print(f'  MATLAB took {matlab_time:.1f}s  P={np.nanmean(trad_prec):.3f}  '
              f'R={np.nanmean(trad_rec):.3f}')
        save = {
            **shared,
            'tradmat_spikes':    np.array(trad_spikes, dtype=object),
            'tradmat_traces':    trad_traces,
            'tradmat_probs':     trad_probs,
            'tradmat_time':      matlab_time,
            'tradmat_precision': trad_prec,
            'tradmat_recall':    trad_rec,
            'tradmat_F1':        trad_F1,
        }
        np.savez(os.path.join(data_dir, _NPZ_NAMES['MATLAB']), **save)

    if run_oasis:

        print('\nRunning OASIS...')
        t0 = time.time()
        oasis_spikes = []
        diff_oasis = np.diff(noisy, axis=1)
        sigmas = np.median(np.abs(diff_oasis), axis=1) / (0.6745 * np.sqrt(2))
        sigmas = np.maximum(sigmas, 1e-9)
        for i in range(N_CELLS):
            g = np.exp(-1 / (FS * TAU))
            _, s, _, _, _ = oasis_deconv(noisy[i], g=(g,), sn=sigmas[i], penalty=1)
            oasis_spikes.append(_oasis_spikes_from_s(s, sigmas[i], FS))
        oasis_time = time.time() - t0
        oasis_prec, oasis_rec, oasis_F1 = OMSI.compute_accuracy_strict(true_spikes, oasis_spikes)
        print(f'  OASIS took {oasis_time:.1f}s  P={np.nanmean(oasis_prec):.3f}  '
              f'R={np.nanmean(oasis_rec):.3f}')
        save = {
            **shared,
            'oasis_spikes':    np.array(oasis_spikes, dtype=object),
            'oasis_time':      oasis_time,
            'oasis_precision': oasis_prec,
            'oasis_recall':    oasis_rec,
            'oasis_F1':        oasis_F1,
        }
        np.savez(os.path.join(data_dir, _NPZ_NAMES['OASIS']), **save)

    if run_cascade:

        for _dev, _key in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print(f'\nRunning CASCADE (subprocess, {_dev.upper()})...')
            _probs, _spikes, _time = _run_cascade_inference(
                noisy, FS, N_CELLS, data_dir,
                prefix=f'fig1_cascade_{_dev}', device=_dev
            )
            _prec, _rec, _F1 = OMSI.compute_accuracy_strict(true_spikes, _spikes)
            print(f'  CASCADE ({_dev.upper()}) took {_time:.1f}s  '
                  f'P={np.nanmean(_prec):.3f}  R={np.nanmean(_rec):.3f}')
            np.savez(os.path.join(data_dir, _NPZ_NAMES[_key]), **{
                **shared,
                'cascade_spikes':    np.array(_spikes, dtype=object),
                'cascade_probs':     _probs,
                'cascade_time':      _time,
                'cascade_precision': _prec,
                'cascade_recall':    _rec,
                'cascade_F1':        _F1,
            })

    print('\nTest mode complete.')


def _best_window_sim(raw_trace, fs, true_spk, det_list, window=60.0, target_spikes=20):
    block_frames = int(window * fs)
    n_frames = len(raw_trace)
    best_t0, best_score = 0.0, -np.inf
    t = 0
    while t + block_frames <= n_frames:
        t0 = t / fs
        t1 = t0 + window
        true_win = true_spk[(true_spk >= t0) & (true_spk < t1)]
        n_true   = len(true_win)
        spike_score = float(np.exp(-0.5 * ((n_true - target_spikes) / 8.0) ** 2))
        recall_list = []
        for det_spk in det_list:
            if len(det_spk) == 0 or len(true_win) == 0:
                continue
            det_win = det_spk[(det_spk >= t0 - 0.1) & (det_spk < t1 + 0.1)]
            hits = sum(1 for ts in true_win if np.any(np.abs(det_win - ts) <= 0.1))
            recall_list.append(hits / len(true_win))
        pred_score = float(np.mean(recall_list)) if recall_list else 0.0
        score = (spike_score + pred_score) / 2.0
        if score > best_score:
            best_score = score
            best_t0    = t0
        t += block_frames
    return best_t0


def _select_example_cells(mine_res, oasis_res, cascade_res, matlab_res,
                           n_cells=4, window=60.0, min_spikes=10,
                           target_kurts=(0.2, 0.5, 1.0, 2.0)):
    fs              = float(mine_res['f'])
    true_spikes_arr = list(mine_res['true_spikes'])
    noisy_traces    = mine_res['noisy_traces']
    kurtosis_arr    = mine_res['kurtosis']
    optim_spikes    = list(mine_res['optim_spikes'])
    oasis_spikes    = list(oasis_res['oasis_spikes'])
    cascade_spikes  = list(cascade_res['cascade_spikes'])
    tradmat_spikes  = list(matlab_res['tradmat_spikes'])

    cells = []
    for i in range(len(true_spikes_arr)):
        true_spk = np.atleast_1d(np.asarray(true_spikes_arr[i], dtype=float))
        if len(true_spk) < 3:
            continue
        my_spk  = np.atleast_1d(np.asarray(optim_spikes[i],   dtype=float))
        oas_spk = np.atleast_1d(np.asarray(oasis_spikes[i],   dtype=float))
        cas_spk = np.atleast_1d(np.asarray(cascade_spikes[i], dtype=float))
        mat_spk = np.atleast_1d(np.asarray(tradmat_spikes[i], dtype=float))
        raw     = noisy_traces[i]
        t_start = _best_window_sim(
            raw, fs, true_spk, [my_spk, oas_spk, cas_spk, mat_spk], window=window
        )
        n_win = int(np.sum((true_spk >= t_start) & (true_spk < t_start + window)))
        if n_win < min_spikes:
            continue
        _f = raw[np.isfinite(raw)]
        _mad = float(np.median(np.abs(np.diff(_f)))) / 0.6745 if len(_f) > 1 else 1e-4
        _snr = (float(np.percentile(_f, 99)) - float(np.percentile(_f, 8))) / (_mad + 1e-9)
        cells.append({
            'cell_idx':      i,
            'true_spikes':   true_spk,
            'my_spikes':     my_spk,
            'oasis_spikes':  oas_spk,
            'cascade_spikes': cas_spk,
            'trad_spikes':   mat_spk,
            'raw':           raw,
            'kurtosis':      float(kurtosis_arr[i]),
            'snr':           _snr,
            'fs':            fs,
            't_start':       t_start,
        })

    cells.sort(key=lambda c: c['kurtosis'])
    available_kurts = np.array([c['kurtosis'] for c in cells])
    clipped = np.clip(target_kurts, available_kurts[0], available_kurts[-1])
    if len(np.unique(clipped)) < n_cells:
        effective_targets = np.percentile(available_kurts, np.linspace(10, 90, n_cells))
    else:
        effective_targets = clipped

    selected, used_idx = [], set()
    for tk in effective_targets:
        best_i, best_d = None, np.inf
        for j, c in enumerate(cells):
            if j in used_idx:
                continue
            d = abs(c['kurtosis'] - tk)
            if d < best_d:
                best_d, best_i = d, j
        if best_i is not None:
            used_idx.add(best_i)
            selected.append(cells[best_i])

    selected.sort(key=lambda c: c['kurtosis'])
    return selected


def _plot_raster(ax, cells, window=60.0):
    n = len(cells)
    if n == 0:
        ax.text(0.5, 0.5, 'No trace data', transform=ax.transAxes,
                ha='center', va='center')
        return

    rr     = 0.9
    th     = 2.2
    pad    = 0.25
    gap    = 0.7
    n_rows = 5
    cell_h = n_rows * rr + pad + th + gap

    method_rows = [
        ('OASIS',        'oasis_spikes',    COLORS['OASIS'],       0),
        ('CASCADE',       'cascade_spikes', COLORS['CASCADE_GPU'], 1),
        ('CaImAn',       'trad_spikes',     COLORS['MATLAB'],      2),
        ('OMSI',        'my_spikes',       COLORS['fMCSI'],       3),
        ('Ground Truth', 'true_spikes',     '#111111',             4),
    ]
    label_x = -4.0

    for i, cell in enumerate(cells):
        base = (n - 1 - i) * cell_h
        t0   = cell['t_start']
        t1   = t0 + window

        for row_name, key, color, row_idx in method_rows:
            y_lo  = base + row_idx * rr + 0.05
            y_hi  = base + row_idx * rr + rr * 0.85
            y_mid = base + row_idx * rr + rr * 0.45
            spk = np.atleast_1d(np.asarray(cell.get(key, []), dtype=float))
            in_win = spk[(spk >= t0) & (spk <= t1)] - t0
            if len(in_win) > 0:
                ax.vlines(in_win, y_lo, y_hi, color=color, lw=0.6, alpha=0.9)
            if i == 0:
                ax.text(label_x, y_mid, row_name, va='center', ha='right',
                        color='k' if color == '#111111' else color, size=6)

        trace_y0 = base + n_rows * rr + pad
        raw      = cell['raw']
        fs       = cell['fs']
        t_arr    = np.arange(len(raw)) / fs
        mask     = (t_arr >= t0) & (t_arr <= t1)
        t_plot   = t_arr[mask] - t0
        raw_plot = raw[mask]
        rmin, rmax = np.nanmin(raw_plot), np.nanmax(raw_plot)
        if rmax > rmin:
            raw_norm = (raw_plot - rmin) / (rmax - rmin) * th + trace_y0
        else:
            raw_norm = np.full_like(raw_plot, trace_y0 + th / 2)
        ax.plot(t_plot, raw_norm, color='k', lw=0.7, alpha=0.8)

        if i == 0:
            ax.text(label_x, trace_y0 + th / 2, 'ΔF/F',
                    va='center', ha='right', color='k', size=6)

        ax.text(label_x / 2, trace_y0 + th / 2, str(i + 1),
                va='center', ha='center', color='k', fontsize=5.5,
                fontweight='bold')

        ax.text(window + 0.8, base + cell_h / 2 - gap / 2,
                f'SNR={cell["snr"]:.1f}', va='center', ha='left', fontsize=6)
        if i < n - 1:
            ax.axhline(base - gap / 2, color='0.75', lw=0.4, ls='--')

    ax.set_xlim(label_x - 0.5, window + 6)
    ax.set_ylim(-gap, n * cell_h)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time (s)')


def _with_window_metrics(res, prefix, true_spikes_list):

    pk = f'{prefix}_precision_window'
    d  = dict(res)
    if pk in res.files:
        return d
    spk_key = f'{prefix}_spikes'
    pred = list(res[spk_key])
    prec_w, rec_w, f1_w = OMSI.helpers.compute_accuracy_window(true_spikes_list, pred)
    d[f'{prefix}_precision_window'] = prec_w
    d[f'{prefix}_recall_window']    = rec_w
    d[f'{prefix}_F1_window']        = f1_w
    return d


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    paths = {k: os.path.join(data_dir, v) for k, v in _NPZ_NAMES.items()}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{name} results not found at {path}. Run --mode test first.'
            )

    MINE_RESULTS        = np.load(paths['fMCSI'],       allow_pickle=True)
    MATLAB_RESULTS      = np.load(paths['MATLAB'],      allow_pickle=True)
    OASIS_RESULTS       = np.load(paths['OASIS'],       allow_pickle=True)
    CASCADE_GPU_RESULTS = np.load(paths['CASCADE_GPU'], allow_pickle=True)
    CASCADE_CPU_RESULTS = np.load(paths['CASCADE_CPU'], allow_pickle=True)

    n_cells = int(MINE_RESULTS['n_cells'])

    true_spikes_list = list(MINE_RESULTS['true_spikes'])
    MINE_RESULTS        = _with_window_metrics(MINE_RESULTS,        'optim',   true_spikes_list)
    MATLAB_RESULTS      = _with_window_metrics(MATLAB_RESULTS,      'tradmat', true_spikes_list)
    OASIS_RESULTS       = _with_window_metrics(OASIS_RESULTS,       'oasis',   true_spikes_list)
    CASCADE_GPU_RESULTS = _with_window_metrics(CASCADE_GPU_RESULTS, 'cascade', true_spikes_list)
    CASCADE_CPU_RESULTS = _with_window_metrics(CASCADE_CPU_RESULTS, 'cascade', true_spikes_list)

    if USE_STRICT_ACCURACY:
        METHOD_INFO = [
            ('fMCSI',       MINE_RESULTS,        'optim_F1',    'optim_recall',    'optim_precision',    None, float(MINE_RESULTS['optim_time'])),
            ('MATLAB',      MATLAB_RESULTS,      'tradmat_F1',  'tradmat_recall',  'tradmat_precision',  None, float(MATLAB_RESULTS['tradmat_time'])),
            ('OASIS',       OASIS_RESULTS,       'oasis_F1',    'oasis_recall',    'oasis_precision',    None, float(OASIS_RESULTS['oasis_time'])),
            ('CASCADE_GPU', CASCADE_GPU_RESULTS, 'cascade_F1',  'cascade_recall',  'cascade_precision',  None, float(CASCADE_GPU_RESULTS['cascade_time'])),
        ]
    else:
        METHOD_INFO = [
            ('fMCSI',       MINE_RESULTS,        'optim_F1_window',    'optim_recall_window',    'optim_precision_window',    None, float(MINE_RESULTS['optim_time'])),
            ('MATLAB',      MATLAB_RESULTS,      'tradmat_F1_window',  'tradmat_recall_window',  'tradmat_precision_window',  None, float(MATLAB_RESULTS['tradmat_time'])),
            ('OASIS',       OASIS_RESULTS,       'oasis_F1_window',    'oasis_recall_window',    'oasis_precision_window',    None, float(OASIS_RESULTS['oasis_time'])),
            ('CASCADE_GPU', CASCADE_GPU_RESULTS, 'cascade_F1_window',  'cascade_recall_window',  'cascade_precision_window',  None, float(CASCADE_GPU_RESULTS['cascade_time'])),
        ]

    labels    = [name for name, *_ in METHOD_INFO]
    positions = list(range(len(labels)))
    _display  = {
        'fMCSI': 'OMSI',
        'MATLAB': 'CaImAn',
        'OASIS': 'OASIS',
        'CASCADE_GPU': 'CASCADE',
        'CASCADE_CPU': 'CASCADE (CPU)',
    }
    tick_labels = [_display.get(l, l) for l in labels]

    speed_labels    = labels + ['CASCADE_CPU']
    speed_positions = list(range(len(speed_labels)))
    speed_tick_labels = [_display.get(l, l) for l in speed_labels]

    speed_tick_labels[labels.index('CASCADE_GPU')] = 'CASCADE (GPU)'
    speed_total_times = [t for *_, t in METHOD_INFO] + [float(CASCADE_CPU_RESULTS['cascade_time'])]
    speed_tpc_means   = (
        [total_t / n_cells for *_, total_t in METHOD_INFO]
        + [float(CASCADE_CPU_RESULTS['cascade_time']) / n_cells]
    )
    speed_bar_colors  = [COLORS[n] for n in speed_labels]

    example_cells = _select_example_cells(
        MINE_RESULTS, OASIS_RESULTS, CASCADE_GPU_RESULTS, MATLAB_RESULTS,
        n_cells=4, window=60.0
    )

    fig = plt.figure(figsize=(5.5, 7), dpi=300)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.6)

    raster_ax      = fig.add_subplot(gs[:2, :])
    prec_vs_recall = fig.add_subplot(gs[2, 0])
    F1_dist        = fig.add_subplot(gs[2, 1])
    cosmic_dist    = fig.add_subplot(gs[2, 2])
    total_time     = fig.add_subplot(gs[3, 0])
    time_per_cell  = fig.add_subplot(gs[3, 1])
    time_per_spike = fig.add_subplot(gs[3, 2])

    _plot_raster(raster_ax, example_cells, window=60.0)

    for name, res, f1_k, rec_k, prec_k, tpc_k, total_t in METHOD_INFO:
        prec_vs_recall.scatter(
            np.array(res[rec_k], dtype=float),
            np.array(res[prec_k], dtype=float),
            s=1, c=COLORS[name], label=name, alpha=0.6,
        )
    prec_vs_recall.set_xlabel('Recall')
    prec_vs_recall.set_ylabel('Precision')
    prec_vs_recall.set_ylim([-0.05, 1.1])
    prec_vs_recall.set_xlim([-0.05, 1.1])

    fb_arrays = [
        _fbeta(np.array(res[prec_k], dtype=float), np.array(res[rec_k], dtype=float))
        for _, res, f1_k, rec_k, prec_k, tpc_k, total_t in METHOD_INFO
    ]
    parts = F1_dist.violinplot(fb_arrays, positions=positions,
                               showmedians=True, widths=0.65)
    for pc, name in zip(parts['bodies'], labels):
        pc.set_facecolor(COLORS[name])
        pc.set_alpha(0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[partname].set_color('k')
        parts[partname].set_linewidth(0.8)
    F1_dist.set_xticks(positions)
    F1_dist.set_xticklabels(tick_labels, fontsize=6, rotation=90, ha='right')
    F1_dist.set_ylabel(r'$F_\beta$ score')
    F1_dist.set_ylim([0, 1.1])

    from OMSI.helpers import compute_cosmic
    true_spikes = list(MINE_RESULTS['true_spikes'])
    fs = float(MINE_RESULTS['f'])
    cosmic_spike_keys = [
        ('fMCSI',       MINE_RESULTS,        'optim_spikes'),
        ('MATLAB',      MATLAB_RESULTS,      'tradmat_spikes'),
        ('OASIS',       OASIS_RESULTS,       'oasis_spikes'),
        ('CASCADE_GPU', CASCADE_GPU_RESULTS, 'cascade_spikes'),
    ]
    print('Computing CosMIC scores...')
    cosmic_arrays = []
    for name, res, spk_k in cosmic_spike_keys:
        scores = compute_cosmic(true_spikes, list(res[spk_k]), fs)
        cosmic_arrays.append(scores)
        print(f'  {name}: mean CosMIC = {np.mean(scores):.3f}')
    parts = cosmic_dist.violinplot(cosmic_arrays, positions=positions,
                                   showmedians=True, widths=0.65)
    for pc, name in zip(parts['bodies'], labels):
        pc.set_facecolor(COLORS[name])
        pc.set_alpha(0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[partname].set_color('k')
        parts[partname].set_linewidth(0.8)
    cosmic_dist.set_xticks(positions)
    cosmic_dist.set_xticklabels(tick_labels, fontsize=6, rotation=90, ha='right')
    cosmic_dist.set_ylabel('CosMIC Score')
    cosmic_dist.set_ylim([0, 1.1])

    speed_total_minutes = [t / 60.0 for t in speed_total_times]
    total_time.bar(speed_positions, speed_total_minutes, color=speed_bar_colors, width=0.65)
    total_time.set_xticks(speed_positions)
    total_time.set_xticklabels(speed_tick_labels, fontsize=5, rotation=90, ha='right')
    total_time.set_ylabel('Total time (min)')
    total_time.set_yscale('log')
    total_time.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=100))
    total_time.yaxis.set_major_locator(ticker.LogLocator(numticks=100))
    total_time.tick_params(axis='y', which='minor', length=4, width=0.5, colors='black')
    total_time.tick_params(axis='y', which='major', length=8, width=1,   colors='black')

    print(f'\n{"Method":<14} {"Total time (s)":>16} {"Time/cell (s)":>14}')
    print('-' * 46)
    for name, total_t, tpc in zip(speed_labels, speed_total_times, speed_tpc_means):
        print(f'{name:<14} {total_t:>16.1f} {tpc:>14.3f}')
    time_per_cell.bar(speed_positions, speed_tpc_means, color=speed_bar_colors, width=0.65)
    time_per_cell.set_xticks(speed_positions)
    time_per_cell.set_xticklabels(speed_tick_labels, fontsize=5, rotation=90, ha='right')
    time_per_cell.set_ylabel('time per cell (sec)')
    time_per_cell.set_yscale('log')
    time_per_cell.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=100))
    time_per_cell.yaxis.set_major_locator(ticker.LogLocator(numticks=100))
    time_per_cell.tick_params(axis='y', which='minor', length=4, width=0.5, colors='black')
    time_per_cell.tick_params(axis='y', which='major', length=8, width=1,   colors='black')

    true_spikes_arr = list(MINE_RESULTS['true_spikes'])
    n_true_spikes   = np.array([len(np.atleast_1d(s)) for s in true_spikes_arr], dtype=float)
    my_tpc          = np.array(MINE_RESULTS['optim_times_per_cell'], dtype=float)
    time_per_spike.scatter(n_true_spikes[my_tpc>0], my_tpc[my_tpc>0], s=2, c=COLORS['fMCSI'], alpha=0.6)
    time_per_spike.set_xlabel('# true spikes')
    time_per_spike.set_ylabel('time per cell (sec)')
    time_per_spike.set_xlim([0, 1000])
    time_per_spike.set_ylim([0,15])
    plot_running_median(
        time_per_spike,
        n_true_spikes[(n_true_spikes<1000)*(my_tpc>0)],
        my_tpc[(n_true_spikes<1000)*(my_tpc>0)],
        n_bins=5,
        vertical=False,
        color='k',
        fb=True
    )

    legend_handles = [
        plt.Line2D([0], [0], color=COLORS['fMCSI'],      marker='.', linestyle='-', label='OMSI'),
        plt.Line2D([0], [0], color=COLORS['MATLAB'],      marker='.', linestyle='-', label='CaImAn'),
        plt.Line2D([0], [0], color=COLORS['OASIS'],       marker='.', linestyle='-', label='OASIS'),
        plt.Line2D([0], [0], color=COLORS['CASCADE_GPU'], marker='.', linestyle='-', label='CASCADE'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure1.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def plot_running_median(ax, x, y, n_bins=7, vertical=False, fb=True, color='k'):

    import scipy.stats
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    x_use, y_use = x[mask], y[mask]
    bins = np.linspace(np.min(x_use), np.max(x_use), n_bins)
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x_use, y_use, np.nanmedian, bins=bins)
    bin_std, _, _  = scipy.stats.binned_statistic(x_use, y_use, np.nanstd,    bins=bins)
    hist, _, _     = scipy.stats.binned_statistic(x_use, y_use,
                                                  lambda v: np.sum(~np.isnan(v)), bins=bins)
    tuning_err = bin_std / np.sqrt(hist)
    centers = bin_edges[:-1] + np.median(np.diff(bins)) / 2
    if not vertical:
        ax.plot(centers, bin_means, '-', color=color)
        if fb:
            ax.fill_between(centers, bin_means - tuning_err, bin_means + tuning_err,
                            color=color, alpha=0.2)
    else:
        ax.plot(bin_means, centers, '-', color=color)
        if fb:
            ax.fill_betweenx(centers, bin_means - tuning_err, bin_means + tuning_err,
                             color=color, alpha=0.2)
            
    # do a linear regressiona and print the slope
    if len(x_use) > 1:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_use, y_use)
        print(f"Linear regression slope: {slope:.4f}, R-squared: {r_value**2:.4f}")

    return np.nanmax(bin_means + tuning_err)



def print_stats(data_dir=_DEFAULT_DATA_DIR):

    paths = {k: os.path.join(data_dir, v) for k, v in _NPZ_NAMES.items()}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'{name} results not found at {path}. Run --mode test first.')

    MINE_RESULTS        = np.load(paths['fMCSI'],       allow_pickle=True)
    MATLAB_RESULTS      = np.load(paths['MATLAB'],      allow_pickle=True)
    OASIS_RESULTS       = np.load(paths['OASIS'],       allow_pickle=True)
    CASCADE_GPU_RESULTS = np.load(paths['CASCADE_GPU'], allow_pickle=True)
    CASCADE_CPU_RESULTS = np.load(paths['CASCADE_CPU'], allow_pickle=True)

    n_cells = int(MINE_RESULTS['n_cells'])

    method_entries = [
        ('OMSI',       MINE_RESULTS,        'optim_precision',    'optim_recall',    'optim_spikes',    float(MINE_RESULTS['optim_time'])),
        ('MATLAB',      MATLAB_RESULTS,      'tradmat_precision',  'tradmat_recall',  'tradmat_spikes',  float(MATLAB_RESULTS['tradmat_time'])),
        ('OASIS',       OASIS_RESULTS,       'oasis_precision',    'oasis_recall',    'oasis_spikes',    float(OASIS_RESULTS['oasis_time'])),
        ('CASCADE_GPU', CASCADE_GPU_RESULTS, 'cascade_precision',  'cascade_recall',  'cascade_spikes',  float(CASCADE_GPU_RESULTS['cascade_time'])),
        ('CASCADE_CPU', CASCADE_CPU_RESULTS, 'cascade_precision',  'cascade_recall',  'cascade_spikes',  float(CASCADE_CPU_RESULTS['cascade_time'])),
    ]

    from OMSI.helpers import compute_cosmic
    true_spikes = list(MINE_RESULTS['true_spikes'])
    fs = float(MINE_RESULTS['f'])

    print('\n' + '='*78)
    print('FIGURE 1 STATISTICS')
    print('='*78)

    print(f'\n{"Method":<14}  {"F_beta median":>14}  {"F_beta IQR":>11}  {"CosMIC median":>14}  {"CosMIC IQR":>11}')
    print('-'*70)
    fb_data = {}
    for label, res, prec_k, rec_k, spk_k, total_t in method_entries:
        prec   = np.array(res[prec_k], dtype=float)
        rec    = np.array(res[rec_k],  dtype=float)
        fb     = _fbeta(prec, rec)
        cosmic = compute_cosmic(true_spikes, list(res[spk_k]), fs)
        fb_data[label] = fb
        fb_med = np.nanmedian(fb);     fb_iqr = np.subtract(*np.nanpercentile(fb,     [75, 25]))
        co_med = np.nanmedian(cosmic); co_iqr = np.subtract(*np.nanpercentile(cosmic, [75, 25]))
        print(f'{label:<14}  {fb_med:>14.3f}  {fb_iqr:>11.3f}  '
              f'{co_med:>14.3f}  {co_iqr:>11.3f}')

    print(f'\n{"Method":<14}  {"Total time (min)":>17}  {"Time/cell (sec)":>16}')
    print('-'*52)
    fmcsi_total_s = None
    for label, res, prec_k, rec_k, spk_k, total_t in method_entries:
        if label == 'OMSI':
            fmcsi_total_s = total_t
        total_min = total_t / 60.0
        tpc_sec   = total_t / n_cells
        print(f'{label:<14}  {total_min:>17.3f}  {tpc_sec:>16.3f}')

    print(f'\n{"Method":<14}  {"% diff from fMCSI total time":>30}')
    print('-'*50)
    for label, res, prec_k, rec_k, spk_k, total_t in method_entries:
        if label == 'OMSI':
            print(f'{label:<14}  {"(reference)":>30}')
            continue
        pct = (total_t - fmcsi_total_s) / total_t * 100.0
        sign = '+' if pct >= 0 else ''
        print(f'{label:<14}  {sign}{pct:>28.1f}%')

    matlab_t = float(MATLAB_RESULTS['tradmat_time'])
    if fmcsi_total_s and fmcsi_total_s > 0:
        oom = np.log10(matlab_t / fmcsi_total_s)
        print(f'\nOrder-of-magnitude difference (fMCSI vs MATLAB): {oom:.2f}  '
              f'(MATLAB is ~{10**oom:.1f}x slower, 10^{oom:.2f})')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Figure 1: fixed benchmark on simulated data'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'print'],
                        help='"test" runs inference and writes NPZ files; '
                             '"plot" loads NPZ files and generates the figure; '
                             '"print" prints summary statistics to terminal')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument('--no-fmcsi',   action='store_true', help='Skip fMCSI')
    parser.add_argument('--no-matlab',  action='store_true', help='Skip MATLAB')
    parser.add_argument('--no-oasis',   action='store_true', help='Skip OASIS')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(
            data_dir    = args.data_dir,
            run_fmcsi   = not args.no_fmcsi,
            run_matlab  = not args.no_matlab,
            run_oasis   = not args.no_oasis,
            run_cascade = not args.no_cascade,
        )
    elif args.mode == 'plot':
        plot_figure(data_dir=args.data_dir)
    else:
        print_stats(data_dir=args.data_dir)
