#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allen institute data benchmark.

To run inference
    $ python figure2.py --mode test --data-dir /path/to/save/results/in --allen-data-dir /path/to/allen/data

To create figure:
    $ python figure2.py --mode plot --data-dir /path/to/results

Written DMM, March 2026
"""

import argparse
import os
import subprocess
import sys
import time
import re
import glob as _glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.lines import Line2D
import h5py
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import percentile_filter, gaussian_filter1d
from scipy.stats import kurtosis as sci_kurtosis
from oasis.functions import deconvolve

import OMSI
from run_pnev_MCMC import run_matlab_pnevMCMC

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig3')
_MATLAB_DATA_DIR  = '/home/dylan/Fast2/spike_deconv/figures_output_data_260421/data/fig3'

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7

model_colors = {
    'fMCSI': '#4C72B0',
    'MATLAB': '#DD8452',
    'OASIS':     '#55A868',
    'CASCADE':   '#8172B3',
}
_MODEL_ORDER = ['fMCSI', 'MATLAB', 'CASCADE', 'OASIS']

USE_STRICT_ACCURACY = True  # Hungarian one-to-one matching (compute_accuracy_strict)
BETA = 0.5

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


_CASCADE_CMP_COLOR_7P5  = 'tab:red'
_CASCADE_CMP_COLOR_30   = 'tab:cyan'

_GENO_LS         = {'Cux2': '-', 'Emx1': '--', 'tetO': ':'}
_GENO_LS_DEFAULT = ':'

_EXCLUDED_DATASETS = {'DS29-GCaMP7f-m-V1', 'DS32-GCaMP8s-m-V1', 'DS28-XCaMPgf-m-V1'}


def _save_records(records, path):

    if not records:
        np.savez(path)
        return
    keys = list(records[0].keys())
    arrays = {}
    for k in keys:
        vals = [r.get(k) for r in records]
        if all(isinstance(v, str) or v is None for v in vals):
            arrays[k] = np.array([v if v is not None else '' for v in vals], dtype=object)
        else:
            try:
                arrays[k] = np.array(vals, dtype=np.float64)
            except (TypeError, ValueError):
                arrays[k] = np.array([str(v) for v in vals], dtype=object)
    np.savez(path, **arrays)


def _load_records(path):

    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
    if not keys:
        return []
    n = len(d[keys[0]])
    records = []
    for i in range(n):
        row = {}
        for k in keys:
            v = d[k][i]
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            elif hasattr(v, 'item'):
                v = v.item()
            row[k] = v
        records.append(row)
    return records


def _fbeta(precision, recall):

    p = np.asarray(precision, dtype=float)
    r = np.asarray(recall, dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.where(denom > 0, (1 + b2) * p * r / denom, 0.0))


def normalize_label(label):
    return re.sub(r'^[^_]+_', '', label)


def clean_label(label):
    m = re.search(r'(.*)_\(\'([^\']+)\',\s*(\d+),\s*(\d+)\)frames', label)
    if m:
        return f"{m.group(2)}_{m.group(1)}_{m.group(4)}frames_{m.group(3)}hz"
    return label


def get_genotype(label):
    known_types = ['Cux2', 'Rorb', 'Scnn1a', 'Nr5a1', 'Emx1', 'Slc17a7', 'tetO',
                   'Vip', 'Sst', 'Pvalb']
    for k in known_types:
        if k.lower() in label.lower():
            return k
    return 'Other'


def get_zoom_for_label(label, zoom_lookup=None):
    if 'lowzoom' in label:
        return 'Low Zoom'
    elif 'highzoom' in label:
        return 'High Zoom'
    return 'Unknown'


def save_aggregated_data(slow_data_groups, fast_data_groups, aggregated_h5_path):

    print(f"\nSaving aggregated data to {aggregated_h5_path}...")
    with h5py.File(aggregated_h5_path, 'w') as hf:
        for group_key, h5_group_name in [('slow', 'slow_data'), ('fast', 'fast_data')]:
            src = slow_data_groups if group_key == 'slow' else fast_data_groups
            if not src:
                continue
            h5g = hf.create_group(h5_group_name)
            for (experiment_name, fs_rounded, n_frames), gd in src.items():
                gname = f"{experiment_name}__fs_{fs_rounded}__n_frames_{n_frames}"
                ng = h5g.create_group(gname)
                ng.attrs['experiment_name'] = experiment_name
                ng.attrs['n_frames'] = n_frames
                ng.create_dataset('dff', data=gd['dff'])
                flat = np.concatenate(gd['spikes_list']) if gd['spikes_list'] else np.array([])
                offs = np.cumsum([0] + [len(s) for s in gd['spikes_list']])
                ng.create_dataset('spike_times_flat', data=flat)
                ng.create_dataset('spike_offsets', data=offs)
                ng.attrs['fs']  = gd['fs']
                ng.attrs['tau'] = gd['tau']
    print(f"Aggregated data saved to {aggregated_h5_path}")


def load_aggregated_data(aggregated_h5_path):

    print(f"\nLoading aggregated data from {aggregated_h5_path}...")
    slow_data_groups = {}
    fast_data_groups = {}
    with h5py.File(aggregated_h5_path, 'r') as hf:
        for h5_key, dst in [('slow_data', slow_data_groups),
                             ('fast_data',  fast_data_groups)]:
            if h5_key not in hf:
                continue
            for group_name in hf[h5_key].keys():
                ng = hf[h5_key][group_name]
                if 'experiment_name' in ng.attrs:
                    experiment_name = ng.attrs['experiment_name']
                    n_frames = int(ng.attrs['n_frames'])
                else:
                    n_frames = int(group_name.split('n_frames_')[-1])
                    experiment_name = 'unknown'
                dff  = ng['dff'][:]
                flat = ng['spike_times_flat'][:]
                offs = ng['spike_offsets'][:]
                spikes_list = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
                fs  = ng.attrs['fs']
                tau = ng.attrs['tau']
                fs_rounded = int(round(fs))
                dst[(experiment_name, fs_rounded, n_frames)] = {
                    'dff': dff, 'spikes_list': spikes_list, 'fs': fs, 'tau': tau}
    print(f"Aggregated data loaded from {aggregated_h5_path}")
    return slow_data_groups, fast_data_groups


def compute_accuracy_window(true_spikes, predicted_spikes, tolerance=0.100):

    precisions, recalls, f1s = [], [], []
    for t_spk, p_spk in zip(true_spikes, predicted_spikes):
        t_spk = np.asarray(t_spk, dtype=np.float64).flatten()
        p_spk = np.asarray(p_spk, dtype=np.float64).flatten()
        if len(t_spk) == 0 and len(p_spk) == 0:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
            continue
        if len(p_spk) == 0:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
            continue
        if len(t_spk) == 0:
            precisions.append(0.0); recalls.append(1.0); f1s.append(0.0)
            continue
        n_tp_recall = int(np.sum(
            np.any(np.abs(t_spk[:, None] - p_spk[None, :]) <= tolerance, axis=1)))
        n_tp_prec = int(np.sum(
            np.any(np.abs(p_spk[:, None] - t_spk[None, :]) <= tolerance, axis=1)))
        rec  = n_tp_recall / len(t_spk)
        prec = n_tp_prec   / len(p_spk)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1s)


def _run_cascade_inference(dff, fs, label, data_dir):

    script      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'fig3_cascade_{label}_input.npz')
    output_path = os.path.join(data_dir, f'fig3_cascade_{label}_output.npz')
    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python', script,
         '--mode', 'inference', '--input', input_path, '--output', output_path],
        check=True)
    result = np.load(output_path, allow_pickle=True)
    return (result['cascade_probs'],
            list(result['cascade_spikes']),
            float(result['cascade_time']))


def _run_and_save_allen_group(dff, true_spikes, fs, tau, label, data_dir,
                              run_matlab=False):
    all_results = []
    n_cells = dff.shape[0]

    cell_kurtosis = OMSI.helpers.compute_kurtosis(dff)
    kurtosis_threshold = 0.5
    good_mask = cell_kurtosis >= kurtosis_threshold
    good_idx  = np.where(good_mask)[0]
    n_good    = len(good_idx)
    print(f"  Kurtosis filter: {n_good}/{n_cells} cells pass "
          f"(excess kurtosis >= {kurtosis_threshold})")

    dff         = dff[good_idx]
    true_spikes = [true_spikes[i] for i in good_idx]
    n_cells     = n_good

    if n_cells == 0:
        print(f"  WARNING: No cells pass kurtosis filter for {label}, skipping.")
        return

    true_events    = [OMSI.helpers.make_event_ground_truth(sp, tau)
                      for sp in true_spikes]
    n_events_total = sum(len(e) for e in true_events)
    n_spikes_total = sum(len(s) for s in true_spikes)
    print(f"  Event ground truth: {n_events_total} events from {n_spikes_total} spikes "
          f"({100*n_events_total/max(n_spikes_total,1):.1f}% isolated)")

    print("  Running fMCSI...")
    t0      = time.time()
    tau_rise = 0.05
    g_rise   = float(np.exp(-1.0 / (tau_rise * fs)))
    g_decay  = float(np.exp(-1.0 / (tau * fs)))
    g_ar2    = [g_rise + g_decay, -g_rise * g_decay]
    params   = {
        'f': fs, 'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1,
        'g': g_ar2, 'defg': [g_rise, g_decay],
        'TauStd': [tau_rise * fs, tau * fs], 'lam_scale': 1.0,
    }
    optim_dict = OMSI.deconv(dff, params, true_spikes=true_spikes, benchmark=True)
    my_probs   = optim_dict['optim_prob']
    my_spikes  = list(optim_dict['optim_spikes'])
    time_my    = time.time() - t0

    my_spikes_shifted, _ = OMSI.detect_spikes_from_probs(my_probs, fs, sigma=1.5)
    prec_my, rec_my, f1_my           = OMSI.compute_accuracy_strict(true_spikes, my_spikes, tolerance=0.1)
    prec_my_w, rec_my_w, f1_my_w     = compute_accuracy_window(true_spikes, my_spikes, tolerance=0.1)
    prec_my_e, rec_my_e, f1_my_e     = compute_accuracy_window(true_events,  my_spikes, tolerance=0.1)
    cosmic_my                         = OMSI.helpers.compute_cosmic(true_spikes, my_spikes_shifted, fs)

    print(f"    [fMCSI] strict F1={np.mean(f1_my):.3f}  window F1={np.mean(f1_my_w):.3f}")
    for i in range(n_cells):
        all_results.append({
            'model': 'fMCSI', 'tau': tau, 'cell_id': int(good_idx[i]),
            'time': time_my / n_cells,
            'f1': f1_my[i], 'precision': prec_my[i], 'recall': rec_my[i],
            'f1_window': f1_my_w[i], 'precision_window': prec_my_w[i],
            'recall_window': rec_my_w[i],
            'f1_event': f1_my_e[i], 'precision_event': prec_my_e[i],
            'recall_event': rec_my_e[i],
            'cosmic': cosmic_my[i],
        })

    trad_probs = None
    trad_spikes_out = []
    cosmic_trad = []

    if run_matlab:
        print("  Running MATLAB...")
        t0 = time.time()
        _, _, trad_probs, _ = run_matlab_pnevMCMC(
            dff, fs=fs, tau=tau, n_sweeps=500, true_spikes=true_spikes)
        time_trad = time.time() - t0

        trad_spikes_out, _ = OMSI.detect_spikes_from_probs(trad_probs, fs, sigma=1.5)
        prec_trad, rec_trad, f1_trad       = OMSI.compute_accuracy_strict(
            true_spikes, trad_spikes_out, tolerance=0.1)
        prec_trad_w, rec_trad_w, f1_trad_w = compute_accuracy_window(
            true_spikes, trad_spikes_out, tolerance=0.1)
        prec_trad_e, rec_trad_e, f1_trad_e = compute_accuracy_window(
            true_events, trad_spikes_out, tolerance=0.1)
        cosmic_trad                         = OMSI.helpers.compute_cosmic(
            true_spikes, trad_spikes_out, fs)

        print(f"    [MATLAB] strict F1={np.mean(f1_trad):.3f}  window F1={np.mean(f1_trad_w):.3f}")
        for i in range(n_cells):
            all_results.append({
                'model': 'MATLAB', 'tau': tau, 'cell_id': int(good_idx[i]),
                'time': time_trad / n_cells,
                'f1': f1_trad[i], 'precision': prec_trad[i], 'recall': rec_trad[i],
                'f1_window': f1_trad_w[i], 'precision_window': prec_trad_w[i],
                'recall_window': rec_trad_w[i],
                'f1_event': f1_trad_e[i], 'precision_event': prec_trad_e[i],
                'recall_event': rec_trad_e[i],
                'cosmic': cosmic_trad[i],
            })

    print("  Running OASIS...")
    t0 = time.time()
    diff   = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    oasis_probs  = []
    oasis_spikes = []
    for i in range(dff.shape[0]):
        g = np.exp(-1 / (fs * tau))
        _, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        oasis_probs.append(s)
        oasis_spikes.append(_oasis_spikes_from_s(s, sigmas[i], fs))
    oasis_probs = np.array(oasis_probs)
    time_oasis  = time.time() - t0

    oasis_spikes_shifted, _ = OMSI.detect_spikes_from_probs(oasis_probs, fs, sigma=0.5)
    prec_oasis, rec_oasis, f1_oasis       = OMSI.compute_accuracy_strict(
        true_spikes, oasis_spikes_shifted, tolerance=0.1)
    prec_oasis_w, rec_oasis_w, f1_oasis_w = compute_accuracy_window(
        true_spikes, oasis_spikes_shifted, tolerance=0.1)
    prec_oasis_e, rec_oasis_e, f1_oasis_e = compute_accuracy_window(
        true_events, oasis_spikes_shifted, tolerance=0.1)
    cosmic_oasis                           = OMSI.helpers.compute_cosmic(
        true_spikes, oasis_spikes_shifted, fs)

    print(f"    [OASIS] strict F1={np.mean(f1_oasis):.3f}  window F1={np.mean(f1_oasis_w):.3f}")
    for i in range(n_cells):
        all_results.append({
            'model': 'OASIS', 'tau': tau, 'cell_id': int(good_idx[i]),
            'time': time_oasis / n_cells,
            'f1': f1_oasis[i], 'precision': prec_oasis[i], 'recall': rec_oasis[i],
            'f1_window': f1_oasis_w[i], 'precision_window': prec_oasis_w[i],
            'recall_window': rec_oasis_w[i],
            'f1_event': f1_oasis_e[i], 'precision_event': prec_oasis_e[i],
            'recall_event': rec_oasis_e[i],
            'cosmic': cosmic_oasis[i],
        })

    traces_path = os.path.join(data_dir, f'allen_data_results_{label}_traces.npz')
    print(f"  Saving traces to {traces_path} ...")
    np.savez(
        traces_path,
        dff=dff, true_spikes=np.array(true_spikes, dtype=object),
        my_probs=my_probs, my_spikes=np.array(my_spikes, dtype=object),
        trad_probs=trad_probs if run_matlab else np.array([]),
        trad_spikes=np.array(trad_spikes_out, dtype=object) if run_matlab else np.array([]),
        oasis_probs=oasis_probs,
        oasis_spikes=np.array(oasis_spikes_shifted, dtype=object),
        fs=fs, tau=tau,
        cosmic_my=cosmic_my,
        cosmic_trad=cosmic_trad if run_matlab else np.array([]),
        cosmic_oasis=cosmic_oasis,
    )

    npz_path = os.path.join(data_dir, f'allen_data_results_{label}.npz')
    _save_records(all_results, npz_path)
    print(f"  Saved results -> {npz_path}")

    print("  Running CASCADE (subprocess)...")
    try:
        cascade_probs, cascade_spikes, time_cascade = _run_cascade_inference(
            dff, fs, label, data_dir)

        prec_cas, rec_cas, f1_cas       = OMSI.compute_accuracy_strict(
            true_spikes, cascade_spikes, tolerance=0.1)
        prec_cas_w, rec_cas_w, f1_cas_w = compute_accuracy_window(
            true_spikes, cascade_spikes, tolerance=0.1)
        prec_cas_e, rec_cas_e, f1_cas_e = compute_accuracy_window(
            true_events, cascade_spikes, tolerance=0.1)
        cosmic_cas                       = OMSI.helpers.compute_cosmic(
            true_spikes, cascade_spikes, fs)

        print(f"    [CASCADE] strict F1={np.mean(f1_cas):.3f}  window F1={np.mean(f1_cas_w):.3f}")

        cas_results = []
        for i in range(n_cells):
            cas_results.append({
                'model': 'CASCADE', 'tau': tau, 'cell_id': int(good_idx[i]),
                'time': time_cascade / n_cells,
                'f1': f1_cas[i], 'precision': prec_cas[i], 'recall': rec_cas[i],
                'f1_window': f1_cas_w[i], 'precision_window': prec_cas_w[i],
                'recall_window': rec_cas_w[i],
                'f1_event': f1_cas_e[i], 'precision_event': prec_cas_e[i],
                'recall_event': rec_cas_e[i],
                'cosmic': cosmic_cas[i],
            })

        cas_traces_path = os.path.join(
            data_dir, f'allen_data_results_cascade_{label}_traces.npz')
        np.savez(
            cas_traces_path,
            dff=dff, true_spikes=np.array(true_spikes, dtype=object),
            cascade_probs=cascade_probs,
            cascade_spikes=np.array(cascade_spikes, dtype=object),
            fs=fs, tau=tau,
            cosmic_cas=cosmic_cas,
        )

        cas_npz_path = os.path.join(
            data_dir, f'allen_data_results_cascade_{label}.npz')
        _save_records(cas_results, cas_npz_path)
        print(f"  Saved CASCADE results -> {cas_npz_path}")

    except subprocess.CalledProcessError as exc:
        print(f"  WARNING: CASCADE subprocess failed for {label}: {exc}")


def _run_and_save_fmcsi_group(dff, true_spikes, fs, tau, label, data_dir):

    n_cells = dff.shape[0]

    cell_kurtosis = OMSI.helpers.compute_kurtosis(dff)
    kurtosis_threshold = 0.5
    good_mask = cell_kurtosis >= kurtosis_threshold
    good_idx  = np.where(good_mask)[0]
    n_good    = len(good_idx)
    print(f"  Kurtosis filter: {n_good}/{n_cells} cells pass "
          f"(excess kurtosis >= {kurtosis_threshold})")

    dff         = dff[good_idx]
    true_spikes = [true_spikes[i] for i in good_idx]
    n_cells     = n_good

    if n_cells == 0:
        print(f"  WARNING: No cells pass kurtosis filter for {label}, skipping.")
        return

    true_events = [OMSI.helpers.make_event_ground_truth(sp, tau)
                   for sp in true_spikes]

    print("  Running fMCSI...")
    t0       = time.time()
    tau_rise = 0.05
    g_rise   = float(np.exp(-1.0 / (tau_rise * fs)))
    g_decay  = float(np.exp(-1.0 / (tau * fs)))
    g_ar2    = [g_rise + g_decay, -g_rise * g_decay]
    params   = {
        'f': fs, 'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1,
        'g': g_ar2, 'defg': [g_rise, g_decay],
        'TauStd': [tau_rise * fs, tau * fs], 'lam_scale': 1.0,
    }
    optim_dict = OMSI.deconv(dff, params, true_spikes=true_spikes, benchmark=True)
    my_probs   = optim_dict['optim_prob']
    my_spikes  = list(optim_dict['optim_spikes'])
    time_my    = time.time() - t0

    my_spikes_shifted, _ = OMSI.detect_spikes_from_probs(my_probs, fs, sigma=1.5)
    prec_my, rec_my, f1_my         = OMSI.compute_accuracy_strict(
        true_spikes, my_spikes, tolerance=0.1)
    prec_my_w, rec_my_w, f1_my_w   = compute_accuracy_window(
        true_spikes, my_spikes, tolerance=0.1)
    prec_my_e, rec_my_e, f1_my_e   = compute_accuracy_window(
        true_events, my_spikes, tolerance=0.1)
    cosmic_my                       = OMSI.helpers.compute_cosmic(
        true_spikes, my_spikes_shifted, fs)

    print(f"    [fMCSI] strict F1={np.mean(f1_my):.3f}  window F1={np.mean(f1_my_w):.3f}")

    all_results = []
    for i in range(n_cells):
        all_results.append({
            'model': 'fMCSI', 'tau': tau, 'cell_id': int(good_idx[i]),
            'time': time_my / n_cells,
            'f1': f1_my[i], 'precision': prec_my[i], 'recall': rec_my[i],
            'f1_window': f1_my_w[i], 'precision_window': prec_my_w[i],
            'recall_window': rec_my_w[i],
            'f1_event': f1_my_e[i], 'precision_event': prec_my_e[i],
            'recall_event': rec_my_e[i],
            'cosmic': cosmic_my[i],
        })

    print("  Running OASIS...")
    t0 = time.time()
    diff   = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    oasis_probs  = []
    for i in range(dff.shape[0]):
        g = np.exp(-1 / (fs * tau))
        _, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        oasis_probs.append(s)
    oasis_probs = np.array(oasis_probs)
    time_oasis  = time.time() - t0

    oasis_spikes_shifted, _ = OMSI.detect_spikes_from_probs(oasis_probs, fs, sigma=0.5)
    prec_oasis, rec_oasis, f1_oasis       = OMSI.compute_accuracy_strict(
        true_spikes, oasis_spikes_shifted, tolerance=0.1)
    prec_oasis_w, rec_oasis_w, f1_oasis_w = compute_accuracy_window(
        true_spikes, oasis_spikes_shifted, tolerance=0.1)
    prec_oasis_e, rec_oasis_e, f1_oasis_e = compute_accuracy_window(
        true_events, oasis_spikes_shifted, tolerance=0.1)
    cosmic_oasis                           = OMSI.helpers.compute_cosmic(
        true_spikes, oasis_spikes_shifted, fs)

    print(f"    [OASIS] strict F1={np.mean(f1_oasis):.3f}  window F1={np.mean(f1_oasis_w):.3f}")
    for i in range(n_cells):
        all_results.append({
            'model': 'OASIS', 'tau': tau, 'cell_id': int(good_idx[i]),
            'time': time_oasis / n_cells,
            'f1': f1_oasis[i], 'precision': prec_oasis[i], 'recall': rec_oasis[i],
            'f1_window': f1_oasis_w[i], 'precision_window': prec_oasis_w[i],
            'recall_window': rec_oasis_w[i],
            'f1_event': f1_oasis_e[i], 'precision_event': prec_oasis_e[i],
            'recall_event': rec_oasis_e[i],
            'cosmic': cosmic_oasis[i],
        })

    traces_path = os.path.join(data_dir, f'allen_data_results_fmcsi_{label}_traces.npz')
    np.savez(
        traces_path,
        dff=dff, true_spikes=np.array(true_spikes, dtype=object),
        my_probs=my_probs, my_spikes=np.array(my_spikes, dtype=object),
        oasis_probs=oasis_probs,
        oasis_spikes=np.array(oasis_spikes_shifted, dtype=object),
        fs=fs, tau=tau,
        cosmic_my=cosmic_my,
        cosmic_oasis=cosmic_oasis,
    )
    print(f"  Saved fMCSI+OASIS traces -> {traces_path}")

    npz_path = os.path.join(data_dir, f'allen_data_results_fmcsi_{label}.npz')
    _save_records(all_results, npz_path)
    print(f"  Saved fMCSI+OASIS results -> {npz_path}")


def test_figure(data_dir, allen_data_dir, run_matlab=False):

    os.makedirs(data_dir, exist_ok=True)
    aggregated_h5 = os.path.join(data_dir, 'allen_aggregated_data.h5')

    if os.path.exists(aggregated_h5):
        print(f"Loading preprocessed data from {aggregated_h5}")
        slow_data_groups, fast_data_groups = load_aggregated_data(aggregated_h5)
    else:
        print(f"Preprocessing raw data from {allen_data_dir}")
        slow_data_groups, fast_data_groups = _load_and_preprocess_raw(
            allen_data_dir, aggregated_h5)

    for indicator, data_groups in [('slow', slow_data_groups),
                                   ('fast', fast_data_groups)]:
        if not data_groups:
            continue
        print(f"\n--- Processing {indicator.upper()} cell groups ---")
        for (experiment_name, fs_rounded, n_frames), gd in data_groups.items():
            if experiment_name in _EXCLUDED_DATASETS:
                print(f"\n  Skipping excluded dataset: {experiment_name}")
                continue
            print(f"\n  Group {experiment_name}  {n_frames} frames @ {fs_rounded}Hz "
                  f"({gd['dff'].shape[0]} cells)")
            label = f"{experiment_name}_{indicator}_tau_{n_frames}frames_{fs_rounded}hz"
            _run_and_save_allen_group(
                gd['dff'], gd['spikes_list'], gd['fs'], gd['tau'],
                label, data_dir, run_matlab=run_matlab)


def test_fmcsi(data_dir, allen_data_dir):

    os.makedirs(data_dir, exist_ok=True)
    aggregated_h5 = os.path.join(data_dir, 'allen_aggregated_data.h5')

    if os.path.exists(aggregated_h5):
        print(f"Loading preprocessed data from {aggregated_h5}")
        slow_data_groups, fast_data_groups = load_aggregated_data(aggregated_h5)
    else:
        print(f"Preprocessing raw data from {allen_data_dir}")
        slow_data_groups, fast_data_groups = _load_and_preprocess_raw(
            allen_data_dir, aggregated_h5)

    for indicator, data_groups in [('slow', slow_data_groups),
                                   ('fast', fast_data_groups)]:
        if not data_groups:
            continue
        print(f"\n--- Processing {indicator.upper()} cell groups (fMCSI only) ---")
        for (experiment_name, fs_rounded, n_frames), gd in data_groups.items():
            if experiment_name in _EXCLUDED_DATASETS:
                print(f"\n  Skipping excluded dataset: {experiment_name}")
                continue
            print(f"\n  Group {experiment_name}  {n_frames} frames @ {fs_rounded}Hz "
                  f"({gd['dff'].shape[0]} cells)")
            label = f"{experiment_name}_{indicator}_tau_{n_frames}frames_{fs_rounded}hz"
            _run_and_save_fmcsi_group(
                gd['dff'], gd['spikes_list'], gd['fs'], gd['tau'],
                label, data_dir)


def _load_and_preprocess_raw(data_dir, aggregated_h5):

    slow_data_groups = {}
    fast_data_groups = {}

    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(data_dir)
        for f in fn if f.endswith('.h5')
    ]
    files.sort()
    print(f"Found {len(files)} H5 files.")

    for fpath in files:
        fname = os.path.basename(fpath)
        experiment_name = os.path.basename(os.path.dirname(fpath))
        is_slow = '-s' in fpath
        is_fast = '-f' in fpath
        if not is_slow and not is_fast:
            print(f"  Cannot determine indicator type from {fname}, skipping.")
            continue
        try:
            with h5py.File(fpath, 'r') as f:
                dte     = f['dte'][:]
                dto     = f['dto'][:]
                ephys_dt = float(dte[0]) if dte.size > 0 else float(dte)
                iFrames = f['iFrames'][:].flatten().astype(np.int64)
                if len(iFrames) > 1:
                    twop_fs = (len(iFrames) - 1) / (
                        (float(iFrames[-1]) - float(iFrames[0])) * ephys_dt)
                else:
                    twop_dt = float(dto[0]) if dto.size > 0 else float(dto)
                    twop_fs = 1.0 / twop_dt

                F     = f['f_cell'][:].astype(np.float64)
                F_neu = f['f_np'][:].astype(np.float64)
                if F.ndim == 1:     F     = F.reshape(1, -1)
                if F_neu.ndim == 1: F_neu = F_neu.reshape(1, -1)

                F_corr = F - 0.7 * F_neu
                Wn     = (5.0 * 2) / twop_fs
                b, a   = butter(3, Wn, btype='low')
                F_filt = filtfilt(b, a, F_corr, axis=1)
                win_fr = int(twop_fs * 60)
                if F_corr.shape[1] > win_fr:
                    baselines = percentile_filter(F_filt, percentile=8,
                                                  size=(1, win_fr))
                else:
                    baselines = np.percentile(F_filt, 8, axis=1, keepdims=True)
                dff = (F_corr - baselines) / np.maximum(baselines, 1.0)

                spike_inds = f['iSpk'][:].flatten().astype(np.int64)
                n_frames   = dff.shape[1]
                in_window  = (spike_inds >= iFrames[0]) & (spike_inds <= iFrames[-1])
                spike_inds = spike_inds[in_window]
                insert     = np.searchsorted(iFrames, spike_inds)
                insert     = np.clip(insert, 0, len(iFrames) - 1)
                left_ok    = insert > 0
                left_dist  = np.where(
                    left_ok,
                    np.abs(iFrames[np.maximum(insert - 1, 0)] - spike_inds),
                    np.iinfo(np.int64).max)
                right_dist = np.abs(iFrames[insert] - spike_inds)
                spike_frame_idx = np.where(
                    left_ok & (left_dist < right_dist), insert - 1, insert)
                valid       = (spike_frame_idx >= 0) & (spike_frame_idx < n_frames)
                spike_times = spike_frame_idx[valid].astype(float) / twop_fs
                if len(spike_times) > 1:
                    keep = np.concatenate([[True], np.diff(spike_times) >= 0.003])
                    spike_times = spike_times[keep]

            indicator_type = 'slow' if is_slow else 'fast'
            current_tau    = 1.2 if is_slow else 0.5
            fs_rounded     = int(round(twop_fs))
            key            = (experiment_name, fs_rounded, n_frames)
            dst = slow_data_groups if is_slow else fast_data_groups
            if key not in dst:
                dst[key] = {'dff_list': [], 'spikes_list': [],
                            'fs': twop_fs, 'tau': current_tau}
            dst[key]['dff_list'].append(dff)
            dst[key]['spikes_list'].append(spike_times)

        except Exception as exc:
            print(f"  Error processing {fname}: {exc}")

    for gd in list(slow_data_groups.values()) + list(fast_data_groups.values()):
        gd['dff'] = np.vstack(gd.pop('dff_list'))

    save_aggregated_data(slow_data_groups, fast_data_groups, aggregated_h5)
    return slow_data_groups, fast_data_groups


def _build_cascade_lookup(data_dir):

    lookup = {}
    for npz_path in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_cascade_*_traces.npz')):
        name      = os.path.basename(npz_path)
        label_raw = (name.replace('allen_data_results_cascade_', '')
                        .replace('_traces.npz', ''))
        rec_npz_path = npz_path.replace('_traces.npz', '.npz')
        cell_id_to_row = {}
        if os.path.exists(rec_npz_path):
            try:
                jdata = _load_records(rec_npz_path)
                seen = {}
                for entry in jdata:
                    cid = entry['cell_id']
                    if cid not in seen:
                        seen[cid] = len(seen)
                cell_id_to_row = seen
            except Exception:
                pass
        lookup[label_raw] = (npz_path, cell_id_to_row)
    return lookup


def _build_fmcsi_records_lookup(data_dir):

    lookup = {}
    for fpath in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_fmcsi_*.npz')):
        name = os.path.basename(fpath)
        if '_traces.npz' in name:
            continue
        orig = name.replace('allen_data_results_fmcsi_', '').replace('.npz', '')
        group = os.path.join(data_dir, f'allen_data_results_{orig}.npz')
        if not os.path.exists(group) or \
                os.path.getmtime(fpath) > os.path.getmtime(group):
            lookup[orig] = fpath
    return lookup


def _build_fmcsi_traces_lookup(data_dir):

    lookup = {}
    for fpath in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_fmcsi_*_traces.npz')):
        name = os.path.basename(fpath)
        orig = name.replace('allen_data_results_fmcsi_', '').replace('_traces.npz', '')
        group = os.path.join(data_dir, f'allen_data_results_{orig}_traces.npz')
        if not os.path.exists(group) or \
                os.path.getmtime(fpath) > os.path.getmtime(group):
            lookup[orig] = fpath
    return lookup


def _peaks_from_prob(prob, fs):
    if prob is None or len(prob) == 0 or prob.max() <= 0:
        return np.array([])
    thresh   = max(0.05, float(np.percentile(prob[prob > 0], 90)))
    min_dist = max(1, int(0.05 * fs))
    idx, _   = find_peaks(prob, height=thresh, distance=min_dist)
    return idx / fs


def _best_window(raw_trace, fs, true_spk, det_spikes_list,
                 window=60.0, target_spikes=20, spike_std=8.0):
    block_frames = int(window * fs)
    n_frames     = len(raw_trace)
    results = []
    t = 0
    while t + block_frames <= n_frames:
        t0_s, t1_s = t / fs, t / fs + window
        seg         = raw_trace[t: t + block_frames]
        trace_kurt  = float(sci_kurtosis(seg))
        true_win    = true_spk[(true_spk >= t0_s) & (true_spk < t1_s)]
        spike_score = float(
            np.exp(-0.5 * ((len(true_win) - target_spikes) / spike_std) ** 2))
        recall_list = []
        for det_spk in det_spikes_list:
            if len(det_spk) == 0 or len(true_win) == 0:
                continue
            det_win = det_spk[(det_spk >= t0_s - 0.1) & (det_spk < t1_s + 0.1)]
            hits    = sum(1 for ts in true_win if np.any(np.abs(det_win - ts) <= 0.1))
            recall_list.append(hits / len(true_win))
        pred_score = float(np.mean(recall_list)) if recall_list else 0.0
        results.append((t0_s, trace_kurt, spike_score, pred_score))
        t += block_frames
    if not results:
        return 0.0
    if len(results) == 1:
        return results[0][0]
    kurts   = np.array([r[1] for r in results])
    k_range = max(kurts.max() - kurts.min(), 1.0)
    best_score, best_t0 = -np.inf, 0.0
    for t0_s, kurt, spike_score, pred_score in results:
        score = ((kurt - kurts.min()) / k_range + spike_score + pred_score) / 3.0
        if score > best_score:
            best_score, best_t0 = score, t0_s
    return best_t0


def _load_raster_trace_data(data_dir, label_map, file_path_map,
                             cascade_lookup, n_cells=5,
                             window=60.0, min_spikes=15,
                             fmcsi_traces_lookup=None,
                             matlab_data_dir=None):
    all_cells = []
    for label, norm_label in label_map.items():
        file_label = file_path_map.get(label, norm_label)
        fpath = os.path.join(data_dir, f'allen_data_results_{file_label}_traces.npz')
        if not os.path.exists(fpath):
            continue
        try:
            d = np.load(fpath, allow_pickle=True)
            if 'true_spikes' not in d:
                continue

            df = None
            if fmcsi_traces_lookup and file_label in fmcsi_traces_lookup:
                try:
                    df = np.load(fmcsi_traces_lookup[file_label], allow_pickle=True)
                except Exception:
                    df = None

            if df is not None and 'my_probs' in df:
                my_probs = df['my_probs']
            elif 'my_probs' in d:
                my_probs = d['my_probs']
            else:
                continue

            true_spikes_arr = list(d['true_spikes'])

            trad_probs = d['trad_probs'] if 'trad_probs' in d else None
            if (trad_probs is None or np.asarray(trad_probs).ndim < 2
                    or np.asarray(trad_probs).size == 0):
                trad_probs = None
                if matlab_data_dir and os.path.isdir(matlab_data_dir):
                    m_fpath = os.path.join(
                        matlab_data_dir,
                        f'allen_data_results_{file_label}_traces.npz')
                    if os.path.exists(m_fpath):
                        try:
                            dm = np.load(m_fpath, allow_pickle=True)
                            if ('trad_probs' in dm
                                    and np.asarray(dm['trad_probs']).ndim == 2
                                    and np.asarray(dm['trad_probs']).size > 0):
                                trad_probs = dm['trad_probs']
                        except Exception:
                            pass

            if df is not None and 'oasis_probs' in df:
                oasis_probs = df['oasis_probs']
            elif 'oasis_probs' in d:
                oasis_probs = d['oasis_probs']
            else:
                oasis_probs = None
            fs_m = re.search(r'(\d+)[Hh]z', os.path.basename(fpath))
            fs   = float(fs_m.group(1)) if fs_m else 30.0
            raw  = None
            for k in ['dff', 'f_cell', 'noisy_traces', 'raw']:
                if k in d and hasattr(d[k], 'ndim') and d[k].ndim == 2:
                    raw = d[k]; break
            if my_probs.ndim != 2 or raw is None:
                continue

            dc = cas_probs_all = None
            cas_cell_id_to_row = {}
            if file_label in cascade_lookup:
                fpath_cas, cas_cell_id_to_row = cascade_lookup[file_label]
                try:
                    dc = np.load(fpath_cas, allow_pickle=True)
                    if 'cascade_probs' in dc:
                        cas_probs_all = dc['cascade_probs']
                except Exception:
                    dc = None

            def _load_spk(key, probs_arr, idx, src=d):
                if key in src:
                    arr = list(src[key])
                    if idx < len(arr):
                        return np.atleast_1d(np.asarray(arr[idx], dtype=float))
                if probs_arr is not None and probs_arr.ndim == 2 \
                        and idx < probs_arr.shape[0]:
                    return _peaks_from_prob(probs_arr[idx], fs)
                return np.array([])

            fmcsi_src = df if df is not None else d
            n = my_probs.shape[0]
            for i in range(min(n, len(true_spikes_arr))):
                raw_trace = raw[i]
                kurt      = float(sci_kurtosis(raw_trace))
                spk       = np.atleast_1d(np.asarray(true_spikes_arr[i], dtype=float))
                if len(spk) < 3:
                    continue
                my_spk    = _load_spk('my_spikes',    my_probs,    i, src=fmcsi_src)
                trad_spk  = _load_spk('trad_spikes',  trad_probs,  i)
                oasis_spk = _load_spk('oasis_spikes', oasis_probs, i)
                cas_spk   = np.array([])
                cas_row   = cas_cell_id_to_row.get(i, -1)
                if dc is not None and cas_row >= 0 and 'cascade_spikes' in dc:
                    try:
                        arr = dc['cascade_spikes']
                        if cas_row < len(arr):
                            cas_spk = np.atleast_1d(
                                np.asarray(arr[cas_row], dtype=float))
                    except Exception:
                        pass
                if len(cas_spk) == 0 and cas_probs_all is not None \
                        and cas_probs_all.ndim == 2 and cas_row >= 0 \
                        and cas_row < cas_probs_all.shape[0]:
                    cas_spk = _peaks_from_prob(cas_probs_all[cas_row], fs)
                det_list = [s for s in [my_spk, trad_spk, oasis_spk, cas_spk]
                            if len(s) > 0]
                t_start = _best_window(raw_trace, fs, spk, det_list, window=window)
                _f = raw_trace[np.isfinite(raw_trace)]
                _mad = float(np.median(np.abs(np.diff(_f)))) / 0.6745 if len(_f) > 1 else 1e-4
                _snr = (float(np.percentile(_f, 99)) - float(np.percentile(_f, 8))) / (_mad + 1e-9)
                all_cells.append({
                    'label': label, 'cell_idx': i,
                    'true_spikes': spk, 'my_spikes': my_spk,
                    'trad_spikes': trad_spk, 'oasis_spikes': oasis_spk,
                    'cas_spikes': cas_spk, 'raw': raw_trace,
                    'kurtosis': kurt, 'snr': _snr, 'fs': fs, 't_start': t_start,
                })
        except Exception as exc:
            print(f"Warning: could not load {fpath}: {exc}")

    if not all_cells:
        print("Warning: no trace data found for raster/trace plots.")
        return []

    all_cells.sort(key=lambda c: c['kurtosis'])


    def _win_count(c):
        t0, t1 = c['t_start'], c['t_start'] + window
        return int(np.sum((c['true_spikes'] >= t0) & (c['true_spikes'] < t1)))

    good   = [c for c in all_cells if _win_count(c) >= min_spikes]
    pool   = good if len(good) >= n_cells else all_cells
    target_kurts = [2.0, 5.0, 10.0, 50.0, 100.0]
    selected, used_idx = [], set()
    for tk in target_kurts:
        best_i, best_d = None, np.inf
        for j, c in enumerate(pool):
            if j in used_idx:
                continue
            dk = abs(c['kurtosis'] - tk)
            if dk < best_d:
                best_d, best_i = dk, j
        if best_i is not None:
            used_idx.add(best_i)
            selected.append(pool[best_i])
    if len(selected) < n_cells and len(pool) >= n_cells:
        remaining = [pool[j] for j in range(len(pool)) if j not in used_idx]
        extra_idx = np.linspace(0, len(remaining) - 1,
                                n_cells - len(selected), dtype=int)
        selected += [remaining[k] for k in extra_idx]
    return selected




def _plot_combined_raster_trace(ax, cells, window=60.0):

    n = len(cells)
    if n == 0:
        ax.text(0.5, 0.5, 'No trace data', transform=ax.transAxes,
                ha='center', va='center')
        return
    rr = 0.9; th = 2.2; pad = 0.25; gap = 0.7
    cell_h = 5 * rr + pad + th + gap
    method_rows = [
        ('OASIS',        'oasis_spikes', model_colors['OASIS'],     0),
        ('CASCADE',      'cas_spikes',   model_colors['CASCADE'],   1),
        ('CaImAn',    'trad_spikes',  model_colors['MATLAB'], 2),
        ('OMSI',    'my_spikes',    model_colors['fMCSI'], 3),
        ('Ground Truth', 'true_spikes',  '#111111',                 4),
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
            spk   = cell.get(key, np.array([]))
            if spk is None: spk = np.array([])
            spk    = np.atleast_1d(np.asarray(spk, dtype=float))
            in_win = spk[(spk >= t0) & (spk <= t1)] - t0
            if len(in_win) > 0:
                ax.vlines(in_win, y_lo, y_hi, color=color, lw=0.6, alpha=0.9)
            if i == 0:
                ax.text(label_x, y_mid, row_name, va='center', ha='right',
                        color=color if color != '#111111' else 'k')
        trace_y0 = base + 5 * rr + pad
        raw  = cell['raw']; fs = cell['fs']
        t_arr = np.arange(len(raw)) / fs
        mask  = (t_arr >= t0) & (t_arr <= t1)
        t_plot = t_arr[mask] - t0; raw_plot = raw[mask]
        rmin, rmax = np.nanmin(raw_plot), np.nanmax(raw_plot)
        if rmax > rmin:
            raw_norm = (raw_plot - rmin) / (rmax - rmin) * th + trace_y0
        else:
            raw_norm = np.full_like(raw_plot, trace_y0 + th / 2)
        ax.plot(t_plot, raw_norm, color='k', lw=0.7, alpha=0.8)
        if i == 0:
            ax.text(label_x, trace_y0 + th / 2, 'ΔF/F',
                    va='center', ha='right', color='k')
        ax.text(label_x / 2, trace_y0 + th / 2, str(i + 1),
                va='center', ha='center', color='k', fontsize=5.5,
                fontweight='bold')
        ax.text(window + 0.8, base + cell_h / 2 - gap / 2,
                f'SNR={cell["snr"]:.1f}', va='center', ha='left')
        if i < n - 1:
            ax.axhline(base - gap / 2, color='0.75', lw=0.4, ls='--')
    ax.set_xlim(label_x - 0.5, window + 4.5)
    ax.set_ylim(-gap, n * cell_h)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time (s)')


def _plot_violin_metric(ax, alldata, taus, zoom, metric, ylabel):

    positions, violin_data, violin_colors = [], [], []
    x_pos = 0; tau_ticks, tau_labels = [], []
    for tau_val in taus:
        tau_data     = [d for d in alldata if d['tau'] == tau_val
                        and d['zoom'] == zoom]
        group_center = x_pos + (len(_MODEL_ORDER) - 1) / 2.0
        for model_name in _MODEL_ORDER:
            md   = [d for d in tau_data if d['model'] == model_name]
            if md:
                vals = np.array([d.get(metric, np.nan) for d in md])
                vals = vals[~np.isnan(vals)]
                if len(vals) >= 2:
                    positions.append(x_pos)
                    violin_data.append(vals)
                    violin_colors.append(model_colors.get(model_name, 'k'))
            x_pos += 1
        tau_ticks.append(group_center)
        tau_labels.append(f'τ={tau_val} s')
        x_pos += 0.5
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if pn in parts:
                parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(tau_ticks); ax.set_xticklabels(tau_labels)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel(ylabel)
    ax.tick_params(axis='both')


def _plot_f1_violin(ax, alldata, taus, f1_key='f1'):

    positions, violin_data, violin_colors = [], [], []
    x_pos = 0; group_ticks, group_labels = [], []
    for tau_val in taus:
        for zoom in ['High Zoom', 'Low Zoom']:
            subset       = [d for d in alldata if d['tau'] == tau_val
                            and d['zoom'] == zoom]
            group_center = x_pos + (len(_MODEL_ORDER) - 1) / 2.0
            for model_name in _MODEL_ORDER:
                md   = [d for d in subset if d['model'] == model_name]
                if md:
                    vals = np.array([d.get(f1_key, np.nan) for d in md])
                    vals = vals[~np.isnan(vals)]
                    if len(vals) >= 2:
                        positions.append(x_pos)
                        violin_data.append(vals)
                        violin_colors.append(model_colors.get(model_name, 'k'))
                x_pos += 1
            z_abbrev = 'high zoom' if 'High' in zoom else 'low zoom'
            group_ticks.append(group_center)
            group_labels.append(f'τ={tau_val} s\n{z_abbrev}')
            x_pos += 0.5
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if pn in parts:
                parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(group_ticks); ax.set_xticklabels(group_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'$F_\beta$ (strict)' if f1_key in ('f1', 'fbeta')
                  else r'$F_\beta$')
    ax.tick_params(axis='both')


def _plot_strict_vs_window_f1(ax, alldata):

    positions, violin_data, violin_colors = [], [], []
    tick_positions, tick_labels = [], []
    x_pos = 0
    for model_name in _MODEL_ORDER:
        md    = [d for d in alldata if d['model'] == model_name]
        if not md: continue
        color = model_colors.get(model_name, 'k')
        for metric, short in [('fbeta', 'S'), ('fbeta_window', 'W')]:
            vals = np.array([d.get(metric, np.nan) for d in md])
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 2:
                positions.append(x_pos); violin_data.append(vals)
                violin_colors.append(color)
            tick_positions.append(x_pos); tick_labels.append(short)
            x_pos += 1
        x_pos += 0.8
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if pn in parts:
                parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel(r'$F_\beta$')
    ax.set_title(r'strict (S) vs window (W)')
    ax.tick_params(axis='both', labelsize=6)


def _plot_fbeta_violin(ax, alldata):

    fb_key = 'fbeta' if USE_STRICT_ACCURACY else 'fbeta_window'
    positions, violin_data, violin_colors, tick_labels = [], [], [], []
    for i, model_name in enumerate(_MODEL_ORDER):
        md   = [d for d in alldata if d['model'] == model_name]
        vals = np.array([d.get(fb_key, np.nan) for d in md])
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            positions.append(i); violin_data.append(vals)
            violin_colors.append(model_colors.get(model_name, 'k'))
            tick_labels.append('CaImAn' if model_name == 'MATLAB' else ('OMSI' if model_name == 'fMCSI' else model_name))
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if pn in parts:
                parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=15, ha='right')
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel(r'$F_\beta$ score')
    ax.tick_params(axis='both', labelsize=6)


def _plot_cosmic_violin(ax, alldata):

    positions, violin_data, violin_colors, tick_labels = [], [], [], []
    for i, model_name in enumerate(_MODEL_ORDER):
        md   = [d for d in alldata if d['model'] == model_name]
        vals = np.array([d.get('cosmic', np.nan) for d in md])
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            positions.append(i); violin_data.append(vals)
            violin_colors.append(model_colors.get(model_name, 'k'))
            tick_labels.append('CaImAn' if model_name == 'MATLAB' else ('OMSI' if model_name == 'fMCSI' else model_name))
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if pn in parts:
                parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=15, ha='right')
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel('CosMIC score')
    ax.tick_params(axis='both', labelsize=6)


def _recompute_all_metrics_from_traces(alldata, data_dir, fmcsi_traces_lookup,
                                        matlab_data_dir=None):

    def _from_probs(true_spikes, true_events, probs, fs, sigma):
        spikes, _ = OMSI.detect_spikes_from_probs(probs, fs, sigma=sigma)
        prec,   rec,   f1   = OMSI.compute_accuracy_strict(
            true_spikes, spikes, tolerance=0.1)
        prec_w, rec_w, f1_w = compute_accuracy_window(
            true_spikes, spikes, tolerance=0.1)
        prec_e, rec_e, f1_e = compute_accuracy_window(
            true_events, spikes, tolerance=0.1)
        cosmic = OMSI.helpers.compute_cosmic(true_spikes, spikes, fs)
        return prec, rec, f1, prec_w, rec_w, f1_w, prec_e, rec_e, f1_e, cosmic

    def _from_spikes(true_spikes, true_events, spikes, fs):
        prec,   rec,   f1   = OMSI.compute_accuracy_strict(
            true_spikes, spikes, tolerance=0.1)
        prec_w, rec_w, f1_w = compute_accuracy_window(
            true_spikes, spikes, tolerance=0.1)
        prec_e, rec_e, f1_e = compute_accuracy_window(
            true_events, spikes, tolerance=0.1)
        cosmic = OMSI.helpers.compute_cosmic(true_spikes, spikes, fs)
        return prec, rec, f1, prec_w, rec_w, f1_w, prec_e, rec_e, f1_e, cosmic

    def _apply(recs, prec, rec, f1, prec_w, rec_w, f1_w,
               prec_e, rec_e, f1_e, cosmic):
        for i, r in enumerate(recs):
            if i >= len(cosmic):
                break
            r['precision']       = float(prec[i]);   r['recall']       = float(rec[i])
            r['f1']              = float(f1[i])
            r['precision_window']= float(prec_w[i]); r['recall_window']= float(rec_w[i])
            r['f1_window']       = float(f1_w[i])
            r['precision_event'] = float(prec_e[i]); r['recall_event'] = float(rec_e[i])
            r['f1_event']        = float(f1_e[i])
            r['cosmic']          = float(cosmic[i])

    updated = {m: 0 for m in _MODEL_ORDER}

    for fpath in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_*_traces.npz')):
        basename = os.path.basename(fpath)
        if 'allen_data_results_fmcsi_'   in basename: continue
        if 'allen_data_results_cascade_' in basename: continue
        orig  = basename.replace('allen_data_results_', '').replace('_traces.npz', '')
        label = clean_label(normalize_label(orig))
        try:
            d = np.load(fpath, allow_pickle=True)
        except Exception as exc:
            print(f"  Warning: could not load {fpath}: {exc}"); continue
        if 'true_spikes' not in d:
            continue
        true_spikes = list(d['true_spikes'])
        fs  = float(d['fs'])  if 'fs'  in d else float(
            re.search(r'(\d+)[Hh]z', basename).group(1)
            if re.search(r'(\d+)[Hh]z', basename) else 30)
        tau = float(d['tau']) if 'tau' in d else 1.2
        true_events = [OMSI.helpers.make_event_ground_truth(sp, tau)
                       for sp in true_spikes]

        df = None
        if orig in fmcsi_traces_lookup:
            try:
                df = np.load(fmcsi_traces_lookup[orig], allow_pickle=True)
            except Exception:
                df = None

        src_my = df if (df is not None and 'my_probs' in df) else d
        if 'my_probs' in src_my:
            recs = [r for r in alldata
                    if r.get('model') == 'fMCSI' and r.get('label') == label]
            if recs:
                rest = _from_probs(
                    true_spikes, true_events, src_my['my_probs'], fs, sigma=1.5)
                _apply(recs, *rest)
                updated['fMCSI'] += len(recs)

        src_oa = df if (df is not None and 'oasis_probs' in df) else d
        if 'oasis_probs' in src_oa:
            recs = [r for r in alldata
                    if r.get('model') == 'OASIS' and r.get('label') == label]
            if recs:
                rest = _from_probs(
                    true_spikes, true_events, src_oa['oasis_probs'], fs, sigma=0.5)
                _apply(recs, *rest)
                updated['OASIS'] += len(recs)

        if ('trad_probs' in d
                and np.asarray(d['trad_probs']).ndim == 2
                and np.asarray(d['trad_probs']).size > 0):
            recs = [r for r in alldata
                    if r.get('model') == 'MATLAB' and r.get('label') == label]
            if recs:
                rest = _from_probs(
                    true_spikes, true_events, d['trad_probs'], fs, sigma=1.5)
                _apply(recs, *rest)
                updated['MATLAB'] += len(recs)

    for orig, fmcsi_path in fmcsi_traces_lookup.items():
        group_fpath = os.path.join(
            data_dir, f'allen_data_results_{orig}_traces.npz')
        if os.path.exists(group_fpath):
            continue
        label = clean_label(normalize_label(orig))
        try:
            df = np.load(fmcsi_path, allow_pickle=True)
        except Exception as exc:
            print(f"  Warning: could not load {fmcsi_path}: {exc}"); continue
        if 'true_spikes' not in df:
            continue
        true_spikes = list(df['true_spikes'])
        fs  = float(df['fs'])  if 'fs'  in df else 30.0
        tau = float(df['tau']) if 'tau' in df else 1.2
        true_events = [OMSI.helpers.make_event_ground_truth(sp, tau)
                       for sp in true_spikes]

        if 'my_probs' in df:
            recs = [r for r in alldata
                    if r.get('model') == 'fMCSI' and r.get('label') == label]
            if recs:
                rest = _from_probs(
                    true_spikes, true_events, df['my_probs'], fs, sigma=1.5)
                _apply(recs, *rest)
                updated['fMCSI'] += len(recs)

        if 'oasis_probs' in df:
            recs = [r for r in alldata
                    if r.get('model') == 'OASIS' and r.get('label') == label]
            if recs:
                rest = _from_probs(
                    true_spikes, true_events, df['oasis_probs'], fs, sigma=0.5)
                _apply(recs, *rest)
                updated['OASIS'] += len(recs)

    cascade_lookup = _build_cascade_lookup(data_dir)
    for label_raw, (cas_fpath, _) in cascade_lookup.items():
        label = clean_label(normalize_label(label_raw))
        try:
            dc = np.load(cas_fpath, allow_pickle=True)
        except Exception as exc:
            print(f"  Warning: could not load {cas_fpath}: {exc}"); continue
        if 'true_spikes' not in dc or 'cascade_spikes' not in dc:
            continue
        true_spikes = list(dc['true_spikes'])
        fs  = float(dc['fs'])  if 'fs'  in dc else 30.0
        tau = float(dc['tau']) if 'tau' in dc else 1.2
        true_events    = [OMSI.helpers.make_event_ground_truth(sp, tau)
                          for sp in true_spikes]
        cascade_spikes = list(dc['cascade_spikes'])

        recs = [r for r in alldata
                if r.get('model') == 'CASCADE' and r.get('label') == label]
        if recs:
            rest = _from_spikes(true_spikes, true_events, cascade_spikes, fs)
            _apply(recs, *rest)
            updated['CASCADE'] += len(recs)

    if matlab_data_dir and os.path.isdir(matlab_data_dir):
        for fpath in _glob.glob(
                os.path.join(matlab_data_dir, 'allen_data_results_*_traces.npz')):
            basename = os.path.basename(fpath)
            if 'allen_data_results_fmcsi_'   in basename: continue
            if 'allen_data_results_cascade_' in basename: continue
            orig  = basename.replace('allen_data_results_', '').replace('_traces.npz', '')
            label = clean_label(normalize_label(orig))
            try:
                dm = np.load(fpath, allow_pickle=True)
            except Exception as exc:
                print(f"  Warning: could not load {fpath}: {exc}"); continue
            if 'trad_probs' not in dm: continue
            trad_arr = np.asarray(dm['trad_probs'])
            if not (trad_arr.ndim == 2 and trad_arr.size > 0): continue
            if 'true_spikes' not in dm: continue
            true_spikes = list(dm['true_spikes'])
            fs_m = re.search(r'(\d+)[Hh]z', basename)
            fs   = float(dm['fs'])  if 'fs'  in dm else (float(fs_m.group(1)) if fs_m else 30.0)
            tau  = float(dm['tau']) if 'tau' in dm else 1.2
            true_events = [OMSI.helpers.make_event_ground_truth(sp, tau)
                           for sp in true_spikes]
            recs = [r for r in alldata
                    if r.get('model') == 'MATLAB' and r.get('label') == label]
            if recs:
                rest = _from_probs(true_spikes, true_events, dm['trad_probs'], fs, sigma=1.5)
                _apply(recs, *rest)
                updated['MATLAB'] += len(recs)

    any_updated = False
    for model in _MODEL_ORDER:
        n = updated[model]
        if n:
            print(f"  {model}: recomputed metrics for {n} cells.")
            any_updated = True
    if not any_updated:
        print("  No trace data found -- metrics unchanged from saved values.")


def plot_figure(data_dir, matlab_data_dir=_MATLAB_DATA_DIR):

    alldata       = []
    label_map     = {}
    file_path_map = {}

    fmcsi_records_lookup = _build_fmcsi_records_lookup(data_dir)
    fmcsi_traces_lookup  = _build_fmcsi_traces_lookup(data_dir)

    fmcsi_data_cache = {}
    for orig_basename, fmcsi_path in fmcsi_records_lookup.items():
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            print(f"Skipping excluded dataset: {orig_basename}")
            continue
        try:
            fmcsi_data_cache[orig_basename] = _load_records(fmcsi_path)
        except Exception as exc:
            print(f"Warning: could not load {fmcsi_path}: {exc}")
            fmcsi_data_cache[orig_basename] = []
    if fmcsi_data_cache:
        print(f"Found {len(fmcsi_data_cache)} newer fMCSI records file(s); "
              f"they will override group-file data for models they contain.")

    for fpath in _glob.glob(os.path.join(data_dir, 'allen_data_results_*.npz')):
        basename = os.path.basename(fpath)
        is_cascade = 'allen_data_results_cascade_' in fpath

        if 'allen_data_results_fmcsi_' in fpath:
            continue
        try:
            data = _load_records(fpath)
        except Exception as exc:
            print(f"Warning: could not load {fpath}: {exc}")
            continue
        orig_basename = (basename
                         .replace('allen_data_results_cascade_', '')
                         .replace('allen_data_results_', '')
                         .replace('.npz', ''))
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            print(f"Skipping excluded dataset: {orig_basename}")
            continue
        norm_label = normalize_label(orig_basename)
        label      = clean_label(norm_label)
        label_map[label]     = norm_label
        file_path_map.setdefault(label, orig_basename)
        geno = get_genotype(orig_basename)

        if not is_cascade and orig_basename in fmcsi_data_cache:
            fmcsi_models = {r.get('model') for r in fmcsi_data_cache[orig_basename]}
            data = [d for d in data if d.get('model') not in fmcsi_models]
        data = [d for d in data if d.get('model') != 'MATLAB']
        for d in data:
            d['label']    = label
            d['genotype'] = geno
        alldata.extend(data)

    if matlab_data_dir and os.path.isdir(matlab_data_dir):
        for fpath in _glob.glob(
                os.path.join(matlab_data_dir, 'allen_data_results_*.npz')):
            basename   = os.path.basename(fpath)
            if 'allen_data_results_fmcsi_'   in fpath: continue
            if 'allen_data_results_cascade_' in fpath: continue
            try:
                mdata = _load_records(fpath)
            except Exception as exc:
                print(f"Warning: could not load {fpath}: {exc}")
                continue
            orig_basename = (basename
                             .replace('allen_data_results_', '')
                             .replace('.npz', ''))
            if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
                print(f"Skipping excluded dataset: {orig_basename}")
                continue
            mdata = [d for d in mdata if d.get('model') == 'MATLAB']
            if not mdata:
                continue
            norm_label = normalize_label(orig_basename)
            label      = clean_label(norm_label)
            label_map.setdefault(label, norm_label)
            file_path_map.setdefault(label, orig_basename)
            geno = get_genotype(orig_basename)
            for d in mdata:
                d['label']    = label
                d['genotype'] = geno
            alldata.extend(mdata)

    for orig_basename, fmcsi_recs in fmcsi_data_cache.items():
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            continue
        norm_label = normalize_label(orig_basename)
        label      = clean_label(norm_label)
        label_map.setdefault(label, norm_label)
        file_path_map.setdefault(label, orig_basename)
        geno = get_genotype(orig_basename)
        for d in fmcsi_recs:
            d['label']    = label
            d['genotype'] = geno
        alldata.extend(fmcsi_recs)

    if not alldata:
        print("No data found in data_dir. Run with --mode test first.")
        return

    print("Recomputing all metrics from traces (CosMIC, precision, recall, F_beta)...")
    _recompute_all_metrics_from_traces(alldata, data_dir, fmcsi_traces_lookup,
                                        matlab_data_dir=matlab_data_dir)

    for d in alldata:
        d['zoom'] = get_zoom_for_label(d['label'])

    for d in alldata:
        d['fbeta']        = _fbeta(d.get('precision',        0.0),
                                   d.get('recall',           0.0))
        d['fbeta_window'] = _fbeta(d.get('precision_window', 0.0),
                                   d.get('recall_window',    0.0))

    n_unique = len(set((d['label'], d['cell_id']) for d in alldata))
    print(f"Loaded {len(alldata)} records, {n_unique} unique cells.")

    cascade_lookup = _build_cascade_lookup(data_dir)

    print("Loading example cells for raster/trace panel...")
    example_cells = _load_raster_trace_data(
        data_dir, label_map, file_path_map, cascade_lookup,
        n_cells=5, window=60.0, min_spikes=15,
        fmcsi_traces_lookup=fmcsi_traces_lookup,
        matlab_data_dir=matlab_data_dir)

    taus    = sorted(set(d['tau'] for d in alldata))
    f1_key  = 'fbeta'          if USE_STRICT_ACCURACY else 'fbeta_window'
    prec_key = 'precision'     if USE_STRICT_ACCURACY else 'precision_window'
    rec_key  = 'recall'        if USE_STRICT_ACCURACY else 'recall_window'



    fig = plt.figure(figsize=(10, 7.5), dpi=300)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.52, wspace=0.44,
                            height_ratios=[1, 1, 1])

    ax_rt = fig.add_subplot(gs[0:2, 0:2])
    _plot_combined_raster_trace(ax_rt, example_cells, window=60.0)

    ax_hz_p = fig.add_subplot(gs[0, 2])
    ax_lz_p = fig.add_subplot(gs[0, 3])
    _plot_violin_metric(ax_hz_p, alldata, taus, 'High Zoom', prec_key, 'Precision')
    ax_hz_p.set_title('high zoom')
    _plot_violin_metric(ax_lz_p, alldata, taus, 'Low Zoom', prec_key, 'Precision')
    ax_lz_p.set_title('low zoom')

    ax_hz_r = fig.add_subplot(gs[1, 2])
    ax_lz_r = fig.add_subplot(gs[1, 3])
    _plot_violin_metric(ax_hz_r, alldata, taus, 'High Zoom', rec_key, 'Recall')
    ax_hz_r.set_title('high zoom')
    _plot_violin_metric(ax_lz_r, alldata, taus, 'Low Zoom', rec_key, 'Recall')
    ax_lz_r.set_title('low zoom')

    ax_fbeta = fig.add_subplot(gs[2, 0])
    _plot_fbeta_violin(ax_fbeta, alldata)

    ax_cosmic = fig.add_subplot(gs[2, 1])
    _plot_cosmic_violin(ax_cosmic, alldata)

    ax_f1 = fig.add_subplot(gs[2, 2:4])
    _plot_f1_violin(ax_f1, alldata, taus, f1_key=f1_key)

    legend_handles = [
        plt.Line2D([0], [0], color=model_colors['fMCSI'],   marker='.', linestyle='-', label='OMSI'),
        plt.Line2D([0], [0], color=model_colors['MATLAB'],  marker='.', linestyle='-', label='CaImAn'),
        plt.Line2D([0], [0], color=model_colors['OASIS'],   marker='.', linestyle='-', label='OASIS'),
        plt.Line2D([0], [0], color=model_colors['CASCADE'], marker='.', linestyle='-', label='CASCADE'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    plt.tight_layout()
    out_svg = os.path.join(data_dir, 'allen_combined_figure.svg')
    out_png = os.path.join(data_dir, 'allen_combined_figure.png')
    plt.savefig(out_svg, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    print(f"Saved -> {out_png}")
    plt.close()


def print_stats(data_dir=_DEFAULT_DATA_DIR, matlab_data_dir=_MATLAB_DATA_DIR):

    fmcsi_records_lookup = _build_fmcsi_records_lookup(data_dir)
    fmcsi_data_cache = {}
    for orig_basename, fmcsi_path in fmcsi_records_lookup.items():
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            continue
        try:
            fmcsi_data_cache[orig_basename] = _load_records(fmcsi_path)
        except Exception as exc:
            print(f"Warning: could not load {fmcsi_path}: {exc}")
            fmcsi_data_cache[orig_basename] = []

    alldata = []
    for fpath in _glob.glob(os.path.join(data_dir, 'allen_data_results_*.npz')):
        if 'allen_data_results_fmcsi_' in fpath:
            continue
        try:
            data = _load_records(fpath)
        except Exception as exc:
            print(f"Warning: could not load {fpath}: {exc}")
            continue
        orig_basename = (os.path.basename(fpath)
                         .replace('allen_data_results_cascade_', '')
                         .replace('allen_data_results_', '')
                         .replace('.npz', ''))
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            continue
        is_cascade = 'allen_data_results_cascade_' in fpath
        if not is_cascade and orig_basename in fmcsi_data_cache:
            fmcsi_models = {r.get('model') for r in fmcsi_data_cache[orig_basename]}
            data = [d for d in data if d.get('model') not in fmcsi_models]
        data = [d for d in data if d.get('model') != 'MATLAB']
        geno = get_genotype(orig_basename)
        label = clean_label(normalize_label(orig_basename))
        for d in data:
            d['label']    = label
            d['genotype'] = geno
            d['zoom']     = get_zoom_for_label(d['label'])
            d['fbeta']        = _fbeta(d.get('precision',        0.0), d.get('recall',        0.0))
            d['fbeta_window'] = _fbeta(d.get('precision_window', 0.0), d.get('recall_window', 0.0))
        alldata.extend(data)

    if matlab_data_dir and os.path.isdir(matlab_data_dir):
        for fpath in _glob.glob(
                os.path.join(matlab_data_dir, 'allen_data_results_*.npz')):
            basename = os.path.basename(fpath)
            if 'allen_data_results_fmcsi_'   in fpath: continue
            if 'allen_data_results_cascade_' in fpath: continue
            try:
                mdata = _load_records(fpath)
            except Exception as exc:
                print(f"Warning: could not load {fpath}: {exc}")
                continue
            orig_basename = (basename
                             .replace('allen_data_results_', '')
                             .replace('.npz', ''))
            if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
                continue
            mdata = [d for d in mdata if d.get('model') == 'MATLAB']
            if not mdata:
                continue
            geno  = get_genotype(orig_basename)
            label = clean_label(normalize_label(orig_basename))
            for d in mdata:
                d['label']    = label
                d['genotype'] = geno
                d['zoom']     = get_zoom_for_label(label)
                d['fbeta']        = _fbeta(d.get('precision',        0.0), d.get('recall',        0.0))
                d['fbeta_window'] = _fbeta(d.get('precision_window', 0.0), d.get('recall_window', 0.0))
            alldata.extend(mdata)

    for orig_basename, fmcsi_recs in fmcsi_data_cache.items():
        if any(orig_basename.startswith(ds) for ds in _EXCLUDED_DATASETS):
            continue
        geno  = get_genotype(orig_basename)
        label = clean_label(normalize_label(orig_basename))
        for d in fmcsi_recs:
            d['label']    = label
            d['genotype'] = geno
            d['zoom']     = get_zoom_for_label(label)
            d['fbeta']        = _fbeta(d.get('precision',        0.0), d.get('recall',        0.0))
            d['fbeta_window'] = _fbeta(d.get('precision_window', 0.0), d.get('recall_window', 0.0))
        alldata.extend(fmcsi_recs)

    if not alldata:
        print("No data found. Run --mode test first.")
        return

    print('\n' + '='*80)
    print('FIGURE 3 STATISTICS')
    print('='*80)

    print(f'\n{"Method":<12}  {"F_beta strict med":>18}  {"strict IQR":>10}  '
          f'{"F_beta window med":>18}  {"window IQR":>10}  '
          f'{"CosMIC med":>11}  {"CosMIC IQR":>10}')
    print('-'*98)
    for model_name in _MODEL_ORDER:
        md = [d for d in alldata if d['model'] == model_name]
        if not md:
            continue
        fb_s   = np.array([d['fbeta']        for d in md], dtype=float)
        fb_w   = np.array([d['fbeta_window'] for d in md], dtype=float)
        cosmic = np.array([d.get('cosmic', np.nan) for d in md], dtype=float)
        fs_med = np.nanmedian(fb_s);   fs_iqr = np.subtract(*np.nanpercentile(fb_s,   [75, 25]))
        fw_med = np.nanmedian(fb_w);   fw_iqr = np.subtract(*np.nanpercentile(fb_w,   [75, 25]))
        co_med = np.nanmedian(cosmic); co_iqr = np.subtract(*np.nanpercentile(cosmic, [75, 25]))
        print(f'{model_name:<12}  {fs_med:>18.3f}  {fs_iqr:>10.3f}  '
              f'{fw_med:>18.3f}  {fw_iqr:>10.3f}  '
              f'{co_med:>11.3f}  {co_iqr:>10.3f}')



def main():

    parser = argparse.ArgumentParser(
        description='Figure 3 -- Allen data benchmark'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'fmcsi', 'plot', 'print'],
                        help='test: run all inference; fmcsi: re-run fMCSI only; '
                             'plot: make figure; print: print stats')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for output data/figures')
    parser.add_argument('--allen-data-dir', default='/home/dylan/Fast2/spike_deconv/allen_results/raw_data',
                        help='Path to raw Allen H5 files (required for test mode)')
    parser.add_argument('--no-matlab', action='store_true',
                        help='Skip traditional MCMC (Matlab) in test mode')
    parser.add_argument('--matlab-data-dir', default=_MATLAB_DATA_DIR,
                        help='Directory containing CaImAn MCMC (MATLAB) results '
                             '(used in plot/print modes)')
    args = parser.parse_args()

    if args.mode == 'test':
        if not args.allen_data_dir:
            parser.error('--allen-data-dir is required for test mode')
        test_figure(
            data_dir=args.data_dir,
            allen_data_dir=args.allen_data_dir,
            run_matlab=not args.no_matlab,
        )
    elif args.mode == 'fmcsi':
        if not args.allen_data_dir:
            parser.error('--allen-data-dir is required for fmcsi mode')
        test_fmcsi(
            data_dir=args.data_dir,
            allen_data_dir=args.allen_data_dir,
        )
    elif args.mode == 'plot':
        plot_figure(data_dir=args.data_dir, matlab_data_dir=args.matlab_data_dir)
    else:
        print_stats(data_dir=args.data_dir, matlab_data_dir=args.matlab_data_dir)


if __name__ == '__main__':

    main()
