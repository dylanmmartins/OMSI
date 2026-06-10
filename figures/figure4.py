#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full CASCADE ground-truth benchmark.

Written DMM, March 2026
"""

import argparse
import os
import subprocess
import sys
import time
from itertools import groupby

import numpy as np
import scipy.io
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Patch

import OMSI
import OMSI.helpers as helpers
from run_pnev_MCMC import run_matlab_pnevMCMC
from oasis.functions import deconvolve as oasis_deconv

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig4')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

BETA               = 0.5
KURTOSIS_THRESHOLD = 0.5
TAU_RISE           = 0.05


def _dff_kurtosis(fluo):
    """Excess kurtosis on a simple dF/F (8th-percentile baseline normalization)."""
    f = np.asarray(fluo, dtype=np.float64)
    b = float(np.percentile(f, 8))
    if abs(b) < 1.0:
        b = 1.0
    dff = (f - b) / abs(b)
    m = np.mean(dff)
    s = np.std(dff)
    if s < 1e-9:
        return 0.0
    return float(np.mean(((dff - m) / s) ** 4) - 3.0)

_METHODS = {
    'fmcsi':       {'label': 'OMSI',   'color': '#4C72B0'},
    'oasis':       {'label': 'OASIS',   'color': '#55A868'},
    'matlab':      {'label': 'CaImAn',  'color': '#DD8452'},
    'cascade_loo': {'label': 'CASCADE', 'color': '#8172B3'},
}
_METHOD_ORDER  = ['fmcsi', 'matlab', 'oasis', 'cascade_loo']
_TRACE_METHODS = ['fmcsi', 'oasis', 'matlab', 'cascade_loo']

_CASCADE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'run_cascade_subprocess.py')

_CASCADE_MODELS = [
    (7.5,  'Global_EXC_7.5Hz_smoothing200ms'),
    (30.0, 'Global_EXC_30Hz_smoothing50ms_causalkernel'),
]

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

def _cascade_model_for_fs(fs):
    """Return the pretrained CASCADE model name closest to the given frame rate."""
    return min(_CASCADE_MODELS, key=lambda x: abs(x[0] - fs))[1]

_SENSOR_ORDER = [
    'GCaMP6f', 'GCaMP6s', 'GCaMP8f', 'GCaMP8m',
    'GCaMP5k', 'OGB1', 'Cal520', 'jGECO', 'XCaMP', 'R-CaMP', 'jRCaMP', 'Other',
]
_EXCLUDED_SENSORS = {'Other', 'Cal520'}

_EXCLUDED_DATASETS = {'DS29-GCaMP7f-m-V1', 'DS32-GCaMP8s-m-V1', 'DS28-XCaMPgf-m-V1'}

_DS_TAU = {
    'DS01': 0.6, 'DS02': 0.6, 'DS03': 0.6, 'DS04': 0.6, 'DS05': 0.6,
    'DS06': 0.5, 'DS07': 0.5, 'DS08': 0.5, 'DS09': 0.5, 'DS10': 0.5,
    'DS11': 0.5, 'DS12': 1.2, 'DS13': 1.2, 'DS14': 1.2, 'DS15': 1.2,
    'DS16': 1.2, 'DS17': 1.0, 'DS18': 0.4, 'DS19': 0.4, 'DS20': 0.7,
    'DS21': 0.5, 'DS22': 0.6, 'DS23': 0.6, 'DS24': 0.5, 'DS25': 0.5,
    'DS26': 0.5, 'DS27': 0.5, 'DS28': 0.3, 'DS29': 0.5, 'DS30': 0.3,
    'DS31': 0.5, 'DS32': 0.8, 'DS33': 0.5, 'DS40': 1.2, 'DS41': 1.2,
}
_NAME_TAU = [
    ('gcaMP6s', 1.2), ('gcaMP8s', 0.8), ('gcaMP8m', 0.5), ('gcaMP8f', 0.3),
    ('gcaMP6f', 0.5), ('gcaMP7f', 0.5), ('gcaMP5k', 1.0), ('xcaMP',   0.3),
    ('jgeco',   0.5), ('ogb',    0.6),  ('cal520', 0.6),  ('rcamp',   0.4),
    ('jrcamp',  0.7),
]
_DS_EPHYS_RATE  = {'DS05': 40000, 'DS28': 20000, 'DS29': 20000,
                   'DS32': 20000, 'DS33': 20000}
_DEFAULT_EPHYS  = 10000

RASTER_SENSORS = ['GCaMP6s', 'GCaMP6f', 'jGECO', 'GCaMP8m']
RASTER_PINS    = [
    ('DS13-GCaMP6s-m-V1-neuropil-corrected', 0),
    ('DS11-GCaMP6f-m-V1-neuropil-corrected', 2),
    ('DS21-jGECO1a-m-V1',                    2),
    ('DS31-GCaMP8m-m-V1',                    6),
]


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
    p, r = float(precision), float(recall)
    b2 = BETA ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


def _get_fbeta(record):
    return _fbeta(record['precision_window'], record['recall_window'])


def get_tau(ds_folder):
    ds_id = ds_folder[:4].upper()
    if ds_id in _DS_TAU:
        return _DS_TAU[ds_id]
    lower = ds_folder.lower()
    for keyword, tau in _NAME_TAU:
        if keyword.lower() in lower:
            return tau
    return 0.7


def get_ephys_rate(ds_folder):
    return _DS_EPHYS_RATE.get(ds_folder[:4].upper(), _DEFAULT_EPHYS)


def _get_sensor(ds_folder):
    s = ds_folder.lower()
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


def compute_accuracy_window(true_spikes, predicted_spikes, tolerance=0.1):

    precs, recs, f1s = [], [], []
    for t, p in zip(true_spikes, predicted_spikes):
        t = np.asarray(t, dtype=np.float64).flatten()
        p = np.asarray(p, dtype=np.float64).flatten()
        if len(t) == 0 and len(p) == 0:
            precs.append(1.0); recs.append(1.0); f1s.append(1.0); continue
        if len(p) == 0:
            precs.append(0.0); recs.append(0.0); f1s.append(0.0); continue
        if len(t) == 0:
            precs.append(0.0); recs.append(1.0); f1s.append(0.0); continue
        n_tp_rec  = int(np.sum(
            np.any(np.abs(t[:, None] - p[None, :]) <= tolerance, axis=1)))
        n_tp_prec = int(np.sum(
            np.any(np.abs(p[:, None] - t[None, :]) <= tolerance, axis=1)))
        rec  = n_tp_rec  / len(t)
        prec = n_tp_prec / len(p)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return np.array(precs), np.array(recs), np.array(f1s)


def _make_event_gt(spike_times_s, tau_s, event_window=0.250):

    t = np.asarray(spike_times_s, dtype=np.float64)
    if len(t) == 0:
        return t.copy()
    quiet_pre, quiet_post = (1.0, 0.5) if tau_s >= 0.8 else (0.3, 0.3)
    isis       = np.diff(t)
    boundaries = np.concatenate([[0], np.where(isis > event_window)[0] + 1, [len(t)]])
    events     = []
    for j in range(len(boundaries) - 1):
        s, e   = boundaries[j], boundaries[j + 1] - 1
        before = (t[s] - t[s - 1]) if s > 0 else np.inf
        after  = (t[e + 1] - t[e]) if e < len(t) - 1 else np.inf
        if before >= quiet_pre and after >= quiet_post:
            events.append(t[s])
    return np.array(events)


def _build_params(fs, tau):

    g_rise  = float(np.exp(-1.0 / (TAU_RISE * fs)))
    g_decay = float(np.exp(-1.0 / (tau * fs)))
    return {
        'f': fs, 'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1,
        'g':       [g_rise + g_decay, -g_rise * g_decay],
        'defg':    [g_rise, g_decay],
        'TauStd':  [TAU_RISE * fs, tau * fs],
        'lam_scale': 1.0,
    }

def process_dataset(ds_folder, ground_truth_dir, model):

    ds_path    = os.path.join(ground_truth_dir, ds_folder)
    tau        = get_tau(ds_folder)
    ephys_rate = get_ephys_rate(ds_folder)
    mat_files  = sorted(f for f in os.listdir(ds_path) if f.endswith('.mat'))
    if not mat_files:
        return [], None

    cells = []
    n_skipped = 0
    for fname in mat_files:
        try:
            mat = scipy.io.loadmat(os.path.join(ds_path, fname))
            if 'CAttached' not in mat:
                continue
            ca   = mat['CAttached'][0, 0]
            fluo = ca['fluo_mean'].flat[0].flatten().astype(np.float32)
            ft   = ca['fluo_time'].flat[0].flatten()
            spk  = ca['events_AP'].flat[0].flatten()
            valid = np.isfinite(ft)
            if valid.sum() < 2:
                continue
            ft   = ft[valid]
            fluo = fluo[valid]
            dt   = float(np.median(np.diff(ft)))
            if not (np.isfinite(dt) and dt > 0):
                continue
            t0  = float(ft[0])
            dur = float(ft[-1]) - t0
            spk = spk[~np.isnan(spk)]
            spk = spk / ephys_rate - t0
            spk = spk[(spk >= 0) & (spk <= dur)]
            kurt = _dff_kurtosis(fluo)
            if kurt < KURTOSIS_THRESHOLD:
                n_skipped += 1
                continue
            cells.append({'fluo': fluo, 'spk': spk,
                          'fs': 1.0 / dt, 'fname': fname, 'kurt': float(kurt)})
        except Exception as exc:
            print(f"    Warning — skipping {fname}: {exc}")

    print(f"  {ds_folder}: {len(cells)} cells pass kurtosis filter "
          f"({n_skipped} skipped), tau={tau}s")
    if not cells:
        return [], None

    fs = float(np.median([c['fs'] for c in cells]))
    if not (np.isfinite(fs) and fs > 0):
        print(f"  Could not determine valid fs for {ds_folder} — skipping.")
        return [], None

    n_cells     = len(cells)
    true_spikes = [c['spk'][np.isfinite(c['spk'])] for c in cells]
    print(f"  Running {model} on {n_cells} cells at {fs:.1f} Hz ...")
    t0_bench = time.time()

    probs_list  = []
    spikes_list = []

    if model == 'fmcsi':
        params = _build_params(fs, tau)
        for cell in cells:
            dff_1 = cell['fluo'][np.newaxis, :]
            try:
                od = OMSI.deconv(dff_1, params, benchmark=True)
                probs_list.append(od['optim_prob'][0])
                spk = np.asarray(list(od['optim_spikes'])[0], dtype=np.float64)
                spikes_list.append(spk[np.isfinite(spk)])
            except Exception as exc:
                print(f"    Warning — inference failed for {cell['fname']}: {exc}")
                probs_list.append(np.zeros(len(cell['fluo']), dtype=np.float32))
                spikes_list.append(np.array([], dtype=np.float64))

    elif model == 'matlab':
        for cell in cells:
            dff_1 = cell['fluo'][np.newaxis, :]
            try:
                spks, _, probs, _ = run_matlab_pnevMCMC(
                    dff_1, fs=cell['fs'], tau=tau, n_sweeps=500)
                probs_list.append(probs[0].astype(np.float32))
                spikes_list.append(np.asarray(spks[0], dtype=np.float64))
            except Exception as exc:
                print(f"    Warning — inference failed for {cell['fname']}: {exc}")
                probs_list.append(np.zeros(len(cell['fluo']), dtype=np.float32))
                spikes_list.append(np.array([], dtype=np.float64))

    elif model == 'oasis':
        g_decay = float(np.exp(-1.0 / (tau * fs)))
        for cell in cells:
            fluo  = cell['fluo'].astype(np.float64)
            diff  = np.diff(fluo)
            sigma = max(float(np.median(np.abs(diff)) / (0.6745 * np.sqrt(2))), 1e-9)
            try:
                _, s, _, _, _ = oasis_deconv(fluo, g=(g_decay,), sn=sigma, penalty=1)
                spikes_list.append(_oasis_spikes_from_s(s, sigma, cell['fs']))
                probs_list.append(s.astype(np.float32))
            except Exception as exc:
                print(f"    Warning — inference failed for {cell['fname']}: {exc}")
                probs_list.append(np.zeros(len(cell['fluo']), dtype=np.float32))
                spikes_list.append(np.array([], dtype=np.float64))

    elif model == 'cascade_loo':
        import tempfile
        model_name = _cascade_model_for_fs(fs)
        print(f"  CASCADE model: {model_name}")
        min_len = min(len(c['fluo']) for c in cells)
        dff_2d = np.stack([c['fluo'][:min_len] for c in cells], axis=0).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, 'cascade_in.npz')
            out_path = os.path.join(tmpdir, 'cascade_out.npz')
            np.savez(in_path, dff=dff_2d, fs=np.float32(fs))
            try:
                subprocess.run(
                    ['conda', 'run', '-n', 'cascade', 'python', _CASCADE_SCRIPT,
                     '--mode', 'inference',
                     '--model', model_name,
                     '--input',  in_path,
                     '--output', out_path,
                     '--device', 'gpu'],
                    check=True,
                )
                cas = np.load(out_path, allow_pickle=True)
                probs_2d_cas  = cas['cascade_probs']   # (n_cells, n_frames)
                spikes_cas    = list(cas['cascade_spikes'])
                for i in range(n_cells):
                    probs_list.append(probs_2d_cas[i].astype(np.float32))
                    spikes_list.append(np.asarray(spikes_cas[i], dtype=np.float64))
            except subprocess.CalledProcessError as exc:
                print(f"  WARNING: CASCADE subprocess failed: {exc}")
                for cell in cells:
                    probs_list.append(np.zeros(len(cell['fluo']), dtype=np.float32))
                    spikes_list.append(np.array([], dtype=np.float64))

    elapsed = time.time() - t0_bench
    print(f"  Finished in {elapsed:.1f}s  ({elapsed/n_cells:.2f}s/cell)")

    n_max    = max(len(p) for p in probs_list)
    probs_2d = np.zeros((n_cells, n_max), dtype=np.float32)
    for i, p in enumerate(probs_list):
        probs_2d[i, :len(p)] = p

    true_events = [_make_event_gt(sp, tau) for sp in true_spikes]
    prec_s, rec_s, f1_s = helpers.compute_accuracy_strict(
        true_spikes, spikes_list, tolerance=0.1)
    prec_w, rec_w, f1_w = compute_accuracy_window(
        true_spikes, spikes_list, tolerance=0.1)
    prec_e, rec_e, f1_e = compute_accuracy_window(
        true_events, spikes_list, tolerance=0.1)
    cosmic = helpers.compute_cosmic(true_spikes, spikes_list, fs)

    print(f"  Strict  P={np.mean(prec_s):.3f}  R={np.mean(rec_s):.3f}  "
          f"F1={np.mean(f1_s):.3f}")
    print(f"  Window  P={np.mean(prec_w):.3f}  R={np.mean(rec_w):.3f}  "
          f"F1={np.mean(f1_w):.3f}")
    print(f"  CosMIC  mean={np.mean(cosmic):.3f}")

    records = []
    for i, cell in enumerate(cells):
        records.append({
            'model':            model,
            'dataset':          ds_folder,
            'fname':            cell['fname'],
            'fs':               fs,
            'tau':              tau,
            'kurtosis':         cell['kurt'],
            'n_true_spikes':    int(len(cell['spk'])),
            'f1':               float(f1_s[i]),
            'precision':        float(prec_s[i]),
            'recall':           float(rec_s[i]),
            'f1_window':        float(f1_w[i]),
            'precision_window': float(prec_w[i]),
            'recall_window':    float(rec_w[i]),
            'f1_event':         float(f1_e[i]),
            'precision_event':  float(prec_e[i]),
            'recall_event':     float(rec_e[i]),
            'cosmic':           float(cosmic[i]),
        })

    traces = {'fs': fs, 'tau': tau, 'n_cells': n_cells}
    for i, cell in enumerate(cells):
        traces[f'dff_{i}']         = cell['fluo']
        traces[f'true_spikes_{i}'] = cell['spk']
        traces[f'pred_spikes_{i}'] = spikes_list[i]
        traces[f'pred_probs_{i}']  = probs_list[i].astype(np.float32)
        traces[f'kurtosis_{i}']    = np.float32(cell['kurt'])

    return records, traces


def test_figure(data_dir, ground_truth_dir, methods=None):
    os.makedirs(data_dir, exist_ok=True)
    if methods is None:
        methods = ['fmcsi', 'oasis', 'matlab', 'cascade_loo']

    ds_folders = sorted(
        d for d in os.listdir(ground_truth_dir)
        if os.path.isdir(os.path.join(ground_truth_dir, d))
        and d not in _EXCLUDED_DATASETS
    )
    print(f"Found {len(ds_folders)} dataset folders "
          f"({len(_EXCLUDED_DATASETS)} excluded).\n")

    for model in methods:
        traces_dir = os.path.join(data_dir, f'ground_truth_traces_{model}')
        os.makedirs(traces_dir, exist_ok=True)

        all_records = []
        t_total = time.time()
        print(f"\n{'='*65}")
        print(f"  Method: {model}")
        print(f"{'='*65}")

        for ds_folder in ds_folders:
            print(f"\n{'─'*55}")
            print(f"  Dataset: {ds_folder}")
            records, traces = process_dataset(ds_folder, ground_truth_dir, model)
            if records:
                all_records.extend(records)
            if traces is not None:
                npz_path = os.path.join(traces_dir, f'{ds_folder}_traces.npz')
                np.savez(npz_path, **traces)
                print(f"  Traces -> {npz_path}")

        print(f"\n{'='*65}")
        print(f"  Total elapsed: {(time.time()-t_total)/60:.1f} min")
        print(f"  Total cells evaluated: {len(all_records)}")

        out_path = os.path.join(data_dir, f'ground_truth_results_{model}.npz')
        _save_records(all_records, out_path)
        print(f"  Results -> {out_path}")


def _traces_dir(data_dir, method_key):
    return os.path.join(data_dir, f'ground_truth_traces_{method_key}')


def _load_all(data_dir):

    all_records = {}
    for method_key in _METHOD_ORDER:
        npz_path = os.path.join(data_dir, f'ground_truth_results_{method_key}.npz')
        if not os.path.exists(npz_path):
            print(f'  (skipping {method_key}: {npz_path} not found)')
            continue
        recs = _load_records(npz_path)
        recs = [r for r in recs if r.get('dataset') not in _EXCLUDED_DATASETS]
        for r in recs:
            r['method'] = method_key
        all_records[method_key] = recs
        print(f'  Loaded {len(recs)} records for {method_key}')
    return all_records


def _best_window_raster(raw, fs, true_spk, pred_spks_list,
                         window=30.0, target_spikes=10):
    block = int(window * fs)
    n     = len(raw)
    best_t0, best_score = 0.0, -np.inf
    for t in range(0, n - block + 1, block):
        t0 = t / fs; t1 = t0 + window
        true_win = true_spk[(true_spk >= t0) & (true_spk < t1)]
        n_true   = len(true_win)
        spike_sc = float(np.exp(-0.5 * ((n_true - target_spikes) / 8.0) ** 2))
        recalls  = []
        for pred in pred_spks_list:
            if n_true == 0 or len(pred) == 0: continue
            det  = pred[(pred >= t0 - 0.1) & (pred < t1 + 0.1)]
            hits = sum(1 for ts in true_win if np.any(np.abs(det - ts) <= 0.1))
            recalls.append(hits / n_true)
        rec_sc = float(np.mean(recalls)) if recalls else 0.0
        score  = (spike_sc + rec_sc) / 2.0
        if score > best_score:
            best_score = score; best_t0 = t0
    return best_t0


def _load_raster_cells(data_dir, raster_cells_npz, window=30.0, min_spikes=5):

    _REQUIRED = ['fmcsi', 'oasis', 'matlab']

    try:
        sets = [
            set(f.replace('_traces.npz', '')
                for f in os.listdir(_traces_dir(data_dir, m))
                if f.endswith('_traces.npz'))
            for m in _REQUIRED
            if os.path.isdir(_traces_dir(data_dir, m))
        ]
    except Exception:
        sets = []
    if not sets:
        print("  No trace directories found.")
        return []
    common_ds = sorted(sets[0].intersection(*sets[1:]))

    cas_td       = _traces_dir(data_dir, 'cascade_loo')
    has_cascade  = os.path.isdir(cas_td)

    by_sensor = {s: [] for s in RASTER_SENSORS}
    for ds in common_ds:
        sensor = _get_sensor(ds)
        if sensor not in by_sensor:
            continue
        ref_path = os.path.join(_traces_dir(data_dir, 'fmcsi'),
                                f'{ds}_traces.npz')
        try:
            ref_npz = np.load(ref_path, allow_pickle=False)
        except Exception:
            continue
        n_c = int(ref_npz['n_cells'])
        fs  = float(ref_npz['fs'])
        for ci in range(n_c):
            kurt     = _dff_kurtosis(ref_npz[f'dff_{ci}'])
            true_spk = ref_npz[f'true_spikes_{ci}']
            true_spk = true_spk[np.isfinite(true_spk)]
            if len(true_spk) < min_spikes:
                continue
            pred_by_method = {}
            ok = True
            for m in _REQUIRED:
                try:
                    npz_m = np.load(
                        os.path.join(_traces_dir(data_dir, m),
                                     f'{ds}_traces.npz'),
                        allow_pickle=False)
                    pred_by_method[m] = npz_m[f'pred_spikes_{ci}']
                except Exception:
                    ok = False; break
            if not ok:
                continue

            if has_cascade:
                cas_path = os.path.join(cas_td, f'{ds}_traces.npz')
                try:
                    npz_cas = np.load(cas_path, allow_pickle=False)
                    pred_by_method['cascade_loo'] = npz_cas[f'pred_spikes_{ci}']
                except Exception:
                    pass
            t_start = _best_window_raster(
                ref_npz[f'dff_{ci}'], fs, true_spk,
                list(pred_by_method.values()), window=window)
            n_win = int(np.sum(
                (true_spk >= t_start) & (true_spk < t_start + window)))
            if n_win < min_spikes:
                continue
            by_sensor[sensor].append({
                'ds': ds, 'cell_idx': ci, 'sensor': sensor,
                'kurtosis': kurt, 'fs': fs,
                'raw': ref_npz[f'dff_{ci}'],
                'true_spikes': true_spk,
                'pred_spikes': pred_by_method,
                't_start': t_start,
            })

    selected = []
    for slot_i, sensor in enumerate(RASTER_SENSORS):
        pool = by_sensor[sensor]
        pin  = RASTER_PINS[slot_i] if slot_i < len(RASTER_PINS) else None
        if pin is not None:
            ds_pin, ci_pin = pin
            match = next(
                (c for c in pool if c['ds'] == ds_pin and c['cell_idx'] == ci_pin),
                None)
            if match is None:
                print(f'  WARNING: pinned cell ({ds_pin}, {ci_pin}) not found; '
                      f'falling back to auto.')
                pin = None
            else:
                selected.append(match)
        if pin is None:
            if not pool:
                print(f'  WARNING: no candidate for sensor {sensor}')
                continue
            mean_kurt = np.mean([c['kurtosis'] for c in pool])
            selected.append(
                min(pool, key=lambda c: abs(c['kurtosis'] - mean_kurt)))

    os.makedirs(os.path.dirname(raster_cells_npz), exist_ok=True)
    save = {'n_cells': len(selected)}
    for i, c in enumerate(selected):
        save[f'dff_{i}']         = c['raw'].astype(np.float32)
        save[f'true_spikes_{i}'] = c['true_spikes'].astype(np.float64)
        save[f'fs_{i}']          = np.float32(c['fs'])
        save[f't_start_{i}']     = np.float32(c['t_start'])
        save[f'dataset_{i}']     = np.bytes_(c['ds'].encode())
        save[f'cell_idx_{i}']    = np.int32(c['cell_idx'])
    np.savez(raster_cells_npz, **save)
    print(f'  Saved raster cell info -> {raster_cells_npz}')
    return selected


def _plot_raster(ax, cells, window=60.0):
    n = len(cells)
    if n == 0:
        ax.text(0.5, 0.5, 'No trace data found', transform=ax.transAxes,
                ha='center', va='center')
        return
    rr = 0.9; th = 2.2; pad = 0.25; gap = 0.7
    has_cascade = any('cascade_loo' in c['pred_spikes'] for c in cells)
    bottom_to_top = []
    if has_cascade:
        bottom_to_top.append((
            _METHODS['cascade_loo']['label'], 'cascade_loo',
            _METHODS['cascade_loo']['color']))
    for m in ['oasis', 'matlab', 'fmcsi']:
        bottom_to_top.append(
            (_METHODS[m]['label'], m, _METHODS[m]['color']))
    bottom_to_top.append(('Ground Truth', None, '#111111'))
    n_rows = len(bottom_to_top)
    cell_h = n_rows * rr + pad + th + gap
    label_x = -3.5

    for i, cell in enumerate(cells):
        base = (n - 1 - i) * cell_h
        t0   = cell['t_start']
        for row_i, (row_name, method_key, color) in enumerate(bottom_to_top):
            y_lo  = base + row_i * rr + 0.05
            y_hi  = base + row_i * rr + rr * 0.85
            y_mid = base + row_i * rr + rr * 0.45
            if method_key is None:
                spk = cell['true_spikes']
            else:
                spk = cell['pred_spikes'].get(method_key, np.array([]))
            spk    = np.atleast_1d(np.asarray(spk, dtype=float))
            in_win = spk[(spk >= t0) & (spk <= t0 + window)] - t0
            if len(in_win):
                ax.vlines(in_win, y_lo, y_hi, color=color, lw=0.6, alpha=0.9)
            if i == 0:
                ax.text(label_x, y_mid, row_name, va='center', ha='right',
                        color=color, fontsize=6)
        trace_y0 = base + n_rows * rr + pad
        raw   = cell['raw']; fs = cell['fs']
        t_arr = np.arange(len(raw)) / fs
        mask  = (t_arr >= t0) & (t_arr <= t0 + window)
        t_pl  = t_arr[mask] - t0; r_pl = raw[mask]
        lo, hi = np.nanmin(r_pl), np.nanmax(r_pl)
        r_norm = ((r_pl - lo) / (hi - lo) * th + trace_y0
                  if hi > lo else np.full_like(r_pl, trace_y0 + th / 2))
        ax.plot(t_pl, r_norm, color='k', lw=0.7, alpha=0.8)
        if i == 0:
            ax.text(label_x, trace_y0 + th / 2, 'ΔF/F',
                    va='center', ha='right', color='k', fontsize=6)
        ax.text(label_x / 2, trace_y0 + th / 2, str(i + 1),
                va='center', ha='center', color='k', fontsize=5.5,
                fontweight='bold')
        ax.text(window + 0.8, base + cell_h / 2 - gap / 2,
                f'{cell["ds"].split("-")[0]}\n{cell["sensor"]}\n'
                f'kurt={cell["kurtosis"]:.1f}',
                va='center', ha='left', fontsize=5, linespacing=1.3)
        if i < n - 1:
            ax.axhline(base - gap / 2, color='0.75', lw=0.4, ls='--')
    ax.set_xlim(label_x - 0.5, window + 8)
    ax.set_ylim(-gap, n * cell_h)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    tick_step = 10 if window >= 30 else 5
    ax.set_xticks(np.arange(0, window + 1, tick_step))
    ax.set_xticklabels(
        [f'{int(t)}' for t in np.arange(0, window + 1, tick_step)], fontsize=6)
    ax.set_xlabel('Time (s)', fontsize=6)


def _draw_grouped_violins(ax, all_records, values_fn, ylabel):
    all_flat     = [r for recs in all_records.values() for r in recs]
    present      = sorted(set(_get_sensor(r['dataset']) for r in all_flat)
                          - _EXCLUDED_SENSORS)
    sensor_order = [s for s in _SENSOR_ORDER if s in present]
    sensor_order += [s for s in present if s not in sensor_order]
    n_sensors      = len(sensor_order)
    group_spacing  = 1.4
    n_meth         = len([m for m in _METHOD_ORDER if m in all_records])
    method_offsets = np.linspace(-0.35, 0.35, max(n_meth, 1))
    violin_width   = 0.55 / max(n_meth, 1) * 2
    mi_map = {m: i for i, m in enumerate(
        [m for m in _METHOD_ORDER if m in all_records])}

    for method_key in _METHOD_ORDER:
        if method_key not in all_records:
            continue
        recs  = all_records[method_key]
        color = _METHODS[method_key]['color']
        mi    = mi_map[method_key]
        positions, data = [], []
        for si, sensor in enumerate(sensor_order):
            vals = np.array(
                [values_fn(r) for r in recs
                 if _get_sensor(r['dataset']) == sensor],
                dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            positions.append(si * group_spacing + method_offsets[mi])
            data.append(vals)
        if not data:
            continue
        parts = ax.violinplot(data, positions=positions,
                              showmedians=True, widths=violin_width)
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.72)
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color('k'); parts[key].set_linewidth(0.8)

    group_centers = [si * group_spacing for si in range(n_sensors)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sensor_order, fontsize=6, rotation=35, ha='right')
    ax.set_xlim(-0.7, (n_sensors - 1) * group_spacing + 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)


def plot_figure(data_dir):
    raster_cells_npz = os.path.join(data_dir, 'raster_cells.npz')

    print('Loading results...')
    all_records = _load_all(data_dir)
    if not all_records:
        print(f'No result files found in {data_dir}. Run --mode test first.')
        return
    print(f'Loaded {sum(len(v) for v in all_records.values())} records '
          f'across {len(all_records)} methods.')

    print('  Loading example cells for raster...')
    cells = _load_raster_cells(data_dir, raster_cells_npz, window=30.0)
    print(f'  Found {len(cells)} example cells.')

    fig = plt.figure(figsize=(6.0, 7.0), dpi=200)
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[2.0, 1.0, 1.0],
                            hspace=0.25)

    ax_raster = fig.add_subplot(gs[0])
    _plot_raster(ax_raster, cells, window=30.0)

    ax_fb = fig.add_subplot(gs[1])
    _draw_grouped_violins(ax_fb, all_records, _get_fbeta, r'$F_{\beta}$')

    ax_cs = fig.add_subplot(gs[2])
    _draw_grouped_violins(ax_cs, all_records,
                          lambda r: r.get('cosmic', np.nan), 'CosMIC')

    fig.align_ylabels([ax_fb, ax_cs])

    legend_handles = [
        plt.Line2D([0], [0], color=_METHODS[m]['color'], marker='.', linestyle='-',
                   label=_METHODS[m]['label'])
        for m in ['fmcsi', 'matlab', 'oasis', 'cascade_loo']
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, f'full_ground_truth_comparison.{ext}')
        plt.savefig(out, bbox_inches='tight')
        print(f'  Saved {out}')
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description='Figure 4 — Full CASCADE ground-truth benchmark'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot'],
                        help='test: run inference; plot: make figure')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for output data/figures')
    parser.add_argument('--ground-truth-dir', default='/home/dylan/Documents/Github/Cascade/Ground_truth',
                        help='Path to CASCADE Ground_truth/ folder (test mode)')
    parser.add_argument('--method', nargs='+',
                        choices=['fmcsi', 'matlab', 'oasis', 'cascade_loo'],
                        default=None,
                        help='Method(s) to run in test mode '
                             '(default: all four)')
    args = parser.parse_args()

    if args.mode == 'test':
        if not args.ground_truth_dir:
            parser.error('--ground-truth-dir is required for test mode')
        test_figure(
            data_dir=args.data_dir,
            ground_truth_dir=args.ground_truth_dir,
            methods=args.method,
        )
    else:
        plot_figure(data_dir=args.data_dir)


if __name__ == '__main__':

    main()
