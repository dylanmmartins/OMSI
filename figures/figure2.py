#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling and sensitivity benchmarks

To run inference
    $ python figure2.py --mode test --data-dir /path/to/results

To create figure:
    $ python figure2.py --mode plot --data-dir /path/to/results

Written DMM, March 2026
"""

import argparse
import json
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from oasis.functions import deconvolve

import OMSI
import OMSI.helpers as helpers
from run_pnev_MCMC import run_matlab_pnevMCMC
from simulation_helpers import generate_synthetic_data

_MATLAB_PRECOMPUTED_DIR    = '/home/dylan/Fast2/spike_deconv/sweeping_benchmarks/all_other_methods'
_CASCADE_DURATION_ALT_DIR  = '/home/dylan/Fast2/spike_deconv/sweeping_benchmarks/cascade'

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig2')

mpl.rcParams['axes.spines.top']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

np.random.seed(3)

BETA = 0.5
USE_STRICT_ACCURACY = True  # Hungarian one-to-one matching (compute_accuracy_strict)

COLORS = {
    'fMCSI':        '#4C72B0',
    'CaImAn MCMC':  '#DD8452',
    'OASIS':        '#55A868',
    'CASCADE_GPU':  '#8172B3',
    'CASCADE_CPU':  '#B39DDB',
}

OASIS_SPIKE_DETECTION = 'peaks'


def _oasis_spikes_from_s(s, sigma, fs, height=1.0):
    thresh = height * sigma
    if OASIS_SPIKE_DETECTION == 'peaks':
        min_dist = max(1, int(0.05 * fs))
        peaks, _ = find_peaks(s, height=thresh, distance=min_dist)
        return peaks / fs
    return np.where(s > thresh)[0] / fs


def _run_cascade_inference(dff, fs, data_dir, prefix, device='gpu'):

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, f'{prefix}_input.npz')
    output_path = os.path.join(data_dir, f'{prefix}_output.npz')

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python', script,
         '--mode', 'inference', '--input', input_path, '--output', output_path,
         '--device', device],
        check=True
    )
    result = np.load(output_path, allow_pickle=True)
    cascade_probs  = result['cascade_probs']
    cascade_spikes = list(result['cascade_spikes'])
    cascade_time   = float(result['cascade_time'])
    return cascade_probs, cascade_spikes, cascade_time


def _metrics(true_spk, pred_spk, true_ev, fs_):
    prec,   rec,   f1   = OMSI.compute_accuracy_strict(true_spk, pred_spk, tolerance=0.1)
    prec_w, rec_w, f1_w = helpers.compute_accuracy_window(true_spk, pred_spk)
    prec_e, rec_e, f1_e = helpers.compute_accuracy_window(true_ev,  pred_spk)
    cosmic = helpers.compute_cosmic(true_spk, pred_spk, fs_)
    return (np.mean(prec),   np.mean(rec),   np.mean(f1),
            np.mean(prec_w), np.mean(rec_w), np.mean(f1_w),
            np.mean(prec_e), np.mean(rec_e), np.mean(f1_e),
            np.mean(cosmic))


def _row(exp, model, tau_, fs_, time_, m, sweeps=0, n_cells=None, duration=None,
         mean_kurtosis=None, **extra):
    p, r, f, pw, rw, fw, pe, re_, fe, cos = m
    d = {
        'Experiment': exp, 'Model': model, 'Tau': tau_, 'Fs': fs_,
        'Time': time_, 'Sweeps': sweeps,
        'F1': f, 'Precision': p, 'Recall': r,
        'F1_window': fw, 'Precision_window': pw, 'Recall_window': rw,
        'F1_event': fe, 'Precision_event': pe, 'Recall_event': re_,
        'COSMIC': cos,
    }
    if n_cells is not None:
        d['N_Cells'] = n_cells
    if duration is not None:
        d['Duration'] = duration
    if mean_kurtosis is not None:
        d['Mean_Kurtosis'] = mean_kurtosis
    d.update(extra)
    return d

def _save_records(records, path):
    if not records:
        if not os.path.exists(path):
            np.savez(path)
        return

    new_tbl = _records_to_tbl(records)

    if os.path.exists(path):
        try:
            existing = _load_records(path)
            if existing and 'Model' in existing and 'Model' in new_tbl:
                new_models = set(str(m) for m in new_tbl['Model'])
                mask = np.array([str(m) not in new_models for m in existing['Model']], dtype=bool)
                if mask.sum() > 0:
                    existing_filtered = {k: v[mask] for k, v in existing.items()}
                    combined = _tbl_concat([existing_filtered, new_tbl])
                else:
                    combined = new_tbl
            else:
                combined = new_tbl
        except Exception:
            combined = new_tbl
    else:
        combined = new_tbl

    np.savez(path, **combined)


def _load_records(path):

    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _tbl_len(tbl):
    return len(next(iter(tbl.values()))) if tbl else 0


def _tbl_filter(tbl, col, val):
    mask = tbl[col] == val
    return {k: v[mask] for k, v in tbl.items()}


def _tbl_sort(tbl, col):
    idx = np.argsort(tbl[col].astype(float))
    return {k: v[idx] for k, v in tbl.items()}


def _records_to_tbl(records):
    if not records:
        return {}
    keys = list(dict.fromkeys(k for r in records for k in r))
    out = {}
    for k in keys:
        vals = [r.get(k, None) for r in records]
        if any(isinstance(v, str) for v in vals if v is not None):
            out[k] = np.array([str(v) if v is not None else '' for v in vals], dtype=object)
        else:
            out[k] = np.array([float(v) if v is not None else np.nan for v in vals],
                               dtype=np.float64)
    return out


def _tbl_concat(tbls):
    tbls = [t for t in tbls if t]
    if not tbls:
        return {}
    all_keys = list(dict.fromkeys(k for t in tbls for k in t))
    result = {}
    for k in all_keys:
        parts = []
        for t in tbls:
            if k in t:
                parts.append(t[k])
            else:
                n = _tbl_len(t)
                ref = next((t2[k] for t2 in tbls if k in t2), None)
                if ref is not None and ref.dtype == object:
                    parts.append(np.array([''] * n, dtype=object))
                else:
                    parts.append(np.full(n, np.nan))
        if any(p.dtype == object for p in parts):
            result[k] = np.concatenate([p.astype(object) for p in parts])
        else:
            result[k] = np.concatenate([p.astype(np.float64) for p in parts])
    return result


def _oasis_spikes(dff, fs, tau, n_cells):
    diff = np.diff(dff, axis=1)
    sigmas = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    spikes, calcium = [], []
    for i in range(n_cells):
        g = np.exp(-1 / (fs * tau))
        c, s, _, _, _ = deconvolve(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        spikes.append(_oasis_spikes_from_s(s, sigmas[i], fs))
        calcium.append(c)
    return spikes, calcium


def _trad_mcmc_from_json(json_path):

    if not os.path.exists(json_path):
        print(f'  WARNING: precomputed file not found: {json_path}')
        return []
    with open(json_path) as f:
        data = json.load(f)
    out = []
    for r in data:
        key = 'Model' if 'Model' in r else 'model'
        if r.get(key) == 'Trad MCMC':
            r = dict(r)
            r[key] = 'CaImAn MCMC'
            out.append(r)
    return out


def _load_external_matlab_data():

    d = _MATLAB_PRECOMPUTED_DIR
    return {
        'sweeps':            _trad_mcmc_from_json(os.path.join(d, 'benchmark_sweeps_partial.json')),
        'scalability':       _trad_mcmc_from_json(os.path.join(d, 'benchmark_scalability_partial.json')),
        'params':            _trad_mcmc_from_json(os.path.join(d, 'benchmark_params_partial.json')),
        'noise_sensitivity': [],
        'firing_rate':       _trad_mcmc_from_json(os.path.join(d, 'firing_rate_sensitivity_partial.json')),
        'sweeps_traces_npz':      os.path.join(d, 'benchmark_sweeps_traces.npz'),
        'firing_rate_traces_npz': os.path.join(d, 'firing_rate_sensitivity_traces.npz'),
    }


def benchmark_sweeps(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True, matlab_records=None):

    n_cells     = 50
    duration    = 600
    fs          = 30.0
    tau         = 1.2
    sweeps_list = [10, 50, 100, 250, 500, 1000, 2000, 3000]

    print(f'Generating synthetic data (n_cells={n_cells}, duration={duration}s)...')
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    n_frames   = dff.shape[1]
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

    results   = []
    npz_spikes = {'true': true_spikes}
    npz_calcium = {}

    partial_path = os.path.join(data_dir, 'benchmark_sweeps_partial.npz')

    if run_oasis:
        print('\nRunning OASIS (baseline)...')
        t0 = time.time()
        oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
        time_oasis = time.time() - t0
        results.append(_row('Sweeps', 'OASIS', tau, fs, time_oasis,
                            _metrics(true_spikes, oas_spk, true_events, fs),
                            sweeps=0, n_cells=n_cells, duration=duration,
                            Samples_per_sec=np.nan))
        npz_spikes['oasis'] = oas_spk
        npz_calcium['oasis'] = np.array(oas_cal)
        print(f'  OASIS: {time_oasis:.1f}s  F1={results[-1]["F1"]:.3f}')
        _save_records(results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print(f'\nRunning CASCADE (subprocess, {_dev.upper()}, baseline)...')
            _, cascade_spikes, time_cascade = _run_cascade_inference(
                dff, fs, data_dir, f'bench_sweeps_cascade_baseline_{_dev}', device=_dev
            )
            results.append(_row('Sweeps', _model, tau, fs, time_cascade,
                                _metrics(true_spikes, cascade_spikes, true_events, fs),
                                sweeps=0, n_cells=n_cells, duration=duration,
                                Samples_per_sec=np.nan))
            npz_spikes[f'cascade_{_dev}'] = cascade_spikes
            print(f'  CASCADE ({_dev.upper()}): {time_cascade:.1f}s  F1={results[-1]["F1"]:.3f}')
            _save_records(results, partial_path)

    print(f'\n--- Varying sweeps: {sweeps_list} ---')
    for s in sweeps_list:
        print(f'  sweeps={s}...')
        if run_mine:
            try:
                t0 = time.time()
                burn_in = int(s * 0.25)
                params  = {'f': fs, 'p': 2, 'Nsamples': s - burn_in, 'B': burn_in, 'auto_stop': False}
                res = OMSI.deconv(dff, params=params, benchmark=True)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'fMCSI', tau, fs, elapsed,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['my_method'] = res['optim_spikes']
                print(f'    fMCSI: {elapsed:.1f}s  F1={results[-1]["F1"]:.3f}')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps=s)
                elapsed = time.time() - t0
                sps = (s * n_cells * n_frames) / elapsed
                results.append(_row('Sweeps', 'CaImAn MCMC', tau, fs, elapsed,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=s, n_cells=n_cells, duration=duration,
                                    Samples_per_sec=sps))
                npz_spikes['trad_mcmc'] = trad_spikes
                print(f'    CaImAn MCMC: {elapsed:.1f}s  F1={results[-1]["F1"]:.3f}')
            except Exception as exc:
                print(f'    CaImAn MCMC failed: {exc}')

        _save_records(results, partial_path)

    if run_matlab and matlab_records is not None:
        print(f'\nInjecting {len(matlab_records)} precomputed CaImAn MCMC (sweeps) records...')
        results.extend(matlab_records)
        _save_records(results, partial_path)

    npz_save = {'dff': dff, 'fs': fs, 'tau': tau}
    for k, v in npz_spikes.items():
        npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
    for k, v in npz_calcium.items():
        npz_save[f'calcium_{k}'] = v
    np.savez(os.path.join(data_dir, 'benchmark_sweeps_traces.npz'), **npz_save)
    return


def benchmark_scalability(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                           run_cascade=True, matlab_records=None):

    fs  = 30.0
    tau = 1.2
    cell_counts    = [50, 200, 500, 1000, 2000, 3000]
    fixed_duration = 300.0
    durations      = [300, 1800, 3600, 7200]
    fixed_cells    = 100

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_scalability_partial.npz')

    print('\n--- Cell-count scaling ---')
    for n_cells in cell_counts:
        print(f'  n_cells={n_cells}...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=fixed_duration, tau=tau
            )
        except MemoryError:
            print(f'  Skipping n_cells={n_cells}: MemoryError')
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * n_cells * n_frames) / t_my
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'fMCSI',
                                'N_Cells': n_cells, 'Time': t_my, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    fMCSI: {t_my:.1f}s')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * n_cells * n_frames) / t_trad
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'CaImAn MCMC',
                                'N_Cells': n_cells, 'Time': t_trad, 'Samples_per_sec': sps,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    CaImAn MCMC: {t_trad:.1f}s')
            except Exception as exc:
                print(f'    CaImAn MCMC failed: {exc}')

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Cell_Scaling', 'Model': 'OASIS',
                                'N_Cells': n_cells, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'Duration': fixed_duration, 'Frames': n_frames})
                print(f'    OASIS: {t_oasis:.1f}s')
            except Exception as exc:
                print(f'    OASIS failed: {exc}')

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_cells_{n_cells}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Cell_Scaling', 'Model': _model,
                                    'N_Cells': n_cells, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'Duration': fixed_duration, 'Frames': n_frames})
                    print(f'    CASCADE ({_dev.upper()}): {t_cascade:.1f}s')
                except Exception as exc:
                    print(f'    CASCADE ({_dev.upper()}) failed: {exc}')

        _save_records(results, partial_path)

    print('\n--- Duration scaling ---')
    for dur in durations:
        print(f'  duration={dur}s...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=fixed_cells, fs=fs, duration=dur, tau=tau
            )
        except MemoryError:
            print(f'  Skipping duration={dur}: MemoryError')
            continue
        n_frames = dff.shape[1]

        if run_mine:
            try:
                t0 = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                sps  = (np.mean(res['optim_nsamples']) * fixed_cells * n_frames) / t_my
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'fMCSI',
                                'Duration': dur, 'Time': t_my, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    fMCSI: {t_my:.1f}s')
            except Exception as exc:
                print(f'    fMCSI failed: {exc}')

        if run_matlab and matlab_records is None:
            try:
                t0 = time.time()
                _, _, _, sweeps = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                sps    = (np.mean(sweeps) * fixed_cells * n_frames) / t_trad
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'CaImAn MCMC',
                                'Duration': dur, 'Time': t_trad, 'Samples_per_sec': sps,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    CaImAn MCMC: {t_trad:.1f}s')
            except Exception as exc:
                print(f'    CaImAn MCMC failed: {exc}')

        if run_oasis:
            try:
                t0 = time.time()
                _oasis_spikes(dff, fs, tau, fixed_cells)
                t_oasis = time.time() - t0
                results.append({'Experiment': 'Duration_Scaling', 'Model': 'OASIS',
                                'Duration': dur, 'Time': t_oasis,
                                'Samples_per_sec': np.nan,
                                'N_Cells': fixed_cells, 'Frames': n_frames})
                print(f'    OASIS: {t_oasis:.1f}s')
            except Exception as exc:
                print(f'    OASIS failed: {exc}')

        if run_cascade:
            for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                try:
                    _, _, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_scale_dur_{dur}_{_dev}', device=_dev)
                    results.append({'Experiment': 'Duration_Scaling', 'Model': _model,
                                    'Duration': dur, 'Time': t_cascade,
                                    'Samples_per_sec': np.nan,
                                    'N_Cells': fixed_cells, 'Frames': n_frames})
                    print(f'    CASCADE ({_dev.upper()}): {t_cascade:.1f}s')
                except Exception as exc:
                    print(f'    CASCADE ({_dev.upper()}) failed: {exc}')

        _save_records(results, partial_path)

    if run_matlab and matlab_records is not None:
        print(f'\nInjecting {len(matlab_records)} precomputed CaImAn MCMC (scalability) records...')
        results.extend(matlab_records)
        _save_records(results, partial_path)

    return


def benchmark_params(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                     run_cascade=True, matlab_records=None):
    
    n_cells  = 50
    duration = 300
    tau_values = [0.2, 0.5, 0.8, 1.2, 2.0]
    fixed_fs   = 30.0
    fs_values  = [7.5, 10, 20, 30, 50, 100]
    fixed_tau  = 1.2

    results = []
    partial_path = os.path.join(data_dir, 'benchmark_params_partial.npz')

    print('\n--- Tau sensitivity ---')
    for tau in tau_values:
        print(f'  tau={tau}s...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fixed_fs, duration=duration, tau=tau
            )
            true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fixed_fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'fMCSI', tau, fixed_fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fixed_fs),
                                    sweeps=np.mean(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fixed_fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'CaImAn MCMC', tau, fixed_fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fixed_fs),
                                    sweeps=np.mean(sweeps),
                                    n_cells=n_cells, duration=duration))
                print(f'    CaImAn MCMC: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fixed_fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Tau_Sensitivity', 'OASIS', tau, fixed_fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fixed_fs),
                                    n_cells=n_cells, duration=duration))
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fixed_fs, data_dir, f'bench_tau_{tau}_{_dev}', device=_dev)
                    results.append(_row('Tau_Sensitivity', _model, tau, fixed_fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fixed_fs),
                                        n_cells=n_cells, duration=duration))
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for tau={tau}: {exc}')
        _save_records(results, partial_path)

    print('\n--- Frame-rate sensitivity ---')
    for fs in fs_values:
        print(f'  fs={fs}Hz...')
        try:
            dff, true_spikes, _, _, _, _ = generate_synthetic_data(
                n_cells=n_cells, fs=fs, duration=duration, tau=fixed_tau
            )
            true_events = [helpers.make_event_ground_truth(s, fixed_tau) for s in true_spikes]

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'fMCSI', fixed_tau, fs, t_my,
                                    _metrics(true_spikes, res['optim_spikes'], true_events, fs),
                                    sweeps=np.mean(res['optim_nsamples']),
                                    n_cells=n_cells, duration=duration))
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, sweeps = run_matlab_pnevMCMC(
                    dff, fs=fs, tau=fixed_tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'CaImAn MCMC', fixed_tau, fs, t_trad,
                                    _metrics(true_spikes, trad_spikes, true_events, fs),
                                    sweeps=np.mean(sweeps),
                                    n_cells=n_cells, duration=duration))
                print(f'    CaImAn MCMC: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, fixed_tau, n_cells)
                t_oasis = time.time() - t0
                results.append(_row('Fs_Sensitivity', 'OASIS', fixed_tau, fs, t_oasis,
                                    _metrics(true_spikes, oas_spk, true_events, fs),
                                    n_cells=n_cells, duration=duration))
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_fs_{fs}_{_dev}', device=_dev)
                    results.append(_row('Fs_Sensitivity', _model, fixed_tau, fs, t_cascade,
                                        _metrics(true_spikes, cascade_spikes, true_events, fs),
                                        n_cells=n_cells, duration=duration))
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for fs={fs}: {exc}')
        _save_records(results, partial_path)

    if run_matlab and matlab_records is not None:
        print(f'\nInjecting {len(matlab_records)} precomputed CaImAn MCMC (params) records...')
        results.extend(matlab_records)
        _save_records(results, partial_path)

    return


def benchmark_noise_sensitivity(data_dir, run_oasis=True, run_matlab=True, run_mine=True,
                                  run_cascade=True, matlab_records=None, cells_only=False):

    n_cells    = 50
    duration   = 300
    fs         = 30.0
    tau        = 1.2
    snr_levels = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]

    results      = []
    cell_records = []
    partial_path = os.path.join(data_dir, 'benchmark_noise_sensitivity_partial.npz')
    cells_path   = os.path.join(data_dir, 'benchmark_noise_sensitivity_cells.npz')

    _metric_keys = ['F1','Precision','Recall',
                    'F1_window','Precision_window','Recall_window',
                    'F1_event','Precision_event','Recall_event','COSMIC']

    print(f'Generating fixed cell population (n_cells={n_cells}, duration={duration}s)...')
    _, true_spikes, clean_traces, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau, snr=1e6
    )
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]

    peak_signals = np.array([
        np.percentile(clean_traces[i], 99) - np.percentile(clean_traces[i], 1)
        for i in range(n_cells)
    ])
    peak_signals = np.maximum(peak_signals, 1e-9)

    def _append_cell_rows(model_name, snr_val, pred_spk):
        p, r, _   = OMSI.compute_accuracy_strict(true_spikes, pred_spk, tolerance=0.1)
        pw, rw, _ = helpers.compute_accuracy_window(true_spikes, pred_spk)
        for i in range(n_cells):
            cell_records.append({
                'Model':            model_name,
                'SNR':              float(snr_val),
                'Precision':        float(p[i]),
                'Recall':           float(r[i]),
                'Precision_window': float(pw[i]),
                'Recall_window':    float(rw[i]),
            })

    for snr_val in snr_levels:
        print(f'  SNR={snr_val}...')
        try:
            sigmas = peak_signals / snr_val
            dff = clean_traces + np.random.normal(0, sigmas[:, None],
                                                   size=clean_traces.shape)

            base = {'Experiment': 'Noise_Sensitivity', 'SNR': snr_val,
                    'N_Cells': n_cells, 'Duration': duration}

            def km(pred_spk):
                return _metrics(true_spikes, pred_spk, true_events, fs)

            if run_mine:
                t0  = time.time()
                res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                       benchmark=True)
                t_my = time.time() - t0
                results.append({**base, 'Model': 'fMCSI', 'Time': t_my,
                                 **dict(zip(_metric_keys, km(res['optim_spikes'])))})
                _append_cell_rows('fMCSI', snr_val, res['optim_spikes'])
                print(f'    fMCSI: F1={results[-1]["F1"]:.3f}')

            if run_matlab and matlab_records is None:
                t0 = time.time()
                trad_spikes, _, _, _ = run_matlab_pnevMCMC(dff, fs=fs, tau=tau, n_sweeps='auto')
                t_trad = time.time() - t0
                results.append({**base, 'Model': 'CaImAn MCMC', 'Time': t_trad,
                                 **dict(zip(_metric_keys, km(trad_spikes)))})
                _append_cell_rows('CaImAn MCMC', snr_val, trad_spikes)
                print(f'    CaImAn MCMC: F1={results[-1]["F1"]:.3f}')

            if run_oasis:
                t0 = time.time()
                oas_spk, _ = _oasis_spikes(dff, fs, tau, n_cells)
                t_oasis = time.time() - t0
                results.append({**base, 'Model': 'OASIS', 'Time': t_oasis,
                                 **dict(zip(_metric_keys, km(oas_spk)))})
                _append_cell_rows('OASIS', snr_val, oas_spk)
                print(f'    OASIS: F1={results[-1]["F1"]:.3f}')

            if run_cascade:
                for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
                    _, cascade_spikes, t_cascade = _run_cascade_inference(
                        dff, fs, data_dir, f'bench_noise_snr{snr_val}_{_dev}', device=_dev)
                    results.append({**base, 'Model': _model, 'Time': t_cascade,
                                     **dict(zip(_metric_keys, km(cascade_spikes)))})
                    _append_cell_rows(_model, snr_val, cascade_spikes)
                    print(f'    CASCADE ({_dev.upper()}): F1={results[-1]["F1"]:.3f}')

        except Exception as exc:
            print(f'  Failed for SNR={snr_val}: {exc}')
        if not cells_only:
            _save_records(results, partial_path)

    if not cells_only and run_matlab and matlab_records is not None:
        print(f'\nInjecting {len(matlab_records)} precomputed CaImAn MCMC (noise) records...')
        results.extend(matlab_records)
        _save_records(results, partial_path)

    if cell_records:
        _save_records(cell_records, cells_path)
        print(f'  Saved {len(cell_records)} cell-level rows -> {cells_path}')

    return


def benchmark_firing_rate_sensitivity(data_dir, run_oasis=True, run_matlab=True,
                                      run_mine=True, run_cascade=True, matlab_records=None):
    
    n_cells  = 250
    duration = 300
    fs       = 30.0
    tau      = 1.2

    print(f'Generating synthetic data (n_cells={n_cells}, duration={duration}s)...')
    dff, true_spikes, _, _, firing_rates, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=tau
    )
    true_events = [helpers.make_event_ground_truth(s, tau) for s in true_spikes]
    npz_spikes  = {'true': true_spikes}
    npz_calcium = {}

    all_results  = []
    partial_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')

    def per_cell(model, i, pred_spk_i, time_i):
        prec,   rec,   f1   = OMSI.compute_accuracy_strict([true_spikes[i]], [pred_spk_i])
        prec_w, rec_w, f1_w = helpers.compute_accuracy_window([true_spikes[i]], [pred_spk_i])
        prec_e, rec_e, f1_e = helpers.compute_accuracy_window([true_events[i]], [pred_spk_i])
        cosmic = helpers.compute_cosmic([true_spikes[i]], [pred_spk_i], fs)
        return {
            'model': model, 'cell_id': i, 'firing_rate': float(firing_rates[i]),
            'precision': prec[0], 'recall': rec[0], 'f1': f1[0],
            'precision_window': prec_w[0], 'recall_window': rec_w[0], 'f1_window': f1_w[0],
            'precision_event': prec_e[0], 'recall_event': rec_e[0], 'f1_event': f1_e[0],
            'cosmic': cosmic[0], 'time': time_i,
        }

    if run_mine:
        print('\nRunning fMCSI...')
        try:
            t0  = time.time()
            res = OMSI.deconv(dff, params={'f': fs, 'p': 2, 'auto_stop': True},
                                   benchmark=True)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('fMCSI', i, res['optim_spikes'][i],
                                             res['optim_times_per_cell'][i]))
            npz_spikes['my_method'] = res['optim_spikes']
            print(f'  Finished in {total_time:.1f}s')
        except Exception as exc:
            print(f'  fMCSI failed: {exc}')
        _save_records(all_results, partial_path)

    if run_matlab:
        if matlab_records is not None:
            print(f'\nInjecting {len(matlab_records)} precomputed CaImAn MCMC (firing rate) records...')
            all_results.extend(matlab_records)
            _save_records(all_results, partial_path)
        else:
            print('\nRunning CaImAn MCMC...')
            trad_spikes_all = []
            for i in range(n_cells):
                print(f'  Processing cell {i+1}/{n_cells}...', end='\r')
                try:
                    t0 = time.time()
                    trad_spk, _, _, _ = run_matlab_pnevMCMC(
                        dff[i:i+1], fs=fs, tau=tau, n_sweeps='auto')
                    time_taken = time.time() - t0
                    trad_spikes_all.append(trad_spk[0])
                    all_results.append(per_cell('CaImAn MCMC', i, trad_spk[0], time_taken))
                except Exception as exc:
                    print(f'\n  CaImAn MCMC failed on cell {i}: {exc}')
                    trad_spikes_all.append(np.array([]))
            print('\n  Finished.')
            npz_spikes['trad_mcmc'] = trad_spikes_all
            _save_records(all_results, partial_path)

    if run_oasis:
        print('\nRunning OASIS...')
        try:
            t0 = time.time()
            oas_spk, oas_cal = _oasis_spikes(dff, fs, tau, n_cells)
            total_time = time.time() - t0
            for i in range(n_cells):
                all_results.append(per_cell('OASIS', i, oas_spk[i], np.nan))
            npz_spikes['oasis']  = oas_spk
            npz_calcium['oasis'] = np.array(oas_cal)
            print(f'  Finished in {total_time:.1f}s')
        except Exception as exc:
            print(f'  OASIS failed: {exc}')
        _save_records(all_results, partial_path)

    if run_cascade:
        for _dev, _model in [('gpu', 'CASCADE_GPU'), ('cpu', 'CASCADE_CPU')]:
            print(f'\nRunning CASCADE (subprocess, {_dev.upper()})...')
            try:
                _, cascade_spikes, t_cascade = _run_cascade_inference(
                    dff, fs, data_dir, f'bench_firerate_cascade_{_dev}', device=_dev)
                for i in range(n_cells):
                    all_results.append(per_cell(_model, i, cascade_spikes[i], np.nan))
                npz_spikes[f'cascade_{_dev}'] = cascade_spikes
                print(f'  CASCADE ({_dev.upper()}) finished in {t_cascade:.1f}s')
            except Exception as exc:
                print(f'  CASCADE ({_dev.upper()}) failed: {exc}')
            _save_records(all_results, partial_path)

    if all_results:
        npz_save = {'dff': dff, 'fs': fs, 'tau': tau,
                    'firing_rates': np.array(firing_rates)}
        for k, v in npz_spikes.items():
            npz_save[f'spikes_{k}'] = np.array(v, dtype=object)
        for k, v in npz_calcium.items():
            npz_save[f'calcium_{k}'] = v
        np.savez(os.path.join(data_dir, 'firing_rate_sensitivity_traces.npz'), **npz_save)

    return

_CASCADE_SR_SEED     = 77
_CASCADE_SR_N_CELLS  = 50
_CASCADE_SR_DURATION = 300
_CASCADE_SR_TAU      = 1.2


def benchmark_cascade_sample_rate(data_dir, run_cascade=True):

    from simulation_helpers import generate_synthetic_data

    out_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    rng = np.random.default_rng(_CASCADE_SR_SEED)

    results = {}
    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        print(f'  {fs} Hz...')
        np.random.seed(int(rng.integers(0, 2**31)))
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=_CASCADE_SR_N_CELLS, fs=fs,
            duration=_CASCADE_SR_DURATION, tau=_CASCADE_SR_TAU)
        true_events = [helpers.make_event_ground_truth(s, _CASCADE_SR_TAU)
                       for s in true_spikes]

        ts_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_true_spikes.npz')
        np.savez(ts_path, true_spikes=np.array(true_spikes, dtype=object),
                 tau=_CASCADE_SR_TAU)

        if not run_cascade:
            results[f'fb_{suffix}']     = np.full(_CASCADE_SR_N_CELLS, np.nan)
            results[f'cosmic_{suffix}'] = np.full(_CASCADE_SR_N_CELLS, np.nan)
            continue

        cascade_spikes = None
        for dev in ('gpu', 'cpu'):
            out_file = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_{dev}_output.npz')
            if os.path.exists(out_file):
                try:
                    _d = np.load(out_file, allow_pickle=True)
                    cascade_spikes = list(_d['cascade_spikes'])
                    print(f'    Loaded existing CASCADE ({dev.upper()}) output')
                    break
                except Exception:
                    pass
            try:
                _, cascade_spikes, _ = _run_cascade_inference(
                    dff, fs, data_dir, f'cascade_samplerate_{fs}hz_{dev}', device=dev)
                print(f'    CASCADE ({dev.upper()}) done')
                break
            except Exception as exc:
                print(f'    CASCADE {dev.upper()} failed: {exc}')

        if cascade_spikes is None:
            results[f'fb_{suffix}']     = np.full(n_cells, np.nan)
            results[f'cosmic_{suffix}'] = np.full(n_cells, np.nan)
            continue

        prec, rec, _ = OMSI.compute_accuracy_strict(true_spikes, cascade_spikes,
                                                      tolerance=0.1)
        b2    = BETA ** 2
        denom = b2 * prec + rec
        fb    = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        cosmic = helpers.compute_cosmic(true_spikes, cascade_spikes, fs)

        results[f'fb_{suffix}']     = fb
        results[f'cosmic_{suffix}'] = cosmic
        print(f'    mean Fβ={np.nanmean(fb):.3f}  CosMIC={np.nanmean(cosmic):.3f}')

    np.savez(out_path, **results)
    print(f'  Saved -> {out_path}')


def run_test(data_dir=_DEFAULT_DATA_DIR, run_fmcsi=True, run_matlab=True,
             run_oasis=True, run_cascade=True):

    os.makedirs(data_dir, exist_ok=True)

    ext = None
    if run_matlab:
        print(f'loading pre-computed caiman MCMC data from:\n  {_MATLAB_PRECOMPUTED_DIR}')
        ext = _load_external_matlab_data()
        total = sum(len(ext[k]) for k in ('sweeps', 'scalability', 'params', 'noise_sensitivity', 'firing_rate'))
        print(f'  Loaded {total} CaImAn MCMC records across all benchmarks.')

    kw_shared = dict(run_oasis=run_oasis, run_mine=run_fmcsi, run_cascade=run_cascade,
                     run_matlab=run_matlab)

    print('=== Sweeps benchmark ===')
    benchmark_sweeps(data_dir, **kw_shared,
                     matlab_records=ext['sweeps'] if ext else None)
    print('\n=== Scalability benchmark ===')
    benchmark_scalability(data_dir, **kw_shared,
                          matlab_records=ext['scalability'] if ext else None)
    print('\n=== Parameter sensitivity benchmark ===')
    benchmark_params(data_dir, **kw_shared,
                     matlab_records=ext['params'] if ext else None)
    print('\n=== Noise sensitivity benchmark ===')
    benchmark_noise_sensitivity(data_dir, **kw_shared,
                                matlab_records=ext['noise_sensitivity'] if ext else None)
    print('\n=== Firing-rate sensitivity benchmark ===')
    benchmark_firing_rate_sensitivity(data_dir, **kw_shared,
                                      matlab_records=ext['firing_rate'] if ext else None)
    print('\n=== CASCADE 7.5 Hz vs 30 Hz comparison ===')
    benchmark_cascade_sample_rate(data_dir, run_cascade=run_cascade)
    print('\nTest mode complete.')


def _fbeta(prec, rec):
    p  = np.asarray(prec, dtype=float)
    r  = np.asarray(rec,  dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _fit_scaling(x, y):
    if len(x) < 3:
        return np.nan, np.nan, 'N/A'
    x, y = np.array(x, float), np.array(y, float)
    def r2(y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 0.0 if ss_tot < 1e-9 else 1 - ss_res / ss_tot
    try:
        r2_lin  = r2(np.poly1d(np.polyfit(x, y, 1))(x))
    except Exception:
        r2_lin  = -np.inf
    try:
        r2_poly = r2(np.poly1d(np.polyfit(x, y, 2))(x))
    except Exception:
        r2_poly = -np.inf
    conclusion = 'Linear' if r2_lin >= r2_poly - 0.02 else 'Polynomial'
    return r2_lin, r2_poly, conclusion


def _set_three_ticks_x(ax):
    all_x = [v for line in ax.get_lines() for v in line.get_xdata()]
    if len(all_x) < 2:
        return
    lo, hi = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    if lo == hi:
        return
    ax.set_xticks([lo, (lo + hi) / 2, hi])
    ax.set_xticklabels([f'{v:.3g}' for v in [lo, (lo + hi) / 2, hi]])


def _filter_cascade_shared_x(cascade_sub, full_tbl, xcol):
    non_cascade = np.array([not str(m).startswith('CASCADE') for m in full_tbl['Model']], dtype=bool)
    vals = full_tbl[xcol][non_cascade].astype(float)
    other_x = set(vals[~np.isnan(vals)].tolist())
    mask = np.isin(cascade_sub[xcol].astype(float), list(other_x))
    return {k: v[mask] for k, v in cascade_sub.items()}


_CASCADE_CMP_COLOR_7P5 = 'tab:red'
_CASCADE_CMP_COLOR_30  = 'tab:cyan'


def _rebuild_cascade_sample_rate_data(data_dir):
    out_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    results  = {}

    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        ts_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_true_spikes.npz')
        if not os.path.exists(ts_path):
            print(f'  No saved true_spikes for {fs} Hz — re-run --mode test to regenerate.')
            return

        try:
            td          = np.load(ts_path, allow_pickle=True)
            true_spikes = list(td['true_spikes'])
            tau         = float(td['tau']) if 'tau' in td else _CASCADE_SR_TAU
        except Exception as exc:
            print(f'  Could not load {ts_path}: {exc}')
            return

        cascade_spikes = None
        for dev in ('gpu', 'cpu'):
            out_file = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_{dev}_output.npz')
            if os.path.exists(out_file):
                try:
                    d = np.load(out_file, allow_pickle=True)
                    cascade_spikes = list(d['cascade_spikes'])
                    break
                except Exception:
                    pass
        if cascade_spikes is None:
            print(f'  No CASCADE output found for {fs} Hz — re-run --mode test.')
            return

        prec, rec, _ = OMSI.compute_accuracy_strict(true_spikes, cascade_spikes,
                                                      tolerance=0.1)
        b2    = BETA ** 2
        denom = b2 * prec + rec
        fb    = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
        cosmic = helpers.compute_cosmic(true_spikes, cascade_spikes, fs)
        results[f'fb_{suffix}']     = fb
        results[f'cosmic_{suffix}'] = cosmic
        print(f'  Rebuilt {fs} Hz: mean Fβ={np.nanmean(fb):.3f}  CosMIC={np.nanmean(cosmic):.3f}')

    if len(results) == 4:
        np.savez(out_path, **results)
        print(f'  Saved rebuilt data -> {out_path}')


def _plot_cascade_comparison(ax, data_dir):
    from matplotlib.patches import Patch

    npz_path = os.path.join(data_dir, 'cascade_7p5_vs_30hz_data.npz')
    _needs_rebuild = not os.path.exists(npz_path)
    if not _needs_rebuild:
        _d = np.load(npz_path)
        if all(not np.any(np.isfinite(_d[k])) for k in _d.files):
            _needs_rebuild = True
    if _needs_rebuild:
        print('  Attempting to rebuild cascade sample-rate data from output files...')
        _rebuild_cascade_sample_rate_data(data_dir)
    if not os.path.exists(npz_path):
        ax.text(0.5, 0.5, 'No data\n(run --mode test)',
                transform=ax.transAxes, ha='center', va='center', fontsize=7)
        return

    data     = np.load(npz_path)
    fb_7     = data['fb_7'];     fb_30     = data['fb_30']
    cosmic_7 = data['cosmic_7']; cosmic_30 = data['cosmic_30']

    all_pos      = [1, 2, 3.15, 4.15]
    all_datasets = [
        (fb_7[np.isfinite(fb_7)],          _CASCADE_CMP_COLOR_7P5),
        (fb_30[np.isfinite(fb_30)],         _CASCADE_CMP_COLOR_30),
        (cosmic_7[np.isfinite(cosmic_7)],   _CASCADE_CMP_COLOR_7P5),
        (cosmic_30[np.isfinite(cosmic_30)], _CASCADE_CMP_COLOR_30),
    ]
    pos      = [p for p, (d, _) in zip(all_pos, all_datasets) if len(d) > 0]
    datasets = [(d, c) for d, c in all_datasets if len(d) > 0]
    if not datasets:
        ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes,
                ha='center', va='center', fontsize=7)
        return
    parts = ax.violinplot([d for d, _ in datasets], positions=pos,
                          showmedians=True, widths=0.65)
    for pc, (_, col) in zip(parts['bodies'], datasets):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[partname].set_color('k'); parts[partname].set_linewidth(0.8)

    ax.set_xticks([(pos[0] + pos[1]) / 2, (pos[2] + pos[3]) / 2])
    ax.set_xticklabels([r'$F_\beta$', 'CosMIC'])
    ax.legend(handles=[
        Patch(facecolor=_CASCADE_CMP_COLOR_7P5, alpha=0.75, label='7.5 Hz'),
        Patch(facecolor=_CASCADE_CMP_COLOR_30,  alpha=0.75, label='30 Hz'),
    ], loc='upper right', handlelength=1.0, handleheight=0.8,
       borderpad=0.4, labelspacing=0.2, frameon=False)
    ax.set_ylabel('score')
    ax.set_ylim(0, 1.1)


def _running_median_se(x, y, x_out, bandwidth):
    medians = np.full(len(x_out), np.nan)
    ses     = np.full(len(x_out), np.nan)
    for i, xc in enumerate(x_out):
        mask = (x >= xc - bandwidth) & (x <= xc + bandwidth)
        if mask.sum() >= 5:
            vals        = y[mask]
            medians[i]  = np.median(vals)
            ses[i]      = vals.std(ddof=1) / np.sqrt(len(vals))
    return medians, ses


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    partial_files = [
        'benchmark_sweeps_partial.npz',
        'benchmark_scalability_partial.npz',
        'benchmark_params_partial.npz',
        'benchmark_noise_sensitivity_partial.npz',
    ]

    all_records = []
    for fname in partial_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            try:
                tbl = _load_records(fpath)
                all_records.append(tbl)
                print(f'Loaded {fpath}  ({_tbl_len(tbl)} rows)')
            except Exception as exc:
                print(f'Error reading {fpath}: {exc}')

    if not all_records:
        raise RuntimeError(f'No partial benchmark files found in {data_dir}. '
                           'Run --mode test first.')

    combined = _tbl_concat(all_records)

    ext = _load_external_matlab_data()
    ext_benchmark_keys = ['sweeps', 'scalability', 'params', 'noise_sensitivity']
    ext_records = []
    for key in ext_benchmark_keys:
        ext_records.extend(ext[key])
    if ext_records:
        ext_tbl = _records_to_tbl(ext_records)
        combined = _tbl_concat([combined, ext_tbl])
        print(f'Injected {len(ext_records)} CaImAn MCMC records from external files.')

    dur_cascade_tbl = {}
    for _ext in ('.npz', '.json'):
        _alt_sc_path = os.path.join(_CASCADE_DURATION_ALT_DIR,
                                    f'benchmark_scalability_partial{_ext}')
        if not os.path.exists(_alt_sc_path):
            continue
        try:
            if _ext == '.json':
                import json as _json
                with open(_alt_sc_path) as _f:
                    _raw = _json.load(_f)
                _rows = [r for r in _raw
                         if str(r.get('Model', '')).startswith('CASCADE')
                         and r.get('Experiment') == 'Duration_Scaling']

                for _r in _rows:
                    if _r['Model'] == 'CASCADE':
                        _r['Model'] = 'CASCADE_GPU'
                if _rows:
                    dur_cascade_tbl = _records_to_tbl(_rows)
            else:
                _alt = _load_records(_alt_sc_path)
                _mask = np.array([
                    str(m).startswith('CASCADE') and str(e) == 'Duration_Scaling'
                    for m, e in zip(_alt.get('Model', []), _alt.get('Experiment', []))
                ], dtype=bool)
                if _mask.sum() > 0:
                    dur_cascade_tbl = {k: v[_mask] for k, v in _alt.items()}
            if dur_cascade_tbl:
                print(f'Loaded {_tbl_len(dur_cascade_tbl)} CASCADE duration rows from alt dir.')
                break
        except Exception as _exc:
            print(f'Warning: could not load alt CASCADE duration data: {_exc}')

    fr_tbl = {}
    fr_path = os.path.join(data_dir, 'firing_rate_sensitivity_partial.npz')
    if os.path.exists(fr_path):
        try:
            fr_tbl = _load_records(fr_path)
            print(f'Loaded {fr_path}  ({_tbl_len(fr_tbl)} rows)')
        except Exception as exc:
            print(f'Error reading {fr_path}: {exc}')
    fr_ext_records = ext['firing_rate']
    if fr_ext_records:
        fr_ext_tbl = _records_to_tbl(fr_ext_records)
        fr_tbl = _tbl_concat([fr_tbl, fr_ext_tbl])
        print(f'Injected {len(fr_ext_records)} CaImAn MCMC firing-rate records from external files.')

    f1_col   = 'F1'        if USE_STRICT_ACCURACY else 'F1_window'
    prec_col = 'Precision' if USE_STRICT_ACCURACY else 'Precision_window'
    rec_col  = 'Recall'    if USE_STRICT_ACCURACY else 'Recall_window'

    scaling_stats = []

    _legend_labels = {
        'fMCSI': 'OMSI', 'CaImAn MCMC': 'CaImAn MCMC', 'OASIS': 'OASIS',
        'CASCADE_GPU': 'CASCADE (GPU)', 'CASCADE_CPU': 'CASCADE (CPU)',
    }
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS[m], marker='.', linestyle='-',
                   label=_legend_labels[m])
        for m in ['fMCSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU']
    ]

    _legend_labels1 = {
        'fMCSI': 'OMSI', 'CaImAn MCMC': 'CaImAn MCMC', 'OASIS': 'OASIS',
        'CASCADE_CPU': 'CASCADE',
    }
    legend_handles1 = [
        plt.Line2D([0], [0], color=COLORS[m], marker='.', linestyle='-',
                   label=_legend_labels1[m])
        for m in ['fMCSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_CPU']
    ]


    mosaic_A = [
        ['sweeps',   'sweeps',   'sweeps',   'sweeps',
         'cells',    'cells',    'cells',    'cells',
         'duration', 'duration', 'duration', 'duration'],
        ['tau_p',    'tau_p',    'tau_p',    'tau_r',
         'tau_r',    'tau_r',    'noise_p',  'noise_p',
         'noise_p',  'noise_r',  'noise_r',  'noise_r'],
    ]

    figA, axA = plt.subplot_mosaic(mosaic_A, figsize=(7, 3.5), dpi=300,
                                    gridspec_kw={'height_ratios': [3, 2]})

    for model in ['fMCSI', 'CaImAn MCMC']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Sweeps'), 'Sweeps')
        if _tbl_len(subset) > 0:
            axA['sweeps'].plot(subset['Sweeps'], subset['Time'] / 60.0,
                               '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Sweeps'], subset['Time'])
            scaling_stats.append({'Experiment': 'Sweeps', 'Model': model,
                                   'Variable': 'Sweeps', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axA['sweeps'].set_xlabel('# sweeps')
    axA['sweeps'].set_ylabel('compute time (min)')
    axA['sweeps'].set_yscale('log')

    for model in ['CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Cell_Scaling'), 'N_Cells')
        if _tbl_len(subset) > 0:
            axA['cells'].plot(subset['N_Cells'], subset['Time'] / 60.,
                              '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['N_Cells'], subset['Time'])
            scaling_stats.append({'Experiment': 'Cell_Scaling', 'Model': model,
                                   'Variable': 'N_Cells', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})
    axA['cells'].set_xlabel('# cells')
    axA['cells'].set_ylabel('compute time (min)')
    axA['cells'].set_yscale('log')

    for model in ['CaImAn MCMC', 'OASIS', 'CASCADE_GPU', 'CASCADE_CPU', 'fMCSI']:
        if model.startswith('CASCADE') and dur_cascade_tbl:
            src = dur_cascade_tbl
        else:
            src = combined
        m_rows = _tbl_filter(src, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Duration_Scaling'), 'Duration')
        if _tbl_len(subset) > 0:
            axA['duration'].plot(subset['Duration'] / 60., subset['Time'] / 60.,
                                 '.-', label=model, color=COLORS.get(model, 'k'))
            r2l, r2p, c = _fit_scaling(subset['Duration'], subset['Time'])
            scaling_stats.append({'Experiment': 'Duration_Scaling', 'Model': model,
                                   'Variable': 'Duration', 'Lin_R2': r2l,
                                   'Poly_R2': r2p, 'Conclusion': c})

    _extrap = {'CaImAn MCMC': 496.0, 'CASCADE_CPU': 5.0}
    for model, t_extrap in _extrap.items():
        color = COLORS.get(model, 'k')

        src = dur_cascade_tbl if model.startswith('CASCADE') and dur_cascade_tbl else combined
        m_rows = _tbl_filter(src, 'Model', model)
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Duration_Scaling'), 'Duration')
        if _tbl_len(subset) > 0:
            last_dur  = float(subset['Duration'][-1]) / 60.
            last_time = float(subset['Time'][-1])    / 60.
            axA['duration'].plot([last_dur, 120.], [last_time, t_extrap],
                                 '-', color=color)
        axA['duration'].plot(120., t_extrap, '.', color=color)

    axA['duration'].set_xlabel('recording duration (min)')
    axA['duration'].set_ylabel('compute time (min)')
    axA['duration'].set_yscale('log')
    axA['duration'].set_xticks([0, 60, 120])
    axA['duration'].set_xticklabels(['0', '60', '120'])

    for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Tau_Sensitivity'), 'Tau')
        if model.startswith('CASCADE'):
            subset = _filter_cascade_shared_x(
                subset, _tbl_filter(combined, 'Experiment', 'Tau_Sensitivity'), 'Tau')
        if _tbl_len(subset) > 0:
            axA['tau_p'].plot(subset['Tau'][:-1], subset[prec_col][:-1], '.-',
                              color=COLORS.get(model, 'k'))
            axA['tau_r'].plot(subset['Tau'][:-1], subset[rec_col][:-1],  '.-',
                              color=COLORS.get(model, 'k'))

    for ax_key, xlabel, ylabel in [
        ('tau_p', '$\tau$ (s)', 'Precision'),
        ('tau_r', '$\tau$ (s)', 'Recall'),
    ]:
        axA[ax_key].set_xlabel(xlabel)
        axA[ax_key].set_ylabel(ylabel)
        axA[ax_key].set_ylim(0.45, 1.05)
        _set_three_ticks_x(axA[ax_key])

    _cells_path = os.path.join(data_dir, 'benchmark_noise_sensitivity_cells.npz')
    _noise_cells = None
    if os.path.exists(_cells_path):
        try:
            _noise_cells = _load_records(_cells_path)
            print(f'Loaded {_tbl_len(_noise_cells)} cell-level noise rows.')
        except Exception as _exc:
            print(f'Warning: could not load noise cells file: {_exc}')

    _prec_col_cell = 'Precision_window' if not USE_STRICT_ACCURACY else 'Precision'
    _rec_col_cell  = 'Recall_window'    if not USE_STRICT_ACCURACY else 'Recall'

    if _noise_cells is not None and _tbl_len(_noise_cells) > 0:
        _all_snr = _noise_cells['SNR'].astype(float)
        _snr_levels_sorted = np.sort(np.unique(_all_snr))[::-1]

        for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
            _mask_m = _noise_cells['Model'] == model
            if _mask_m.sum() == 0:
                continue
            color = COLORS.get(model, 'k')
            snr_vals, mean_p, se_p, mean_r, se_r = [], [], [], [], []
            for snr_val in _snr_levels_sorted:
                _mask_sn = _mask_m & (_all_snr == snr_val)
                if _mask_sn.sum() < 2:
                    continue
                _py = _noise_cells[_prec_col_cell].astype(float)[_mask_sn]
                _ry = _noise_cells[_rec_col_cell].astype(float)[_mask_sn]
                snr_vals.append(snr_val)
                mean_p.append(np.mean(_py))
                se_p.append(np.std(_py, ddof=1) / np.sqrt(len(_py)))
                mean_r.append(np.mean(_ry))
                se_r.append(np.std(_ry, ddof=1) / np.sqrt(len(_ry)))
            if len(snr_vals) >= 2:
                sv   = np.array(snr_vals)
                mp_  = np.array(mean_p); sp_ = np.array(se_p)
                mr_  = np.array(mean_r); sr_ = np.array(se_r)
                axA['noise_p'].plot(sv, mp_, '.-', color=color)
                axA['noise_p'].fill_between(sv, mp_ - sp_, mp_ + sp_,
                                             color=color, alpha=0.25, linewidth=0)
                axA['noise_r'].plot(sv, mr_, '.-', color=color)
                axA['noise_r'].fill_between(sv, mr_ - sr_, mr_ + sr_,
                                             color=color, alpha=0.25, linewidth=0)
    else:
        for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
            m_rows = _tbl_filter(combined, 'Model', model)
            if _tbl_len(m_rows) == 0:
                continue
            subset = _tbl_filter(m_rows, 'Experiment', 'Noise_Sensitivity')
            if _tbl_len(subset) == 0 or 'SNR' not in subset:
                continue
            subset = _tbl_sort(subset, 'SNR')
            if _tbl_len(subset) > 0:
                axA['noise_p'].plot(subset['SNR'], subset[prec_col], '.-',
                                    color=COLORS.get(model, 'k'))
                axA['noise_r'].plot(subset['SNR'], subset[rec_col], '.-',
                                    color=COLORS.get(model, 'k'))

    for ax_key, xlabel, ylabel in [
        ('noise_p', 'SNR', 'Precision'),
        ('noise_r', 'SNR', 'Recall'),
    ]:
        axA[ax_key].set_xlabel(xlabel)
        axA[ax_key].set_ylabel(ylabel)
        axA[ax_key].set_ylim(-0.05, 1.05)
        axA[ax_key].set_xscale('log')

    figA.legend(handles=legend_handles, loc='upper center', ncol=5,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    print('\nScaling statistics:')
    print(f'{"Experiment":<22} {"Model":<12} {"Variable":<10} '
          f'{"Lin R^2":<8} {"Poly R^2":<8} {"Fit":<12}')
    print('-' * 80)
    for s in scaling_stats:
        print(f'{s["Experiment"]:<22} {s["Model"]:<12} {s["Variable"]:<10} '
              f'{s["Lin_R2"]:<8.4f} {s["Poly_R2"]:<8.4f} {s["Conclusion"]:<12}')
        
    figA.subplots_adjust(wspace=20., hspace=0.55, top=0.88)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure2A.{sfx}')
        figA.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(figA)


    mosaic_B = [
        ['fs_p',  'fs_r'     ],
        ['fs_fb', 'fs_cosmic'],
        ['casc',  'casc'     ],
    ]
    figB, axB = plt.subplot_mosaic(
        mosaic_B, figsize=(3.25, 4.5), dpi=300,
        gridspec_kw={'height_ratios': [2, 2, 3]},
    )

    for model in ['CaImAn MCMC', 'CASCADE_GPU', 'CASCADE_CPU', 'OASIS', 'fMCSI']:
        m_rows = _tbl_filter(combined, 'Model', model)
        if _tbl_len(m_rows) == 0:
            continue
        subset_fs = _tbl_sort(_tbl_filter(m_rows, 'Experiment', 'Fs_Sensitivity'), 'Fs')
        subset_fs = {k: v[np.array(subset_fs['Fs'], dtype=float) != 100.0]
                     for k, v in subset_fs.items()}
        if model.startswith('CASCADE'):
            subset_fs = _filter_cascade_shared_x(
                subset_fs, _tbl_filter(combined, 'Experiment', 'Fs_Sensitivity'), 'Fs')
        if _tbl_len(subset_fs) > 0:
            fb_fs = _fbeta(subset_fs[prec_col], subset_fs[rec_col])
            axB['fs_p'].plot(subset_fs['Fs'], subset_fs[prec_col], '.-',
                             color=COLORS.get(model, 'k'))
            axB['fs_r'].plot(subset_fs['Fs'], subset_fs[rec_col],  '.-',
                             color=COLORS.get(model, 'k'))
            axB['fs_fb'].plot(subset_fs['Fs'], fb_fs, '.-',
                              color=COLORS.get(model, 'k'))
            axB['fs_cosmic'].plot(subset_fs['Fs'], subset_fs['COSMIC'], '.-',
                                  color=COLORS.get(model, 'k'))
            
    axB['fs_p'].set_ylim([0.45, 1.05])
    axB['fs_r'].set_ylim([0.45, 1.05])
    axB['fs_fb'].set_ylim([0.45, 1.05])
    axB['fs_cosmic'].set_ylim(-0.05, 1.05)

    for ax_key, xlabel, ylabel in [
        ('fs_p',      'sample rate (Hz)', 'Precision'),
        ('fs_r',      'sample rate (Hz)', 'Recall'),
        ('fs_fb',     'sample rate (Hz)', r'$F_\beta$'),
        ('fs_cosmic', 'sample rate (Hz)', 'CosMIC'),
    ]:
        axB[ax_key].set_xlabel(xlabel)
        axB[ax_key].set_ylabel(ylabel)
        _set_three_ticks_x(axB[ax_key])

    _plot_cascade_comparison(axB['casc'], data_dir)

    figB.legend(handles=legend_handles1, loc='upper center', ncol=5,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)
    figB.tight_layout()

    figA.subplots_adjust(top=0.88)

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, f'figure2B.{sfx}')
        figB.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(figB)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Figure 2: scaling and sensitivity benchmarks'
    )
    parser.add_argument('--mode', required=True,
                        choices=['test', 'plot', 'noise-cells'],
                        help='"test" runs all benchmarks; "plot" generates the figure; '
                             '"noise-cells" runs only the noise sensitivity benchmark and writes '
                             'benchmark_noise_sensitivity_cells.npz without touching other result files')
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
    elif args.mode == 'noise-cells':
        os.makedirs(args.data_dir, exist_ok=True)
        print('=== Noise sensitivity cell-level benchmark (cells_only) ===')
        benchmark_noise_sensitivity(
            args.data_dir,
            run_oasis   = not args.no_oasis,
            run_matlab  = not args.no_matlab,
            run_mine    = not args.no_fmcsi,
            run_cascade = not args.no_cascade,
            cells_only  = True,
        )
    else:
        plot_figure(data_dir=args.data_dir)
