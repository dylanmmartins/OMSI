# -*- coding: utf-8 -*-
"""
figures/timing_calibration.py

DMM, September 2026
"""

import argparse
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, 'data', 'timing')

_POP_NPZ     = 'timing_population.npz'
_SUMMARY_NPZ = 'timing_calibration.npz'
_FIX_NPZ     = 'timing_caiman_fix_thr.npz'
_FR_NPZ      = 'timing_framerate.npz'
_FR_FIX_NPZ  = 'timing_framerate_caiman_fix_thr.npz'

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

FS   = 30.0
TAU  = 1.2
BETA = 0.5

METHODS = {
    'OMSI':       ('OMSI',                  '#4C72B0'),
    'MATLAB':     ('CaImAn',                '#DD8452'),
    'MATLAB_FIX': ('CaImAn, fixed',         '#DD8452'),
    'OASIS':      ('OASIS',                 '#55A868'),
    'CASCADE':    ('CASCADE',               '#8172B3'),
}

METHOD_LS = {'MATLAB_FIX': '--'}
CAIMAN_INIT_THR = 0.70
CAIMAN_WORKERS = 12

POP_CELLS    = 200
POP_DURATION = 600.0
POP_SEED     = 5
KURTOSIS_RANGE = (0.0, 25.0)

BURSTY_MIN_PAIRS = 3

OMSI_PARAMS   = {'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1}
MATLAB_SWEEPS = 500

MATCH_TOL = 0.100

TOL_GRID = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.033, 0.040,
                     0.050, 0.060, 0.075, 0.100])

DEBIAS = False

SHIFT_GRID = np.arange(-4.0, 4.0001, 0.25)

N_BINS = 10
N_BOOT = 200

FR_RATES    = np.array([7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0])
FR_CELLS    = 40
FR_DURATION = 300.0
FR_SEED     = 11

FR_OFFSET_TOL = 0.250

CASCADE_MODELS = {
    7.5:  'Global_EXC_7.5Hz_smoothing200ms_causalkernel',
    10.0: 'Global_EXC_10Hz_smoothing100ms_causalkernel',
    15.0: 'Global_EXC_15Hz_smoothing100ms_causalkernel',
    20.0: 'Global_EXC_20Hz_smoothing100ms_causalkernel',
    30.0: 'Global_EXC_30Hz_smoothing50ms_causalkernel',
    40.0: 'Global_EXC_40Hz_smoothing50ms_causalkernel',
}
CASCADE_FALLBACK_FS = 40.0

_POST_NPZ         = 'timing_posterior.npz'
_POST_SUMMARY_NPZ = 'timing_posterior_summary.npz'
POST_CELLS   = 40
POST_SEED    = 17
POST_METHODS = ['OMSI', 'MATLAB_FIX', 'MATLAB']

TRUE_TAU_RISE = 0.05

COVERAGE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

MIN_SAMPLES_CI = 5

EXIST_EDGES = np.linspace(0.0, 1.0, 6)

PAIR_EDGES_MS = np.array([0.0, 5.0, 12.0, 22.0, 35.0, 55.0, 80.0, 120.0, 200.0, 350.0,
                          600.0])
PAIR_WIN = MATCH_TOL

_OASIS_S_NPZ = 'timing_population_oasis_s.npz'

PAIR_ISO = 0.5

EXAMPLE_CELL  = 2
EXAMPLE_START = 80.357
EXAMPLE_WIN   = 8.0
N_RASTER      = 20


def _fbeta(precision, recall):
    p  = np.asarray(precision, dtype=float)
    r  = np.asarray(recall,    dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _mad(x, axis=None):
    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _match_pairs(t, p, tol):
    t = np.asarray(t, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    empty = np.array([], dtype=int)
    if len(t) == 0 or len(p) == 0:
        return empty, empty

    ot, op = np.argsort(t), np.argsort(p)
    ts, ps = t[ot], p[op]

    ev   = np.concatenate([ts, ps])
    kind = np.concatenate([np.zeros(len(ts), dtype=int), np.ones(len(ps), dtype=int)])
    src  = np.concatenate([np.arange(len(ts)), np.arange(len(ps))])
    order = np.argsort(ev, kind='stable')
    ev, kind, src = ev[order], kind[order], src[order]

    breaks = np.where(np.diff(ev) > tol)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(ev)]])

    large = 1e6
    out_t, out_p = [], []
    for s, e in zip(starts, ends):
        ii = src[s:e]
        k  = kind[s:e]
        ti, pi = ii[k == 0], ii[k == 1]
        if len(ti) == 0 or len(pi) == 0:
            continue
        cost = np.abs(ts[ti][:, None] - ps[pi][None, :])
        cost[cost > tol] = large
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] <= tol
        out_t.append(ot[ti[r[ok]]])
        out_p.append(op[pi[c[ok]]])

    if not out_t:
        return empty, empty
    return np.concatenate(out_t), np.concatenate(out_p)


def _signed_errors(true_spikes, called, tol=MATCH_TOL):
    errs = []
    for t, p in zip(true_spikes, called):
        t = np.asarray(t, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        it, ip = _match_pairs(t, p, tol)
        errs.append(p[ip] - t[it])
    return np.concatenate(errs) if errs else np.array([])


def _prf_by_cell(t, p, tol):
    if len(p) == 0:
        return 0.0, (0.0 if len(t) > 0 else 1.0)
    if len(t) == 0:
        return 0.0, 1.0
    it, _ = _match_pairs(t, p, tol)
    n_tp = len(it)
    return n_tp / len(p), n_tp / len(t)


def _timing_analysis(true_spikes, called):
    raw = _signed_errors(true_spikes, called)
    offset = float(np.median(raw)) if (DEBIAS and len(raw) > 0) else 0.0
    shifted = [np.asarray(p, dtype=np.float64).ravel() - offset for p in called]
    err = _signed_errors(true_spikes, shifted)

    n_cells = len(true_spikes)
    prec = np.full((len(TOL_GRID), n_cells), np.nan)
    rec  = np.full((len(TOL_GRID), n_cells), np.nan)
    for c in range(n_cells):
        t = np.asarray(true_spikes[c], dtype=np.float64).ravel()
        if len(t) == 0:
            continue
        for k, tol in enumerate(TOL_GRID):
            prec[k, c], rec[k, c] = _prf_by_cell(t, shifted[c], tol)

    return {
        'offset_s': offset,
        'err_s':    err,
        'n_true':   int(sum(len(np.ravel(t)) for t in true_spikes)),
        'n_called': int(sum(len(np.ravel(p)) for p in called)),
        'fb':       _fbeta(prec, rec),
        'prec':     prec,
        'rec':      rec,
    }


def _true_occupancy(t, n_frames, shift):
    idx = np.floor(np.asarray(t, dtype=np.float64).ravel() * FS + shift).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    occ = np.zeros(n_frames, dtype=bool)
    occ[idx] = True
    return occ


def _best_shift(true_spikes, probs):
    n_cells, n_frames = probs.shape
    p = np.clip(np.nan_to_num(probs.astype(np.float64)), 0.0, 1.0)
    brier = np.zeros(len(SHIFT_GRID))
    for k, s in enumerate(SHIFT_GRID):
        tot = 0.0
        for c in range(n_cells):
            occ = _true_occupancy(true_spikes[c], n_frames, s)
            tot += np.sum((p[c] - occ) ** 2)
        brier[k] = tot / (n_cells * n_frames)
    return float(SHIFT_GRID[int(np.argmin(brier))]), brier


def _calibration_analysis(true_spikes, probs):
    n_cells, n_frames = probs.shape
    shift, brier_by_shift = _best_shift(true_spikes, probs)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    n    = np.zeros((n_cells, N_BINS))
    hits = np.zeros((n_cells, N_BINS))
    psum = np.zeros((n_cells, N_BINS))
    sq   = 0.0
    for c in range(n_cells):
        p = np.clip(np.nan_to_num(probs[c].astype(np.float64)), 0.0, 1.0)
        occ = _true_occupancy(true_spikes[c], n_frames, shift)
        b = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
        n[c]    = np.bincount(b, minlength=N_BINS)
        hits[c] = np.bincount(b, weights=occ.astype(float), minlength=N_BINS)
        psum[c] = np.bincount(b, weights=p, minlength=N_BINS)
        sq += np.sum((p - occ) ** 2)

    return {
        'shift':          shift,
        'brier_by_shift': brier_by_shift,
        'brier':          sq / (n_cells * n_frames),
        'n':              n,
        'hits':           hits,
        'psum':           psum,
    }


def _simulate(n_cells, fs, duration, seed, return_sigma=False):
    from simulation_helpers import generate_synthetic_data

    np.random.seed(seed)
    noisy, true_spikes, clean, _, _, _ = generate_synthetic_data(
        n_cells=n_cells, fs=fs, duration=duration, tau=TAU,
        target_kurtosis_range=KURTOSIS_RANGE)
    true_spikes = [np.asarray(t, dtype=np.float64).ravel() for t in true_spikes]

    burst_isi = 2.0 / (10.0 * fs)
    bursty = np.array([np.sum(np.abs(np.diff(np.sort(t)) - burst_isi) < 1e-9)
                       >= BURSTY_MIN_PAIRS for t in true_spikes])
    sigma = np.std(noisy - clean, axis=1)
    if bursty.any() and (~bursty).any():
        pool = sigma[~bursty]
        for i in np.where(bursty)[0]:
            noisy[i] = clean[i] + np.random.normal(0.0, np.random.choice(pool),
                                                   size=clean.shape[1])
    print('  Bursty cells: {} of {} -- noise redrawn from non-bursty cells.'.format(
        int(bursty.sum()), n_cells))
    if return_sigma:
        return noisy, true_spikes, np.std(noisy - clean, axis=1)
    return noisy, true_spikes


def _ensure_cascade_model(model):
    code = ("import os, cascade2p.cascade as c; "
            "f = os.path.join(os.path.dirname(os.path.dirname(c.__file__)), "
            "'Pretrained_models'); "
            "os.path.isdir(os.path.join(f, {m!r})) or "
            "c.download_model({m!r}, model_folder=f, verbose=1)").format(m=model)
    subprocess.run(['conda', 'run', '-n', 'cascade', 'python', '-c', code], check=True)


def _run_cascade_at(dff, fs, work_dir, device='gpu'):
    from fractions import Fraction
    from scipy.signal import resample_poly

    model = CASCADE_MODELS.get(float(fs))
    fs_in, resampled = float(fs), model is None
    if resampled:
        fs_in = CASCADE_FALLBACK_FS
        frac = Fraction(fs_in / fs).limit_denominator(100)
        dff = resample_poly(dff, frac.numerator, frac.denominator, axis=1)
        model = CASCADE_MODELS[fs_in]
    _ensure_cascade_model(model)

    inp  = os.path.join(work_dir, 'fr_cascade_input.npz')
    outp = os.path.join(work_dir, 'fr_cascade_output.npz')
    np.savez(inp, dff=dff.astype(np.float32), fs=np.float32(fs_in))
    subprocess.run(
        ['conda', 'run', '-n', 'cascade', 'python',
         os.path.join(_HERE, 'run_cascade_subprocess.py'),
         '--mode', 'inference', '--input', inp, '--output', outp,
         '--model', model, '--device', device],
        check=True)
    r = np.load(outp, allow_pickle=True)
    spikes = [np.asarray(s, dtype=np.float64) for s in r['cascade_spikes']]
    return spikes, np.asarray(r['cascade_probs'], dtype=np.float32), resampled


def _run_oasis_at(dff, fs, return_s=False):
    from oasis.functions import deconvolve as oasis_deconv
    from figure1 import _oasis_spikes_from_s

    sigmas = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    g = np.exp(-1 / (fs * TAU))
    spikes, s_all = [], []
    for i in range(dff.shape[0]):
        _, s, _, _, _ = oasis_deconv(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        spikes.append(_oasis_spikes_from_s(s, sigmas[i], fs))
        s_all.append(s)
    if return_s:
        return spikes, np.asarray(s_all, dtype=np.float32), sigmas
    return spikes


def _run_population(data_dir, n_cells=POP_CELLS, duration=POP_DURATION,
                    run_matlab=True, run_cascade=True, device='gpu'):
    import OMSI
    from run_pnev_MCMC import run_matlab_pnevMCMC

    def _obj(xs):
        arr = np.empty(len(xs), dtype=object)
        for i, x in enumerate(xs):
            arr[i] = np.asarray(x)
        return arr

    print('Simulating {} cells x {:.0f} s at {:.0f} Hz...'.format(n_cells, duration, FS))
    noisy, true_spikes = _simulate(n_cells, FS, duration, POP_SEED)
    out = {'fs': np.array([FS]), 'duration': np.array([duration]),
           'true_spikes': _obj(true_spikes)}

    print('Running OMSI...')
    res = OMSI.deconv(noisy, dict(OMSI_PARAMS, f=FS), benchmark=True)
    out['OMSI_spikes'] = _obj(list(res['optim_spikes']))
    out['OMSI_probs']  = np.asarray(res['optim_prob'], dtype=np.float32)

    print('Running OASIS...')
    out['OASIS_spikes'] = _obj(_run_oasis_at(noisy, FS))

    if run_cascade:
        print('Running CASCADE...')
        work_dir = os.path.join(data_dir, 'population_work')
        os.makedirs(work_dir, exist_ok=True)
        spk, probs, _ = _run_cascade_at(noisy, FS, work_dir, device)
        out['CASCADE_spikes'] = _obj(spk)
        out['CASCADE_probs']  = probs

    if run_matlab:
        print('Running CaImAn (MATLAB)...')
        spk, _, probs, _ = run_matlab_pnevMCMC(noisy, fs=FS, tau=TAU,
                                               n_sweeps=MATLAB_SWEEPS)
        if sum(len(s) for s in spk) == 0:
            print('CaImAn returned no spikes -- left out.')
        else:
            out['MATLAB_spikes'] = _obj(spk)
            out['MATLAB_probs']  = np.asarray(probs, dtype=np.float32)

    out_path = os.path.join(data_dir, _POP_NPZ)
    np.savez(out_path, **out)
    print('Saved to {}.'.format(out_path))


def _run_caiman_parallel(dff, fs, work_dir, n_workers=CAIMAN_WORKERS, **kwargs):

    from concurrent.futures import ThreadPoolExecutor
    from run_pnev_MCMC import run_matlab_pnevMCMC

    chunks = [c for c in np.array_split(np.arange(dff.shape[0]), n_workers) if len(c)]

    def _one(k):
        spk, _, probs, _ = run_matlab_pnevMCMC(
            dff[chunks[k]], fs=fs, tau=TAU, n_sweeps=MATLAB_SWEEPS,
            work_dir=os.path.join(work_dir, 'w{}'.format(k)), verbose=(k == 0), **kwargs)
        return spk, probs

    with ThreadPoolExecutor(len(chunks)) as pool:
        res = list(pool.map(_one, range(len(chunks))))
    spikes = [s for spk, _ in res for s in spk]
    probs = np.concatenate([p for _, p in res], axis=0)
    return spikes, probs


def _run_caiman_fix(data_dir, n_workers=CAIMAN_WORKERS):


    pop_path = os.path.join(data_dir, _POP_NPZ)
    if not os.path.exists(pop_path):
        raise FileNotFoundError(
            'No data at {}. Run --mode test --stages population first.'.format(pop_path))
    pop = np.load(pop_path, allow_pickle=True)
    saved = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    duration = float(pop['duration'][0])

    print('Regenerating {} cells x {:.0f} s...'.format(len(saved), duration))
    noisy, true_spikes = _simulate(len(saved), FS, duration, POP_SEED)
    same = all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(saved, true_spikes))
    if not same:
        raise ValueError('Regenerated population differs from {}. Rerun the population '
                         'stage with the current code first.'.format(pop_path))

    print('Running CaImAn (MATLAB), rise-time fix + init threshold {:g}...'.format(
        CAIMAN_INIT_THR))
    spk, probs = _run_caiman_parallel(noisy, FS, os.path.join(data_dir, 'caiman_fix_work'),
                                      n_workers, fix_rise_bug=True,
                                      init_thr=CAIMAN_INIT_THR)
    n_empty = sum(len(s) == 0 for s, t in zip(spk, true_spikes) if len(t) > 0)
    if sum(len(s) for s in spk) == 0:
        raise RuntimeError('Fixed CaImAn returned no spikes.')
    if n_empty:
        print('  {} cells with true spikes got no calls -- check the MATLAB logs.'.format(
            n_empty))

    arr = np.empty(len(spk), dtype=object)
    for i, x in enumerate(spk):
        arr[i] = np.asarray(x)
    out_path = os.path.join(data_dir, _FIX_NPZ)
    np.savez(out_path, n_true=np.array([len(t) for t in true_spikes]),
             init_thr=np.array([CAIMAN_INIT_THR]),
             MATLAB_FIX_spikes=arr, MATLAB_FIX_probs=np.asarray(probs, dtype=np.float32))
    print('Saved to {}.'.format(out_path))


def _run_caiman_fix_framerate(data_dir, n_workers=CAIMAN_WORKERS):

    fr_path = os.path.join(data_dir, _FR_NPZ)
    if not os.path.exists(fr_path):
        raise FileNotFoundError(
            'No data at {}. Run --mode test --stages framerate first.'.format(fr_path))
    fr = np.load(fr_path)
    rates = fr['rates']
    n_cells, duration = int(fr['n_cells'][0]), float(fr['duration'][0])

    out = {'rates': rates, 'init_thr': np.array([CAIMAN_INIT_THR])}
    for r, fs in enumerate(rates):
        print('\nFrame rate {:.1f} Hz: regenerating {} cells...'.format(fs, n_cells))
        noisy, true_spikes = _simulate(n_cells, fs, duration, FR_SEED + r)
        if not np.array_equal(fr['n_true_{}'.format(r)], [len(t) for t in true_spikes]):
            raise ValueError('Regenerated {:g} Hz population differs from {}. Rerun the '
                             'framerate stage with the current code first.'.format(
                                 fs, fr_path))

        print('  Running CaImAn (MATLAB), fixed...')
        spk, _ = _run_caiman_parallel(noisy, fs, os.path.join(data_dir, 'caiman_fix_work'),
                                      n_workers, fix_rise_bug=True,
                                      init_thr=CAIMAN_INIT_THR)
        if sum(len(s) for s in spk) == 0:
            print('  Fixed CaImAn returned no spikes at {:.1f} Hz -- skipped.'.format(fs))
            continue
        err, cell, offset = _centered_errors(true_spikes, spk, fs)
        out['MATLAB_FIX_{}_err'.format(r)]      = err
        out['MATLAB_FIX_{}_cell'.format(r)]     = cell
        out['MATLAB_FIX_{}_offset'.format(r)]   = np.array([offset])
        out['MATLAB_FIX_{}_n_called'.format(r)] = np.array([len(s) for s in spk])

    out_path = os.path.join(data_dir, _FR_FIX_NPZ)
    np.savez(out_path, **out)
    print('\nSaved to {}.'.format(out_path))


def _load_framerate(data_dir):

    path = os.path.join(data_dir, _FR_NPZ)
    if not os.path.exists(path):
        return None
    f = np.load(path)
    fr = {k: f[k] for k in f.files}
    fix_path = os.path.join(data_dir, _FR_FIX_NPZ)
    if os.path.exists(fix_path):
        fix = np.load(fix_path)
        if np.array_equal(fix['rates'], fr['rates']):
            fr.update({k: fix[k] for k in fix.files if k.startswith('MATLAB_FIX_')})
        else:
            print('{} is from a different sweep -- rerun --mode caiman_fix.'.format(fix_path))
    return fr


def _refresh_summary(data_dir):

    summary = os.path.join(data_dir, _SUMMARY_NPZ)
    raws = [os.path.join(data_dir, f) for f in (_POP_NPZ, _FIX_NPZ)]
    raws = [f for f in raws if os.path.exists(f)]
    if not raws:
        return
    if (not os.path.exists(summary)
            or max(os.path.getmtime(f) for f in raws) > os.path.getmtime(summary)):
        print('Raw results changed -- reanalyzing...')
        _analyze_population(data_dir)


def _analyze_population(data_dir):

    path = os.path.join(data_dir, _POP_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'No data at {}. Run --mode test --stages population first.'.format(path))
    pop = np.load(path, allow_pickle=True)
    raw = {k: pop[k] for k in pop.files}
    true_spikes = [np.asarray(t, dtype=np.float64) for t in raw['true_spikes']]


    fix_path = os.path.join(data_dir, _FIX_NPZ)
    if os.path.exists(fix_path):
        fix = np.load(fix_path, allow_pickle=True)
        if np.array_equal(fix['n_true'], [len(t) for t in true_spikes]):
            raw.update({k: fix[k] for k in fix.files if k.startswith('MATLAB_FIX_')})
        else:
            print('{} is from a different population -- rerun --mode caiman_fix.'.format(
                fix_path))

    out = {'debias': np.array([DEBIAS]), 'tol_grid': TOL_GRID, 'shift_grid': SHIFT_GRID,
           'n_cells': np.array([len(true_spikes)]),
           'duration': np.array([float(raw['duration'][0])])}
    for key in METHODS:
        if '{}_spikes'.format(key) not in raw:
            continue
        called = [np.asarray(p, dtype=np.float64) for p in raw['{}_spikes'.format(key)]]

        print('{}: timing analysis...'.format(key))
        res = _timing_analysis(true_spikes, called)
        for k, v in res.items():
            out['{}_{}'.format(key, k)] = np.asarray(v)

        if '{}_probs'.format(key) in raw:
            print('{}: calibration analysis...'.format(key))
            cal = _calibration_analysis(true_spikes, raw['{}_probs'.format(key)])
            for k, v in cal.items():
                out['{}_cal_{}'.format(key, k)] = np.asarray(v)

    out_path = os.path.join(data_dir, _SUMMARY_NPZ)
    np.savez(out_path, **out)
    print('Saved to {}.'.format(out_path))


def _centered_errors(true_spikes, called, fs):
    raw = _signed_errors(true_spikes, called, FR_OFFSET_TOL)
    offset = float(np.median(raw)) if len(raw) > 0 else 0.0
    tol = max(MATCH_TOL, 1.0 / fs)
    err, cell = [], []
    for i, (t, p) in enumerate(zip(true_spikes, called)):
        p = np.asarray(p, dtype=np.float64).ravel() - offset
        it, ip = _match_pairs(t, p, tol)
        err.append(p[ip] - t[it])
        cell.append(np.full(len(it), i))
    return np.concatenate(err), np.concatenate(cell), offset


def _run_framerate(data_dir, rates=FR_RATES, n_cells=FR_CELLS, duration=FR_DURATION,
                   run_matlab=True, run_cascade=True, device='gpu'):
    import OMSI
    from run_pnev_MCMC import run_matlab_pnevMCMC

    rates = np.asarray(rates, dtype=float)
    out = {'rates': rates, 'n_cells': np.array([n_cells]),
           'duration': np.array([duration]),
           'cascade_resampled': np.zeros(len(rates), dtype=bool)}
    work_dir = os.path.join(data_dir, 'framerate_work')
    os.makedirs(work_dir, exist_ok=True)

    for r, fs in enumerate(rates):
        print('\nFrame rate {:.1f} Hz: simulating {} cells...'.format(fs, n_cells))
        noisy, true_spikes = _simulate(n_cells, fs, duration, FR_SEED + r)
        out['n_true_{}'.format(r)] = np.array([len(t) for t in true_spikes])

        called = {}
        print('  Running OMSI...')
        res = OMSI.deconv(noisy, dict(OMSI_PARAMS, f=fs), benchmark=True)
        called['OMSI'] = list(res['optim_spikes'])

        print('  Running OASIS...')
        called['OASIS'] = _run_oasis_at(noisy, fs)

        if run_cascade:
            print('  Running CASCADE...')
            called['CASCADE'], _, out['cascade_resampled'][r] = _run_cascade_at(
                noisy, fs, work_dir, device)

        if run_matlab:
            print('  Running CaImAn (MATLAB)...')
            spk, _, _, _ = run_matlab_pnevMCMC(noisy, fs=fs, tau=TAU,
                                               n_sweeps=MATLAB_SWEEPS)
            if sum(len(s) for s in spk) == 0:
                print('  CaImAn returned no spikes at {:.1f} Hz -- skipped.'.format(fs))
            else:
                called['MATLAB'] = spk

        for key, spk in called.items():
            err, cell, offset = _centered_errors(true_spikes, spk, fs)
            out['{}_{}_err'.format(key, r)]    = err
            out['{}_{}_cell'.format(key, r)]   = cell
            out['{}_{}_offset'.format(key, r)] = np.array([offset])
            out['{}_{}_n_called'.format(key, r)] = np.array([len(s) for s in spk])

    out_path = os.path.join(data_dir, _FR_NPZ)
    np.savez(out_path, **out)
    print('\nSaved to {}.'.format(out_path))


def _as_obj(xs):
    arr = np.empty(len(xs), dtype=object)
    for i, x in enumerate(xs):
        arr[i] = np.asarray(x)
    return arr


def _pack_samples(cells):
    trains = [np.asarray(s, dtype=np.float32).ravel() for c in cells for s in c]
    flat = np.concatenate(trains) if trains else np.zeros(0, dtype=np.float32)
    return (flat, np.array([len(s) for s in trains], dtype=np.int64),
            np.array([len(c) for c in cells], dtype=np.int64))


def _unpack_samples(flat, lens, n_samp):
    flat = np.asarray(flat, dtype=np.float64)
    trains = np.split(flat, np.cumsum(lens)[:-1]) if len(lens) else []
    out, k = [], 0
    for n in n_samp:
        out.append(trains[k:k + n])
        k += n
    return out


def _unpack_rows(flat, counts):
    return np.split(np.asarray(flat), np.cumsum(counts)[:-1]) if len(counts) else []


def _omsi_posterior_cell(task):
    from OMSI.sampler import cont_ca_sampler
    from OMSI.deconv import spikes_from_samples

    y, seed = task
    y = np.asarray(y, dtype=np.float64)
    S = cont_ca_sampler(y, dict(OMSI_PARAMS, f=FS, auto_stop=True, seed=int(seed)))
    _, called = spikes_from_samples(S['ss'], len(y), FS, lag_s=0.0)
    samples = [(np.asarray(sp, dtype=np.float64).ravel() - 1.0) / FS for sp in S['ss']]
    tau = np.asarray(S['g'], dtype=np.float64).reshape(-1, 2) / FS
    sn = np.sqrt(np.asarray(S['sn2'], dtype=np.float64)).ravel() if 'sn2' in S \
        else np.zeros(0)
    return samples, np.asarray(called, dtype=np.float64), tau, sn


def _run_omsi_posterior(dff, seeds, workers=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    thread_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'NUMBA_NUM_THREADS')
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: '1' for v in thread_vars})
    try:
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count(),
                                 mp_context=mp.get_context('spawn')) as ex:
            return list(ex.map(_omsi_posterior_cell,
                               [(dff[i], int(seeds[i])) for i in range(dff.shape[0])]))
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _run_caiman_posterior(dff, work_dir, n_workers=CAIMAN_WORKERS, **kwargs):
    from concurrent.futures import ThreadPoolExecutor
    from run_pnev_MCMC import run_matlab_pnevMCMC

    chunks = [c for c in np.array_split(np.arange(dff.shape[0]), n_workers) if len(c)]

    def _one(k):
        spk, _, _, _, ss, prm = run_matlab_pnevMCMC(
            dff[chunks[k]], fs=FS, tau=TAU, n_sweeps=MATLAB_SWEEPS,
            work_dir=os.path.join(work_dir, 'w{}'.format(k)), verbose=(k == 0),
            return_samples=True, return_params=True, **kwargs)
        return spk, ss, prm

    with ThreadPoolExecutor(len(chunks)) as pool:
        res = list(pool.map(_one, range(len(chunks))))

    out = []
    for spk, ss, prm in res:
        for c in range(len(spk)):
            samples = [np.asarray(st, dtype=np.float64).ravel() / FS for st in ss[c]]
            g = prm[c].get('g', np.zeros((0, 2)))
            tau = np.asarray(g, dtype=np.float64).reshape(-1, 2) / FS
            sn = np.sqrt(np.asarray(prm[c]['sn2'], dtype=np.float64).ravel()) \
                if 'sn2' in prm[c] else np.zeros(0)
            out.append((samples, np.asarray(spk[c], dtype=np.float64), tau, sn))
    return out


def _store_posterior(out, key, res):
    samples, called, tau, sn = zip(*res)
    flat, lens, n_samp = _pack_samples(samples)
    out['{}_ss'.format(key)]     = flat
    out['{}_ss_len'.format(key)] = lens
    out['{}_ss_n'.format(key)]   = n_samp
    out['{}_spikes'.format(key)] = _as_obj(called)
    out['{}_tau'.format(key)]    = np.concatenate([t.reshape(-1, 2) for t in tau])
    out['{}_tau_n'.format(key)]  = np.array([len(t) for t in tau])
    out['{}_sn'.format(key)]     = np.concatenate(sn)
    out['{}_sn_n'.format(key)]   = np.array([len(s) for s in sn])


def _run_posterior(data_dir, n_cells=POST_CELLS, n_workers=CAIMAN_WORKERS,
                   run_matlab=True):
    pop_path = os.path.join(data_dir, _POP_NPZ)
    if not os.path.exists(pop_path):
        raise FileNotFoundError(
            'No data at {}. Run --mode test --stages population first.'.format(pop_path))
    pop = np.load(pop_path, allow_pickle=True)
    saved = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    duration = float(pop['duration'][0])

    print('Regenerating {} cells x {:.0f} s...'.format(len(saved), duration))
    noisy, true_spikes, sigma = _simulate(len(saved), FS, duration, POP_SEED,
                                          return_sigma=True)
    if not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(saved, true_spikes)):
        raise ValueError('Regenerated population differs from {}. Rerun the population '
                         'stage with the current code first.'.format(pop_path))

    idx = np.arange(min(n_cells, len(saved)))
    dff = noisy[idx]
    out = {'cells': idx, 'fs': np.array([FS]), 'duration': np.array([duration]),
           'dff': dff.astype(np.float32), 'true_spikes': _as_obj([true_spikes[i] for i in idx]),
           'sigma_true': sigma[idx]}

    if 'CASCADE_probs' in pop.files:
        out['CASCADE_probs']  = np.asarray(pop['CASCADE_probs'][idx], dtype=np.float32)
        out['CASCADE_spikes'] = _as_obj([pop['CASCADE_spikes'][i] for i in idx])
    print('Running OASIS...')
    spk, s, sig = _run_oasis_at(dff, FS, return_s=True)
    out['OASIS_spikes'], out['OASIS_s'], out['OASIS_sigma'] = _as_obj(spk), s, sig

    seeds = np.random.default_rng(POST_SEED).integers(0, 2 ** 31 - 1, size=len(idx))
    print('Running OMSI on {} cells, samples kept...'.format(len(idx)))
    _store_posterior(out, 'OMSI', _run_omsi_posterior(dff, seeds))

    if run_matlab:
        for key, kw in (('MATLAB', {}),
                        ('MATLAB_FIX', {'fix_rise_bug': True, 'init_thr': CAIMAN_INIT_THR})):
            print('Running {}, samples kept...'.format(METHODS[key][0]))
            res = _run_caiman_posterior(
                dff, os.path.join(data_dir, 'posterior_work', key.lower()), n_workers, **kw)
            if sum(len(r[0]) for r in res) == 0:
                print('  {} returned no samples -- left out.'.format(METHODS[key][0]))
                continue
            _store_posterior(out, key, res)

    out_path = os.path.join(data_dir, _POST_NPZ)
    np.savez(out_path, **out)
    print('Saved to {}.'.format(out_path))


def _assign_to_anchors(anchors, samples, tol=MATCH_TOL):
    idx_all, t_all = [np.zeros(0, dtype=int)], [np.zeros(0)]
    if len(anchors) == 0:
        return idx_all[0], t_all[0]
    last = len(anchors) - 1
    for s in samples:
        s = np.asarray(s, dtype=np.float64)
        if len(s) == 0:
            continue
        j = np.searchsorted(anchors, s)
        lo, hi = np.clip(j - 1, 0, last), np.clip(j, 0, last)
        near = np.where(np.abs(s - anchors[lo]) <= np.abs(s - anchors[hi]), lo, hi)
        d = np.abs(s - anchors[near])
        ok = d <= tol
        near, d, s = near[ok], d[ok], s[ok]
        order = np.lexsort((d, near))
        near, s = near[order], s[order]
        first = np.r_[True, near[1:] != near[:-1]] if len(near) else np.zeros(0, bool)
        idx_all.append(near[first])
        t_all.append(s[first])
    return np.concatenate(idx_all), np.concatenate(t_all)


def _anchor_posteriors(true, called, samples):
    a = np.sort(np.asarray(called, dtype=np.float64).ravel())
    n_a, n_l = len(a), len(COVERAGE_LEVELS)
    idx, t = _assign_to_anchors(a, samples)
    order = np.argsort(idx, kind='stable')
    idx, t = idx[order], t[order]
    counts = np.bincount(idx, minlength=n_a) if n_a else np.zeros(0, dtype=int)
    starts = np.r_[0, np.cumsum(counts)[:-1]] if n_a else np.zeros(0, dtype=int)

    mean, sd = np.full(n_a, np.nan), np.full(n_a, np.nan)
    lo, hi = np.full((n_a, n_l), np.nan), np.full((n_a, n_l), np.nan)
    qs = np.r_[(1.0 - COVERAGE_LEVELS) / 2.0, (1.0 + COVERAGE_LEVELS) / 2.0]
    for k in np.where(counts >= MIN_SAMPLES_CI)[0]:
        x = t[starts[k]:starts[k] + counts[k]]
        mean[k], sd[k] = x.mean(), x.std(ddof=1)
        q = np.quantile(x, qs)
        lo[k], hi[k] = q[:n_l], q[n_l:]

    t_true = np.full(n_a, np.nan)
    it, ip = _match_pairs(true, a, MATCH_TOL)
    t_true[ip] = np.asarray(true, dtype=np.float64)[it]
    return {'time': a, 'p': counts / max(len(samples), 1), 'n': counts, 'mean': mean,
            'sd': sd, 'lo': lo, 'hi': hi, 'true': t_true}


def _greedy_hits(true, times, scores, tol=MATCH_TOL):
    t = np.sort(np.asarray(true, dtype=np.float64).ravel())
    used = np.zeros(len(t), dtype=bool)
    hit = np.zeros(len(times), dtype=bool)
    for c in np.argsort(-np.asarray(scores), kind='stable'):
        x = times[c]
        j = np.searchsorted(t, x)
        best, bd = -1, np.inf
        k = j - 1
        while k >= 0 and x - t[k] <= tol:
            if not used[k]:
                best, bd = k, x - t[k]
                break
            k -= 1
        k = j
        while k < len(t) and t[k] - x <= tol:
            if not used[k]:
                if t[k] - x < bd:
                    best = k
                break
            k += 1
        if best >= 0:
            used[best] = True
            hit[c] = True
    return hit


def _peak_candidates(trace, fs, floor=0.0):
    from scipy.signal import find_peaks

    trace = np.nan_to_num(np.asarray(trace, dtype=np.float64))
    pk, prop = find_peaks(trace, height=floor + 1e-12, distance=max(1, int(0.05 * fs)))
    return pk / fs, prop['peak_heights']


def _load_population_spikes(data_dir):
    path = os.path.join(data_dir, _POP_NPZ)
    if not os.path.exists(path):
        return None, {}
    pop = np.load(path, allow_pickle=True)
    raw = {k: pop[k] for k in pop.files}
    true_spikes = [np.asarray(t, dtype=np.float64) for t in raw['true_spikes']]
    fix_path = os.path.join(data_dir, _FIX_NPZ)
    if os.path.exists(fix_path):
        fix = np.load(fix_path, allow_pickle=True)
        if np.array_equal(fix['n_true'], [len(t) for t in true_spikes]):
            raw.update({k: fix[k] for k in fix.files if k.startswith('MATLAB_FIX_')})
    called = {key: [np.asarray(p, dtype=np.float64) for p in raw['{}_spikes'.format(key)]]
              for key in METHODS if '{}_spikes'.format(key) in raw}
    return true_spikes, called


def _window_counts(times, starts, ends):
    times = np.sort(np.asarray(times, dtype=np.float64).ravel())
    return (np.searchsorted(times, ends, side='right')
            - np.searchsorted(times, starts, side='left')).astype(float)


def _window_sums(trace, starts, ends):
    trace = np.nan_to_num(np.asarray(trace, dtype=np.float64).ravel())
    csum = np.r_[0.0, np.cumsum(trace)]
    lo = np.clip(np.ceil(np.asarray(starts) * FS).astype(int), 0, len(trace))
    hi = np.clip(np.floor(np.asarray(ends) * FS).astype(int) + 1, 0, len(trace))
    return csum[np.maximum(hi, lo)] - csum[lo]


def _population_count_traces(data_dir, true_spikes):
    traces = {}
    if true_spikes is None:
        return traces
    pop = np.load(os.path.join(data_dir, _POP_NPZ), allow_pickle=True)
    if 'CASCADE_probs' in pop.files:
        traces['CASCADE'] = np.asarray(pop['CASCADE_probs'], dtype=np.float64)

    n_true = np.array([len(t) for t in true_spikes])
    path = os.path.join(data_dir, _OASIS_S_NPZ)
    if os.path.exists(path):
        f = np.load(path)
        if np.array_equal(f['n_true'], n_true):
            traces['OASIS'] = f['s'].astype(np.float64)
            return traces
    print('Regenerating population for OASIS deconvolved signal...')
    noisy, regen = _simulate(len(true_spikes), FS, float(pop['duration'][0]), POP_SEED)
    if not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(true_spikes, regen)):
        print('  Regenerated population differs -- OASIS pairs use called spikes.')
        return traces
    _, s, _ = _run_oasis_at(noisy, FS, return_s=True)
    np.savez(path, n_true=n_true, s=s)
    traces['OASIS'] = s.astype(np.float64)
    return traces


def _analyze_posterior(data_dir):
    out = {}

    true_spikes, called = _load_population_spikes(data_dir)
    traces = _population_count_traces(data_dir, true_spikes)
    for key, spk in called.items():
        print('{}: spike-pair analysis...'.format(key))
        raw = _signed_errors(true_spikes, spk)
        offset = float(np.median(raw)) if len(raw) else 0.0
        spacing, n_pair, n_single, cell = [], [], [], []
        for c, (t, p) in enumerate(zip(true_spikes, spk)):
            t = np.sort(t)
            if len(t) < 2:
                continue
            gap = np.diff(t)
            before = np.r_[np.inf, gap[:-1]]
            after = np.r_[gap[1:], np.inf]
            iso = (before >= PAIR_ISO) & (after >= PAIR_ISO)
            single = (np.r_[np.inf, gap] >= PAIR_ISO) & (np.r_[gap, np.inf] >= PAIR_ISO)
            a = np.r_[t[:-1][iso], t[single]] - PAIR_WIN
            b = np.r_[t[1:][iso], t[single]] + PAIR_WIN
            if key in traces:
                n = _window_sums(traces[key][c], a + offset, b + offset)
            else:
                n = _window_counts(p, a + offset, b + offset)
            n_iso = int(iso.sum())
            spacing.append(gap[iso])
            n_pair.append(n[:n_iso])
            n_single.append(n[n_iso:])
            cell.append(np.full(n_iso, c))
        n_single = np.concatenate(n_single)
        scale = float(np.median(n_single)) if len(n_single) else 1.0
        scale = scale if scale > 0 else 1.0
        n_pair = np.concatenate(n_pair) / scale
        n_single = n_single / scale
        print('  offset {:+.1f} ms, count scale {:.3f}, isolated singles counted as one: '
              '{:.2f}'.format(offset * 1e3, scale,
                              np.mean(np.round(n_single) == 1) if len(n_single) else np.nan))
        out['pair_{}_spacing'.format(key)] = np.concatenate(spacing)
        out['pair_{}_count'.format(key)]   = n_pair
        out['pair_{}_ok'.format(key)]      = np.round(n_pair) == 2
        out['pair_{}_cell'.format(key)]    = np.concatenate(cell)
        out['pair_{}_offset'.format(key)]  = np.array([offset])
        out['pair_{}_scale'.format(key)]   = np.array([scale])
        out['pair_{}_single_ok'.format(key)] = np.round(n_single) == 1

    post_path = os.path.join(data_dir, _POST_NPZ)
    if os.path.exists(post_path):
        post = np.load(post_path, allow_pickle=True)
        true = [np.asarray(t, dtype=np.float64) for t in post['true_spikes']]
        n_cells = len(true)
        out['post_n_true'] = np.array([len(t) for t in true])
        out['post_sigma_true'] = post['sigma_true']

        for key in POST_METHODS:
            if '{}_ss'.format(key) not in post.files:
                continue
            print('{}: posterior analysis...'.format(key))
            samples = _unpack_samples(post['{}_ss'.format(key)],
                                      post['{}_ss_len'.format(key)],
                                      post['{}_ss_n'.format(key)])
            cols = {k: [] for k in ('cell', 'time', 'p', 'n', 'mean', 'sd', 'lo', 'hi',
                                    'true', 'hit')}
            for c in range(n_cells):
                r = _anchor_posteriors(true[c], post['{}_spikes'.format(key)][c], samples[c])
                r['cell'] = np.full(len(r['time']), c)
                r['hit'] = _greedy_hits(true[c], r['time'], r['p'])
                for k in cols:
                    cols[k].append(r[k])
            for k, v in cols.items():
                out['{}_a_{}'.format(key, k)] = np.concatenate(v)

            tau = _unpack_rows(post['{}_tau'.format(key)], post['{}_tau_n'.format(key)])
            sn  = _unpack_rows(post['{}_sn'.format(key)], post['{}_sn_n'.format(key)])
            par = np.full((n_cells, 3, 3), np.nan)
            for c in range(n_cells):
                for j, x in enumerate((tau[c][:, 0] if len(tau[c]) else [],
                                       tau[c][:, 1] if len(tau[c]) else [],
                                       sn[c] if len(sn) else [])):
                    if len(x):
                        par[c, j] = np.percentile(x, [50.0, 2.5, 97.5])
            out['{}_par'.format(key)] = par

        traces = {}
        if 'CASCADE_probs' in post.files:
            traces['CASCADE'] = (post['CASCADE_probs'], np.ones(n_cells), 1e-3)
        if 'OASIS_s' in post.files:
            traces['OASIS'] = (post['OASIS_s'], post['OASIS_sigma'], 1e-3)
        for key, (tr, scale, floor) in traces.items():
            print('{}: ranking analysis...'.format(key))
            cols = {'cell': [], 'time': [], 'p': [], 'hit': []}
            for c in range(n_cells):
                tm, sc = _peak_candidates(tr[c] / scale[c], FS, floor)
                cols['cell'].append(np.full(len(tm), c))
                cols['time'].append(tm)
                cols['p'].append(sc)
                cols['hit'].append(_greedy_hits(true[c], tm, sc))
            for k, v in cols.items():
                out['{}_a_{}'.format(key, k)] = np.concatenate(v)


    out_path = os.path.join(data_dir, _POST_SUMMARY_NPZ)
    np.savez(out_path, **out)
    print('Saved to {}.'.format(out_path))


def _refresh_posterior(data_dir):
    summary = os.path.join(data_dir, _POST_SUMMARY_NPZ)
    raws = [os.path.join(data_dir, f) for f in (_POP_NPZ, _FIX_NPZ, _POST_NPZ)]
    raws = [f for f in raws if os.path.exists(f)]
    if not raws:
        return
    if (not os.path.exists(summary)
            or max(os.path.getmtime(f) for f in raws) > os.path.getmtime(summary)):
        print('Raw results changed -- reanalyzing posteriors...')
        _analyze_posterior(data_dir)


def _pick_example(true, post):
    if EXAMPLE_CELL is not None and EXAMPLE_START is not None:
        return int(EXAMPLE_CELL), float(EXAMPLE_START)
    duration = float(post['duration'][0])
    best, best_score = (0, 0.0), -np.inf
    for c, t in enumerate(true):
        t = np.sort(t)
        for t0 in t - 0.25 * EXAMPLE_WIN:
            if t0 < 0 or t0 + EXAMPLE_WIN > duration:
                continue
            w = t[(t >= t0) & (t < t0 + EXAMPLE_WIN)]
            if len(w) < 2:
                continue
            n_close = int(np.sum(np.diff(w) < 1.0 / FS))
            if n_close == 0 or n_close > 2 or np.any(np.abs(t - t0) < 0.05) \
                    or np.any(np.abs(t - t0 - EXAMPLE_WIN) < 0.05):
                continue
            score = -abs(len(w) - 1.5 * EXAMPLE_WIN)
            if score > best_score:
                best, best_score = (c, float(t0)), score
    return best


def _cell_rate_band(hits, totals, rng):
    import warnings

    n_cells = hits.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rate = hits.sum(0) / totals.sum(0)
        boot = np.empty((N_BOOT, hits.shape[1]))
        for b in range(N_BOOT):
            pick = rng.randint(0, n_cells, n_cells)
            boot[b] = hits[pick].sum(0) / totals[pick].sum(0)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return rate, lo, hi


def _per_cell_counts(cell, bins, ok, n_cells, n_bins):
    keep = (bins >= 0) & (bins < n_bins)
    cell, bins, ok = cell[keep], bins[keep], ok[keep]
    flat = cell * n_bins + bins
    totals = np.bincount(flat, minlength=n_cells * n_bins).reshape(n_cells, n_bins)
    hits = np.bincount(flat, weights=ok.astype(float),
                       minlength=n_cells * n_bins).reshape(n_cells, n_bins)
    return hits, totals


def _coverage(d, key):
    tt, lo, hi = d['{}_a_true'.format(key)], d['{}_a_lo'.format(key)], d['{}_a_hi'.format(key)]
    cell = d['{}_a_cell'.format(key)]
    use = np.isfinite(tt) & np.isfinite(lo[:, 0])
    n_cells = int(cell.max()) + 1 if len(cell) else 0
    inside = (lo[use] <= tt[use, None]) & (tt[use, None] <= hi[use])
    hits = np.zeros((n_cells, len(COVERAGE_LEVELS)))
    totals = np.zeros_like(hits)
    np.add.at(hits, cell[use], inside.astype(float))
    np.add.at(totals, cell[use], 1.0)
    return hits, totals


def _pr_curve(score, hit, n_true):
    order = np.argsort(-score, kind='stable')
    s, h = score[order], hit[order].astype(float)
    tp = np.cumsum(h)
    n = np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]
    prec, rec = tp[last] / n[last], tp[last] / max(n_true, 1)
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return prec, rec, s[last], ap


def _plot_pairs(ax, d, rng):
    n_pb = len(PAIR_EDGES_MS) - 1
    pair_keys = [k for k in METHODS if 'pair_{}_ok'.format(k) in d.files]
    for key in pair_keys:
        color = METHODS[key][1]
        sp = d['pair_{}_spacing'.format(key)] * 1e3
        cell = d['pair_{}_cell'.format(key)]
        b = np.digitize(sp, PAIR_EDGES_MS) - 1
        hits, totals = _per_cell_counts(cell, b, d['pair_{}_ok'.format(key)],
                                        int(cell.max()) + 1, n_pb)
        rate, lo, hi = _cell_rate_band(hits, totals, rng)
        x = np.array([np.median(sp[b == k]) if np.any(b == k) else np.nan
                      for k in range(n_pb)])
        ok = np.isfinite(x) & (totals.sum(0) >= 10)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.15, linewidth=0)
        ax.plot(x[ok], rate[ok], METHOD_LS.get(key, '-'), color=color, lw=1.0)
    ax.axvline(1000.0 / FS, color='k', ls='--', lw=0.7, alpha=0.6)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel('spike spacing (ms)')
    ax.set_ylabel('pairs counted as two spikes')
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(['5', '10', '20', '50', '100', '200', '500'])
    ax.minorticks_off()
    return pair_keys


def _plot_existence(ax, d, key, rng, marker='o'):
    n_eb = len(EXIST_EDGES) - 1
    color, ls = METHODS[key][1], METHOD_LS.get(key, '-')
    pr = np.clip(d['{}_a_p'.format(key)], 0.0, 1.0)
    hit = d['{}_a_hit'.format(key)]
    cell = d['{}_a_cell'.format(key)]
    b = np.clip(np.digitize(pr, EXIST_EDGES) - 1, 0, n_eb - 1)
    hits, totals = _per_cell_counts(cell, b, hit, int(cell.max()) + 1, n_eb)
    rate, lo, hi = _cell_rate_band(hits, totals, rng)
    x = np.array([pr[b == k].mean() if np.any(b == k) else np.nan for k in range(n_eb)])
    ok = np.isfinite(x)
    ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.15, linewidth=0)
    ax.plot(x[ok], rate[ok], ls, color=color, lw=1.0, marker=marker, ms=2)
    return totals.sum(0)


def _plot_example(fig, spec, d, post):
    cell, t0 = _pick_example([np.asarray(t, dtype=np.float64) for t in post['true_spikes']],
                             post)
    t1 = t0 + EXAMPLE_WIN
    short = EXAMPLE_WIN <= 3.0
    true = np.asarray(post['true_spikes'][cell], dtype=np.float64)
    tw = true[(true >= t0) & (true < t1)] - t0
    mcmc = [k for k in POST_METHODS if '{}_ss'.format(k) in post.files]
    rows = ['trace'] + mcmc + [k for k in ('CASCADE', 'OASIS')
                               if '{}_spikes'.format(k) in post.files]
    ratios = {'trace': 1.1, 'CASCADE': 0.55, 'OASIS': 0.55}
    gs = gridspec.GridSpecFromSubplotSpec(
        len(rows), 1, subplot_spec=spec, hspace=0.12,
        height_ratios=[ratios.get(r, 1.0) for r in rows])
    frames = np.arange(int(np.ceil(t0 * FS)), int(np.floor(t1 * FS)) + 1)
    i95 = int(np.argmin(np.abs(COVERAGE_LEVELS - 0.95)))

    axes = []
    for r, row in enumerate(rows):
        ax = fig.add_subplot(gs[r], sharex=axes[0] if axes else None)
        axes.append(ax)
        for f in (frames if short else []):
            ax.axvline(f / FS - t0, color='0.92', lw=0.4, zorder=0)
        for t in tw:
            ax.axvline(t, color='0.55', lw=0.6, ls=':', zorder=1)
        ax.spines['left'].set_visible(row == 'trace')
        ax.tick_params(axis='x', labelbottom=(r == len(rows) - 1))
        label = 'dF/F' if row == 'trace' else METHODS[row][0]
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=6)

        if row == 'trace':
            y = post['dff'][cell, frames]
            ax.plot(frames / FS - t0, y, '-', color='0.4', lw=0.7)
            if short:
                ax.plot(frames / FS - t0, y, 'o', color='0.4', ms=1.2)
            ax.plot(tw, np.full(len(tw), y.max() * 1.15), 'v', color='k', ms=3)
            ax.set_yticks([])
        elif row in mcmc:
            color, ls = METHODS[row][1], METHOD_LS.get(row, '-')
            samples = _unpack_samples(post['{}_ss'.format(row)],
                                      post['{}_ss_len'.format(row)],
                                      post['{}_ss_n'.format(row)])[cell]
            pick = np.linspace(0, len(samples) - 1, min(N_RASTER, len(samples))).astype(int)
            for j, k in enumerate(pick):
                s = samples[k]
                s = s[(s >= t0) & (s < t1)] - t0
                ax.vlines(s, j + 0.1, j + 0.9, color=color, lw=0.8, alpha=0.8)
            sel = (d['{}_a_cell'.format(row)] == cell)
            at = d['{}_a_time'.format(row)][sel]
            inw = (at >= t0) & (at < t1)
            p = d['{}_a_p'.format(row)][sel][inw]
            mean = d['{}_a_mean'.format(row)][sel][inw] - t0
            lo = d['{}_a_lo'.format(row)][sel][inw, i95] - t0
            hi = d['{}_a_hi'.format(row)][sel][inw, i95] - t0
            yb = -1.6
            for m, a, b, pp in zip(mean, lo, hi, p):
                if not np.isfinite(m):
                    continue
                ax.plot([a, b], [yb, yb], ls, color=color, lw=1.2, solid_capstyle='butt')
                ax.plot(m, yb, 'o', color=color, ms=2.2)
                if short or pp < 0.9:
                    ax.text(m, yb - 1.1, '{:.2f}'.format(pp), ha='center', va='top',
                            fontsize=4.5, color=color)
            ax.set_ylim(-3.6, len(pick) + 0.3)
            ax.set_yticks([])
        elif row in ('CASCADE', 'OASIS'):
            if row == 'CASCADE':
                tr = post['CASCADE_probs'][cell, frames]
            else:
                tr = post['OASIS_s'][cell, frames] / post['OASIS_sigma'][cell]
            ax.step(frames / FS - t0, tr, where='mid', color=METHODS[row][1], lw=0.8)
            sp = np.asarray(post['{}_spikes'.format(row)][cell], dtype=np.float64)
            sp = sp[(sp >= t0) & (sp < t1)] - t0
            top = max(1.0, float(np.nanmax(tr)))
            ax.plot(sp, np.full(len(sp), top * 1.1), 'v', color=METHODS[row][1], ms=2.5)
            ax.set_ylim(0, top * 1.25)
            ax.set_yticks([])
        if r < len(rows) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)

    axes[-1].set_xlim(0, EXAMPLE_WIN)
    axes[-1].set_xlabel('time (s)')


def plot_posterior_figure(data_dir=_DEFAULT_DATA_DIR):
    _refresh_posterior(data_dir)
    path = os.path.join(data_dir, _POST_SUMMARY_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError('No data at {}. Run --mode test first.'.format(path))
    d = np.load(path)
    post_path = os.path.join(data_dir, _POST_NPZ)
    post = np.load(post_path, allow_pickle=True) if os.path.exists(post_path) else None
    mcmc = [k for k in POST_METHODS if '{}_a_sd'.format(k) in d.files]
    rng = np.random.RandomState(0)

    fig = plt.figure(figsize=(5.6, 11.6))
    outer = gridspec.GridSpec(5, 2, figure=fig, wspace=0.45, hspace=0.55, top=0.96,
                              height_ratios=[2.3, 1.0, 1.0, 0.9, 1.0])
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[2, 0])
    ax_e = fig.add_subplot(outer[2, 1])
    ax_g = fig.add_subplot(outer[4, 0])
    ax_h = fig.add_subplot(outer[4, 1])

    if post is not None and mcmc:
        _plot_example(fig, outer[0, :], d, post)
    else:
        _placeholder(fig.add_subplot(outer[0, :]), 'posterior')

    ax_b.plot([0, 1], [0, 1], '--', color='k', lw=0.7, alpha=0.6)
    for key in mcmc:
        color = METHODS[key][1]
        hits, totals = _coverage(d, key)
        rate, lo, hi = _cell_rate_band(hits, totals, rng)
        ax_b.fill_between(COVERAGE_LEVELS, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax_b.plot(COVERAGE_LEVELS, rate, METHOD_LS.get(key, '-'), color=color, lw=1.0)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel('credible level')
    ax_b.set_ylabel('true spikes inside interval')
    if not mcmc:
        _placeholder(ax_b, 'posterior')

    sd_edges = np.geomspace(0.3, 100.0, 13)
    ax_c.plot([0.3, 100], [0.3, 100], '--', color='k', lw=0.7, alpha=0.6)
    for key in mcmc:
        color = METHODS[key][1]
        sd = d['{}_a_sd'.format(key)] * 1e3
        err = (d['{}_a_mean'.format(key)] - d['{}_a_true'.format(key)]) * 1e3
        ok = np.isfinite(sd) & np.isfinite(err)
        b = np.digitize(sd[ok], sd_edges) - 1
        xs, ys = [], []
        for k in range(len(sd_edges) - 1):
            m = b == k
            if m.sum() >= 20:
                xs.append(np.median(sd[ok][m]))
                ys.append(np.sqrt(np.mean(err[ok][m] ** 2)))
        ax_c.plot(xs, ys, METHOD_LS.get(key, '-'), color=color, lw=1.0, marker='o', ms=2)
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlim(0.3, 100)
    ax_c.set_ylim(0.3, 100)
    ax_c.set_xlabel('posterior SD (ms)')
    ax_c.set_ylabel('RMS timing error (ms)')
    if not mcmc:
        _placeholder(ax_c, 'posterior')

    pair_keys = _plot_pairs(ax_d, d, rng)
    if not pair_keys:
        _placeholder(ax_d, 'population')

    n_true = int(d['post_n_true'].sum()) if 'post_n_true' in d.files else 0
    default_thr = {'CASCADE': 0.5, 'OASIS': 1.0}
    rank_keys = [k for k in METHODS if '{}_a_hit'.format(k) in d.files]
    for key in rank_keys:
        color = METHODS[key][1]
        prec, rec, thr, _ = _pr_curve(d['{}_a_p'.format(key)], d['{}_a_hit'.format(key)],
                                      n_true)
        ax_e.plot(rec, prec, METHOD_LS.get(key, '-'), color=color, lw=1.0)
        if key in default_thr:
            j = int(np.argmin(np.abs(thr - default_thr[key])))
        else:
            j = len(thr) - 1
        ax_e.plot(rec[j], prec[j], 'o', color=color, ms=3, mec='k', mew=0.3)
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1.02)
    ax_e.set_xlabel('recall')
    ax_e.set_ylabel('precision')
    if not rank_keys:
        _placeholder(ax_e, 'posterior')

    gs_f = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3, :], wspace=0.6)
    par_names = [('rise time (ms)', 1e3, True), ('decay time (s)', 1.0, False),
                 ('noise SD / true', 1.0, True)]
    sig = d['post_sigma_true'] if 'post_sigma_true' in d.files else None
    for j, (name, scale, logy) in enumerate(par_names):
        ax = fig.add_subplot(gs_f[j])
        if not mcmc:
            _placeholder(ax, 'posterior')
            continue
        truth_line = {0: TRUE_TAU_RISE * scale, 1: TAU * scale, 2: 1.0}[j]
        ax.axhline(truth_line, color='k', ls='--', lw=0.7, alpha=0.6)
        for m, key in enumerate(mcmc):
            color = METHODS[key][1]
            par = d['{}_par'.format(key)][:, j, :]
            if j == 2:
                par = par / sig[:, None]
            truth = {0: TRUE_TAU_RISE, 1: TAU, 2: 1.0}[j]
            ok = np.all(np.isfinite(par), axis=1)
            if not ok.any():
                continue
            x = m + np.linspace(-0.3, 0.3, ok.sum())
            order = np.argsort(par[ok, 0])
            pm = par[ok][order] * scale
            ax.vlines(x, pm[:, 1], pm[:, 2], color=color, lw=0.5, alpha=0.6)
            ax.plot(x, pm[:, 0], 'o', color=color, ms=1.5)
            cov = np.mean((par[ok, 1] <= truth) & (truth <= par[ok, 2]))
            ax.text(m, 1.02, '{:.0%}'.format(cov), transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=5.5, color=color)
        ax.set_xticks(range(len(mcmc)))
        ax.set_xticklabels([METHODS[k][0] for k in mcmc], rotation=30, ha='right',
                           fontsize=5.5)
        ax.set_xlim(-0.6, len(mcmc) - 0.4)
        if logy:
            ax.set_yscale('log')
            fmt = mpl.ticker.FuncFormatter(lambda v, _: '{:g}'.format(v))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylabel(name)

    rel_keys = [k for k in mcmc + ['CASCADE'] if '{}_a_hit'.format(k) in d.files]
    ax_g.plot([0, 1], [0, 1], '--', color='k', lw=0.7, alpha=0.6)
    for m, key in enumerate(rel_keys):
        totals = _plot_existence(ax_g, d, key, rng)
        width = (EXIST_EDGES[1] - EXIST_EDGES[0]) / (len(rel_keys) + 1)
        xb = EXIST_EDGES[:-1] + width * (m + 1)
        ax_h.bar(xb, totals, width=width, color=METHODS[key][1],
                 alpha=0.8 if METHOD_LS.get(key, '-') == '-' else 0.4, linewidth=0)
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1)
    ax_g.set_xlabel('spike existence probability')
    ax_g.set_ylabel('called spikes that are real')
    ax_h.set_yscale('log')
    ax_h.set_xlim(0, 1)
    ax_h.set_xticks(EXIST_EDGES)
    ax_h.set_xlabel('spike existence probability')
    ax_h.set_ylabel('called spikes')
    if not rel_keys:
        _placeholder(ax_g, 'posterior')
        _placeholder(ax_h, 'posterior')

    handles = [plt.Line2D([], [], color=c, ls=METHOD_LS.get(k, '-'), lw=1.0, label=l)
               for k, (l, c) in METHODS.items()
               if k in pair_keys or k in mcmc or k in rank_keys]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, 0.995))

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'timing_posterior.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def _print_posterior_stats(data_dir):
    _refresh_posterior(data_dir)
    path = os.path.join(data_dir, _POST_SUMMARY_NPZ)
    if not os.path.exists(path):
        return
    d = np.load(path)

    pair_keys = [k for k in METHODS if 'pair_{}_ok'.format(k) in d.files]
    if pair_keys:
        n_pb = len(PAIR_EDGES_MS) - 1
        print('\nIsolated spike pairs counted as two spikes (offset removed, window '
              '+-{:.0f} ms around the pair, no other spike within {:.1f} s):'.format(
                  PAIR_WIN * 1e3, PAIR_ISO))
        heads = ['{:g}-{:g}'.format(PAIR_EDGES_MS[k], PAIR_EDGES_MS[k + 1])
                 for k in range(n_pb)]
        print('  {:<22}'.format('method') + ''.join('{:>9}'.format(h) for h in heads)
              + '   (ms)')
        sp0 = d['pair_{}_spacing'.format(pair_keys[0])] * 1e3
        b0 = np.digitize(sp0, PAIR_EDGES_MS) - 1
        print('  {:<22}'.format('n pairs') + ''.join(
            '{:>9d}'.format(int(np.sum(b0 == k))) for k in range(n_pb)))
        for key in pair_keys:
            sp = d['pair_{}_spacing'.format(key)] * 1e3
            ok = d['pair_{}_ok'.format(key)]
            b = np.digitize(sp, PAIR_EDGES_MS) - 1
            row = [np.mean(ok[b == k]) if np.any(b == k) else np.nan for k in range(n_pb)]
            print('  {:<22}'.format(METHODS[key][0]) + ''.join('{:>9.2f}'.format(v)
                                                               for v in row)
                  + '   singles as one: {:.2f}'.format(
                      np.mean(d['pair_{}_single_ok'.format(key)])))

    mcmc = [k for k in POST_METHODS if '{}_a_sd'.format(k) in d.files]
    if not mcmc:
        return
    print('\nPosterior stage: {} cells, {} true spikes.'.format(
        len(d['post_n_true']), int(d['post_n_true'].sum())))

    print('\nTiming interval coverage (matched spikes with >= {} samples):'.format(
        MIN_SAMPLES_CI))
    show = [0.5, 0.8, 0.9, 0.95]
    print('  {:<22} {:>8}'.format('method', 'n') + ''.join(
        '{:>8.0%}'.format(q) for q in show) + '{:>11}{:>11}{:>9}'.format(
        'med SD', 'RMS err', 'RMS z'))
    for key in mcmc:
        hits, totals = _coverage(d, key)
        rate = hits.sum(0) / np.maximum(totals.sum(0), 1)
        sd = d['{}_a_sd'.format(key)] * 1e3
        err = (d['{}_a_mean'.format(key)] - d['{}_a_true'.format(key)]) * 1e3
        ok = np.isfinite(sd) & np.isfinite(err) & (sd > 0)
        print('  {:<22} {:>8d}'.format(METHODS[key][0], int(totals[:, 0].sum())) + ''.join(
            '{:>8.1%}'.format(rate[int(np.argmin(np.abs(COVERAGE_LEVELS - q)))])
            for q in show) + '{:>9.1f}ms{:>9.1f}ms{:>9.2f}'.format(
            np.median(sd[ok]), np.sqrt(np.mean(err[ok] ** 2)),
            np.sqrt(np.mean((err[ok] / sd[ok]) ** 2))))

    n_true = int(d['post_n_true'].sum())
    print('\nRanking spikes by confidence (greedy match, +-{:.0f} ms):'.format(
        MATCH_TOL * 1e3))
    print('  {:<22} {:>8} {:>10} {:>10} {:>10}'.format(
        'method', 'AP', 'max rec', 'P at max', 'cands'))
    for key in METHODS:
        if '{}_a_hit'.format(key) not in d.files:
            continue
        prec, rec, _, ap = _pr_curve(d['{}_a_p'.format(key)], d['{}_a_hit'.format(key)],
                                     n_true)
        print('  {:<22} {:>8.3f} {:>10.3f} {:>10.3f} {:>10d}'.format(
            METHODS[key][0], ap, rec[-1], prec[-1], len(d['{}_a_p'.format(key)])))

    print('\nExistence probability vs. fraction of called spikes that are true:')
    edges = EXIST_EDGES
    print('  {:<22}'.format('method') + ''.join(
        '{:>11}'.format('{:.1f}-{:.1f}'.format(edges[k], edges[k + 1])) for k in range(5)))
    for key in mcmc:
        p, h = d['{}_a_p'.format(key)], d['{}_a_hit'.format(key)]
        b = np.clip(np.digitize(p, edges) - 1, 0, 4)
        print('  {:<22}'.format(METHODS[key][0]) + ''.join(
            '{:>11}'.format('{:.2f} ({:d})'.format(np.mean(h[b == k]), int(np.sum(b == k)))
                            if np.any(b == k) else '--') for k in range(5)))

    sig = d['post_sigma_true']
    print('\nParameter posteriors: median over cells of the posterior median, and '
          'fraction of cells whose 95% interval holds the truth:')
    print('  {:<22} {:>18} {:>18} {:>18}'.format(
        'method', 'rise (true {:.0f} ms)'.format(TRUE_TAU_RISE * 1e3),
        'decay (true {:.1f} s)'.format(TAU), 'noise SD / true'))
    for key in mcmc:
        par = d['{}_par'.format(key)]
        cells = []
        for j, (truth, scale) in enumerate(((TRUE_TAU_RISE, 1e3), (TAU, 1.0), (None, 1.0))):
            pj = par[:, j, :] / (sig[:, None] if truth is None else 1.0)
            t = 1.0 if truth is None else truth
            ok = np.all(np.isfinite(pj), axis=1)
            if not ok.any():
                cells.append('--')
                continue
            cov = np.mean((pj[ok, 1] <= t) & (t <= pj[ok, 2]))
            cells.append('{:.3g}, {:.0%}'.format(np.median(pj[ok, 0]) * scale, cov))
        print('  {:<22} {:>18} {:>18} {:>18}'.format(METHODS[key][0], *cells))


def run_test(data_dir=_DEFAULT_DATA_DIR, stages=('population', 'framerate'),
             pop_cells=POP_CELLS, pop_duration=POP_DURATION,
             fr_rates=FR_RATES, fr_cells=FR_CELLS, fr_duration=FR_DURATION,
             run_matlab=True, run_cascade=True, device='gpu', post_cells=POST_CELLS,
             workers=CAIMAN_WORKERS):
    os.makedirs(data_dir, exist_ok=True)
    if 'population' in stages:
        print('Stage population...')
        _run_population(data_dir, pop_cells, pop_duration, run_matlab, run_cascade, device)
    if 'population' in stages or 'analyze' in stages:
        print('\nStage analyze...')
        _analyze_population(data_dir)
    if 'framerate' in stages:
        print('\nStage framerate...')
        _run_framerate(data_dir, fr_rates, fr_cells, fr_duration,
                       run_matlab, run_cascade, device)
    if 'posterior' in stages:
        print('\nStage posterior...')
        _run_posterior(data_dir, post_cells, workers, run_matlab)
        _analyze_posterior(data_dir)


def _present(d):
    return [(k, v) for k, v in METHODS.items() if '{}_fb'.format(k) in d.files]


def _ece(n, hits, psum):
    n, hits, psum = (np.asarray(a, dtype=float).sum(axis=0) if np.ndim(a) > 1
                     else np.asarray(a, dtype=float) for a in (n, hits, psum))
    ok = n > 0
    return float(np.sum(np.abs(hits[ok] - psum[ok])) / np.sum(n[ok]))


def _placeholder(ax, stage):
    ax.text(0.5, 0.5, 'run --mode test\n--stages {}'.format(stage), ha='center',
            va='center', transform=ax.transAxes, color='0.5')
    ax.set_xticks([])
    ax.set_yticks([])


def _peak_offset(err_ms, bins, smooth=5):
    if len(err_ms) == 0:
        return 0.0
    counts, _ = np.histogram(err_ms, bins=bins)
    counts = np.convolve(counts, np.ones(smooth) / smooth, mode='same')
    centers = 0.5 * (bins[:-1] + bins[1:])
    return float(centers[int(np.argmax(counts))])


def _centered_spread(err, cell, rng):
    if len(err) == 0:
        return np.nan, np.nan, np.nan
    cells = np.unique(cell)
    by_cell = [err[cell == c] for c in cells]
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.randint(0, len(cells), len(cells))
        boot[b] = _mad(np.concatenate([by_cell[i] for i in pick]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return _mad(err) * 1e3, lo * 1e3, hi * 1e3


def plot_figure(data_dir=_DEFAULT_DATA_DIR):
    _refresh_summary(data_dir)
    path = os.path.join(data_dir, _SUMMARY_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError('No data at {}. Run --mode test first.'.format(path))
    d = np.load(path)
    fr = _load_framerate(data_dir)

    tol_ms = d['tol_grid'] * 1e3
    debias = bool(d['debias'][0])
    rng = np.random.RandomState(0)

    fig = plt.figure(figsize=(8.4, 4.8))
    outer = gridspec.GridSpec(2, 3, figure=fig, wspace=0.45, hspace=0.55)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 2])
    ax_e = fig.add_subplot(outer[0, 2])
    ax_f = fig.add_subplot(outer[1, 1])
    for ax in (ax_e, ax_f):
        ax.set_box_aspect(1)

    step = 1000.0 / (FS * 10.0)
    bins = np.arange(-150.0 - step / 2, 150.0 + step, step)
    centers = 0.5 * (bins[:-1] + bins[1:])
    handles = []
    for key, (label, color) in _present(d):
        err_ms = d['{}_err_s'.format(key)] * 1e3
        peak = _peak_offset(err_ms, bins)
        print('{}: panel a peak at {:+.1f} ms, shifted to 0.'.format(key, peak))
        dens, _ = np.histogram(err_ms - peak, bins=bins, density=True)
        off = d['{}_offset_s'.format(key)] * 1e3
        name = '{} ({:+.0f} ms)'.format(label, off) if debias else label
        h, = ax_a.plot(centers, dens, METHOD_LS.get(key, '-'), color=color, lw=1.0,
                       label=name)
        handles.append(h)

    ax_a.axvspan(-500.0 / FS, 500.0 / FS, color='0.5', alpha=0.10, linewidth=0)
    ax_a.set_xlim(-30, 30)
    ax_a.axvline(0, color='k', ls='--', lw=0.7, alpha=0.6)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel('timing error, peak-centered (ms)')
    ax_a.set_ylabel('matched spikes (fraction / ms)')
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, 0.99))

    for key, (label, color) in _present(d):
        fb = d['{}_fb'.format(key)]
        med = np.nanmedian(fb, axis=1)
        mad = _mad(fb, axis=1)
        ax_b.fill_between(tol_ms, np.clip(med - mad, 0, 1), np.clip(med + mad, 0, 1),
                          color=color, alpha=0.15, linewidth=0)
        ax_b.plot(tol_ms, med, METHOD_LS.get(key, '-'), color=color, lw=1.0, label=label)
    ax_b.axvline(1000.0 / FS, color='k', ls='--', lw=0.7, alpha=0.6)
    ax_b.set_xlabel('coincidence window (ms)')
    ax_b.set_ylabel('$F_\\beta$')
    ax_b.set_xlim(0, tol_ms.max())
    ax_b.set_ylim(0, 1)


    ax_c.plot([0, 1], [0, 1], '--', color='k', lw=0.7, alpha=0.6)
    for key, (label, color) in _present(d):
        if '{}_cal_n'.format(key) not in d.files:
            continue
        n    = d['{}_cal_n'.format(key)]
        hits = d['{}_cal_hits'.format(key)]
        psum = d['{}_cal_psum'.format(key)]
        n_cells = n.shape[0]

        ns, hs, ps = n.sum(0), hits.sum(0), psum.sum(0)
        ok = ns > 0
        x, y = ps[ok] / ns[ok], hs[ok] / ns[ok]

        boot = np.full((N_BOOT, N_BINS), np.nan)
        for b in range(N_BOOT):
            idx = rng.randint(0, n_cells, n_cells)
            nb, hb = n[idx].sum(0), hits[idx].sum(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                boot[b] = np.where(nb > 0, hb / nb, np.nan)
        lo = np.nanpercentile(boot, 2.5, axis=0)[ok]
        hi = np.nanpercentile(boot, 97.5, axis=0)[ok]

        ax_c.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax_c.plot(x, y, METHOD_LS.get(key, '-'), color=color, lw=1.0)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel('predicted spike probability')
    ax_c.set_ylabel('P(spike in frame)')

    if fr is None:
        _placeholder(ax_d, 'framerate')
    else:
        rates = fr['rates']
        fine = np.geomspace(rates.min() * 0.8, rates.max() * 1.25, 50)
        ax_d.plot(fine, 250.0 / fine, '--', color='k', lw=0.7, alpha=0.6)
        for key, (label, color) in METHODS.items():
            spread = np.full((len(rates), 3), np.nan)
            for r in range(len(rates)):
                if '{}_{}_err'.format(key, r) in fr:
                    spread[r] = _centered_spread(fr['{}_{}_err'.format(key, r)],
                                                 fr['{}_{}_cell'.format(key, r)], rng)
            if np.all(np.isnan(spread[:, 0])):
                continue
            ax_d.fill_between(rates, spread[:, 1], spread[:, 2], color=color, alpha=0.2,
                              linewidth=0)
            resampled = fr['cascade_resampled'] if key == 'CASCADE' \
                else np.zeros(len(rates), dtype=bool)
            for r in range(len(rates) - 1):
                ax_d.plot(rates[r:r + 2], spread[r:r + 2, 0], color=color, lw=1.0,
                          ls='--' if resampled[r + 1] else METHOD_LS.get(key, '-'))
        ax_d.set_xscale('log')
        ax_d.set_yscale('log')
        ax_d.set_xticks(rates)
        ax_d.set_xticklabels(['{:g}'.format(v) for v in rates])
        ax_d.minorticks_off()

        lo, hi = ax_d.get_ylim()
        yt = [v for v in (1, 2, 5, 10, 20, 50, 100) if lo <= v <= hi]
        ax_d.set_yticks(yt)
        ax_d.set_yticklabels(['{:g}'.format(v) for v in yt])
        ax_d.set_xlabel('frame rate (Hz)')
        ax_d.set_ylabel('timing spread, MAD (ms)')

    post_path = os.path.join(data_dir, _POST_SUMMARY_NPZ)
    _refresh_posterior(data_dir)
    if not os.path.exists(post_path):
        _placeholder(ax_e, 'posterior')
        _placeholder(ax_f, 'posterior')
    else:
        dp = np.load(post_path)
        if not _plot_pairs(ax_e, dp, rng):
            _placeholder(ax_e, 'population')
        ax_e.set_xticks([5, 10, 20, 50, 100, 500])
        ax_e.set_xticklabels(['5', '10', '20', '50', '100', '500'])
        rel_keys = [k for k in POST_METHODS if '{}_a_hit'.format(k) in dp.files]
        ax_f.plot([0, 1], [0, 1], '--', color='k', lw=0.7, alpha=0.6)
        for key in rel_keys:
            _plot_existence(ax_f, dp, key, rng, marker=None)
        ax_f.set_xlim(0, 1)
        ax_f.set_ylim(0, 1)
        ax_f.set_xticks(np.linspace(0, 1, 6))
        ax_f.set_xlabel('spike probability')
        ax_f.set_ylabel('frac real spikes')
        if not rel_keys:
            _placeholder(ax_f, 'posterior')


    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'timing_calibration.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    _refresh_summary(data_dir)
    d = np.load(os.path.join(data_dir, _SUMMARY_NPZ))
    tol_ms = d['tol_grid'] * 1e3
    frame_ms = 1000.0 / FS
    print('Population: {} cells x {:.0f} s. Frame period {:.1f} ms. Offsets removed: {}.'.format(
        int(d['n_cells'][0]), float(d['duration'][0]), frame_ms, bool(d['debias'][0])))

    print('\nTiming of matched spikes (+-{:.0f} ms window):'.format(MATCH_TOL * 1e3))
    step = 1000.0 / (FS * 10.0)
    bins = np.arange(-150.0 - step / 2, 150.0 + step, step)
    print('  {:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}'.format(
        'method', 'offset', 'peak', 'MAD', 'n match', 'med |e|', '90% |e|', '<5 ms',
        '<10 ms'))
    for key, (label, _) in _present(d):
        signed = d['{}_err_s'.format(key)] * 1e3
        off = np.median(signed) + d['{}_offset_s'.format(key)] * 1e3
        err = np.abs(signed)
        print('  {:<22} {:>+7.1f}ms {:>+7.1f}ms {:>7.1f}ms {:>9d} {:>7.1f}ms {:>7.1f}ms '
              '{:>9.1%} {:>9.1%}'.format(
            label, float(off), _peak_offset(signed, bins), float(_mad(signed)), len(err),
            np.median(err), np.percentile(err, 90), np.mean(err < 5.0),
            np.mean(err < 10.0)))

    print('\nMedian F_beta by coincidence window:')
    print('  {:<22}'.format('method') + ''.join('{:>7.0f}'.format(t) for t in tol_ms) + '   (ms)')
    for key, (label, _) in _present(d):
        med = np.nanmedian(d['{}_fb'.format(key)], axis=1)
        print('  {:<22}'.format(label) + ''.join('{:>7.3f}'.format(v) for v in med))

    print('\nPer-frame probability calibration:')
    print('  {:<22} {:>10} {:>8} {:>8}'.format('method', 'shift', 'ECE', 'Brier'))
    for key, (label, _) in _present(d):
        if '{}_cal_n'.format(key) not in d.files:
            continue
        ece = _ece(d['{}_cal_n'.format(key)], d['{}_cal_hits'.format(key)],
                   d['{}_cal_psum'.format(key)])
        print('  {:<22} {:>7.2f} fr {:>8.4f} {:>8.5f}'.format(
            label, float(d['{}_cal_shift'.format(key)]), ece,
            float(d['{}_cal_brier'.format(key)])))

    f = _load_framerate(data_dir)
    if f is not None:
        rates = f['rates']
        rng = np.random.RandomState(0)
        print('\nTiming spread (MAD, ms) vs. frame rate, {} cells x {:.0f} s each:'.format(
            int(f['n_cells'][0]), float(f['duration'][0])))
        print('  {:<22}'.format('method') + ''.join('{:>8g}'.format(r) for r in rates) + '   (Hz)')
        print('  {:<22}'.format('floor') + ''.join('{:>8.1f}'.format(250.0 / r) for r in rates))
        for key, (label, _) in METHODS.items():
            row = []
            for r in range(len(rates)):
                k = '{}_{}_err'.format(key, r)
                row.append(_centered_spread(f[k], f['{}_{}_cell'.format(key, r)], rng)[0]
                           if k in f else np.nan)
            print('  {:<22}'.format(label) + ''.join('{:>8.1f}'.format(v) for v in row))
        if np.any(f['cascade_resampled']):
            print('  CASCADE resampled to {:.0f} Hz at: {} Hz.'.format(
                CASCADE_FALLBACK_FS, ', '.join('{:g}'.format(r)
                                                for r in rates[f['cascade_resampled']])))

    _print_posterior_stats(data_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sub-frame timing and calibration benchmarks on simulated data'
    )
    parser.add_argument('--mode', required=True,
                        choices=['test', 'caiman_fix', 'plot', 'print', 'stats'],
                        help='"caiman_fix" runs only the fixed CaImAn on the saved '
                             'population and frame-rate sweep; "print" (alias '
                             '"stats") prints summary statistics')
    parser.add_argument('--stages', nargs='+', default=['population', 'framerate'],
                        choices=['population', 'analyze', 'framerate', 'posterior'],
                        help='Test stages to run (test mode); in caiman_fix mode, '
                             'population and/or framerate')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing results')
    parser.add_argument('--pop-cells', type=int, default=POP_CELLS,
                        help='Cells in the test population (population)')
    parser.add_argument('--pop-duration', type=float, default=POP_DURATION,
                        help='Test population duration in seconds (population)')
    parser.add_argument('--fr-rates', type=float, nargs='+', default=list(FR_RATES),
                        help='Frame rates in Hz (framerate)')
    parser.add_argument('--fr-cells', type=int, default=FR_CELLS,
                        help='Cells per frame rate (framerate)')
    parser.add_argument('--fr-duration', type=float, default=FR_DURATION,
                        help='Recording duration in seconds (framerate)')
    parser.add_argument('--no-matlab', action='store_true', help='Skip CaImAn')
    parser.add_argument('--no-cascade', action='store_true', help='Skip CASCADE')
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'],
                        help='CASCADE device')
    parser.add_argument('--post-cells', type=int, default=POST_CELLS,
                        help='Population cells rerun with samples kept (posterior)')
    parser.add_argument('--workers', type=int, default=CAIMAN_WORKERS,
                        help='Parallel MATLAB sessions (caiman_fix, posterior)')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(args.data_dir, args.stages, args.pop_cells, args.pop_duration,
                 args.fr_rates, args.fr_cells, args.fr_duration,
                 not args.no_matlab, not args.no_cascade, args.device,
                 args.post_cells, args.workers)
    elif args.mode == 'caiman_fix':
        if 'population' in args.stages:
            _run_caiman_fix(args.data_dir, args.workers)
        if 'framerate' in args.stages:
            _run_caiman_fix_framerate(args.data_dir, args.workers)
    elif args.mode == 'plot':
        plot_figure(args.data_dir)
        plot_posterior_figure(args.data_dir)
    elif args.mode in ('print', 'stats'):
        print_stats(args.data_dir)
