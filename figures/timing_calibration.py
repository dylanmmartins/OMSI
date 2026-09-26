# -*- coding: utf-8 -*-
"""
figures/timing_calibration.py

Sub-frame spike timing and probability calibration benchmarks on simulated data.

Simulates its own population (same generator and settings as figure 1, own seed) and
runs every method on it, so nothing is read from other figures' results. Asks what
the 100 ms coincidence metrics hide: how close called spikes are to true times,
whether per-frame spike probabilities are calibrated, and how timing precision
scales with frame rate.

Test stages
-----------
population
    Simulate the population, run all methods, save raw results, then analyze.
analyze
    Reanalyze saved population results without rerunning inference.
framerate
    Simulate at several frame rates and run every method on each.

Bug-fixed CaImAn (panels a-c) runs on its own with --mode caiman_fix, on the saved
population, so nothing else is rerun. Plot and stats reanalyze automatically when
either raw file is newer than the summary.

Functions
---------
_fbeta
    Compute F-beta score from precision and recall arrays.
_mad
    Compute the median absolute deviation, ignoring NaNs.
_match_pairs
    One-to-one spike matches within a tolerance, as index arrays.
_signed_errors
    Signed timing errors of matched spikes, pooled over cells.
_prf_by_cell
    Per-cell precision and recall at one tolerance.
_timing_analysis
    Timing offset, error distribution, and F-beta vs. tolerance for one method.
_true_occupancy
    Boolean per-frame occupancy of true spikes at a fractional frame shift.
_best_shift
    Frame shift that best aligns a probability trace with true spikes.
_calibration_analysis
    Reliability counts and Brier score for one method.
_simulate
    Simulate a seeded population with the figure 1 generator.
_ensure_cascade_model
    Download a pretrained CASCADE model if it is not installed.
_run_cascade_at
    Run CASCADE at any frame rate, resampling when no model matches.
_run_oasis_at
    Run OASIS the way figure1.py does, at any frame rate.
_run_population
    Simulate the test population, run every method, and save raw results.
_run_caiman_fix
    Run only the rise-time-fixed CaImAn on the saved population.
_analyze_population
    Timing and calibration analysis of saved population results.
_refresh_summary
    Reanalyze if any raw result file is newer than the summary.
_centered_errors
    Per-cell timing errors after removing the pooled offset.
_run_framerate
    Simulate at several frame rates and collect timing errors for every method.
run_test
    Run the requested test stages.
_present
    Methods with results in a summary file.
_ece
    Expected calibration error from binned reliability counts.
_placeholder
    Mark a panel whose test stage has not been run.
_centered_spread
    Median absolute deviation of centered errors, with a bootstrap band over cells.
plot_figure
    Load summary results and render the figure.
print_stats
    Print summary statistics to the terminal.

To simulate, run all methods, and analyze (or pick stages with --stages):
    $ python timing_calibration.py --mode test
    $ python timing_calibration.py --mode test --stages population

To run only the bug-fixed CaImAn on the saved population:
    $ python timing_calibration.py --mode caiman_fix

To create figure (reanalyzes first if raw results changed):
    $ python timing_calibration.py --mode plot


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
_FIX_NPZ     = 'timing_caiman_fix.npz'
_FR_NPZ      = 'timing_framerate.npz'

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

FS   = 30.0
TAU  = 1.2
BETA = 0.5

# key: (label, color). OASIS has no probability trace.
METHODS = {
    'OMSI':       ('OMSI',                  '#4C72B0'),
    'MATLAB':     ('CaImAn',                '#DD8452'),
    'MATLAB_FIX': ('CaImAn, rise-time fix', '#DD8452'),
    'OASIS':      ('OASIS',                 '#55A868'),
    'CASCADE':    ('CASCADE',               '#8172B3'),
}

# CaImAn's rise-time Metropolis step scores proposals with the current calcium shape
# (cont_ca_sampler.m, Gs instead of Gs_), so every rise-time proposal is accepted and
# spike times drift with it. MATLAB_FIX reruns CaImAn on a copy with that one line
# fixed (panels a-c only, --mode caiman_fix); the installed CaImAn is untouched.
METHOD_LS = {'MATLAB_FIX': '--'}

# Test population: figure 1 generator and settings, own seed. Smaller than figure 1
# (500 cells x 20 min) because CaImAn runs one MATLAB cell at a time.
POP_CELLS    = 200
POP_DURATION = 600.0
POP_SEED     = 5
KURTOSIS_RANGE = (0.0, 25.0)

# Generator sets each cell's noise to hit a target kurtosis, and bursts make the
# clean trace heavy-tailed, so bursty cells get far more noise than others. Here
# bursty cells get noise sigma drawn from non-bursty cells instead (same per-spike
# amplitude, so same SNR distribution). Bursty: at least BURSTY_MIN_PAIRS ISIs of
# exactly the generator's 2 fine-grid steps.
BURSTY_MIN_PAIRS = 3

# Same sampler settings figure1.py passes to OMSI.deconv, and same sweep count it
# passes to CaImAn.
OMSI_PARAMS   = {'p': 2, 'Nsamples': 200, 'B': 75, 'marg': 0, 'upd_gam': 1}
MATLAB_SWEEPS = 500

# Coincidence window used to pair called and true spikes when measuring offsets and
# timing errors. Same as every accuracy metric in the other figures.
MATCH_TOL = 0.100

# Tolerances for the F-beta curve, in seconds.
TOL_GRID = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.033, 0.040,
                     0.050, 0.060, 0.075, 0.100])

# Offsets are part of each method's error: users can't measure them without ground
# truth, so raw times are scored. True subtracts each method's median offset first.
DEBIAS = False

# Fractional frame shifts tried when aligning a probability trace with true spikes.
SHIFT_GRID = np.arange(-4.0, 4.0001, 0.25)

N_BINS = 10
N_BOOT = 200

# Frame-rate sweep. Shorter and smaller than the population -- CaImAn runs one
# MATLAB cell at a time.
FR_RATES    = np.array([7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0])
FR_CELLS    = 40
FR_DURATION = 300.0
FR_SEED     = 11

# Offset is estimated with a wide window first, so slow frame rates with a large
# lag don't get their error distribution truncated.
FR_OFFSET_TOL = 0.250

# Same causal-kernel family as the figure 1 model, smoothing scaled with frame rate.
# No Global_EXC model exists above 40 Hz, so faster traces are resampled to the
# nearest model rate, CASCADE_FALLBACK_FS, as CASCADE's docs suggest.
CASCADE_MODELS = {
    7.5:  'Global_EXC_7.5Hz_smoothing200ms_causalkernel',
    10.0: 'Global_EXC_10Hz_smoothing100ms_causalkernel',
    15.0: 'Global_EXC_15Hz_smoothing100ms_causalkernel',
    20.0: 'Global_EXC_20Hz_smoothing100ms_causalkernel',
    30.0: 'Global_EXC_30Hz_smoothing50ms_causalkernel',
    40.0: 'Global_EXC_40Hz_smoothing50ms_causalkernel',
}
CASCADE_FALLBACK_FS = 40.0


def _fbeta(precision, recall):
    """ Compute F-beta score from precision and recall arrays.

    Parameters
    ----------
    precision : array-like
        Precision values.
    recall : array-like
        Recall values.

    Returns
    -------
    np.ndarray
        F-beta scores.
    """

    p  = np.asarray(precision, dtype=float)
    r  = np.asarray(recall,    dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _mad(x, axis=None):
    """ Compute the median absolute deviation, ignoring NaNs.

    Parameters
    ----------
    x : array-like
        Input values.
    axis : int or None, optional
        Axis along which to compute; None flattens the input.

    Returns
    -------
    float or np.ndarray
        Median of |x - median(x)| along axis.
    """

    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _match_pairs(t, p, tol):
    """ One-to-one spike matches within a tolerance, as index arrays.

    Same assignment as helpers.compute_accuracy_strict (Hungarian, out-of-tolerance
    pairs never matched), but solved per cluster of nearby spikes so long, dense
    spike trains don't need one huge cost matrix.

    Parameters
    ----------
    t : array-like
        True spike times in seconds.
    p : array-like
        Called spike times in seconds.
    tol : float
        Maximum time difference for a match, in seconds.

    Returns
    -------
    it : np.ndarray
        Indices into t of matched true spikes.
    ip : np.ndarray
        Indices into p of the matching called spikes.
    """

    t = np.asarray(t, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    empty = np.array([], dtype=int)
    if len(t) == 0 or len(p) == 0:
        return empty, empty

    ot, op = np.argsort(t), np.argsort(p)
    ts, ps = t[ot], p[op]

    # Merge both trains in time order. A gap longer than tol between neighbors means
    # no true/called pair can straddle it, so each cluster solves independently.
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
    """ Signed timing errors (called minus true) of matched spikes, pooled over cells.

    Parameters
    ----------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    called : list of np.ndarray
        Per-cell called spike times in seconds.
    tol : float, optional
        Matching window in seconds.

    Returns
    -------
    np.ndarray
        Signed errors in seconds.
    """

    errs = []
    for t, p in zip(true_spikes, called):
        t = np.asarray(t, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        it, ip = _match_pairs(t, p, tol)
        errs.append(p[ip] - t[it])
    return np.concatenate(errs) if errs else np.array([])


def _prf_by_cell(t, p, tol):
    """ Per-cell precision and recall at one tolerance.

    Empty-train conventions follow helpers.compute_accuracy_strict.

    Parameters
    ----------
    t : np.ndarray
        True spike times in seconds.
    p : np.ndarray
        Called spike times in seconds.
    tol : float
        Matching window in seconds.

    Returns
    -------
    prec, rec : float
        Precision and recall for this cell.
    """

    if len(p) == 0:
        return 0.0, (0.0 if len(t) > 0 else 1.0)
    if len(t) == 0:
        return 0.0, 1.0
    it, _ = _match_pairs(t, p, tol)
    n_tp = len(it)
    return n_tp / len(p), n_tp / len(t)


def _timing_analysis(true_spikes, called):
    """ Timing offset, error distribution, and F-beta vs. tolerance for one method.

    Parameters
    ----------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    called : list of np.ndarray
        Per-cell called spike times in seconds.

    Returns
    -------
    dict
        offset_s: median signed error removed from called times (0 if DEBIAS is off).
        err_s: signed errors of matched spikes after the offset is removed.
        n_true, n_called: spike counts pooled over cells.
        fb: F-beta, shape (len(TOL_GRID), n_cells); NaN for cells with no true spikes.
        prec, rec: same shape as fb.
    """

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
    """ Boolean per-frame occupancy of true spikes at a fractional frame shift.

    Parameters
    ----------
    t : np.ndarray
        True spike times in seconds.
    n_frames : int
        Number of frames.
    shift : float
        Shift in frames added to each spike time before assigning it to a frame.

    Returns
    -------
    np.ndarray
        Boolean array, True where at least one true spike falls in the frame.
    """

    idx = np.floor(np.asarray(t, dtype=np.float64).ravel() * FS + shift).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    occ = np.zeros(n_frames, dtype=bool)
    occ[idx] = True
    return occ


def _best_shift(true_spikes, probs):
    """ Frame shift that best aligns a probability trace with true spikes.

    Each method indexes frames its own way (rounding vs. flooring, 0 vs. 1 based,
    indicator lag), so which frame a spike belongs to is fit by minimizing the pooled
    Brier score. One free parameter for the whole population.

    Parameters
    ----------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    probs : np.ndarray
        Per-frame probabilities, shape (n_cells, n_frames).

    Returns
    -------
    best : float
        Best shift in frames.
    brier : np.ndarray
        Pooled Brier score at every shift in SHIFT_GRID.
    """

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
    """ Reliability counts and Brier score for one method.

    Frame probabilities are binned into N_BINS equal-width bins on [0, 1]; per cell,
    per bin, this keeps frame count, number of frames holding a true spike, and summed
    predicted probability, so cells can be resampled for confidence bands.

    Parameters
    ----------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    probs : np.ndarray
        Per-frame probabilities, shape (n_cells, n_frames).

    Returns
    -------
    dict
        shift: fitted alignment in frames.
        brier_by_shift: pooled Brier score at every shift in SHIFT_GRID.
        brier: pooled Brier score at the fitted shift.
        n, hits, psum: arrays of shape (n_cells, N_BINS).
    """

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


def _simulate(n_cells, fs, duration, seed):
    """ Simulate a seeded population with the figure 1 generator.

    Bursty cells get their noise redrawn from the non-bursty cells' noise levels.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    fs : float
        Frame rate in Hz.
    duration : float
        Recording duration in seconds.
    seed : int
        Random seed.

    Returns
    -------
    noisy : np.ndarray
        Noisy dF/F traces, shape (n_cells, n_frames).
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    """

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
    return noisy, true_spikes


def _ensure_cascade_model(model):
    """ Download a pretrained CASCADE model if it is not installed.

    Parameters
    ----------
    model : str
        CASCADE model name.
    """

    code = ("import os, cascade2p.cascade as c; "
            "f = os.path.join(os.path.dirname(os.path.dirname(c.__file__)), "
            "'Pretrained_models'); "
            "os.path.isdir(os.path.join(f, {m!r})) or "
            "c.download_model({m!r}, model_folder=f, verbose=1)").format(m=model)
    subprocess.run(['conda', 'run', '-n', 'cascade', 'python', '-c', code], check=True)


def _run_cascade_at(dff, fs, work_dir, device='gpu'):
    """ Run CASCADE at any frame rate, resampling when no model matches.

    Parameters
    ----------
    dff : np.ndarray
        dF/F traces, shape (n_cells, n_frames).
    fs : float
        Frame rate in Hz.
    work_dir : str
        Directory for subprocess input and output files.
    device : str, optional
        'gpu' or 'cpu'.

    Returns
    -------
    spikes : list of np.ndarray
        Called spike times in seconds.
    probs : np.ndarray
        Per-frame spike probability at the rate CASCADE was run at.
    resampled : bool
        True if traces were resampled to CASCADE_FALLBACK_FS.
    """

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


def _run_oasis_at(dff, fs):
    """ Run OASIS the way figure1.py does, at any frame rate.

    Parameters
    ----------
    dff : np.ndarray
        dF/F traces, shape (n_cells, n_frames).
    fs : float
        Frame rate in Hz.

    Returns
    -------
    list of np.ndarray
        Called spike times in seconds.
    """

    from oasis.functions import deconvolve as oasis_deconv
    from figure1 import _oasis_spikes_from_s

    sigmas = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    g = np.exp(-1 / (fs * TAU))
    spikes = []
    for i in range(dff.shape[0]):
        _, s, _, _, _ = oasis_deconv(dff[i], g=(g,), sn=sigmas[i], penalty=1)
        spikes.append(_oasis_spikes_from_s(s, sigmas[i], fs))
    return spikes


def _run_population(data_dir, n_cells=POP_CELLS, duration=POP_DURATION,
                    run_matlab=True, run_cascade=True, device='gpu'):
    """ Simulate the test population, run every method, and save raw results.

    Parameters
    ----------
    data_dir : str
        Directory where the raw npz and subprocess files are written.
    n_cells : int, optional
        Cells to simulate.
    duration : float, optional
        Recording duration in seconds.
    run_matlab : bool, optional
        Whether to run CaImAn (MATLAB).
    run_cascade : bool, optional
        Whether to run CASCADE.
    device : str, optional
        CASCADE device, 'gpu' or 'cpu'.
    """

    import OMSI
    from run_pnev_MCMC import run_matlab_pnevMCMC

    def _obj(xs):
        """ Ragged list to object array. """
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


def _run_caiman_fix(data_dir):
    """ Run only the rise-time-fixed CaImAn on the saved population.

    Regenerates the population from its seed, checks it matches the saved ground
    truth, and writes results to their own file so nothing else is rerun.

    Parameters
    ----------
    data_dir : str
        Directory holding the population npz; results are written alongside.
    """

    from run_pnev_MCMC import run_matlab_pnevMCMC

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

    print('Running CaImAn (MATLAB), rise-time fix...')
    spk, _, probs, _ = run_matlab_pnevMCMC(noisy, fs=FS, tau=TAU, n_sweeps=MATLAB_SWEEPS,
                                           fix_rise_bug=True)
    if sum(len(s) for s in spk) == 0:
        raise RuntimeError('CaImAn with rise-time fix returned no spikes.')

    arr = np.empty(len(spk), dtype=object)
    for i, x in enumerate(spk):
        arr[i] = np.asarray(x)
    out_path = os.path.join(data_dir, _FIX_NPZ)
    np.savez(out_path, n_true=np.array([len(t) for t in true_spikes]),
             MATLAB_FIX_spikes=arr, MATLAB_FIX_probs=np.asarray(probs, dtype=np.float32))
    print('Saved to {}.'.format(out_path))


def _refresh_summary(data_dir):
    """ Reanalyze if any raw result file is newer than the summary.

    Parameters
    ----------
    data_dir : str
        Directory holding the result files.
    """

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
    """ Timing and calibration analysis of saved population results.

    Parameters
    ----------
    data_dir : str
        Directory holding the raw population npz (and bug-fixed CaImAn npz, if run);
        summary is written alongside.
    """

    path = os.path.join(data_dir, _POP_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'No data at {}. Run --mode test --stages population first.'.format(path))
    pop = np.load(path, allow_pickle=True)
    raw = {k: pop[k] for k in pop.files}
    true_spikes = [np.asarray(t, dtype=np.float64) for t in raw['true_spikes']]

    # Bug-fixed CaImAn lives in its own file; use it only if it matches this population.
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
    """ Per-cell timing errors after removing the pooled offset.

    Offset comes from a wide first pass (FR_OFFSET_TOL); errors are then rescored
    within max(MATCH_TOL, one frame) so a whole frame of quantization error fits.

    Parameters
    ----------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    called : list of np.ndarray
        Per-cell called spike times in seconds.
    fs : float
        Frame rate in Hz.

    Returns
    -------
    err : np.ndarray
        Centered signed errors in seconds, pooled over cells.
    cell : np.ndarray
        Cell index of each error.
    offset : float
        Removed offset in seconds.
    """

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
    """ Simulate at several frame rates and collect timing errors for every method.

    Parameters
    ----------
    data_dir : str
        Directory where the summary npz and subprocess files are written.
    rates : array-like, optional
        Frame rates in Hz.
    n_cells : int, optional
        Cells simulated per frame rate.
    duration : float, optional
        Recording duration in seconds.
    run_matlab : bool, optional
        Whether to run CaImAn (MATLAB).
    run_cascade : bool, optional
        Whether to run CASCADE.
    device : str, optional
        CASCADE device, 'gpu' or 'cpu'.
    """

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


def run_test(data_dir=_DEFAULT_DATA_DIR, stages=('population', 'framerate'),
             pop_cells=POP_CELLS, pop_duration=POP_DURATION,
             fr_rates=FR_RATES, fr_cells=FR_CELLS, fr_duration=FR_DURATION,
             run_matlab=True, run_cascade=True, device='gpu'):
    """ Run the requested test stages.

    Parameters
    ----------
    data_dir : str, optional
        Directory where result npz files are written.
    stages : sequence of str, optional
        Any of 'population', 'analyze', 'framerate'. 'population' also analyzes.
    pop_cells : int, optional
        Cells in the test population.
    pop_duration : float, optional
        Test population recording duration, in seconds.
    fr_rates : array-like, optional
        Frame rates for the framerate stage.
    fr_cells : int, optional
        Cells per frame rate.
    fr_duration : float, optional
        Recording duration per frame rate, in seconds.
    run_matlab : bool, optional
        Include CaImAn.
    run_cascade : bool, optional
        Include CASCADE.
    device : str, optional
        CASCADE device, 'gpu' or 'cpu'.
    """

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


def _present(d):
    """ METHODS entries with results in summary npz d, as (key, (label, color)). """

    return [(k, v) for k, v in METHODS.items() if '{}_fb'.format(k) in d.files]


def _ece(n, hits, psum):
    """ Expected calibration error: frame-weighted mean gap between predicted and observed.

    Parameters
    ----------
    n, hits, psum : np.ndarray
        Per-bin frame counts, hit counts, and summed probabilities, pooled or per cell.

    Returns
    -------
    float
        Expected calibration error.
    """

    n, hits, psum = (np.asarray(a, dtype=float).sum(axis=0) if np.ndim(a) > 1
                     else np.asarray(a, dtype=float) for a in (n, hits, psum))
    ok = n > 0
    return float(np.sum(np.abs(hits[ok] - psum[ok])) / np.sum(n[ok]))


def _placeholder(ax, stage):
    """ Mark a panel whose test stage has not been run. """

    ax.text(0.5, 0.5, 'run --mode test\n--stages {}'.format(stage), ha='center',
            va='center', transform=ax.transAxes, color='0.5')
    ax.set_xticks([])
    ax.set_yticks([])


def _centered_spread(err, cell, rng):
    """ Median absolute deviation of centered errors, with a bootstrap band over cells.

    Parameters
    ----------
    err : np.ndarray
        Centered errors in seconds.
    cell : np.ndarray
        Cell index of each error.
    rng : np.random.RandomState
        Random state for resampling cells.

    Returns
    -------
    spread, lo, hi : float
        Spread and 95% band, in ms.
    """

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
    """ Load summary results and render the figure.

    Panels: (a) timing error of matched spikes, (b) F-beta vs. coincidence window,
    (c) reliability of per-frame spike probability, (d) timing precision vs. frame
    rate.

    Parameters
    ----------
    data_dir : str, optional
        Directory holding the summary npz files; figure is saved alongside them.
    """

    _refresh_summary(data_dir)
    path = os.path.join(data_dir, _SUMMARY_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError('No data at {}. Run --mode test first.'.format(path))
    d = np.load(path)
    fr_path = os.path.join(data_dir, _FR_NPZ)
    fr = np.load(fr_path) if os.path.exists(fr_path) else None

    tol_ms = d['tol_grid'] * 1e3
    debias = bool(d['debias'][0])
    rng = np.random.RandomState(0)

    fig = plt.figure(figsize=(5.6, 4.8))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.45, hspace=0.55)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    # Timing error distribution over matched spikes. Bin width is the ground-truth
    # grid spacing (frame period / 10) so frame-quantized methods don't alias into a
    # comb; shaded band is +-half a frame.
    step = 1000.0 / (FS * 10.0)
    bins = np.arange(-100.0 - step / 2, 100.0 + step, step)
    centers = 0.5 * (bins[:-1] + bins[1:])
    handles = []
    for key, (label, color) in _present(d):
        err_ms = d['{}_err_s'.format(key)] * 1e3
        dens, _ = np.histogram(err_ms, bins=bins, density=True)
        off = d['{}_offset_s'.format(key)] * 1e3
        name = '{} ({:+.0f} ms)'.format(label, off) if debias else label
        h, = ax_a.plot(centers, dens, METHOD_LS.get(key, '-'), color=color, lw=1.0,
                       label=name)
        handles.append(h)

    ax_a.axvspan(-500.0 / FS, 500.0 / FS, color='0.5', alpha=0.10, linewidth=0)
    ax_a.set_xlim(-100, 100)
    ax_a.axvline(0, color='k', ls='--', lw=0.7, alpha=0.6)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel('timing error (ms)' if not debias else 'timing error, offset removed (ms)')
    ax_a.set_ylabel('density')
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, 0.99))

    # F-beta vs. coincidence window; band is median absolute deviation over cells.
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

    # Reliability diagram. Band: 95% interval from resampling cells.
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

    # Timing precision vs. frame rate; band is 95% interval from resampling cells.
    # Dashed floor: a perfect frame-quantized caller has uniform error over one frame,
    # whose MAD is a quarter frame. Dashed CASCADE segments: traces resampled to its
    # 40 Hz model.
    if fr is None:
        _placeholder(ax_d, 'framerate')
    else:
        rates = fr['rates']
        fine = np.geomspace(rates.min() * 0.8, rates.max() * 1.25, 50)
        ax_d.plot(fine, 250.0 / fine, '--', color='k', lw=0.7, alpha=0.6)
        for key, (label, color) in METHODS.items():
            spread = np.full((len(rates), 3), np.nan)
            for r in range(len(rates)):
                if '{}_{}_err'.format(key, r) in fr.files:
                    spread[r] = _centered_spread(fr['{}_{}_err'.format(key, r)],
                                                 fr['{}_{}_cell'.format(key, r)], rng)
            if np.all(np.isnan(spread[:, 0])):
                continue
            ax_d.fill_between(rates, spread[:, 1], spread[:, 2], color=color, alpha=0.2,
                              linewidth=0)
            # Dashed where the segment ends at a resampled CASCADE rate.
            resampled = fr['cascade_resampled'] if key == 'CASCADE' \
                else np.zeros(len(rates), dtype=bool)
            for r in range(len(rates) - 1):
                ax_d.plot(rates[r:r + 2], spread[r:r + 2, 0], color=color, lw=1.0,
                          ls='--' if resampled[r + 1] else '-')
        ax_d.set_xscale('log')
        ax_d.set_yscale('log')
        ax_d.set_xticks(rates)
        ax_d.set_xticklabels(['{:g}'.format(v) for v in rates])
        ax_d.minorticks_off()
        ax_d.set_xlabel('frame rate (Hz)')
        ax_d.set_ylabel('timing spread, MAD (ms)')

    # for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), 'abcd'):
        # ax.text(-0.28, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'timing_calibration.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):
    """ Print summary statistics to the terminal.

    Parameters
    ----------
    data_dir : str, optional
        Directory holding the summary npz files.
    """

    _refresh_summary(data_dir)
    d = np.load(os.path.join(data_dir, _SUMMARY_NPZ))
    tol_ms = d['tol_grid'] * 1e3
    frame_ms = 1000.0 / FS
    print('Population: {} cells x {:.0f} s. Frame period {:.1f} ms. Offsets removed: {}.'.format(
        int(d['n_cells'][0]), float(d['duration'][0]), frame_ms, bool(d['debias'][0])))

    print('\nTiming of matched spikes (+-{:.0f} ms window):'.format(MATCH_TOL * 1e3))
    print('  {:<22} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}'.format(
        'method', 'offset', 'n match', 'med |e|', '90% |e|', '<5 ms', '<10 ms'))
    for key, (label, _) in _present(d):
        # Offset: median signed error of raw times, whether or not it was removed.
        signed = d['{}_err_s'.format(key)] * 1e3
        off = np.median(signed) + d['{}_offset_s'.format(key)] * 1e3
        err = np.abs(signed)
        print('  {:<22} {:>7.1f}ms {:>9d} {:>7.1f}ms {:>7.1f}ms {:>9.1%} {:>9.1%}'.format(
            label, float(off), len(err), np.median(err), np.percentile(err, 90),
            np.mean(err < 5.0), np.mean(err < 10.0)))

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

    fr_path = os.path.join(data_dir, _FR_NPZ)
    if os.path.exists(fr_path):
        f = np.load(fr_path)
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
                           if k in f.files else np.nan)
            print('  {:<22}'.format(label) + ''.join('{:>8.1f}'.format(v) for v in row))
        if np.any(f['cascade_resampled']):
            print('  CASCADE resampled to {:.0f} Hz at: {} Hz.'.format(
                CASCADE_FALLBACK_FS, ', '.join('{:g}'.format(r)
                                                for r in rates[f['cascade_resampled']])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sub-frame timing and calibration benchmarks on simulated data'
    )
    parser.add_argument('--mode', required=True,
                        choices=['test', 'caiman_fix', 'plot', 'stats'],
                        help='"caiman_fix" runs only the bug-fixed CaImAn on the saved '
                             'population')
    parser.add_argument('--stages', nargs='+', default=['population', 'framerate'],
                        choices=['population', 'analyze', 'framerate'],
                        help='Test stages to run (test mode)')
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
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(args.data_dir, args.stages, args.pop_cells, args.pop_duration,
                 args.fr_rates, args.fr_cells, args.fr_duration,
                 not args.no_matlab, not args.no_cascade, args.device)
    elif args.mode == 'caiman_fix':
        _run_caiman_fix(args.data_dir)
    elif args.mode == 'plot':
        plot_figure(args.data_dir)
    elif args.mode == 'stats':
        print_stats(args.data_dir)
