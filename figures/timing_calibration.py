# -*- coding: utf-8 -*-
"""
figures/timing_calibration.py

Sub-frame spike timing and posterior calibration from the figure 1 simulation.

Reads the per-method result files written by figure1.py (500 simulated cells, ground
truth on a 10x finer grid than the frame rate) and asks two questions the standard
100 ms coincidence metrics hide: how close are called spikes to true spike times, and
are per-frame spike probabilities calibrated.

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
_ece
    Expected calibration error from binned reliability counts.
_load_method
    Load ground truth, called spikes, and probabilities for one method.
run_test
    Analyze all methods and save summary results.
plot_figure
    Load summary results and render the figure.
print_stats
    Print summary statistics to the terminal.

To analyze figure 1 results:
    $ python timing_calibration.py --mode test

To create figure:
    $ python timing_calibration.py --mode plot

DMM, September 2026
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIG1_DIR = os.path.join(_HERE, 'data', 'fig1')
_DEFAULT_DATA_DIR = os.path.join(_HERE, 'data', 'timing')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

FS   = 30.0
BETA = 0.5

# key: (label, color, figure1 file, spikes key, probability key or None).
# OASIS has no probability trace. CASCADE GPU and CPU run the same network.
METHODS = {
    'OMSI':    ('OMSI',    '#4C72B0', 'fixed_benchmark_OMSI.npz',
                'optim_spikes',   'optim_prob'),
    'MATLAB':  ('CaImAn',  '#DD8452', 'fixed_benchmark_MATLAB.npz',
                'tradmat_spikes', 'tradmat_probs'),
    'OASIS':   ('OASIS',   '#55A868', 'fixed_benchmark_OASIS.npz',
                'oasis_spikes',   None),
    'CASCADE': ('CASCADE', '#8172B3', 'fixed_benchmark_CASCADE_GPU.npz',
                'cascade_spikes', 'cascade_probs'),
}

# Coincidence window used to pair called and true spikes when measuring offsets and
# timing errors. Same as every accuracy metric in the other figures.
MATCH_TOL = 0.100

# Tolerances for the F-beta curve, in seconds.
TOL_GRID = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.033, 0.040,
                     0.050, 0.060, 0.075, 0.100])

# Each method has its own fixed timing offset (indicator lag, frame indexing
# convention). It is constant and removable, so it is estimated and subtracted before
# scoring jitter. Set False to score raw times.
DEBIAS = True

# Fractional frame shifts tried when aligning a probability trace with true spikes.
SHIFT_GRID = np.arange(-4.0, 4.0001, 0.25)

N_BINS = 10
N_BOOT = 200


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


def _load_method(fig1_dir, key, n_cells=None):
    """ Load ground truth, called spikes, and probabilities for one method.

    Parameters
    ----------
    fig1_dir : str
        Directory holding figure1.py result files.
    key : str
        Method key in METHODS.
    n_cells : int, optional
        Keep only the first n_cells cells.

    Returns
    -------
    true_spikes : list of np.ndarray
        Per-cell true spike times in seconds.
    called : list of np.ndarray
        Per-cell called spike times in seconds.
    probs : np.ndarray or None
        Per-frame probabilities, shape (n_cells, n_frames), or None if the method
        has none.
    """

    _, _, fname, spikes_key, prob_key = METHODS[key]
    path = os.path.join(fig1_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'No data at {}. Run figure1.py --mode test first.'.format(path))

    d = np.load(path, allow_pickle=True)
    sl = slice(None, n_cells)
    true_spikes = [np.asarray(x, dtype=np.float64) for x in d['true_spikes'][sl]]
    called = [np.asarray(x, dtype=np.float64) for x in d[spikes_key][sl]]
    probs = np.asarray(d[prob_key][sl]) if prob_key is not None else None
    return true_spikes, called, probs


def run_test(fig1_dir=_DEFAULT_FIG1_DIR, data_dir=_DEFAULT_DATA_DIR, n_cells=None):
    """ Analyze all methods and save summary results.

    Parameters
    ----------
    fig1_dir : str, optional
        Directory holding figure1.py result files.
    data_dir : str, optional
        Directory where timing_calibration.npz is written.
    n_cells : int, optional
        Analyze only the first n_cells cells. Default is all.
    """

    os.makedirs(data_dir, exist_ok=True)

    out = {'debias': np.array([DEBIAS]), 'tol_grid': TOL_GRID, 'shift_grid': SHIFT_GRID}
    reference = None
    for key in METHODS:
        print('Loading {}...'.format(key))
        true_spikes, called, probs = _load_method(fig1_dir, key, n_cells)

        # Every method file must hold the same simulated cells.
        counts = np.array([len(t) for t in true_spikes])
        if reference is None:
            reference = counts
        elif not np.array_equal(reference, counts):
            raise ValueError(
                '{} was simulated with different ground truth than the first method. '
                'Rerun figure1.py so all methods share one simulation.'.format(key))

        print('  Timing analysis ({} cells)...'.format(len(true_spikes)))
        res = _timing_analysis(true_spikes, called)
        for k, v in res.items():
            out['{}_{}'.format(key, k)] = np.asarray(v)

        if probs is not None:
            print('  Calibration analysis...')
            cal = _calibration_analysis(true_spikes, probs)
            for k, v in cal.items():
                out['{}_cal_{}'.format(key, k)] = np.asarray(v)

    out_path = os.path.join(data_dir, 'timing_calibration.npz')
    np.savez(out_path, **out)
    print('\nSaved to {}.'.format(out_path))


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


def plot_figure(data_dir=_DEFAULT_DATA_DIR):
    """ Load summary results and render the figure.

    Panels: (a) timing error distribution of matched spikes, (b) F-beta vs. coincidence
    window, (c) reliability of per-frame spike probability with frame counts below.

    Parameters
    ----------
    data_dir : str, optional
        Directory holding timing_calibration.npz; figure is saved alongside it.
    """

    path = os.path.join(data_dir, 'timing_calibration.npz')
    if not os.path.exists(path):
        raise FileNotFoundError('No data at {}. Run --mode test first.'.format(path))
    d = np.load(path)
    tol_ms = d['tol_grid'] * 1e3
    debias = bool(d['debias'][0])
    rng = np.random.RandomState(0)

    fig = plt.figure(figsize=(7.4, 2.5))
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 2], height_ratios=[3, 1], hspace=0.12)
    ax_c = fig.add_subplot(inner[0])
    ax_h = fig.add_subplot(inner[1], sharex=ax_c)

    # Timing error distribution over matched spikes. Bin width is the ground-truth
    # grid spacing (frame period / 10) so frame-quantized methods don't alias into a
    # comb; shaded band is +-half a frame.
    step = 1000.0 / (FS * 10.0)
    bins = np.arange(-60.0 - step / 2, 60.0 + step, step)
    centers = 0.5 * (bins[:-1] + bins[1:])
    handles = []
    for key, (label, color, *_rest) in METHODS.items():
        err_ms = d['{}_err_s'.format(key)] * 1e3
        dens, _ = np.histogram(err_ms, bins=bins, density=True)
        off = d['{}_offset_s'.format(key)] * 1e3
        name = '{} ({:+.0f} ms)'.format(label, off) if debias else label
        h, = ax_a.plot(centers, dens, '-', color=color, lw=1.0, label=name)
        handles.append(h)
    ax_a.axvspan(-500.0 / FS, 500.0 / FS, color='0.5', alpha=0.10, linewidth=0)
    ax_a.set_xlim(-60, 60)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel('timing error (ms)' if not debias else 'timing error, offset removed (ms)')

    ax_a.set_ylabel('density')
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.42, 1.11))

    # F-beta vs. coincidence window; band is median absolute deviation over cells.
    for key, (label, color, *_rest) in METHODS.items():
        fb = d['{}_fb'.format(key)]
        med = np.nanmedian(fb, axis=1)
        mad = _mad(fb, axis=1)
        ax_b.fill_between(tol_ms, np.clip(med - mad, 0, 1), np.clip(med + mad, 0, 1),
                          color=color, alpha=0.15, linewidth=0)
        ax_b.plot(tol_ms, med, '.-', color=color, ms=3, lw=1.0, label=label)
    ax_b.axvline(1000.0 / FS, color='k', ls='--', lw=0.7, alpha=0.6)
    ax_b.set_xlabel('coincidence window (ms)')
    ax_b.set_ylabel('$F_\\beta$')
    ax_b.set_xlim(0, tol_ms.max())
    ax_b.set_ylim(0, 1)

    # Reliability diagram. Error bars: 95% band from resampling cells.
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    ax_c.plot([0, 1], [0, 1], '--', color='k', lw=0.7, alpha=0.6)
    for key, (label, color, *_rest) in METHODS.items():
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

        ece = _ece(n, hits, psum)
        ax_c.errorbar(x, y, yerr=[y - lo, hi - y], fmt='.-', color=color, ms=3, lw=1.0,
                      capsize=1.5, label='{} (ECE {:.3f})'.format(label, ece))
        ax_h.step(np.concatenate([edges[:-1], [1.0]]),
                  np.concatenate([np.maximum(ns, 1), [max(ns[-1], 1)]]),
                  where='post', color=color, lw=1.0)

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel('P(spike in frame)')
    ax_c.legend(frameon=False, fontsize=5, loc='upper left', handlelength=1.2)
    plt.setp(ax_c.get_xticklabels(), visible=False)
    ax_h.set_yscale('log')
    ax_h.set_xlabel('predicted spike probability')
    ax_h.set_ylabel('frames')

    for ax, letter, x0 in zip((ax_a, ax_b, ax_c), 'abc', (-0.28, -0.28, -0.34)):
        ax.text(x0, 1.05, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')

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
        Directory holding timing_calibration.npz.
    """

    d = np.load(os.path.join(data_dir, 'timing_calibration.npz'))
    tol_ms = d['tol_grid'] * 1e3
    frame_ms = 1000.0 / FS
    print('Frame period {:.1f} ms. Offsets removed: {}.'.format(frame_ms, bool(d['debias'][0])))

    print('\nTiming of matched spikes (+-{:.0f} ms window):'.format(MATCH_TOL * 1e3))
    print('  {:<9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}'.format(
        'method', 'offset', 'n match', 'med |e|', '90% |e|', '<5 ms', '<10 ms'))
    for key, (label, *_rest) in METHODS.items():
        err = np.abs(d['{}_err_s'.format(key)]) * 1e3
        off = d['{}_offset_s'.format(key)] * 1e3
        print('  {:<9} {:>7.1f}ms {:>9d} {:>7.1f}ms {:>7.1f}ms {:>9.1%} {:>9.1%}'.format(
            label, float(off), len(err), np.median(err), np.percentile(err, 90),
            np.mean(err < 5.0), np.mean(err < 10.0)))

    print('\nMedian F_beta by coincidence window:')
    print('  {:<9}'.format('method') + ''.join('{:>7.0f}'.format(t) for t in tol_ms) + '   (ms)')
    for key, (label, *_rest) in METHODS.items():
        med = np.nanmedian(d['{}_fb'.format(key)], axis=1)
        print('  {:<9}'.format(label) + ''.join('{:>7.3f}'.format(v) for v in med))

    print('\nPer-frame probability calibration:')
    print('  {:<9} {:>10} {:>8} {:>8}'.format('method', 'shift', 'ECE', 'Brier'))
    for key, (label, *_rest) in METHODS.items():
        if '{}_cal_n'.format(key) not in d.files:
            continue
        ece = _ece(d['{}_cal_n'.format(key)], d['{}_cal_hits'.format(key)],
                   d['{}_cal_psum'.format(key)])
        print('  {:<9} {:>7.2f} fr {:>8.4f} {:>8.5f}'.format(
            label, float(d['{}_cal_shift'.format(key)]), ece,
            float(d['{}_cal_brier'.format(key)])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Sub-frame timing and calibration analysis of the figure 1 simulation'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'stats'])
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing summary results')
    parser.add_argument('--fig1-dir', default=_DEFAULT_FIG1_DIR,
                        help='Directory holding figure1.py result files')
    parser.add_argument('--n-cells', type=int, default=None,
                        help='Analyze only the first N cells (test mode)')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(args.fig1_dir, args.data_dir, args.n_cells)
    elif args.mode == 'plot':
        plot_figure(args.data_dir)
    elif args.mode == 'stats':
        print_stats(args.data_dir)
