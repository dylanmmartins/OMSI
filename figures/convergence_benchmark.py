# -*- coding: utf-8 -*-
"""
figures/convergence_benchmark.py

Checks OMSI's auto-stop rule against conventional multi-chain MCMC diagnostics.

For each cell, runs several long fixed-length reference chains from deliberately
different spike-train starts, then computes rank-normalized split R-hat, bulk and
tail ESS, and autocorrelation on scalar summaries (spike count, amplitude,
baseline, noise, decay tau, log-likelihood) and split R-hat on 100 ms binned
spike counts and on 100 ms spike occupancy (any spike in bin). Counts check how
many spikes sit at each event; occupancy checks only where events are. Separately runs the default auto-stop sampler
with several seeds and compares where it stops and what it outputs against the
reference chains.

All chains for a cell share the same NNLS-derived hyperparameters (prior mean
for amplitude/baseline, firing rate, amplitude floor) so they target the same
distribution -- only the starting spike train differs.

To run the benchmark:
    $ python convergence_benchmark.py --mode test --data-dir /path/to/results

Quick pilot on a few cells with short chains:
    $ python convergence_benchmark.py --mode test --pilot

To create figure:
    $ python convergence_benchmark.py --mode plot --data-dir /path/to/results

To print summary statistics:
    $ python convergence_benchmark.py --mode print --data-dir /path/to/results

To run the shifted-start timing test (OMSI vs CaImAn, each from its own start):
    $ python convergence_benchmark.py --mode perturb --data-dir /path/to/results

Functions
---------
_split_chains
    Split each chain in half, doubling chain count.
_rank_normalize
    Rank-normalize draws across all chains to standard normal scores.
_rhat_basic
    Classic potential scale reduction along axis 1, vectorized over trailing axes.
rhat_rank
    Rank-normalized split R-hat (max of bulk and folded).
_ess_core
    Multi-chain ESS from Geyer initial monotone sequence.
ess_bulk
    Bulk ESS on rank-normalized split chains.
ess_tail
    Tail ESS: min ESS of 5% and 95% quantile indicators.
iat
    Integrated autocorrelation time of a single chain, in sweeps.
mean_acf
    Autocorrelation function averaged over chains.
_binned_counts
    Per-sweep spike counts in fixed-width time bins.
_loglik
    Gaussian log-likelihood per sweep from residual SSE and noise std.
_scalar_traces
    Scalar per-sweep summaries used for diagnostics, incl. log-posterior.
_convergence_sweep
    First chain length after which max split R-hat stays below threshold.
_prob_corr
    Pearson correlation of binned spike probability traces.
_score
    Window precision/recall, F-beta, and CosMIC against a reference spike train.
_pairwise
    Mean of a symmetric-ish agreement function over all pairs.
_events
    Merge spikes closer than EVENT_GAP_S into single events.
_event_agree
    One-to-one F1 between two chains' event lists.
_log_idx
    Log-spaced sweep indices for storing traces.
_make_inits
    Build overdispersed starting samples that share one set of hyperparameters.
_run_chain
    Run one sampler chain with a fixed seed and return full chain plus outputs.
_run_cell
    Run reference chains and auto-stop runs for one cell and save diagnostics.
_synthetic_tasks
    Generate synthetic cells for each condition and build task dicts.
_allen_tasks
    Sample Allen ground-truth cells and build task dicts.
run_test
    Build all tasks, run them in parallel, and save per-cell results.
_median_signed_error
    Median signed timing error of one-to-one matched spikes.
_match
    One-to-one matches within PERTURB_TOL via the timing figure's matcher.
_perturb_tasks
    Simulate non-bursty cells and build shifted starting samples for each.
_perturb_cell_omsi
    OMSI chains from every shifted start for one cell.
run_perturb
    Start OMSI and CaImAn from their own starts, shifted, and track timing error.
_load_results
    Load per-cell result files for current groups.
_by_group
    Map group to per-cell values over non-skipped cells.
_strip
    Strip plot with median bar per group, optionally several series.
_plot_example
    Spike count by sweep, trace, and called events per chain for one cell.
plot_figure
    Load per-cell results and generate the supplementary convergence figure.
_mm
    Median +/- MAD string over finite values.
print_stats
    Print per-group convergence and agreement statistics.
main
    Parse command-line arguments and dispatch.


DMM, September 2026
"""

import os

# One thread per worker -- parallelism is across cells.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import glob
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.stats import rankdata, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import OMSI.helpers as helpers
from OMSI.sampler import cont_ca_sampler
from OMSI.get_init_sample import get_init_sample
from OMSI.deconv import spikes_from_samples
from OMSI._win_perf import no_power_throttling

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'convergence')
_DEFAULT_ALLEN_H5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'fig3', 'allen_aggregated_data.h5')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

BETA     = 0.5
BIN_S    = 0.100
RHAT_THR = 1.01
ESS_THR  = 400
RHAT_CAP = 1000.0

# Example cells in the supplementary figure, as group:idx: one SNR 3 and one 7.5 Hz
# synthetic cell and one Allen cell, each with default-start chains that agree on
# events (F1 >= 0.93) but R-hat above 1.1. The Allen cell is the one among those
# whose chains move most in spike count. Indices refer to the seed-0 run.
EXAMPLE_CELLS = 'snr3:1,low_fs_snr5:2,allen_slow:0'

# Synthetic conditions where the NNLS init is well short of ground truth. Picked
# from a screen (32 cells/condition, 120 s, median F-beta NNLS init -> auto-stop):
# SNR 3 0.73 -> 0.89, SNR 2 0.43 -> 0.57, fast tau SNR 2 0.52 -> 0.89, 7.5 Hz
# SNR 5 0.73 -> 0.80. At 7.5 Hz, MCMC gains nothing at SNR 5 and loses to NNLS
# below it; the generator's default noise there saturates init (0.98).
SYNTH_CONDITIONS = {
    'snr3':          {'fs': 30.0, 'tau': 1.2, 'snr': 3.0},
    'snr2':          {'fs': 30.0, 'tau': 1.2, 'snr': 2.0},
    'fast_tau_snr2': {'fs': 30.0, 'tau': 0.4, 'snr': 2.0},
    'low_fs_snr5':   {'fs': 7.5,  'tau': 1.2, 'snr': 5.0},
}

GROUP_ORDER  = ['snr3', 'snr2', 'fast_tau_snr2', 'low_fs_snr5', 'allen_slow', 'allen_fast']
GROUP_NAMES  = {
    'snr3':          'SNR 3',
    'snr2':          'SNR 2',
    'fast_tau_snr2': 'fast tau, SNR 2',
    'low_fs_snr5':   '7.5 Hz, SNR 5',
    'allen_slow':    'Allen slow',
    'allen_fast':    'Allen fast',
}
# Blue, orange, green, and purple are reserved for deconvolution methods in the
# other figures -- conditions use reds, yellows, pinks, and browns only.
GROUP_COLORS = {
    'snr3':          '#C0182B',
    'snr2':          '#E0B000',
    'fast_tau_snr2': '#EE6FA8',
    'low_fs_snr5':   '#8C5A3C',
    'allen_slow':    '#5A0F2E',
    'allen_fast':    '#C8A77E',
}

INIT_LABELS = ['nnls', 'empty', 'foopsi', 'dense']

SCALAR_KEYS = ['ns', 'Am', 'Cb', 'sg', 'tau_decay', 'loglik', 'logpost']

# Chains started the way OMSI.deconv starts them (NNLS, re-jittered).
DEFAULT_STARTS = ('nnls', 'nnls_jit')

# Spikes closer than this are merged into one event for event-level agreement.
EVENT_GAP_S = 0.100

# Checkpoints for accuracy vs sweeps; clipped to chain length.
SWEEP_CHECKPOINTS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Bump when per-cell result contents change so old files get rerun.
RESULT_VERSION = 4

# Timing perturbation test: every spike of each method's own starting sample shifted
# by these many frames, fixed-length chains with no burn-in discarded.
PERTURB_COND   = {'fs': 30.0, 'tau': 1.2, 'snr': 6.0}
PERTURB_SHIFTS = [-2, -1, 0, 1, 2]
PERTURB_CELLS  = 20
# CaImAn's usual budget in our wrapper: 500 kept + 250 burn-in.
PERTURB_SWEEPS = 750
# Pairing window for timing error; wider than 100 ms so a 2-frame start still pairs.
PERTURB_TOL    = 0.150
# Bursty cells fail for a different reason (one amplitude per cell) -- left out.
# Bursty: at least this many ISIs of exactly the generator's 2 fine-grid steps.
BURSTY_MIN_PAIRS = 3
PERTURB_METHOD_NAMES = {'omsi': 'OMSI', 'caiman': 'CaImAn'}


def _split_chains(x):
    """Split each chain in half, doubling chain count. x: (M, N, ...)."""

    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half:]], axis=0)


def _rank_normalize(x):
    """Rank-normalize draws across all chains to standard normal scores."""

    r = rankdata(x, method='average').reshape(x.shape)
    return norm.ppf((r - 0.375) / (x.size + 0.25))


def _rhat_basic(x):
    """ Classic potential scale reduction along axis 1, vectorized over trailing axes.

    Parameters
    ----------
    x : np.ndarray
        Draws, shape (C, n, ...).

    Returns
    -------
    np.ndarray or float
        R-hat per trailing index. NaN where every draw is identical, inf where
        chains are each constant but disagree.
    """

    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    var_plus = (n - 1) / n * W + B / n
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sqrt(var_plus / W)
    r = np.where((W == 0) & (B == 0), np.nan, r)
    r = np.where((W == 0) & (B > 0), np.inf, r)
    return r if r.ndim else float(r)


def rhat_rank(x):
    """ Rank-normalized split R-hat (Vehtari et al. 2021), max of bulk and folded.

    Parameters
    ----------
    x : np.ndarray
        Draws, shape (M, N).

    Returns
    -------
    float
        R-hat, NaN if draws are constant, capped at RHAT_CAP.
    """

    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return np.nan
    xs = _split_chains(x)
    bulk = _rhat_basic(_rank_normalize(xs))
    fold = _rhat_basic(_rank_normalize(np.abs(xs - np.median(xs))))
    # Frozen chains give W ~ float noise and absurd ratios -- cap them.
    return float(min(np.nanmax([bulk, fold]), RHAT_CAP))


def _ess_core(x):
    """ Multi-chain ESS from Geyer initial monotone sequence.

    Parameters
    ----------
    x : np.ndarray
        Draws, shape (C, n).

    Returns
    -------
    float
        Effective sample size, NaN if draws are constant.
    """

    x = np.asarray(x, dtype=float)
    C, n = x.shape
    if n < 4:
        return np.nan

    y = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(y, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=1)[:, :n].real / n

    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = chain_var.mean()
    var_plus = mean_var * (n - 1) / n
    if C > 1:
        var_plus += x.mean(axis=1).var(ddof=1)
    if var_plus <= 0:
        return np.nan

    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0

    # Geyer: sum adjacent pairs while positive, then force monotone.
    pairs = []
    t = 0
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        pairs.append(p)
        t += 2
    if not pairs:
        return float(C * n)
    pairs = np.minimum.accumulate(np.array(pairs))
    tau = max(-1.0 + 2.0 * pairs.sum(), 1.0 / np.log10(C * n))
    return float(C * n / tau)


def ess_bulk(x):
    """Bulk ESS on rank-normalized split chains. x: (M, N)."""

    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return np.nan
    return _ess_core(_rank_normalize(_split_chains(x)))


def ess_tail(x):
    """Tail ESS: min ESS of 5% and 95% quantile indicators. x: (M, N)."""

    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return np.nan
    xs = _split_chains(x)
    out = []
    for q in (0.05, 0.95):
        ind = (xs <= np.quantile(x, q)).astype(float)
        if np.ptp(ind) == 0:
            continue
        out.append(_ess_core(ind))
    return float(np.nanmin(out)) if out else np.nan


def iat(x):
    """Integrated autocorrelation time of a single chain, in sweeps. x: (N,)."""

    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        return np.nan
    e = _ess_core(x[None, :])
    return float(len(x) / e) if e and np.isfinite(e) else np.nan


def mean_acf(x, max_lag):
    """ Autocorrelation function averaged over chains.

    Parameters
    ----------
    x : np.ndarray
        Draws, shape (M, N).
    max_lag : int
        Largest lag returned.

    Returns
    -------
    np.ndarray
        ACF for lags 0..max_lag, NaN if draws are constant.
    """

    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    max_lag = min(max_lag, n - 1)
    y = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(y, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=1)[:, :max_lag + 1].real
    with np.errstate(divide='ignore', invalid='ignore'):
        acf = acov / acov[:, :1]
    return np.nanmean(acf, axis=0)


def _binned_counts(ss, n_frames, bin_frames):
    """ Per-sweep spike counts in fixed-width time bins.

    Parameters
    ----------
    ss : list of np.ndarray
        Spike times per sweep, in frame units.
    n_frames : int
        Trace length in frames.
    bin_frames : int
        Bin width in frames.

    Returns
    -------
    np.ndarray
        Counts, shape (n_sweeps, n_bins), uint8 (clipped at 255).
    """

    n_bins = int(np.ceil(n_frames / bin_frames))
    n_sw = len(ss)
    lens = np.array([len(s) for s in ss], dtype=np.int64)
    if lens.sum() == 0:
        return np.zeros((n_sw, n_bins), dtype=np.uint8)
    t = np.concatenate([np.asarray(s, dtype=float) for s in ss])
    b = np.clip((t // bin_frames).astype(np.int64), 0, n_bins - 1)
    sweep = np.repeat(np.arange(n_sw, dtype=np.int64), lens)
    counts = np.bincount(sweep * n_bins + b, minlength=n_sw * n_bins)
    return np.minimum(counts, 255).astype(np.uint8).reshape(n_sw, n_bins)


def _loglik(sse, n_valid, sg):
    """Gaussian log-likelihood per sweep from residual SSE and noise std."""

    sg = np.asarray(sg, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = -sse / (2.0 * sg ** 2) - 0.5 * n_valid * np.log(2.0 * np.pi * sg ** 2)
    return np.where(sg > 0, ll, np.nan)


def _scalar_traces(chain, fs, lam_scale):
    """ Scalar per-sweep summaries used for diagnostics, keyed by SCALAR_KEYS.

    logpost adds the Poisson spike-count prior the birth-death moves use,
    n * log(lam * lam_scale), to the log-likelihood. Amplitude/baseline priors
    left out -- weak next to the likelihood.

    Parameters
    ----------
    chain : dict
        SAMPLES['chain'] from cont_ca_sampler.
    fs : float
        Frame rate in Hz.
    lam_scale : float
        Rate scale the sampler ran with.

    Returns
    -------
    dict of str to np.ndarray
        One trace per key in SCALAR_KEYS.
    """

    ns = np.asarray(chain['ns'], dtype=float)
    ll = _loglik(chain['sse'], chain['n_valid'], chain['sg'])
    lam_val = np.maximum(np.asarray(chain['lam'], dtype=float) * lam_scale, 1e-300)
    return {
        'ns':        ns,
        'Am':        np.asarray(chain['Am'], dtype=float),
        'Cb':        np.asarray(chain['Cb'], dtype=float),
        'sg':        np.asarray(chain['sg'], dtype=float),
        'tau_decay': np.asarray(chain['tau'], dtype=float)[:, 1] / fs,
        'loglik':    ll,
        'logpost':   ll + ns * np.log(lam_val),
    }


def _convergence_sweep(traces, step, thr):
    """ First chain length after which max split R-hat stays below threshold.

    At each grid length n, first half is treated as warmup and R-hat is taken
    on draws n/2..n -- same rule used for the full reference diagnostics.

    Parameters
    ----------
    traces : dict of str to np.ndarray
        Scalar traces, each shape (M, N).
    step : int
        Grid spacing in sweeps.
    thr : float
        R-hat threshold.

    Returns
    -------
    n_conv : float
        Chain length in sweeps, NaN if never reached.
    grid : np.ndarray
        Grid lengths evaluated.
    rmax : np.ndarray
        Max R-hat over scalars at each grid length.
    """

    N = next(iter(traces.values())).shape[1]
    grid = np.arange(max(step, 40), N + 1, step)
    rmax = np.full(len(grid), np.nan)
    for gi, n in enumerate(grid):
        vals = [rhat_rank(tr[:, n // 2:n]) for tr in traces.values()]
        vals = [v for v in vals if np.isfinite(v) or v == np.inf]
        rmax[gi] = max(vals) if vals else np.nan

    ok = ~(rmax > thr)
    # Last grid point that fails; converged from the next one on.
    bad = np.where(~ok)[0]
    if len(bad) == 0:
        return float(grid[0]), grid, rmax
    if bad[-1] == len(grid) - 1:
        return np.nan, grid, rmax
    return float(grid[bad[-1] + 1]), grid, rmax


def _prob_corr(p1, p2, bin_frames):
    """Pearson correlation of spike probability traces summed into bins."""

    n = (min(len(p1), len(p2)) // bin_frames) * bin_frames
    if n == 0:
        return np.nan
    a = np.asarray(p1[:n], dtype=float).reshape(-1, bin_frames).sum(axis=1)
    b = np.asarray(p2[:n], dtype=float).reshape(-1, bin_frames).sum(axis=1)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _score(ref_sp, pred_sp, fs):
    """ Window precision/recall, F-beta, and CosMIC against a reference spike train.

    Parameters
    ----------
    ref_sp : np.ndarray
        Reference spike times in seconds (ground truth or another run).
    pred_sp : np.ndarray
        Predicted spike times in seconds.
    fs : float
        Frame rate in Hz.

    Returns
    -------
    dict
        precision, recall, fbeta (window), fbeta_strict (one-to-one), cosmic.
    """

    def _fb(p, r):
        b2 = BETA ** 2
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom > 0 else 0.0

    ref_sp, pred_sp = np.asarray(ref_sp), np.asarray(pred_sp)
    p, r, _ = helpers.compute_accuracy_window([ref_sp], [pred_sp])
    p, r = float(p[0]), float(r[0])
    ps, rs, _ = helpers.compute_accuracy_strict([ref_sp], [pred_sp], tolerance=0.1)
    cos = float(helpers.compute_cosmic([ref_sp], [pred_sp], fs)[0])
    return {'precision': p, 'recall': r, 'fbeta': _fb(p, r),
            'fbeta_strict': _fb(float(ps[0]), float(rs[0])), 'cosmic': cos}


def _pairwise(items, fn):
    """Mean of fn(a, b) over all unordered pairs, NaN if fewer than two items."""

    vals = [fn(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _events(sp):
    """Merge spikes closer than EVENT_GAP_S into single events at their mean time."""

    sp = np.sort(np.asarray(sp, dtype=float))
    if len(sp) < 2:
        return sp
    groups = np.concatenate([[0], np.cumsum(np.diff(sp) > EVENT_GAP_S)])
    return np.array([sp[groups == g].mean() for g in range(groups[-1] + 1)])


def _event_agree(a, b):
    """ One-to-one F1 between two chains' event lists, +/-100 ms.

    Stacked spikes merged first, so this asks only whether both chains found
    the same events, not how many spikes they put at each.

    Parameters
    ----------
    a, b : np.ndarray
        Spike times in seconds.

    Returns
    -------
    float
        F1 in [0, 1]; 1 if both are empty.
    """

    ea, eb = _events(a), _events(b)
    if len(ea) == 0 and len(eb) == 0:
        return 1.0
    if len(ea) == 0 or len(eb) == 0:
        return 0.0
    return float(helpers.compute_accuracy_strict([ea], [eb], tolerance=0.1)[2][0])


def _log_idx(n, n_pts=400):
    """Log-spaced unique sweep indices in [0, n), always including 0."""

    idx = np.unique(np.geomspace(1, n, n_pts).astype(int) - 1)
    return idx[idx < n]


def _make_inits(y, params, seeds):
    """ Build overdispersed starting samples that share one set of hyperparameters.

    cont_ca_sampler takes its amplitude prior mean, firing rate, and amplitude
    floor from the init sample, so every chain reuses the NNLS values for those
    and only the starting spike train changes.

    Parameters
    ----------
    y : np.ndarray
        dF/F trace.
    params : dict
        Sampler params as the user would pass them.
    seeds : list of int
        One seed per chain.

    Returns
    -------
    inits : list of dict
        Init samples, one per chain.
    labels : list of str
        Init type per chain.
    """

    T = len(y)
    np.random.seed(seeds[0])
    base = get_init_sample(y, dict(params))

    inits, labels = [], []
    for k, seed in enumerate(seeds):
        np.random.seed(seed)
        sam = dict(base)
        label = INIT_LABELS[k] if k < len(INIT_LABELS) else 'nnls_jit'
        if label == 'nnls':
            sp = base['spiketimes_']
        elif label == 'empty':
            sp = np.empty(0)
        elif label == 'foopsi':
            sp = get_init_sample(y, dict(params, init_method='foopsi'))['spiketimes_']
        elif label == 'dense':
            # Extra uniform spikes, at least 2% of frames, so this chain starts
            # well above the spike count the others start near.
            n_extra = max(len(base['spiketimes_']), int(0.02 * T))
            sp = np.concatenate([base['spiketimes_'], np.random.rand(n_extra) * T])
        else:
            sp = get_init_sample(y, dict(params))['spiketimes_']
        sam['spiketimes_'] = np.sort(np.asarray(sp, dtype=np.float64))
        inits.append(sam)
        labels.append(label)
    return inits, labels


def _run_chain(y, params, seed, init=None, n_sweeps=None):
    """ Run one sampler chain with a fixed seed and return full chain plus outputs.

    Parameters
    ----------
    y : np.ndarray
        dF/F trace.
    params : dict
        Sampler params as the user would pass them.
    seed : int
        Seed for both numpy (init jitter) and numba (sampler) RNGs.
    init : dict, optional
        Init sample. None lets the sampler build its own, as OMSI.deconv does.
    n_sweeps : int, optional
        Fixed chain length with no burn-in discard. None runs auto-stop.

    Returns
    -------
    dict or None
        chain, time, p_switched, lam_scale. None if the sampler skipped the cell.
    """

    p = dict(params)
    p['seed'] = int(seed)
    p['return_full'] = True
    if init is not None:
        p['init'] = init
    if n_sweeps is None:
        p['auto_stop'] = True
    else:
        p['auto_stop'] = False
        p['Nsamples'] = int(n_sweeps)
        p['B'] = 0

    p_in = p.get('p', 2)
    np.random.seed(seed)
    t0 = time.time()
    S = cont_ca_sampler(y, p)
    elapsed = time.time() - t0
    if 'chain' not in S:
        return None
    return {
        'chain': S['chain'],
        'time': elapsed,
        # Fast-indicator path rebuilds the init, overriding the spike start.
        'p_switched': bool(p.get('p', p_in) != p_in),
        'lam_scale': float(p['lam_scale']),
    }


def _run_cell(task):
    """ Run reference chains and auto-stop runs for one cell and save diagnostics.

    Parameters
    ----------
    task : dict
        Keys: group, cell_idx, dff, true_spikes, fs, params, seeds, n_chains,
        n_sweeps, n_auto, warmup_frac, out_path.

    Returns
    -------
    str
        Status message.
    """

    y = np.asarray(task['dff'], dtype=np.float64)
    fs = float(task['fs'])
    params = task['params']
    T = len(y)
    bin_frames = max(1, int(round(BIN_S * fs)))
    lag_s = 0.0
    true_sp = np.asarray(task['true_spikes'], dtype=float)
    M, N, K = task['n_chains'], task['n_sweeps'], task['n_auto']
    seeds = task['seeds']
    t_cell = time.time()

    result = {
        'group': task['group'], 'cell_idx': task['cell_idx'], 'fs': fs,
        'n_frames': T, 'n_true': len(true_sp), 'n_chains': M, 'n_sweeps': N,
        'skipped': False, 'version': RESULT_VERSION,
    }

    inits, labels = _make_inits(y, params, seeds[:M])
    refs = []
    for k in range(M):
        r = _run_chain(y, params, seeds[k], init=inits[k], n_sweeps=N)
        if r is None:
            result['skipped'] = True
            np.save(task['out_path'], np.array(result, dtype=object), allow_pickle=True)
            return '{} cell {}: skipped by SNR gate.'.format(task['group'], task['cell_idx'])
        refs.append(r)

    result['init_labels'] = labels
    result['p_switched'] = any(r['p_switched'] for r in refs)
    result['ref_time'] = float(np.mean([r['time'] for r in refs]))

    # Scalar diagnostics on post-warmup draws.
    w0 = int(task['warmup_frac'] * N)
    tr_each = [_scalar_traces(r['chain'], fs, r['lam_scale']) for r in refs]
    tr_full = {k: np.stack([t[k] for t in tr_each]) for k in SCALAR_KEYS}
    del tr_each
    tr_post = {k: v[:, w0:] for k, v in tr_full.items()}
    result['rhat']     = {k: rhat_rank(v) for k, v in tr_post.items()}
    result['ess_bulk'] = {k: ess_bulk(v) for k, v in tr_post.items()}
    result['ess_tail'] = {k: ess_tail(v) for k, v in tr_post.items()}
    result['iat']      = {k: float(np.nanmedian([iat(c) for c in v]))
                          for k, v in tr_post.items()}
    finite = [v for v in result['rhat'].values() if not np.isnan(v)]
    result['rhat_max'] = max(finite) if finite else np.nan

    # Same, restricted to chains started the way OMSI.deconv starts them.
    # NaN here means every draw identical (frozen, identical chains).
    dflt = [i for i, l in enumerate(labels) if l in DEFAULT_STARTS]
    result['default_idx'] = dflt
    if len(dflt) >= 2:
        result['rhat_default'] = {k: rhat_rank(v[dflt]) for k, v in tr_post.items()}
        finite = [v for v in result['rhat_default'].values() if not np.isnan(v)]
        result['rhat_default_max'] = max(finite) if finite else np.nan
    else:
        result['rhat_default'] = {k: np.nan for k in tr_post}
        result['rhat_default_max'] = np.nan
    result['acf_ns']     = mean_acf(tr_post['ns'], 300)
    result['acf_loglik'] = mean_acf(tr_post['loglik'], 300)
    # ESS and ACF on default-start chains too. Constant chains have no ACF, so
    # also count how many default chains never change spike count after warmup.
    if dflt:
        result['ess_bulk_default'] = {k: ess_bulk(v[dflt]) for k, v in tr_post.items()}
        result['acf_ns_default']     = mean_acf(tr_post['ns'][dflt], 300)
        result['acf_loglik_default'] = mean_acf(tr_post['loglik'][dflt], 300)
        result['frozen_default'] = float(np.mean([np.ptp(c) == 0 for c in tr_post['ns'][dflt]]))
    else:
        result['ess_bulk_default'] = {k: np.nan for k in tr_post}
        result['acf_ns_default'] = result['acf_loglik_default'] = np.full(301, np.nan)
        result['frozen_default'] = np.nan

    step = max(50, N // 60)
    for thr, key in [(RHAT_THR, 'n_conv'), (1.05, 'n_conv_105')]:
        n_conv, grid, rmax = _convergence_sweep(tr_full, step, thr)
        result[key] = n_conv
    result['conv_grid'] = grid
    result['conv_rmax'] = rmax

    # Spike-time posterior: split R-hat per active bin, on spike counts and on
    # occupancy. Chains can agree on where events are (occupancy) while
    # disagreeing on how many spikes sit at each one (counts). Constant bins
    # (NaN) count as converged.
    counts = np.stack([_binned_counts(r['chain']['ss'][w0:], T, bin_frames)
                       for r in refs])
    active = counts.max(axis=(0, 1)) > 0
    result['n_bins_active'] = int(active.sum())
    for tag, arr in [('bin', counts[:, :, active].astype(np.float32)),
                     ('occ', (counts[:, :, active] > 0).astype(np.float32))]:
        rb = _rhat_basic(_split_chains(arr)) if active.any() else np.array([])
        rb = np.where(np.isnan(rb), 1.0, rb)
        result[tag + '_frac_101'] = float(np.mean(rb < RHAT_THR)) if rb.size else np.nan
        result[tag + '_frac_105'] = float(np.mean(rb < 1.05)) if rb.size else np.nan
        result[tag + '_rhat_max'] = float(np.max(rb)) if rb.size else np.nan
    # Bins where chains' spike probability differs by more than 0.5.
    p_bin = (counts > 0).mean(axis=1)
    spread = p_bin.max(axis=0) - p_bin.min(axis=0)
    result['bin_disagree'] = int(np.sum(spread[active] > 0.5))
    del counts

    # Reference outputs per chain, same spike calling as OMSI.deconv. Chains
    # don't necessarily mix, so no pooling -- compare against each chain and
    # against the one with highest mean post-warmup log-posterior.
    per_chain = [spikes_from_samples(r['chain']['ss'][w0:], T, fs, lag_s=lag_s) for r in refs]
    ref_probs = [pr for pr, _ in per_chain]
    ref_spikes = [sp for _, sp in per_chain]
    best = int(np.nanargmax(np.nanmean(tr_post['logpost'], axis=1)))
    result['ref_best'] = best
    result['ref_best_label'] = labels[best]
    result['ref_chain_score'] = [_score(true_sp, sp, fs) for sp in ref_spikes]
    result['ref_best_score'] = result['ref_chain_score'][best]
    result['ref_chain_ns'] = [float(v) for v in tr_post['ns'].mean(axis=1)]
    # Called spikes per chain and the trace itself, for example-cell panels.
    result['ref_spikes'] = [np.asarray(sp, dtype=np.float32) for sp in ref_spikes]
    result['true_spikes'] = true_sp.astype(np.float32)
    result['dff'] = y.astype(np.float32)
    result['ref_chain_logpost'] = [float(v) for v in np.nanmean(tr_post['logpost'], axis=1)]
    result['ref_ref_cosmic'] = _pairwise(
        ref_spikes, lambda a, b: _score(a, b, fs)['cosmic'])
    result['ref_ref_probcorr'] = _pairwise(
        ref_probs, lambda a, b: _prob_corr(a, b, bin_frames))
    result['event_agree_all'] = _pairwise(ref_spikes, _event_agree)
    result['event_agree_default'] = _pairwise([ref_spikes[i] for i in dflt], _event_agree)
    result['ref_default_score'] = {
        m: float(np.median([result['ref_chain_score'][i][m] for i in dflt]))
        for m in ('fbeta', 'fbeta_strict', 'cosmic')} if dflt else None

    # Accuracy vs sweeps: output as if the chain stopped at each checkpoint,
    # first half dropped as warmup. Column 0 is the starting spike train.
    grid = [c for c in SWEEP_CHECKPOINTS if c < N] + [N]
    result['sweep_grid'] = np.array([0] + grid)
    sweep_score = {m: np.full((M, len(grid) + 1), np.nan)
                   for m in ('fbeta', 'fbeta_strict', 'cosmic')}
    for k, r in enumerate(refs):
        # Sampler times are 1-based; see spikes_from_samples.
        init_sp = np.clip(inits[k]['spiketimes_'] - 1.0 - lag_s * fs, 0, T - 1) / fs
        sc = _score(true_sp, init_sp, fs)
        for m in sweep_score:
            sweep_score[m][k, 0] = sc[m]
        for gi, n in enumerate(grid, 1):
            _, sp = spikes_from_samples(r['chain']['ss'][n // 2:n], T, fs, lag_s=lag_s)
            sc = _score(true_sp, sp, fs)
            for m in sweep_score:
                sweep_score[m][k, gi] = sc[m]
    result['sweep_score'] = sweep_score

    # Log-spaced traces for plotting, warmup included.
    tidx = _log_idx(N)
    result['trace_idx'] = tidx
    result['ref_traces'] = {k: tr_full[k][:, tidx].astype(np.float32)
                            for k in ('ns', 'loglik', 'Am')}
    del refs, per_chain

    # Ground-truth spike counts per frame, on the same 0-based frame clock as the
    # probability traces, so agreement with truth uses the same measures as
    # agreement between runs.
    true_counts = np.bincount(np.clip(np.floor(true_sp * fs).astype(int), 0, T - 1),
                              minlength=T).astype(float)

    # Auto-stop runs, as a user would get them from OMSI.deconv.
    autos = []
    for j in range(K):
        r = _run_chain(y, params, seeds[M + j])
        if r is None:
            continue
        ch = r['chain']
        b0, stop = ch['B_final'], ch['stop_idx']
        prob, sp = spikes_from_samples(ch['ss'][b0:stop], T, fs, lag_s=lag_s)
        tr = _scalar_traces(ch, fs, r['lam_scale'])
        autos.append({
            'B_final': b0, 'stop_idx': stop, 'time': r['time'],
            'hit_max': stop >= int(params.get('max_sweeps', 2000)),
            'prob': prob, 'spikes': sp,
            'score': _score(true_sp, sp, fs),
            'event_vs_truth': _event_agree(true_sp, sp),
            'probcorr_vs_truth': _prob_corr(prob, true_counts, bin_frames),
            'vs_best_cosmic': _score(ref_spikes[best], sp, fs)['cosmic'],
            'vs_best_probcorr': _prob_corr(prob, ref_probs[best], bin_frames),
            'vs_chains_cosmic': float(np.nanmean(
                [_score(rs, sp, fs)['cosmic'] for rs in ref_spikes])),
            'vs_chains_probcorr': float(np.nanmean(
                [_prob_corr(prob, rp, bin_frames) for rp in ref_probs])),
            'ns_mean': float(np.mean(tr['ns'][b0:stop])),
            'logpost_mean': float(np.nanmean(tr['logpost'][b0:stop])),
            'kept': {k: tr[k][b0:stop] for k in SCALAR_KEYS},
            'trace_idx': _log_idx(stop),
            'trace_ns': tr['ns'][_log_idx(stop)].astype(np.float32),
        })

    result['auto'] = [{k: v for k, v in a.items() if k not in ('prob', 'spikes', 'kept')}
                      for a in autos]
    result['auto_auto_cosmic'] = _pairwise(
        [a['spikes'] for a in autos], lambda a, b: _score(a, b, fs)['cosmic'])
    result['auto_auto_probcorr'] = _pairwise(
        [a['prob'] for a in autos], lambda a, b: _prob_corr(a, b, bin_frames))
    result['event_agree_auto'] = _pairwise([a['spikes'] for a in autos], _event_agree)

    # Do independent auto-stop runs agree with each other at the point they stop?
    # Kept windows truncated to common length; same quantities as reference R-hat.
    if len(autos) >= 2:
        L = min(len(a['kept']['ns']) for a in autos)
        result['auto_rhat'] = {
            k: rhat_rank(np.stack([a['kept'][k][:L] for a in autos])) if L >= 8 else np.nan
            for k in SCALAR_KEYS}
    else:
        result['auto_rhat'] = {k: np.nan for k in SCALAR_KEYS}
    finite = [v for v in result['auto_rhat'].values() if not np.isnan(v)]
    result['auto_rhat_max'] = max(finite) if finite else np.nan

    result['cell_time'] = time.time() - t_cell
    np.save(task['out_path'], np.array(result, dtype=object), allow_pickle=True)
    return '{} cell {}: max R-hat {:.3f}, n_conv {}, auto stop {} ({:.0f}s).'.format(
        task['group'], task['cell_idx'], result['rhat_max'], result['n_conv'],
        [a['stop_idx'] for a in autos], result['cell_time'])


def _synthetic_tasks(n_cells, duration, seed):
    """ Generate synthetic cells for each condition and build task dicts.

    Parameters
    ----------
    n_cells : int
        Cells per condition.
    duration : float
        Trace duration in seconds.
    seed : int
        Base seed.

    Returns
    -------
    list of dict
        Partial task dicts (no run settings yet).
    """

    from simulation_helpers import generate_synthetic_data

    tasks = []
    for gi, (group, cond) in enumerate(SYNTH_CONDITIONS.items()):
        print('Generating {} cells for {}...'.format(n_cells, group))
        np.random.seed(seed + gi)
        dff, true_spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=n_cells, fs=cond['fs'], duration=duration,
            tau=cond['tau'], snr=cond['snr'])
        # Same params as the figure 2 OMSI runs.
        params = {'f': cond['fs'], 'p': 2}
        for i in range(n_cells):
            tasks.append({'group': group, 'gi': gi, 'cell_idx': i, 'dff': dff[i],
                          'true_spikes': true_spikes[i], 'fs': cond['fs'],
                          'params': params})
    return tasks


def _allen_tasks(h5_path, n_cells, seed):
    """ Sample Allen ground-truth cells and build task dicts.

    Uses figure 3's dataset exclusions, kurtosis filter, and sampler params.
    Splits n_cells evenly between slow and fast indicators.

    Parameters
    ----------
    h5_path : str
        Aggregated Allen HDF5 file written by figure3.py.
    n_cells : int
        Total Allen cells to sample.
    seed : int
        Base seed.

    Returns
    -------
    list of dict
        Partial task dicts (no run settings yet).
    """

    from figure3 import load_aggregated_data, _EXCLUDED_DATASETS

    slow, fast = load_aggregated_data(h5_path)
    rng = np.random.default_rng(seed)
    tasks = []
    for gi, (group, groups) in enumerate([('allen_slow', slow), ('allen_fast', fast)]):
        pool = []
        for (exp_name, _, _), gd in groups.items():
            if exp_name in _EXCLUDED_DATASETS:
                continue
            kurt = helpers.compute_kurtosis(gd['dff'])
            for i in np.where(kurt >= 0.5)[0]:
                pool.append((gd, int(i)))
        n_take = min(n_cells // 2, len(pool))
        print('Sampling {} of {} eligible {} cells...'.format(n_take, len(pool), group))
        for ci, pi in enumerate(rng.choice(len(pool), size=n_take, replace=False)):
            gd, i = pool[pi]
            fs, tau = float(gd['fs']), float(gd['tau'])
            tau_rise = 0.05
            g_rise  = float(np.exp(-1.0 / (tau_rise * fs)))
            g_decay = float(np.exp(-1.0 / (tau * fs)))
            params = {
                'f': fs, 'p': 2, 'marg': 0, 'upd_gam': 1,
                'g': [g_rise + g_decay, -g_rise * g_decay], 'defg': [g_rise, g_decay],
                'TauStd': [tau_rise * fs, tau * fs], 'lam_scale': 1.0,
            }
            tasks.append({'group': group, 'gi': 10 + gi, 'cell_idx': ci,
                          'dff': gd['dff'][i], 'true_spikes': gd['spikes_list'][i],
                          'fs': fs, 'params': params})
    return tasks


def run_test(data_dir=_DEFAULT_DATA_DIR, n_cells=16, duration=300.0, n_chains=10,
             n_sweeps=10000, n_auto=10, warmup_frac=0.5, n_allen=16,
             allen_h5=_DEFAULT_ALLEN_H5, workers=None, seed=0, overwrite=False):
    """ Build all tasks, run them in parallel, and save per-cell results.

    Parameters
    ----------
    data_dir : str, optional
        Output directory. Per-cell results go in data_dir/cells.
    n_cells : int, optional
        Synthetic cells per condition.
    duration : float, optional
        Synthetic trace duration in seconds.
    n_chains : int, optional
        Reference chains per cell.
    n_sweeps : int, optional
        Reference chain length in sweeps.
    n_auto : int, optional
        Auto-stop runs per cell.
    warmup_frac : float, optional
        Fraction of each reference chain discarded as warmup.
    n_allen : int, optional
        Allen cells in total, split across slow and fast. 0 skips Allen.
    allen_h5 : str, optional
        Aggregated Allen HDF5 file.
    workers : int, optional
        Worker processes. Defaults to CPU count.
    seed : int, optional
        Base seed.
    overwrite : bool, optional
        Rerun cells that already have results.
    """

    cell_dir = os.path.join(data_dir, 'cells')
    os.makedirs(cell_dir, exist_ok=True)

    tasks = _synthetic_tasks(n_cells, duration, seed)
    if n_allen > 0:
        if os.path.exists(allen_h5):
            tasks += _allen_tasks(allen_h5, n_allen, seed)
        else:
            print('No Allen file at {} -- skipping Allen cells.'.format(allen_h5))

    todo = []
    for t in tasks:
        t['out_path'] = os.path.join(cell_dir, '{}_{:03d}.npy'.format(t['group'], t['cell_idx']))
        if os.path.exists(t['out_path']) and not overwrite:
            # Reuse only if saved with same chain settings.
            prev = np.load(t['out_path'], allow_pickle=True).item()
            same = (prev.get('n_sweeps') == n_sweeps and prev.get('n_chains') == n_chains
                    and len(prev.get('auto', [])) == n_auto)
            if prev.get('version') == RESULT_VERSION and (prev.get('skipped') or same):
                continue
        rng = np.random.default_rng([seed, t['gi'], t['cell_idx']])
        t['seeds'] = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n_chains + n_auto)]
        t.update({'n_chains': n_chains, 'n_sweeps': n_sweeps, 'n_auto': n_auto,
                  'warmup_frac': warmup_frac})
        todo.append(t)

    print('{} cells total, {} to run ({} already done).'.format(
        len(tasks), len(todo), len(tasks) - len(todo)))
    if not todo:
        return

    # Longest traces first so the slow ones start early.
    todo.sort(key=lambda t: -len(t['dff']))
    workers = workers or os.cpu_count()
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(_run_cell, t): t for t in todo}
        for n_done, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                msg = fut.result()
            except Exception as exc:
                msg = '{} cell {} failed: {}'.format(t['group'], t['cell_idx'], exc)
            print('  [{}/{}] {}'.format(n_done, len(todo), msg), flush=True)
    print('Finished in {:.1f} min. Results in {}.'.format((time.time() - t0) / 60, cell_dir))


def _median_signed_error(true_sp, pred_sp):
    """ Median signed timing error (ms) of one-to-one matched spikes, NaN if none.

    Parameters
    ----------
    true_sp, pred_sp : np.ndarray
        True and predicted spike times in seconds.

    Returns
    -------
    float
        Median of predicted minus true over matched pairs, in ms.
    """

    it, ip = _match(true_sp, pred_sp)
    if len(it) == 0:
        return np.nan
    return float(np.median(pred_sp[ip] - true_sp[it])) * 1000.0


def _match(t, p):
    """ One-to-one matches within PERTURB_TOL via the timing figure's matcher. """

    from timing_calibration import _match_pairs
    return _match_pairs(np.asarray(t, dtype=float), np.asarray(p, dtype=float), PERTURB_TOL)


def _perturb_tasks(n_cells, duration, seed):
    """ Simulate non-bursty cells and build shifted starting samples for each.

    Parameters
    ----------
    n_cells : int
        Non-bursty cells to keep.
    duration : float
        Trace duration in seconds.
    seed : int
        Base seed.

    Returns
    -------
    list of dict
        One task per cell: dff, true spikes, fs, params, and one init per shift.
    """

    from simulation_helpers import generate_synthetic_data

    cond = PERTURB_COND
    fs = cond['fs']
    np.random.seed(seed + 100)
    dff, true_spikes, _, _, _, _ = generate_synthetic_data(
        n_cells=int(np.ceil(n_cells * 1.5)), fs=fs, duration=duration,
        tau=cond['tau'], snr=cond['snr'])

    burst_isi = 2.0 / (10.0 * fs)
    params = {'f': fs, 'p': 2}
    tasks = []
    for i in range(len(true_spikes)):
        t = np.sort(np.asarray(true_spikes[i], dtype=float))
        if np.sum(np.abs(np.diff(t) - burst_isi) < 1e-9) >= BURSTY_MIN_PAIRS or len(t) == 0:
            continue
        y = np.asarray(dff[i], dtype=np.float64)
        np.random.seed(seed + 1000 + i)
        base = get_init_sample(y, dict(params))
        inits = []
        for sh in PERTURB_SHIFTS:
            sam = dict(base)
            # Sampler times live on (0, T]; keep shifted spikes inside.
            sam['spiketimes_'] = np.clip(np.asarray(base['spiketimes_'], dtype=float) + sh,
                                         1e-3, len(y))
            inits.append(sam)
        tasks.append({'cell_idx': i, 'dff': y, 'true_spikes': t, 'fs': fs,
                      'params': params, 'inits': inits, 'seed': seed + 2000 + i})
        if len(tasks) == n_cells:
            break
    return tasks


def _perturb_cell_omsi(task):
    """ OMSI chains from every shifted start for one cell.

    Parameters
    ----------
    task : dict
        From _perturb_tasks, plus n_sweeps and sweep_idx.

    Returns
    -------
    np.ndarray
        Median signed error (ms), shape (n_shifts, len(sweep_idx) + 1); column 0 is
        the start itself.
    """

    fs, t, idx = task['fs'], task['true_spikes'], task['sweep_idx']
    out = np.full((len(PERTURB_SHIFTS), len(idx) + 1), np.nan)
    for k, init in enumerate(task['inits']):
        # Sampler times are 1-based; see spikes_from_samples.
        out[k, 0] = _median_signed_error(t, (np.sort(init['spiketimes_']) - 1.0) / fs)
        r = _run_chain(task['dff'], task['params'], task['seed'] + k, init=dict(init),
                       n_sweeps=task['n_sweeps'])
        if r is None:
            continue
        ss = r['chain']['ss']
        for j, i in enumerate(idx):
            if i < len(ss):
                out[k, j + 1] = _median_signed_error(
                    t, (np.sort(np.asarray(ss[i], dtype=float)) - 1.0) / fs)
    return out


def run_perturb(data_dir=_DEFAULT_DATA_DIR, n_cells=PERTURB_CELLS, duration=120.0,
                n_sweeps=PERTURB_SWEEPS, workers=None, seed=0, run_matlab=True):
    """ Start OMSI and CaImAn from their own starts, shifted, and track timing error.

    Parameters
    ----------
    data_dir : str, optional
        Output directory; results go to data_dir/perturb.npz.
    n_cells : int, optional
        Non-bursty cells.
    duration : float, optional
        Trace duration in seconds.
    n_sweeps : int, optional
        Sweeps per chain, none discarded.
    workers : int, optional
        Worker processes for OMSI. Defaults to CPU count.
    seed : int, optional
        Base seed.
    run_matlab : bool, optional
        Whether to run CaImAn.
    """

    os.makedirs(data_dir, exist_ok=True)
    tasks = _perturb_tasks(n_cells, duration, seed)
    idx = _log_idx(n_sweeps, 60)
    print('Perturbation test: {} cells x {} shifts x {} sweeps...'.format(
        len(tasks), len(PERTURB_SHIFTS), n_sweeps))

    for t in tasks:
        t.update({'n_sweeps': n_sweeps, 'sweep_idx': idx})
    out = {'shifts': np.array(PERTURB_SHIFTS), 'sweep_idx': np.asarray(idx),
           'fs': np.array([tasks[0]['fs']]), 'cell_idx': np.array([t['cell_idx'] for t in tasks])}

    print('Running OMSI...')
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count(), mp_context=ctx) as ex:
        res = list(ex.map(_perturb_cell_omsi, tasks))
    out['omsi'] = np.stack(res, axis=1)
    print('  OMSI took {:.1f} min.'.format((time.time() - t0) / 60))

    if run_matlab:
        from run_pnev_MCMC import run_matlab_pnevMCMC

        print('Running CaImAn (MATLAB) from its own shifted starts...')
        dff = np.stack([t['dff'] for t in tasks for _ in PERTURB_SHIFTS])
        shifts = [float(sh) for _ in tasks for sh in PERTURB_SHIFTS]
        _, _, _, _, samples, init_sp = run_matlab_pnevMCMC(
            dff, fs=tasks[0]['fs'], tau=PERTURB_COND['tau'],
            n_sweeps=n_sweeps, return_samples=True, init_shifts=shifts, burn_in=0,
            verbose=False)
        cai = np.full((len(PERTURB_SHIFTS), len(tasks), len(idx) + 1), np.nan)
        fs = tasks[0]['fs']
        for c, t in enumerate(tasks):
            for k in range(len(PERTURB_SHIFTS)):
                row = c * len(PERTURB_SHIFTS) + k
                ss = samples[row]
                # Wrapper already shifted CaImAn start and samples to 0-based frames.
                cai[k, c, 0] = _median_signed_error(t['true_spikes'], np.sort(init_sp[row]) / fs)
                for j, i in enumerate(idx):
                    if i < len(ss):
                        cai[k, c, j + 1] = _median_signed_error(
                            t['true_spikes'], np.sort(ss[i]) / fs)
        out['caiman'] = cai

    out_path = os.path.join(data_dir, 'perturb.npz')
    np.savez(out_path, **out)
    print('Saved {}.'.format(out_path))


def _load_results(data_dir):
    """Load per-cell result files for current groups, ordered by group then cell."""

    paths = sorted(glob.glob(os.path.join(data_dir, 'cells', '*.npy')))
    res = [np.load(p, allow_pickle=True).item() for p in paths]
    stale = [r for r in res if r['group'] not in GROUP_ORDER]
    if stale:
        print('Ignoring {} result files from groups no longer in the benchmark ({}).'.format(
            len(stale), ', '.join(sorted({r['group'] for r in stale}))))
    res = [r for r in res if r['group'] in GROUP_ORDER]
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    res.sort(key=lambda r: (order.get(r['group'], 99), r['cell_idx']))
    return res


def _by_group(results, fn):
    """Map group to array of fn(r) over non-skipped cells, in GROUP_ORDER."""

    out = {}
    for g in GROUP_ORDER:
        vals = [fn(r) for r in results if r['group'] == g and not r['skipped']]
        if vals:
            out[g] = np.asarray(vals, dtype=float)
    return out


def _strip(ax, series, ylabel, hline=None, log=False):
    """ Strip plot with median bar per group, optionally several series side by side.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    series : list of (str, dict, dict)
        (label, group-to-values dict, scatter style kwargs) per series.
    ylabel : str
        Y-axis label.
    hline : float or list of float, optional
        Reference lines.
    log : bool, optional
        Log y-axis.
    """

    rng = np.random.default_rng(0)
    groups = [g for g in GROUP_ORDER if any(g in d for _, d, _ in series)]
    n_s = len(series)
    width = 0.8 / n_s
    for si, (label, data, style) in enumerate(series):
        for xi, g in enumerate(groups):
            v = data.get(g, np.array([]))
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            xc = xi - 0.4 + width * (si + 0.5)
            kw = dict(s=6, linewidths=0.6)
            kw.update(style)
            if kw.pop('filled', True):
                kw.update(color=GROUP_COLORS[g])
            else:
                kw.update(facecolors='none', edgecolors=GROUP_COLORS[g])
            ax.scatter(xc + rng.uniform(-width / 3, width / 3, len(v)), v, **kw)
            ax.plot([xc - width / 2.2, xc + width / 2.2], [np.median(v)] * 2, color='k', lw=0.8)
    if hline is not None:
        for h in np.atleast_1d(hline):
            ax.axhline(h, color='0.5', lw=0.6, ls='--')
    if log:
        ax.set_yscale('log')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_NAMES[g] for g in groups], rotation=30, ha='right', fontsize=6)
    ax.set_ylabel(ylabel)
    if n_s > 1:
        # Gray proxies -- series differ by marker/fill, groups by color.
        for label, _, style in series:
            filled = style.get('filled', True)
            ax.scatter([], [], s=8, marker=style.get('marker', 'o'), linewidths=0.6,
                       facecolors='0.3' if filled else 'none', edgecolors='0.3', label=label)
        ax.legend(frameon=False, fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)


def _plot_example(fig, spec, r, window_s=20.0):
    """ Spike count by sweep, trace, and called events per chain for one cell.

    Default-start chains only. Event raster is shown over the window_s stretch of
    the trace holding the most true spikes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    spec : matplotlib.gridspec.SubplotSpec
        Region for this example.
    r : dict
        Per-cell result.
    window_s : float, optional
        Raster window length in seconds.

    Returns
    -------
    list of matplotlib.axes.Axes
        Count, trace, and raster axes.
    """

    fs = r['fs']
    dflt = r['default_idx']
    color = GROUP_COLORS[r['group']]
    # Row 1 is a spacer so the sweep axis labels clear the trace.
    sub = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=spec, hspace=0.15,
                                           height_ratios=[1.3, 0.45, 0.8, 1.0])

    ax_n = fig.add_subplot(sub[0])
    tidx = r['trace_idx']
    for k in dflt:
        ax_n.plot(tidx + 1, r['ref_traces']['ns'][k], color=color, lw=0.6, alpha=0.8)
    ax_n.axvspan(r['n_sweeps'] // 2, r['n_sweeps'], color='0.5', alpha=0.1, linewidth=0)
    ax_n.set_xscale('log')
    ax_n.set_ylim(bottom=0)
    ax_n.set_xlabel('sweep', labelpad=1)
    ax_n.set_ylabel('spike count')
    ax_n.set_title('{} #{}\n$\\hat{{R}}$ = {:.1f}, event agreement = {:.2f}'.format(
        GROUP_NAMES[r['group']], r['cell_idx'], r['rhat_default_max'],
        r['event_agree_default']), fontsize=6.5)

    true_sp = np.asarray(r['true_spikes'], dtype=float)
    dur = r['n_frames'] / fs
    w = min(window_s, dur)
    starts = np.arange(0.0, dur - w + 1e-9, 1.0)
    t0 = starts[int(np.argmax([np.sum((true_sp >= s) & (true_sp < s + w)) for s in starts]))]

    ax_y = fig.add_subplot(sub[2])
    t = np.arange(r['n_frames']) / fs
    m = (t >= t0) & (t < t0 + w)
    ax_y.plot(t[m] - t0, r['dff'][m], color='0.3', lw=0.5)
    ax_y.set_xlim(0, w)
    ax_y.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_y.set_yticks([])
    ax_y.spines['left'].set_visible(False)
    ax_y.spines['bottom'].set_visible(False)

    ax_r = fig.add_subplot(sub[3], sharex=ax_y)
    rows = [('true', true_sp, 'k')] + [('', np.asarray(r['ref_spikes'][k], dtype=float), color)
                                       for k in dflt]
    for i, (_, sp, c) in enumerate(rows):
        sp = sp[(sp >= t0) & (sp < t0 + w)] - t0
        ax_r.vlines(sp, i + 0.1, i + 0.9, color=c, lw=0.6)
    ax_r.set_ylim(len(rows), 0)
    ax_r.set_yticks([0.5, len(rows) / 2 + 0.5])
    ax_r.set_yticklabels(['true', 'chains'], fontsize=5.5)
    ax_r.tick_params(axis='y', length=0)
    ax_r.set_xlabel('time (s)', labelpad=1)
    ax_r.spines['left'].set_visible(False)
    return ax_n, ax_y, ax_r


def plot_figure(data_dir=_DEFAULT_DATA_DIR, examples=None):
    """ Load per-cell results and generate the supplementary convergence figure.

    Top: three example cells whose default-start chains call the same events but
    have high R-hat. Below, per condition: split R-hat; bulk ESS, spike-count spread
    between chains, and spike-count autocorrelation; event agreement and spike
    probability correlation between independent auto-stop runs.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the cells/ results folder.
    examples : str, optional
        Example cells as comma-separated 'group:idx'. Defaults to EXAMPLE_CELLS.
    """

    results = _load_results(data_dir)
    done = [r for r in results if not r['skipped']]
    if not done:
        raise FileNotFoundError('No results in {}. Run --mode test first.'.format(data_dir))
    old = [r for r in done if r.get('version') != RESULT_VERSION]
    if old:
        raise RuntimeError('{} result files are from an older version -- rerun --mode test.'.format(
            len(old)))

    exs = []
    for e in (examples or EXAMPLE_CELLS).split(','):
        g, i = e.split(':')
        exs.append(next(r for r in done if r['group'] == g and r['cell_idx'] == int(i)))
    print('Example cells: {}.'.format(', '.join(
        '{}:{}'.format(r['group'], r['cell_idx']) for r in exs)))
    groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in done)]

    fig = plt.figure(figsize=(6.5, 9.4), dpi=300)
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6, height_ratios=[1.9, 0.9, 0.9, 0.9])
    gs_ex = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.35)
    for j, r in enumerate(exs):
        _plot_example(fig, gs_ex[0, j], r)

    # R-hat. NaN means every draw identical across chains -- R-hat undefined, so
    # those cells are counted under each group instead of plotted.
    ax_c = fig.add_subplot(gs[1])
    _strip(ax_c, [
        ('NNLS starts, long runs', _by_group(done, lambda r: r['rhat_default_max']),
         {'filled': False}),
        ('NNLS starts, auto-stop', _by_group(done, lambda r: r['auto_rhat_max']),
         {'marker': '^'}),
        ('all starts, long runs', _by_group(done, lambda r: r['rhat_max']), {}),
    ], 'max split $\\hat{R}$', hline=RHAT_THR, log=True)
    ax_c.set_ylim(0.7, 2e3)
    ax_c.set_yticks([1, 10, 100, 1000])
    ax_c.set_yticklabels(['1', '10', '100', '$\\geq$1000'])
    ax_c.minorticks_off()
    ax_c.set_xticklabels([GROUP_NAMES[g] for g in groups], rotation=0, ha='center', fontsize=6)
    for xi, g in enumerate(groups):
        n_nan = sum(np.isnan(r['rhat_default_max']) for r in done if r['group'] == g)
        if n_nan:
            ax_c.text(xi, 0.75, '{} identical'.format(n_nan), ha='center', va='bottom',
                      fontsize=5, color='0.4')
    ax_c.legend(frameon=False, fontsize=6, handletextpad=0.2, loc='upper left',
                bbox_to_anchor=(1.0, 1.0))

    # ESS spans two of three columns so its two series per condition have room.
    gs_mix = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.5)
    gs_agr = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3], wspace=0.5)

    # Bulk ESS of spike count and log-likelihood, default-start chains.
    ax_e = fig.add_subplot(gs_mix[0, 0:2])
    _strip(ax_e, [
        ('spike count', _by_group(done, lambda r: r['ess_bulk_default']['ns']), {'filled': False}),
        ('log-lik.', _by_group(done, lambda r: r['ess_bulk_default']['loglik']), {}),
    ], 'bulk ESS', hline=ESS_THR, log=True)
    ax_e.minorticks_off()

    # Spike-count ACF, default-start chains; median over cells with a defined ACF.
    ax_f = fig.add_subplot(gs_agr[0, 2])
    for g in groups:
        acfs = np.stack([r['acf_ns_default'] for r in done if r['group'] == g])
        with np.errstate(all='ignore'):
            med = np.nanmedian(acfs, axis=0)
        ax_f.plot(np.arange(len(med)), med, color=GROUP_COLORS[g], lw=0.9,
                  label=GROUP_NAMES[g])
    ax_f.axhline(0, color='0.5', lw=0.6, ls='--')
    ax_f.set_xlabel('lag (sweeps)')
    ax_f.set_ylabel('spike-count ACF')
    ax_f.set_ylim(-0.2, 1.02)
    ax_f.legend(frameon=False, fontsize=5.5, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                handlelength=1.2)

    # Agreement between independent auto-stop runs (NNLS start, different seeds),
    # as a user would get them from OMSI.deconv.
    ax_g = fig.add_subplot(gs_agr[0, 0])
    _strip(ax_g, [('', _by_group(done, lambda r: r['event_agree_auto']), {})],
           'event agreement\nbetween runs (F1)')
    ax_h = fig.add_subplot(gs_agr[0, 1])
    _strip(ax_h, [('', _by_group(done, lambda r: r['auto_auto_probcorr']), {})],
           'spike prob. correlation\nbetween runs')
    for ax in (ax_g, ax_h):
        ax.set_ylim(0, 1.03)

    # How far apart default-start chains settle in spike count, relative to the
    # count -- the size of the disagreement that low R-hat and ESS reflect.
    ax_s = fig.add_subplot(gs_mix[0, 2])
    _strip(ax_s, [('', _by_group(done, lambda r: 100.0 * np.std(
        np.asarray(r['ref_chain_ns'])[r['default_idx']], ddof=1) / max(np.mean(
            np.asarray(r['ref_chain_ns'])[r['default_idx']]), 1.0)), {})],
        'spike-count spread\nbetween chains (% of count)')
    ax_s.set_ylim(bottom=0)


    # F-beta vs truth at the NNLS start and after sampling. Off for now; to restore,
    # uncomment and give it a free grid slot (the agreement row is full).
    # ax_i = fig.add_subplot(gs_agr[0, 2])
    # _strip(ax_i, [
    #     ('NNLS start', _by_group(done, lambda r: np.median(
    #         r['sweep_score']['fbeta'][r['default_idx'], 0])), {'filled': False}),
    #     ('OMSI', _by_group(done, lambda r: np.median(
    #         [a['score']['fbeta'] for a in r['auto']])), {}),
    # ], '$F_\\beta$ vs truth')
    # ax_i.set_ylim(0, 1.03)

    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, 'convergence_benchmark.{}'.format(ext))
        fig.savefig(out, bbox_inches='tight')
        print('Saved {}.'.format(out))
    plt.close(fig)


def _mm(v, fmt='.3f'):
    """Median +/- MAD string over finite values."""

    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 'n/a'
    med = np.median(v)
    return '{} +/- {}'.format(format(med, fmt), format(np.median(np.abs(v - med)), fmt))


def print_stats(data_dir=_DEFAULT_DATA_DIR):
    """ Print per-group convergence and agreement statistics.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the cells/ results folder.
    """

    results = _load_results(data_dir)
    if not results:
        raise FileNotFoundError('No results in {}. Run --mode test first.'.format(data_dir))
    old = [r for r in results if not r['skipped'] and r.get('version') != RESULT_VERSION]
    if old:
        raise RuntimeError('{} result files are from an older version -- rerun --mode test.'.format(
            len(old)))

    print('\n' + '=' * 78)
    print('CONVERGENCE BENCHMARK')
    print('=' * 78)
    r0 = next(r for r in results if not r['skipped'])
    print('{} reference chains x {} sweeps (first half warmup); {} auto-stop runs per cell.'.format(
        r0['n_chains'], r0['n_sweeps'], len(r0['auto'])))
    print('Values are median +/- MAD across cells.')

    for g in GROUP_ORDER:
        rs = [r for r in results if r['group'] == g]
        if not rs:
            continue
        done = [r for r in rs if not r['skipped']]
        print('\n--- {} ({} cells, {} skipped by SNR gate, {} with fast-indicator init override) ---'.format(
            g, len(rs), len(rs) - len(done), sum(r.get('p_switched', False) for r in done)))
        if not done:
            continue

        rmax = np.array([r['rhat_max'] for r in done])
        rdef = np.array([r['rhat_default_max'] for r in done])
        n_def = len(done[0]['default_idx'])
        print('  Reference chains')
        print('    Max split R-hat, {} default starts {:>16}   {}/{} cells < {}  '
              '({} undefined: identical frozen chains)'.format(
                  n_def, _mm(rdef), int(np.sum(rdef < RHAT_THR)), len(done), RHAT_THR,
                  int(np.sum(np.isnan(rdef)))))
        print('    Default starts: bulk ESS ns {}   loglik {}   constant spike count {}'.format(
            _mm([r['ess_bulk_default']['ns'] for r in done], '.0f'),
            _mm([r['ess_bulk_default']['loglik'] for r in done], '.0f'),
            _mm([r['frozen_default'] for r in done], '.2f')))
        print('    Max split R-hat (scalars)     {:>20}   {}/{} cells < {}, {}/{} < 1.05'.format(
            _mm(rmax), int(np.sum(rmax < RHAT_THR)), len(done), RHAT_THR,
            int(np.sum(rmax < 1.05)), len(done)))
        for k in SCALAR_KEYS:
            print('    {:<10} R-hat {:>16}   bulk ESS {:>16}   tail ESS {:>16}   IAT {:>14}'.format(
                k, _mm([r['rhat'][k] for r in done]),
                _mm([r['ess_bulk'][k] for r in done], '.0f'),
                _mm([r['ess_tail'][k] for r in done], '.0f'),
                _mm([r['iat'][k] for r in done], '.1f')))
        ess_ns = np.array([r['ess_bulk']['ns'] for r in done])
        print('    Cells with bulk ESS(ns) >= {}  {}/{}'.format(
            ESS_THR, int(np.sum(ess_ns >= ESS_THR)), len(done)))
        print('    Active bins, R-hat < {}, counts     {:>20}   (< 1.05: {})'.format(
            RHAT_THR, _mm([r['bin_frac_101'] for r in done]),
            _mm([r['bin_frac_105'] for r in done])))
        print('    Active bins, R-hat < {}, occupancy  {:>20}   (< 1.05: {})'.format(
            RHAT_THR, _mm([r['occ_frac_101'] for r in done]),
            _mm([r['occ_frac_105'] for r in done])))
        print('    Bins where chain spike prob differs > 0.5   {}   (total over cells: {})'.format(
            _mm([r['bin_disagree'] for r in done], '.0f'),
            int(sum(r['bin_disagree'] for r in done))))
        nc = np.array([r['n_conv'] for r in done])
        print('    Sweeps to R-hat < {}          {:>20}   ({} never within {} sweeps)'.format(
            RHAT_THR, _mm(nc, '.0f'), int(np.sum(~np.isfinite(nc))), r0['n_sweeps']))
        print('    Event agreement (F1): default starts {}   all starts {}   auto-stop runs {}'.format(
            _mm([r['event_agree_default'] for r in done]),
            _mm([r['event_agree_all'] for r in done]),
            _mm([r['event_agree_auto'] for r in done])))

        grid = done[0]['sweep_grid']
        print('  Accuracy vs sweeps, default-start chains (column "start" = NNLS start)')
        print('    {:<14}'.format('') + ''.join(
            '{:>7}'.format('start' if n == 0 else n) for n in grid))
        for m, name in [('fbeta', 'F_beta window'), ('fbeta_strict', 'F_beta strict'),
                        ('cosmic', 'CosMIC')]:
            curves = np.stack([np.median(r['sweep_score'][m][r['default_idx']], axis=0)
                               for r in done])
            print('    {:<14}'.format(name) + ''.join(
                '{:>7.3f}'.format(v) for v in np.median(curves, axis=0)))

        with_auto = [r for r in done if r['auto']]
        if not with_auto:
            continue
        stops = np.array([np.median([a['stop_idx'] for a in r['auto']]) for r in with_auto])
        burns = np.array([np.median([a['B_final'] for a in r['auto']]) for r in with_auto])
        n_early = sum(1 for r in with_auto for a in r['auto']
                      if not np.isfinite(r['n_conv']) or a['stop_idx'] < r['n_conv'])
        n_runs = sum(len(r['auto']) for r in with_auto)
        n_max = sum(a['hit_max'] for r in with_auto for a in r['auto'])
        t_auto = [np.median([a['time'] for a in r['auto']]) for r in with_auto]
        t_ref = [r['ref_time'] for r in with_auto]
        print('  Auto-stop runs')
        print('    Time per run: auto-stop {} s vs {}-sweep chain {} s'.format(
            _mm(t_auto, '.2f'), r0['n_sweeps'], _mm(t_ref, '.2f')))
        print('    Burn-in end sweep             {:>20}'.format(_mm(burns, '.0f')))
        print('    Stop sweep                    {:>20}   ({}/{} runs hit max_sweeps)'.format(
            _mm(stops, '.0f'), n_max, n_runs))
        print('    Runs stopping before ref. R-hat < {}: {}/{}'.format(RHAT_THR, n_early, n_runs))
        ar = np.array([r['auto_rhat_max'] for r in with_auto])
        print('    Max split R-hat across auto runs {:>16}   {}/{} cells < {}  ({} undefined)'.format(
            _mm(ar), int(np.sum(ar < RHAT_THR)), len(with_auto), RHAT_THR,
            int(np.sum(np.isnan(ar)))))
        print('    Agreement, between auto runs vs with truth: events {} vs {}   '
              'prob trace corr {} vs {}'.format(
                  _mm([r['event_agree_auto'] for r in with_auto]),
                  _mm([np.nanmedian([a['event_vs_truth'] for a in r['auto']]) for r in with_auto]),
                  _mm([r['auto_auto_probcorr'] for r in with_auto]),
                  _mm([np.nanmedian([a['probcorr_vs_truth'] for a in r['auto']])
                       for r in with_auto])))
        print('    R-hat across auto runs (ns / Am / loglik)   {} / {} / {}'.format(
            *[_mm([r['auto_rhat'][k] for r in with_auto]) for k in ('ns', 'Am', 'loglik')]))
        print('    Mean spike count, auto vs best ref chain  {} vs {}'.format(
            _mm([np.mean([a['ns_mean'] for a in r['auto']]) for r in with_auto], '.1f'),
            _mm([r['ref_chain_ns'][r['ref_best']] for r in with_auto], '.1f')))
        d_lp = [np.mean([a['logpost_mean'] for a in r['auto']])
                - r['ref_chain_logpost'][r['ref_best']] for r in with_auto]
        print('    Log-posterior, auto minus best ref chain  {}'.format(_mm(d_lp, '.1f')))
        labs = [r['ref_best_label'] for r in with_auto]
        print('    Best ref chain start: {}'.format(
            ', '.join('{} {}'.format(l, labs.count(l)) for l in sorted(set(labs)))))

        print('  Accuracy vs ground truth, end of run (median over chains/runs per cell)')
        print('    {:<22} {:>18} {:>18} {:>18}'.format(
            '', 'F_beta window', 'F_beta strict', 'CosMIC'))
        rows = [
            ('default-start chains', lambda r, m: r['ref_default_score'][m]),
            ('best ref chain', lambda r, m: r['ref_best_score'][m]),
            ('mean over ref chains', lambda r, m: np.mean([s[m] for s in r['ref_chain_score']])),
            ('auto-stop', lambda r, m: np.median([a['score'][m] for a in r['auto']])),
        ]
        for name, fn in rows:
            print('    {:<22} {:>18} {:>18} {:>18}'.format(
                name, *[_mm([fn(r, m) for r in with_auto])
                        for m in ('fbeta', 'fbeta_strict', 'cosmic')]))
        d = [np.median([a['score']['fbeta'] for a in r['auto']]) - r['ref_default_score']['fbeta']
             for r in with_auto]
        print('    F_beta window, auto minus default-start chains  {}'.format(_mm(d)))

        print('  Output agreement (CosMIC between spike trains / prob trace corr)')
        print('    auto vs best ref     {:>18} / {:>18}'.format(
            _mm([np.nanmean([a['vs_best_cosmic'] for a in r['auto']]) for r in with_auto]),
            _mm([np.nanmean([a['vs_best_probcorr'] for a in r['auto']]) for r in with_auto])))
        print('    auto vs ref chains   {:>18} / {:>18}'.format(
            _mm([np.nanmean([a['vs_chains_cosmic'] for a in r['auto']]) for r in with_auto]),
            _mm([np.nanmean([a['vs_chains_probcorr'] for a in r['auto']]) for r in with_auto])))
        print('    ref vs ref chain     {:>18} / {:>18}'.format(
            _mm([r['ref_ref_cosmic'] for r in with_auto]),
            _mm([r['ref_ref_probcorr'] for r in with_auto])))
        print('    auto vs auto         {:>18} / {:>18}'.format(
            _mm([r['auto_auto_cosmic'] for r in with_auto]),
            _mm([r['auto_auto_probcorr'] for r in with_auto])))

    pert_path = os.path.join(data_dir, 'perturb.npz')
    if os.path.exists(pert_path):
        pert = np.load(pert_path)
        print('\n|Timing error| per cell from shifted starts (median over {} cells, ms):'.format(
            pert['omsi'].shape[1]))
        frame_ms = 1000.0 / float(pert['fs'][0])
        for m in ('omsi', 'caiman'):
            if m not in pert.files:
                continue
            print('  {}'.format(PERTURB_METHOD_NAMES[m]))
            for k, sh in enumerate(pert['shifts']):
                med = np.nanmedian(np.abs(pert[m][k]), axis=0)
                print('    start {:+5.0f} ms: at start {:+6.1f}, after 50 sweeps {:+6.1f}, '
                      'final {:+6.1f}  (IQR at final {:.1f} ms)'.format(
                          sh * frame_ms, med[0], med[min(len(med) - 1, np.searchsorted(
                              pert['sweep_idx'], 50) + 1)], med[-1],
                          np.subtract(*np.nanpercentile(np.abs(pert[m][k][:, -1]), [75, 25]))))



def main():
    """Parse command-line arguments and dispatch."""

    parser = argparse.ArgumentParser(
        description='OMSI auto-stop rule vs conventional MCMC convergence diagnostics')
    parser.add_argument('--mode', required=True, choices=['test', 'perturb', 'plot', 'print'],
                        help='"test" runs chains and saves per-cell results; '
                             '"perturb" runs the shifted-start timing test; '
                             '"plot" generates the figure; '
                             '"print" prints summary statistics')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing result files')
    parser.add_argument('--n-cells', type=int, default=16, help='Synthetic cells per condition')
    parser.add_argument('--duration', type=float, default=120., help='Synthetic duration (s)')
    parser.add_argument('--n-chains', type=int, default=10, help='Reference chains per cell')
    parser.add_argument('--n-sweeps', type=int, default=10000, help='Reference chain length')
    parser.add_argument('--n-auto', type=int, default=10, help='Auto-stop runs per cell')
    parser.add_argument('--warmup-frac', type=float, default=0.5,
                        help='Fraction of reference chain discarded as warmup')
    parser.add_argument('--n-allen', type=int, default=16,
                        help='Allen cells in total (split slow/fast); 0 to skip')
    parser.add_argument('--allen-h5', default=_DEFAULT_ALLEN_H5,
                        help='Aggregated Allen HDF5 file from figure3.py')
    parser.add_argument('--workers', type=int, default=None, help='Worker processes')
    parser.add_argument('--seed', type=int, default=0, help='Base seed')
    parser.add_argument('--overwrite', action='store_true', help='Rerun cells with saved results')
    parser.add_argument('--pilot', action='store_true',
                        help='Small run into data-dir/pilot: 3 cells/condition, '
                             '1500 sweeps, 2 Allen cells')
    parser.add_argument('--perturb-cells', type=int, default=PERTURB_CELLS,
                        help='Non-bursty cells in the shifted-start test (perturb)')
    parser.add_argument('--perturb-sweeps', type=int, default=PERTURB_SWEEPS,
                        help='Sweeps per chain in the shifted-start test (perturb)')
    parser.add_argument('--no-matlab', action='store_true',
                        help='Skip CaImAn in the shifted-start test (perturb)')
    parser.add_argument('--examples', default=None,
                        help='Example cells, comma-separated group:idx')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, 'pilot') if args.pilot else args.data_dir

    if args.mode == 'test':
        kw = dict(n_cells=args.n_cells, duration=args.duration, n_chains=args.n_chains,
                  n_sweeps=args.n_sweeps, n_auto=args.n_auto, n_allen=args.n_allen)
        if args.pilot:
            kw.update(n_cells=3, n_sweeps=1500, n_auto=2, n_allen=2)
        run_test(data_dir=data_dir, warmup_frac=args.warmup_frac,
                 allen_h5=args.allen_h5, workers=args.workers, seed=args.seed,
                 overwrite=args.overwrite, **kw)
    elif args.mode == 'perturb':
        run_perturb(data_dir=data_dir, n_cells=args.perturb_cells, duration=args.duration,
                    n_sweeps=args.perturb_sweeps, workers=args.workers, seed=args.seed,
                    run_matlab=not args.no_matlab)
    elif args.mode == 'print':
        print_stats(data_dir=data_dir)
    else:
        plot_figure(data_dir=data_dir, examples=args.examples)


if __name__ == '__main__':
    with no_power_throttling(verbose=True):
        main()
