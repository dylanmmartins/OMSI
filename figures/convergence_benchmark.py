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
_load_results
    Load per-cell result files for current groups.
_by_group
    Map group to per-cell values over non-skipped cells.
_strip
    Strip plot with median bar per group, optionally several series.
_rhat_nan1
    Treat NaN R-hat (every draw identical) as 1.
_pick_example
    Cell from first synthetic group with median true spike count.
plot_figure
    Load per-cell results and generate the convergence figure.
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
from OMSI.deconv import spikes_from_samples, default_lag_s
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

# Synthetic conditions where NNLS init alone falls short and MCMC improves on
# it. Picked from a screen (16 cells/condition, 120 s): high-SNR data is already
# saturated at init, and at 5 Hz or 7.5 Hz + SNR 6 MCMC did worse than NNLS.
SYNTH_CONDITIONS = {
    'snr6':          {'fs': 30.0, 'tau': 1.2, 'snr': 6.0},
    'snr4':          {'fs': 30.0, 'tau': 1.2, 'snr': 4.0},
    'fast_tau_snr6': {'fs': 30.0, 'tau': 0.4, 'snr': 6.0},
    'low_fs':        {'fs': 7.5,  'tau': 1.2, 'snr': None},
}

GROUP_ORDER  = ['snr6', 'snr4', 'fast_tau_snr6', 'low_fs', 'allen_slow', 'allen_fast']
GROUP_NAMES  = {
    'snr6':          'SNR 6',
    'snr4':          'SNR 4',
    'fast_tau_snr6': 'fast tau, SNR 6',
    'low_fs':        '7.5 Hz',
    'allen_slow':    'Allen slow',
    'allen_fast':    'Allen fast',
}
# Blue, orange, green, and purple are reserved for deconvolution methods in the
# other figures -- conditions use reds, yellows, pinks, and browns only.
GROUP_COLORS = {
    'snr6':          '#C0182B',
    'snr4':          '#E0B000',
    'fast_tau_snr6': '#EE6FA8',
    'low_fs':        '#8C5A3C',
    'allen_slow':    '#5A0F2E',
    'allen_fast':    '#C8A77E',
}

INIT_LABELS = ['nnls', 'empty', 'foopsi', 'dense']
# Panel a uses black and grays only (told apart by line style) so it shares no
# color with the condition palette.
INIT_STYLES = {
    'nnls':     {'color': 'k',       'ls': '-'},
    'nnls_jit': {'color': 'k',       'ls': '-'},
    'empty':    {'color': '#555555', 'ls': '--'},
    'foopsi':   {'color': '#555555', 'ls': ':'},
    'dense':    {'color': '#555555', 'ls': '-.'},
}
AUTO_COLOR = '#BBBBBB'

SCALAR_KEYS = ['ns', 'Am', 'Cb', 'sg', 'tau_decay', 'loglik', 'logpost']

# Chains started the way OMSI.deconv starts them (NNLS, re-jittered).
DEFAULT_STARTS = ('nnls', 'nnls_jit')
START_NAMES  = {'nnls': 'NNLS (default)', 'nnls_jit': 'NNLS (default)',
                'empty': 'no spikes', 'foopsi': 'FOOPSI', 'dense': 'extra spikes'}

# Spikes closer than this are merged into one event for event-level agreement.
EVENT_GAP_S = 0.100

# Checkpoints for accuracy vs sweeps; clipped to chain length.
SWEEP_CHECKPOINTS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Bump when per-cell result contents change so old files get rerun.
RESULT_VERSION = 3


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
    lag_s = default_lag_s(params, fs)
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
        init_sp = np.clip(inits[k]['spiketimes_'] - lag_s * fs, 0, T - 1) / fs
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


def _rhat_nan1(v):
    """Treat NaN R-hat (every draw identical across chains) as 1."""

    return 1.0 if np.isnan(v) else v


def _pick_example(done):
    """Cell from first synthetic group with median true spike count."""

    base = [r for r in done if r['group'] == GROUP_ORDER[0]] or done
    med = np.median([r['n_true'] for r in base])
    return min(base, key=lambda r: abs(r['n_true'] - med))


def plot_figure(data_dir=_DEFAULT_DATA_DIR, example=None):
    """ Load per-cell results and generate the convergence figure.

    Layout (2 columns x 4 rows): a (example traces) + shared legends; b
    (accuracy vs sweeps) and e (auto-stop vs long run); c (R-hat) full width;
    d (event agreement) full width.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the cells/ results folder.
    example : str, optional
        Example cell as 'group:idx'. Defaults to first synthetic group's cell
        with median true spike count.
    """

    from matplotlib.lines import Line2D

    results = _load_results(data_dir)
    done = [r for r in results if not r['skipped']]
    if not done:
        raise FileNotFoundError('No results in {}. Run --mode test first.'.format(data_dir))
    old = [r for r in done if r.get('version') != RESULT_VERSION]
    if old:
        raise RuntimeError('{} result files are from an older version -- rerun --mode test.'.format(
            len(old)))

    if example:
        g, i = example.split(':')
        ex = next(r for r in done if r['group'] == g and r['cell_idx'] == int(i))
    else:
        ex = _pick_example(done)
    print('Example cell: {}:{} ({} true spikes).'.format(ex['group'], ex['cell_idx'], ex['n_true']))
    groups = [g for g in GROUP_ORDER if any(r['group'] == g for r in done)]

    fig = plt.figure(figsize=(5., 8.25), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.5,
                           height_ratios=[1, 1, 0.8, 0.8])
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], wspace=0.1)

    # a: example spike-count traces, log sweep axis so early lock-in is visible.
    ax_a = fig.add_subplot(gs_top[0, 0:2])
    tidx = ex['trace_idx']
    for a in ex['auto']:
        ax_a.plot(a['trace_idx'] + 1, a['trace_ns'], color=AUTO_COLOR, lw=0.6, zorder=1)
        ax_a.plot(a['stop_idx'], a['trace_ns'][-1], marker='|', color=AUTO_COLOR, ms=5, zorder=1)
    for k, lab in enumerate(ex['init_labels']):
        st = INIT_STYLES[lab]
        ax_a.plot(tidx + 1, ex['ref_traces']['ns'][k], color=st['color'], ls=st['ls'],
                  lw=0.8, zorder=2)
    ax_a.axhline(ex['n_true'], color='k', lw=0.5, ls=(0, (1, 3)), zorder=3)
    ax_a.text(1.2, ex['n_true'], 'true: {}'.format(ex['n_true']), va='bottom', fontsize=5.5)
    # ax_a.axvline(ex['n_sweeps'] // 2, color='k', lw=0.4, ls='--', alpha=0.5)
    ax_a.set_xscale('log')
    ax_a.set_xlabel('sweep')
    ax_a.set_ylabel('spike count')
    # ax_a.set_title('{} cell {}: spike count by starting point'.format(
    #     GROUP_NAMES[ex['group']], ex['cell_idx']), fontsize=7)

    # Legend column: starting points (a) and conditions (b-e).
    ax_leg = fig.add_subplot(gs_top[0, 2])
    ax_leg.axis('off')
    start_handles = [Line2D([], [], color=INIT_STYLES[k]['color'], ls=INIT_STYLES[k]['ls'],
                            lw=1.0, label=START_NAMES[k])
                     for k in ('nnls', 'empty', 'foopsi', 'dense')]
    start_handles.append(Line2D([], [], color=AUTO_COLOR, lw=1.0, label='auto-stop runs'))
    # Side by side -- stacked they overlap in this row height.
    leg1 = ax_leg.legend(handles=start_handles, title='Starting point (a)', loc='upper left',
                         bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6,
                         title_fontsize=6.5, alignment='left', handlelength=2.2)
    ax_leg.add_artist(leg1)
    cond_handles = [Line2D([], [], color=GROUP_COLORS[g], marker='o', ms=4, lw=1.2,
                           label=GROUP_NAMES[g]) for g in groups]
    leg2 = ax_leg.legend(handles=cond_handles, title='Condition (b-e)', loc='upper left',
                  bbox_to_anchor=(0.80, 1.0), frameon=False, fontsize=6,
                  title_fontsize=6.5, alignment='left', handlelength=1.5)
    ax_leg.add_artist(leg2)

    # Panel b auto-stop marker, own legend straddling both columns below them.
    # Gray since the real markers take the condition color.
    tri = Line2D([], [], ls='none', marker='v', ms=5, color='0.6',
                 markeredgecolor='k', markeredgewidth=0.5, label='auto-stop point (b)')
    ax_leg.legend(handles=[tri], loc='upper center', bbox_to_anchor=(0.80, 0.22),
                  frameon=False, fontsize=6, handlelength=1.5)

    # b: accuracy vs sweeps, NNLS-start chains, median over cells.
    ax_b = fig.add_subplot(gs[1, 0])
    for g in groups:
        rs = [r for r in done if r['group'] == g]
        grid = rs[0]['sweep_grid']
        curves = np.stack([np.median(r['sweep_score']['fbeta'][r['default_idx']], axis=0)
                           for r in rs])
        x = np.where(grid == 0, 1, grid)
        ax_b.plot(x, np.median(curves, axis=0), color=GROUP_COLORS[g], lw=0.9, marker='.', ms=3)
        stops = [np.median([a['stop_idx'] for a in r['auto']]) for r in rs if r['auto']]
        fbs = [np.median([a['score']['fbeta'] for a in r['auto']]) for r in rs if r['auto']]
        if stops:
            ax_b.scatter(np.median(stops), np.median(fbs), s=22, marker='v',
                         color=GROUP_COLORS[g], edgecolors='k', linewidths=0.5, zorder=3)
    ax_b.set_xscale('log')
    ax_b.set_xticks([1, 10, 100, 1000, 10000])
    ax_b.set_xticklabels(['start', '10', '100', '1k', '10k'])
    ax_b.set_ylim(0.5, 1.02)
    ax_b.set_xlabel('sweeps')
    ax_b.set_ylabel(r'$F_\beta$')
    # ax_b.set_title(r'accuracy vs sweeps ($\blacktriangledown$: auto-stop)', fontsize=7)

    # e: auto-stop vs long NNLS-start chains.
    ax_e = fig.add_subplot(gs[1, 1])
    for r in done:
        if not r['auto'] or r['ref_default_score'] is None:
            continue
        ax_e.scatter(r['ref_default_score']['fbeta'],
                     np.median([a['score']['fbeta'] for a in r['auto']]),
                     s=8, color=GROUP_COLORS[r['group']], linewidths=0, zorder=3)
    ax_e.plot([0, 1], [0, 1], color='0.5', lw=0.6, ls='--')
    ax_e.set_xlim(0, 1.02)
    ax_e.set_ylim(0, 1.02)
    ax_e.set_xlabel(r'$F_\beta$, {}-sweep NNLS-start chains'.format(ex['n_sweeps']))
    ax_e.set_ylabel(r'$F_\beta$, auto-stop')
    # ax_e.set_title('auto-stop vs long run', fontsize=7)
    ax_e.axis('equal')

    # c: R-hat across runs, full width.
    ax_c = fig.add_subplot(gs[2, :])
    _strip(ax_c, [
        ('NNLS, long runs', _by_group(done, lambda r: _rhat_nan1(r['rhat_default_max'])),
         {'filled': False}),
        ('NNLS, auto-stop', _by_group(done, lambda r: _rhat_nan1(r['auto_rhat_max'])),
         {'marker': '^'}),
        ('all starts, long runs', _by_group(done, lambda r: _rhat_nan1(r['rhat_max'])), {}),
    ], 'max split $\hat{R}$', hline=RHAT_THR, log=True)
    ax_c.set_ylim(0.7, 2e3)
    ax_c.set_yticks([1, 10, 100, 1000])
    ax_c.set_yticklabels(['1', '10', '100', '1000'])
    ax_c.minorticks_off()
    ax_c.set_xticklabels([GROUP_NAMES[g] for g in groups], rotation=0, ha='center', fontsize=6)
    ax_c.legend(frameon=False, fontsize=6, handletextpad=0.2, loc='upper left',
                bbox_to_anchor=(1.0, 1.0))
    # ax_c.set_title('chain agreement (1 = agree, dashed = {})'.format(RHAT_THR), fontsize=7)

    # d: event-level agreement between runs, full width.
    ax_d = fig.add_subplot(gs[3, :])
    _strip(ax_d, [
        ('NNLS, long runs', _by_group(done, lambda r: r['event_agree_default']), {'filled': False}),
        ('NNLS, auto-stop', _by_group(done, lambda r: r['event_agree_auto']), {'marker': '^'}),
        ('all starts, long runs', _by_group(done, lambda r: r['event_agree_all']), {}),
    ], 'event agreement (F1)')
    ax_d.set_ylim(0, 1.02)
    ax_d.set_xticklabels([GROUP_NAMES[g] for g in groups], rotation=0, ha='center', fontsize=6)
    ax_d.legend(frameon=False, fontsize=6, handletextpad=0.2, loc='upper left',
                bbox_to_anchor=(1.0, 1.0))
    # ax_d.set_title('same events found? (1 = identical events)', fontsize=7)

    # for label, axx, xoff in [('a', ax_a, -0.08), ('b', ax_b, -0.18), ('e', ax_e, -0.18),
    #                          ('c', ax_c, -0.06), ('d', ax_d, -0.06)]:
    #     axx.text(xoff, 1.1, label, transform=axx.transAxes, fontsize=9,
    #              fontweight='bold', ha='right')

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

        rmax = np.array([_rhat_nan1(r['rhat_max']) for r in done])
        rdef = np.array([_rhat_nan1(r['rhat_default_max']) for r in done])
        n_def = len(done[0]['default_idx'])
        print('  Reference chains')
        print('    Max split R-hat, {} default starts {:>16}   {}/{} cells < {}  '
              '(identical frozen chains count as 1)'.format(
                  n_def, _mm(rdef), int(np.sum(rdef < RHAT_THR)), len(done), RHAT_THR))
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
        ar = np.array([_rhat_nan1(r['auto_rhat_max']) for r in with_auto])
        print('    Max split R-hat across auto runs {:>16}   {}/{} cells < {}'.format(
            _mm(ar), int(np.sum(ar < RHAT_THR)), len(with_auto), RHAT_THR))
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


def main():
    """Parse command-line arguments and dispatch."""

    parser = argparse.ArgumentParser(
        description='OMSI auto-stop rule vs conventional MCMC convergence diagnostics')
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'print'],
                        help='"test" runs chains and saves per-cell results; '
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
    parser.add_argument('--example', default=None,
                        help='Example cell for trace panels, as group:idx')
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
    elif args.mode == 'print':
        print_stats(data_dir=data_dir)
    else:
        plot_figure(data_dir=data_dir, example=args.example)


if __name__ == '__main__':
    with no_power_throttling(verbose=True):
        main()
