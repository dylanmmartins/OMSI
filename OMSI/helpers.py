# -*- coding: utf-8 -*-
"""
OMSI/helpers.py

Utility functions for spike detection, calcium simulation, and accuracy scoring.

Functions
---------
_otsu_threshold
    Compute Otsu threshold for optimal binary split of a value array.
detect_spikes_from_probs
    Find spike peaks from a per-cell probability trace.
spikes_to_calcium
    Simulate noisy calcium traces from a binary spike train.
compute_accuracy_strict
    Hungarian-algorithm-based spike matching precision/recall/F1.
compute_cosmic
    CosMIC soft intersection-over-union score.
make_event_ground_truth
    Filter spike times down to clean, isolated events.
compute_accuracy_window
    Window-based spike matching precision/recall/F1.
compute_kurtosis
    Fisher excess kurtosis of one or more fluorescence traces.


DMM, Feb 2026
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment


def _otsu_threshold(values):
    """ Compute Otsu threshold for optimal binary split of a value array.

    Scans every possible split point and picks the one maximizing
    weighted between-class variance.

    Parameters
    ----------
    values : array-like
        1-D array of values to threshold.

    Returns
    -------
    float
        Optimal threshold midpoint.
    """

    x = np.sort(values)
    n = len(x)
    cum = np.cumsum(x)
    total = cum[-1]
    best_var, best_i = -1.0, 0
    for i in range(n - 1):
        w0 = (i + 1) / n
        w1 = 1.0 - w0
        mu0 = cum[i] / (i + 1)
        mu1 = (total - cum[i]) / (n - i - 1)
        v = w0 * w1 * (mu0 - mu1) ** 2
        if v > best_var:
            best_var, best_i = v, i

    return (x[best_i] + x[best_i + 1]) / 2.0


def detect_spikes_from_probs(probs, fs, sigma=1.5, min_thresh=0.001):
    """ Find spike peaks from a per-cell probability trace.

    Optionally smooths the trace with a Gaussian, then finds peaks above
    an Otsu-derived threshold computed from the global peak distribution
    across all cells.

    Parameters
    ----------
    probs : np.ndarray, shape (n_cells, n_frames) or (n_frames,)
        Per-frame spike probability traces.
    fs : float
        Frame rate in Hz. Used to set minimum peak distance (100 ms).
    sigma : float
        Gaussian smoothing sigma in frames. Set to 0 to skip smoothing.
    min_thresh : float
        Minimum allowed threshold, regardless of Otsu result.

    Returns
    -------
    spikes : list of np.ndarray
        Per-cell spike times in seconds.
    thresh : float
        Threshold applied to peak heights.
    """

    probs = np.atleast_2d(probs)
    n_cells, n_frames = probs.shape
    min_dist = max(1, int(0.1 * fs))

    sm = np.array([gaussian_filter1d(probs[i], sigma=sigma) if sigma > 0
                   else probs[i].copy()
                   for i in range(n_cells)])

    # Pool peaks across all cells so threshold is based on global peak
    # distribution, not just whatever one cell happens to have.
    all_peaks = []
    for i in range(n_cells):
        _, props = find_peaks(sm[i], height=0)
        if 'peak_heights' in props:
            all_peaks.extend(props['peak_heights'].tolist())

    # Split peaks into noise and signal using Otsu, then push threshold
    # below the noise cluster to keep sensitivity high.
    if len(all_peaks) >= 2 and max(all_peaks) > 1e-6:
        peaks_arr = np.array(all_peaks)
        otsu = _otsu_threshold(peaks_arr)
        noise_peaks = peaks_arr[peaks_arr < otsu]
        if len(noise_peaks) >= 2:
            thresh = noise_peaks.mean() - 5.0 * noise_peaks.std()
        else:
            thresh = otsu
        thresh = max(min_thresh, thresh)
    else:
        thresh = min_thresh

    spikes = []
    for i in range(n_cells):
        spf, _ = find_peaks(sm[i], height=thresh, distance=min_dist)
        spikes.append(spf / fs)

    return spikes, thresh


def spikes_to_calcium(spikes, fs_in, fs_out, tau, snr):
    """ Simulate noisy calcium traces from a binary spike train.

    Convolves each cell's spike train with a double-exponential kernel
    (fast rise, slow decay), resamples to fs_out, and adds Gaussian noise
    scaled to the requested SNR.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_cells, n_frames_in)
        Binary spike train at fs_in.
    fs_in : float
        Input spike train frame rate in Hz.
    fs_out : float
        Output calcium trace frame rate in Hz.
    tau : float
        Decay time constant in seconds.
    snr : float or array-like of shape (n_cells,)
        Signal-to-noise ratio. Noise std = 1/snr.

    Returns
    -------
    noisy_traces : np.ndarray, shape (n_cells, n_frames_out)
        Simulated calcium traces with noise.
    clean_traces : np.ndarray, shape (n_cells, n_frames_out)
        Noiseless calcium traces.
    """

    n_cells, n_in = spikes.shape

    k_len = int(5 * tau * fs_in)

    # Double-exponential kernel: fast rise (tau_rise) then slow decay (tau).
    # Mimics the shape of a real calcium transient from a spike.
    tau_rise = 0.05
    t_k = np.arange(k_len) / fs_in
    kernel = np.exp(-t_k / tau) - np.exp(-t_k / tau_rise)
    kernel /= np.max(kernel)

    duration = n_in / fs_in
    n_out = int(duration * fs_out)

    clean_traces = np.zeros((n_cells, n_out))

    step = fs_in / fs_out

    for i in range(n_cells):
        tr = np.convolve(spikes[i], kernel, mode='full')[:n_in]

        if step.is_integer():
            s = int(step)
            clean_traces[i] = tr[::s][:n_out]
        else:
            in_times = np.arange(n_in) / fs_in
            out_times = np.arange(n_out) / fs_out
            clean_traces[i] = np.interp(out_times, in_times, tr)

    noisy_traces = np.zeros_like(clean_traces)

    sigma = 1.0 / snr
    if np.ndim(sigma) == 0:
        sigma = np.full(n_cells, sigma)

    for i in range(n_cells):
        noisy_traces[i] = clean_traces[i] + np.random.normal(0, sigma[i], size=n_out)

    return noisy_traces, clean_traces


def compute_accuracy_strict(true_spikes, predicted_spikes, tolerance=0.100):
    """ Hungarian-algorithm-based spike matching precision/recall/F1.

    Finds the optimal one-to-one assignment between true and predicted spikes
    within a tolerance window. Pairs outside tolerance are excluded.

    Parameters
    ----------
    true_spikes : list of array-like
        Per-cell ground-truth spike times in seconds.
    predicted_spikes : list of array-like
        Per-cell predicted spike times in seconds.
    tolerance : float
        Maximum time difference in seconds for a match.

    Returns
    -------
    precisions : np.ndarray
        Per-cell precision.
    recalls : np.ndarray
        Per-cell recall.
    f1s : np.ndarray
        Per-cell F1 score.
    """

    precisions = []
    recalls = []
    f1s = []

    for t_spk, p_spk in zip(true_spikes, predicted_spikes):
        t_spk = np.array(t_spk, dtype=np.float64).flatten()
        p_spk = np.array(p_spk, dtype=np.float64).flatten()

        if len(p_spk) == 0:
            precisions.append(0.0)
            recalls.append(0.0 if len(t_spk) > 0 else 1.0)
            f1s.append(0.0 if len(t_spk) > 0 else 1.0)
            continue
        if len(t_spk) == 0:
            precisions.append(0.0)
            recalls.append(1.0)
            f1s.append(0.0)
            continue

        diffs = np.abs(t_spk[:, None] - p_spk[None, :])

        # Hungarian algorithm for optimal one-to-one matching. Pairs outside
        # tolerance get a large cost so they're never matched.
        cost_matrix = diffs.copy()
        LARGE_VAL = 1e6
        cost_matrix[cost_matrix > tolerance] = LARGE_VAL

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_costs = cost_matrix[row_ind, col_ind]
        valid_mask = matched_costs <= tolerance
        n_tp = np.sum(valid_mask)

        n_fp = len(p_spk) - n_tp
        n_fn = len(t_spk) - n_tp

        prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return np.array(precisions), np.array(recalls), np.array(f1s)


def compute_cosmic(true_spikes, inferred_spikes, fs, tolerance=0.100):
    """ CosMIC soft intersection-over-union score.

    Convolves both spike trains with a triangular kernel then computes soft
    IOU. Gives partial credit for spikes that are close but not exactly
    aligned, unlike the strict binary matching in compute_accuracy_strict.

    Parameters
    ----------
    true_spikes : list of array-like
        Per-cell ground-truth spike times in seconds.
    inferred_spikes : list of array-like
        Per-cell inferred spike times in seconds.
    fs : float
        Frame rate in Hz, used to convert times to bins.
    tolerance : float
        Half-width of the triangular kernel in seconds.

    Returns
    -------
    np.ndarray
        Per-cell CosMIC score in [0, 1].
    """

    from scipy.signal import fftconvolve

    hw_frames = max(tolerance * fs, 1.0)
    r = int(np.ceil(hw_frames))
    t = np.arange(-r, r + 1, dtype=float)
    kernel = np.maximum(0.0, 1.0 - np.abs(t) / hw_frames)

    scores = []
    for t_spk, i_spk in zip(true_spikes, inferred_spikes):
        t_spk = np.asarray(t_spk, dtype=float).ravel()
        i_spk = np.asarray(i_spk, dtype=float).ravel()
        if len(t_spk) == 0 and len(i_spk) == 0:
            scores.append(1.0); continue

        max_t = max(np.max(t_spk) if len(t_spk) > 0 else 0,
                    np.max(i_spk) if len(i_spk) > 0 else 0)
        duration = max_t + 1.0 + tolerance * 2
        n_bins = int(np.ceil(duration * fs))
        u = np.zeros(n_bins); v = np.zeros(n_bins)

        if len(t_spk) > 0:
            np.add.at(u, np.clip(np.round(t_spk * fs).astype(int), 0, n_bins - 1), 1.0)
        if len(i_spk) > 0:
            np.add.at(v, np.clip(np.round(i_spk * fs).astype(int), 0, n_bins - 1), 1.0)

        u_s = fftconvolve(u, kernel, mode='same')
        v_s = fftconvolve(v, kernel, mode='same')

        intersection = np.sum(np.minimum(u_s, v_s))
        total = np.sum(u_s) + np.sum(v_s)
        scores.append(2.0 * intersection / total if total > 1e-9 else 1.0)

    return np.array(scores)


def make_event_ground_truth(spike_times_s, tau_s, event_window=0.250):
    """ Filter spike times down to clean, isolated events.

    Groups spikes into burst events and keeps only those with enough
    quiet time before and after. Quiet windows are longer for slow indicators
    (tau >= 0.8 s) since their transients overlap more.

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds.
    tau_s : float
        Decay time constant in seconds, used to set quiet-window lengths.
    event_window : float
        Maximum inter-spike interval in seconds to group into a burst.

    Returns
    -------
    np.ndarray
        Filtered spike times containing only isolated events.
    """

    t = np.asarray(spike_times_s, dtype=np.float64)
    if len(t) == 0:
        return t.copy()
    quiet_pre, quiet_post = (1.0, 0.5) if tau_s >= 0.8 else (0.3, 0.3)
    isis = np.diff(t)
    boundaries = np.concatenate([[0], np.where(isis > event_window)[0] + 1, [len(t)]])
    event_times = []
    for j in range(len(boundaries) - 1):
        s_idx, e_idx = boundaries[j], boundaries[j + 1] - 1
        gap_before = t[s_idx] - t[s_idx - 1] if s_idx > 0 else np.inf
        gap_after  = t[e_idx + 1] - t[e_idx] if e_idx < len(t) - 1 else np.inf
        if gap_before >= quiet_pre and gap_after >= quiet_post:
            event_times.append(t[s_idx])
    return np.array(event_times)


def compute_accuracy_window(true_spikes, predicted_spikes, tolerance=0.100):
    """ Window-based spike matching precision/recall/F1.

    For each predicted spike, checks whether any true spike falls within
    tolerance (and vice versa). Allows many-to-one matches unlike the strict
    Hungarian version.

    Parameters
    ----------
    true_spikes : list of array-like
        Per-cell ground-truth spike times in seconds.
    predicted_spikes : list of array-like
        Per-cell predicted spike times in seconds.
    tolerance : float
        Maximum time difference in seconds for a match.

    Returns
    -------
    precisions : np.ndarray
        Per-cell precision.
    recalls : np.ndarray
        Per-cell recall.
    f1s : np.ndarray
        Per-cell F1 score.
    """

    precisions, recalls, f1s = [], [], []
    for t_spk, p_spk in zip(true_spikes, predicted_spikes):
        t_spk = np.asarray(t_spk, dtype=np.float64).flatten()
        p_spk = np.asarray(p_spk, dtype=np.float64).flatten()
        if len(t_spk) == 0 and len(p_spk) == 0:
            precisions.append(1.0); recalls.append(1.0); f1s.append(1.0); continue
        if len(p_spk) == 0:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0); continue
        if len(t_spk) == 0:
            precisions.append(0.0); recalls.append(1.0); f1s.append(0.0); continue
        n_tp_recall = int(np.sum(np.any(np.abs(t_spk[:, None] - p_spk[None, :]) <= tolerance, axis=1)))
        n_tp_prec   = int(np.sum(np.any(np.abs(p_spk[:, None] - t_spk[None, :]) <= tolerance, axis=1)))
        rec  = n_tp_recall / len(t_spk)
        prec = n_tp_prec   / len(p_spk)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1s)


def compute_kurtosis(traces):
    """ Fisher excess kurtosis of one or more fluorescence traces.

    Excess kurtosis is zero for a Gaussian, positive for heavy-tailed
    distributions. Useful for checking whether a trace has spike-like events.

    Parameters
    ----------
    traces : np.ndarray, shape (n_cells, n_frames) or (n_frames,)
        Fluorescence traces.

    Returns
    -------
    np.ndarray, shape (n_cells,)
        Excess kurtosis per cell.
    """

    if traces.ndim == 1:
        traces = traces[None, :]

    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True)
    std[std < 1e-9] = 1.0

    fourth_moment = np.mean((traces - mean)**4, axis=1, keepdims=True)
    kurt = fourth_moment / (std**4)

    return (kurt - 3.0).flatten()
