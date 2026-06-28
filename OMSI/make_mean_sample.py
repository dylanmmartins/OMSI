# -*- coding: utf-8 -*-
"""
OMSI/make_mean_sample.py

Reconstruct the mean calcium trace by averaging over posterior spike train samples.

Functions
---------
_iir_filter
    Simple recursive IIR filter: y[n] = x[n] + alpha*y[n-1].
_compute_single_trace
    Compute calcium trace for a single posterior spike train sample.
make_mean_sample
    Average calcium trace over all posterior samples.


DMM, Feb 2026
"""

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def _iir_filter(x, alpha):
    """ Simple recursive IIR: y[n] = x[n] + alpha*y[n-1].

    Equivalent to convolution with a decaying exponential, but faster.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    alpha : float
        Filter coefficient (AR pole).

    Returns
    -------
    y : np.ndarray
        Filtered output, same shape as x.
    """

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + alpha * y[i - 1]
    return y


@nb.njit(cache=True, fastmath=True)
def _compute_single_trace(ss_arr, T, tau0, tau1, am_val, cb_val, cin_val, dt):
    """ Compute calcium trace for a single posterior spike train sample.

    Uses sub-frame timing via the offset between spike time and bin edge,
    then IIR-filters the binned amplitudes to get the double-exponential shape.

    Parameters
    ----------
    ss_arr : np.ndarray
        Spike times in seconds for one posterior sample.
    T : int
        Number of frames.
    tau0 : float
        Rise time constant in seconds.
    tau1 : float
        Decay time constant in seconds.
    am_val : float
        Amplitude.
    cb_val : float
        Baseline.
    cin_val : float
        Initial calcium offset.
    dt : float
        Frame duration in seconds (1 / frame_rate).

    Returns
    -------
    trace : np.ndarray of float32
        Model calcium trace of length T.
    """

    gr0 = np.float32(np.exp(-dt / tau0)) if tau0 > 0.0 else np.float32(0.0)
    gr1 = np.float32(np.exp(-dt / tau1))
    diff_gr = gr1 - gr0   # float32

    ge = np.empty(T, dtype=np.float32)
    ge[0] = np.float32(1.0)
    for k in range(1, T):
        ge[k] = ge[k - 1] * gr1

    s_1 = np.zeros(T, dtype=np.float32)
    s_2 = np.zeros(T, dtype=np.float32)

    for j in range(len(ss_arr)):

        st = ss_arr[j]
        ceil_st = np.ceil(st / dt)
        idx = int(ceil_st) - 1

        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1

        # Sub-frame offset scales starting amplitude so spike timing is precise
        # below the frame resolution.
        offset = st - dt * ceil_st
        if gr0 > 0.0:
            s_1[idx] += np.float32(np.exp(offset / tau0))
        s_2[idx] += np.float32(np.exp(offset / tau1))

    # Apply IIR filter to get the two exponential components, then combine
    # to get the net double-exponential calcium shape.
    G1sp = _iir_filter(s_1, gr0) if gr0 > 0.0 else np.zeros(T, dtype=np.float32)
    G2sp = _iir_filter(s_2, gr1)
    Gs   = (-G1sp + G2sp) / diff_gr   # float32

    am_f  = np.float32(am_val)
    cb_f  = np.float32(cb_val)
    cin_f = np.float32(cin_val)

    trace = np.empty(T, dtype=np.float32)
    for k in range(T):
        trace[k] = cb_f + am_f * Gs[k] + cin_f * ge[k]

    return trace


def make_mean_sample(SAMPLES, Y):
    """ Average calcium trace over all posterior spike train samples.

    Each sample gives a slightly different calcium trace; averaging gives a
    smoother estimate than any single sample.

    Parameters
    ----------
    SAMPLES : dict
        Output of cont_ca_sampler. Must contain 'ns', 'ss', 'params', 'g',
        'Am', 'Cb', 'Cin'.
    Y : np.ndarray
        Observed fluorescence trace -- used only to get length T.

    Returns
    -------
    np.ndarray of float32
        Mean calcium trace, shape (T,).
    """

    T = len(Y)
    N = len(SAMPLES['ns'])
    P = SAMPLES['params']

    f_val = P.get('f', 1.0)
    dt    = 1.0 / f_val
    g_val = np.array(P['g']).flatten()

    if 'g' not in SAMPLES:
        SAMPLES['g'] = np.tile(g_val, (N, 1))

    # If Cb has exactly 2 elements it's stored as [mean, std] (marginalized mode);
    # otherwise it's a full array of per-sample values.
    marg = 1 if len(np.atleast_1d(SAMPLES['Cb'])) == 2 else 0

    C_sum    = np.zeros(T, dtype=np.float32)
    Cin_flat = np.array(SAMPLES['Cin']).flatten()

    for rep in range(N):
        tau = np.atleast_1d(SAMPLES['g'][rep, :])

        ss      = np.atleast_1d(SAMPLES['ss'][rep]).astype(np.float64)
        am_val  = float(SAMPLES['Am'][rep])

        if marg:
            cb_val  = float(SAMPLES['Cb'][0])
            cin_val = float(Cin_flat[0])
        else:
            cb_val  = float(np.atleast_1d(SAMPLES['Cb'])[rep])
            cin_val = float(Cin_flat[rep] if len(Cin_flat) > rep else Cin_flat[-1])

        trace = _compute_single_trace(
            ss, T, float(tau[0]), float(tau[1]),
            am_val, cb_val, cin_val, dt,
        )
        C_sum += trace

    return C_sum / N
