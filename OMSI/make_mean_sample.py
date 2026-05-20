# -*- coding: utf-8 -*-
"""
Construct the mean calcium trace from posterior samples.

Written Feb 2026, DMM
"""

import numpy as np
import numba as nb


# simple recursive filter: y[n] = x[n] + alpha*y[n-1]
# this is equivalent to convolution with a decaying exponential, but way faster
@nb.njit(cache=True, fastmath=True)
def _iir_filter(x, alpha):

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + alpha * y[i - 1]
    return y


@nb.njit(cache=True, fastmath=True)
def _compute_single_trace(ss_arr, T, tau0, tau1, am_val, cb_val, cin_val, dt):

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

        # offset is the sub-frame time within the bin; use it to scale the starting amplitude
        # of each exponential component so spike timing is precise below the frame resolution
        offset = st - dt * ceil_st
        if gr0 > 0.0:
            s_1[idx] += np.float32(np.exp(offset / tau0))
        s_2[idx] += np.float32(np.exp(offset / tau1))

    # apply the iir filter to get the two exponential components of the calcium transient,
    # then combine them to get the net double-exponential shape
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


# reconstruct the mean calcium trace by averaging over all posterior spike train samples.
# each sample gives a slightly different calcium trace, so the mean is a smoother estimate
def make_mean_sample(SAMPLES, Y):

    T = len(Y)
    N = len(SAMPLES['ns'])
    P = SAMPLES['params']

    f_val = P.get('f', 1.0)
    dt    = 1.0 / f_val
    g_val = np.array(P['g']).flatten()

    if 'g' not in SAMPLES:
        SAMPLES['g'] = np.tile(g_val, (N, 1))

    # if Cb has exactly 2 elements it's stored as [mean, std] (marginalized mode),
    # otherwise it's a full array of per-sample values
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
