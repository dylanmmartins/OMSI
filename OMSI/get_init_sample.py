# -*- coding: utf-8 -*-
"""
OMSI/get_init_sample.py

Compute an initial sample for the MCMC sampler via block-wise NNLS deconvolution.

Functions
---------
_get_sn
    Estimate noise std from high-frequency end of the power spectral density.
_estimate_time_constants
    Fit AR(p) time constants from autocorrelation via Yule-Walker equations.
_ar_kernel
    Compute the AR(p) impulse response, truncated at 1% of peak.
_block_nnls_deconv
    Block-wise NNLS deconvolution with cross-block spillover correction.
_foopsi_deconv
    AR(1) FOOPSI deconvolution via L-BFGS-B with L1 spike penalty.
get_init_sample
    Get initial spike times and parameters for MCMC via NNLS or FOOPSI.


DMM, Feb 2026
"""

import numpy as np
from scipy.optimize import nnls as scipy_nnls, minimize as _sp_minimize
from scipy.linalg import toeplitz as sp_toeplitz
from scipy.signal import lfilter


def _get_sn(y, range_ff):
    """ Estimate noise std from high-frequency end of the power spectral density.

    Calcium transients are slow, so high frequencies are dominated by noise
    rather than signal. Uses geometric mean of PSD (mean in log domain) for
    robustness to outlier frequencies.

    Parameters
    ----------
    y : np.ndarray
        Fluorescence trace.
    range_ff : tuple of float
        Normalized frequency range [low, high] in [0, 0.5] to use for estimation.

    Returns
    -------
    float
        Estimated noise standard deviation.
    """

    L = len(y)
    xdft = np.fft.rfft(y)
    psd = (1.0 / L) * np.abs(xdft) ** 2
    psd[1:-1] *= 2
    ff = np.linspace(0, 0.5, len(psd))
    ind = (ff > range_ff[0]) & (ff <= range_ff[1])
    if not np.any(ind):
        return float(np.std(y))
    return float(np.sqrt(np.exp(np.mean(np.log(psd[ind] / 2.0)))))


def _estimate_time_constants(y, p, sn, lags=20):
    """ Fit AR(p) time constants from autocorrelation via Yule-Walker equations.

    The Toeplitz matrix is what autocorrelation looks like under the AR model,
    minus the noise contribution (sn^2 on the diagonal for lag zero).

    Parameters
    ----------
    y : np.ndarray
        Fluorescence trace.
    p : int
        AR model order.
    sn : float
        Noise standard deviation.
    lags : int
        Number of autocorrelation lags to use.

    Returns
    -------
    g : np.ndarray
        AR coefficients of length p.
    """

    lags = lags + p
    yn = y - np.mean(y)
    xc = np.zeros(lags + 2)
    for k in range(lags + 2):
        xc[k] = np.dot(yn[k:], yn[:len(yn) - k])
    xc /= len(y)
    col = xc[1:lags + 1]
    row = xc[1:p + 1]
    A = sp_toeplitz(col, row) - (sn ** 2) * np.eye(lags, p)
    try:
        g = np.linalg.pinv(A) @ xc[2:lags + 2]
    except Exception:
        g = np.array([0.0])
    return g


def _ar_kernel(g, K):
    """ Compute the AR(p) impulse response, truncated at 1% of peak.

    Recursively expands h[k] = sum(g_j * h[k-j-1]). Truncates the tail
    once it drops below 1% of the peak to keep convolution cheap.

    Parameters
    ----------
    g : array-like
        AR coefficients.
    K : int
        Maximum kernel length before truncation.

    Returns
    -------
    h : np.ndarray
        Impulse response, truncated at 1% of peak.
    """

    g = np.atleast_1d(g).flatten()
    h = np.zeros(K)
    h[0] = 1.0
    for k in range(1, K):
        for j, gj in enumerate(g):
            km = k - j - 1
            if km >= 0:
                h[k] += gj * h[km]
    thresh = 0.01 * np.max(np.abs(h))
    below = np.where(np.abs(h) < thresh)[0]
    if len(below) > 0:
        h = h[:below[0]]
    return h


def _block_nnls_deconv(y_corr, h, T, block_size=400):
    """ Block-wise NNLS deconvolution with cross-block spillover correction.

    Processes the trace in chunks to keep memory manageable on long recordings.
    Tracks the tail of each block's calcium response that bleeds into the next
    block and subtracts it before solving -- otherwise spikes near block
    boundaries would be undercounted.

    Parameters
    ----------
    y_corr : np.ndarray
        Baseline- and initial-calcium-corrected fluorescence trace.
    h : np.ndarray
        AR impulse response kernel.
    T : int
        Number of frames.
    block_size : int
        Frames per block.

    Returns
    -------
    sp : np.ndarray
        Nonnegative spike amplitude vector, shape (T,).
    """

    K = len(h)
    sp = np.zeros(T)
    spillover = np.zeros(T + K)

    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        B = end - start

        h_col = np.zeros(B)
        h_col[:min(K, B)] = h[:min(K, B)]
        H_block = sp_toeplitz(h_col, np.zeros(B))

        y_block = y_corr[start:end] - spillover[start:end]

        sp_block, _ = scipy_nnls(H_block, y_block)
        sp[start:end] = sp_block

        if K > 1 and np.any(sp_block > 0):
            tail = np.convolve(sp_block, h)[B:]
            tail_len = min(len(tail), T + K - end)
            spillover[end:end + tail_len] += tail[:tail_len]

    return sp


def _foopsi_deconv(y, g_decay, lam):
    """ AR(1) FOOPSI deconvolution via L-BFGS-B with L1 spike penalty.

    Parameters
    ----------
    y : np.ndarray
        Fluorescence trace.
    g_decay : float
        AR(1) decay coefficient.
    lam : float
        L1 penalty weight on spike amplitudes.

    Returns
    -------
    np.ndarray
        Nonnegative spike vector of length T.
    """

    T = len(y)
    g = float(g_decay)

    def _fwd(s):
        return lfilter([1.0], [1.0, -g], s)

    def _adj(v):
        return lfilter([1.0], [1.0, -g], v[::-1])[::-1]

    def _obj(s):
        c    = _fwd(s)
        res  = c - y
        f    = 0.5 * float(np.dot(res, res)) + lam * float(s.sum())
        grad = _adj(res) + lam
        return f, grad

    result = _sp_minimize(
        _obj, np.zeros(T), method='L-BFGS-B', jac=True,
        bounds=[(0.0, None)] * T,
        options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6},
    )
    return np.maximum(result.x, 0.0)


def get_init_sample(Y, params):
    """ Get initial spike times and parameters for the MCMC sampler.

    Estimates noise, time constants, and baseline from the trace, then
    runs block-wise NNLS (or FOOPSI) to get an initial spike position guess.
    Returns a dict of initial values consumed by cont_ca_sampler.

    Parameters
    ----------
    Y : np.ndarray
        Fluorescence trace (dF/F or raw F).
    params : dict
        Sampler parameters. Relevant keys: 'p' (AR order), 'g' (AR coefficients),
        'sn' (noise std), 'b' (baseline), 'c1' (initial calcium), 'f' (frame rate),
        'bas_nonneg' (enforce nonnegative baseline), 'init_method' ('foopsi' or default).

    Returns
    -------
    SAM : dict
        Initial sample dict with keys: 'lam_', 'spiketimes_', 'A_', 'b_',
        'C_in', 'sg', 'g'.
    """

    options = {'p': params.get('p', 1)}

    required_keys = ['c', 'b', 'c1', 'g', 'sn', 'sp']

    Y = np.atleast_1d(Y).flatten()
    T = len(Y)

    if not any(params.get(k) is None for k in required_keys):
        c   = params['c']
        b   = float(params['b'])
        c1  = float(params['c1'])
        g   = np.atleast_1d(params['g']).flatten()
        sn  = float(params['sn'])
        sp  = params['sp']

    else:
        if params.get('g') is not None:
            g = np.atleast_1d(params['g']).flatten()
        else:
            p = options['p']
            sn_tmp = _get_sn(Y, [0.25, 0.5])
            g = _estimate_time_constants(Y, p, sn_tmp, lags=20)

            roots = np.roots(np.concatenate([[1.0], -g]))
            roots = np.real(roots).clip(0.01, 0.999)
            g = -np.poly(roots)[1:]
        g = np.atleast_1d(g).flatten()

        if params.get('sn') is not None:
            sn = float(params['sn'])
        else:
            # For fast sensors, the calcium signal has non-negligible power in
            # the [0.25, 0.5] PSD band used by _get_sn, which inflates the noise
            # estimate and raises A_lb above real spike amplitudes. MAD of first
            # differences is robust here -- the difference operator attenuates the
            # slow signal and MAD ignores spike outliers.
            _roots_abs = np.abs(np.roots(np.concatenate([[1.0], -g])))
            _g_d = float(np.max(_roots_abs)) if len(_roots_abs) > 0 else float(np.max(g))
            _tau_d_s = -1.0 / (np.log(max(min(_g_d, 0.9999), 1e-6)) * float(params.get('f', 30.0)))
            if _tau_d_s < 0.6:
                sn = float(np.median(np.abs(np.diff(Y))) / (0.6745 * np.sqrt(2.0)))
            else:
                sn = _get_sn(Y, [0.25, 0.5])

        bas_nonneg = params.get('bas_nonneg', 0)
        if params.get('b') is not None:
            b = float(params['b'])
        else:
            b = float(np.nanpercentile(Y, 8))
            if bas_nonneg:
                b = max(b, 0.0)

        c1 = float(params['c1']) if params.get('c1') is not None \
             else max(float(Y[0]) - b, 0.0)

        roots_abs = np.abs(np.roots(np.concatenate([[1.0], -g])))
        g_decay = float(np.max(roots_abs)) if len(roots_abs) > 0 else float(np.max(g))
        g_decay = min(g_decay, 0.9999)
        ge = g_decay ** np.arange(T)

        y_corr = Y - b - c1 * ge

        tau_frames = max(1.0, -1.0 / np.log(max(g_decay, 1e-6)))

        if params.get('init_method') == 'foopsi':
            sp = _foopsi_deconv(y_corr, g_decay, lam=sn)
        else:
            K  = min(T, max(50, int(np.ceil(5 * tau_frames))))
            h  = _ar_kernel(g, K)
            sp = _block_nnls_deconv(y_corr, h, T, block_size=min(400, T))

        c = lfilter([1.0], np.concatenate([[1.0], -g]), sp)

    dt = 1.0
    sp_max = float(np.max(sp)) if len(sp) > 0 else 0.0

    # Keep frames where NNLS response is at least 15% of the peak.
    s_in = (sp > 0.15 * sp_max) if sp_max > 0 else np.zeros(T, dtype=bool)
    indices = np.where(s_in)[0]

    # Jitter spike positions slightly within their frame for sub-frame precision;
    # reflect any that land just outside recording bounds back in.
    spiketimes_ = dt * (indices.astype(float) + np.random.rand(len(indices)) - 0.5)
    oob = spiketimes_ >= T * dt
    spiketimes_[oob] = 2.0 * T * dt - spiketimes_[oob]

    SAM = {}
    SAM['lam_'] = len(spiketimes_) / (T * dt)
    SAM['spiketimes_'] = spiketimes_

    sp_in = sp[s_in]
    if len(sp_in) > 0:
        # Amplitude guess: median of detected spike amplitudes, but at least
        # 1/4 of the max so we don't undershoot on sparse data.
        SAM['A_'] = max(float(np.median(sp_in)), float(np.max(sp_in)) / 4.0)
    else:
        SAM['A_'] = sn

    if len(g) == 2:
        # Rescale for AR(2): peak of impulse response isn't 1, it depends on g values.
        denom = g[0] ** 2 + 4 * g[1]
        if denom > 0:
            SAM['A_'] = SAM['A_'] / np.sqrt(denom)

    y_range = float(np.max(Y)) - float(np.min(Y))
    SAM['b_']   = max(b, float(np.min(Y)) + y_range / 25.0)
    SAM['C_in'] = max(c1, (float(Y[0]) - b) / 10.0)
    SAM['sg']   = sn
    SAM['g']    = g

    return SAM
