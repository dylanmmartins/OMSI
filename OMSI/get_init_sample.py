# -*- coding: utf-8 -*-
"""
Get an initial sample for the MCMC sampler. Uses block-wise unregularized
non-negative least squares (NNLS) deconvolution instead of FOOPSI.

Written Feb 2026, DMM
"""


import numpy as np
from scipy.optimize import nnls as scipy_nnls, minimize as _sp_minimize
from scipy.linalg import toeplitz as sp_toeplitz
from scipy.signal import lfilter


# estimate noise std from the high-frequency end of the power spectral density.
# calcium transients are slow, so high frequencies are dominated by noise rather than signal.
# we use geometric mean of the psd (mean in log domain) which is more robust to outlier frequencies
def _get_sn(y, range_ff):

    L = len(y)
    xdft = np.fft.rfft(y)
    psd = (1.0 / L) * np.abs(xdft) ** 2
    psd[1:-1] *= 2
    ff = np.linspace(0, 0.5, len(psd))
    ind = (ff > range_ff[0]) & (ff <= range_ff[1])
    if not np.any(ind):
        return float(np.std(y))
    return float(np.sqrt(np.exp(np.mean(np.log(psd[ind] / 2.0)))))


# fits ar(p) time constants from the autocorrelation structure using yule-walker equations.
# the toeplitz matrix is what the autocorrelation should look like under the ar model,
# minus the noise contribution (sn^2 on the diagonal for lag zero)
def _estimate_time_constants(y, p, sn, lags=20):

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



# compute the impulse response of the ar filter by recursively expanding h[k] = sum(g_j * h[k-j-1]).
# cuts off the tail once it decays below 1% of the peak to keep convolution cheap
def _ar_kernel(g, K):

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



# runs nnls block by block to keep memory manageable on long recordings.
# spillover tracks the tail of each block's calcium response that bleeds into the next block,
# so subtract it before solving the next block (otherwise it would undercount spikes near boundaries)
def _block_nnls_deconv(y_corr, h, T, block_size=400):

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


# AR(1) FOOPSI: L-BFGS-B with L1 spike penalty. Returns spike vector.
def _foopsi_deconv(y, g_decay, lam):
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
            # for fast sensors the calcium signal has non-negligible power in
            # the [0.25, 0.5] PSD band used by _get_sn, inflating the noise
            # estimate and raising A_lb above real spike amplitudes.
            # MAD of first differences is robust to this because the difference
            # operator attenuates the slow signal and MAD ignores spike outliers.
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
    
    # keep frames where the nnls response is at least 15% of the peak
    s_in = (sp > 0.15 * sp_max) if sp_max > 0 else np.zeros(T, dtype=bool)
    indices = np.where(s_in)[0]

    # jitter spike positions slightly within their frame to get sub-frame precision
    # and reflect any that land just outside the recording bounds back in
    spiketimes_ = dt * (indices.astype(float) + np.random.rand(len(indices)) - 0.5)
    oob = spiketimes_ >= T * dt
    spiketimes_[oob] = 2.0 * T * dt - spiketimes_[oob]

    SAM = {}
    SAM['lam_'] = len(spiketimes_) / (T * dt)
    SAM['spiketimes_'] = spiketimes_

    sp_in = sp[s_in]
    if len(sp_in) > 0:
        # amplitude guess is the median of detected spike amplitudes,
        # but at least 1/4 of the max so we dont undershot on sparse data
        SAM['A_'] = max(float(np.median(sp_in)), float(np.max(sp_in)) / 4.0)
    else:
        SAM['A_'] = sn

    if len(g) == 2:
        # rescale for ar(2): the peak of the impulse response isnt 1 but depends on g values
        denom = g[0] ** 2 + 4 * g[1]
        if denom > 0:
            SAM['A_'] = SAM['A_'] / np.sqrt(denom)

    y_range = float(np.max(Y)) - float(np.min(Y))
    SAM['b_']   = max(b, float(np.min(Y)) + y_range / 25.0)
    SAM['C_in'] = max(c1, (float(Y[0]) - b) / 10.0)
    SAM['sg']   = sn
    SAM['g']    = g

    return SAM

