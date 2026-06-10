# -*- coding: utf-8 -*-
"""
MCMC sampler for continuous-time spike inference.

Written Feb 2026, DMM
"""

import numpy as np
import numba as nb
from numba import typed, types

from .get_init_sample import get_init_sample
from .get_next_spikes import get_next_spikes
from .HMC import HMC_exact2


@nb.njit(cache=True, fastmath=True)
def _iir_filter(x, alpha, out):

    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + alpha * out[i - 1]


@nb.njit(cache=True, fastmath=True)
def _compute_ge(gr_max, T):

    ge = np.empty(T, dtype=np.float32)
    ge[0] = 1.0
    for i in range(1, T):
        ge[i] = ge[i - 1] * gr_max
    return ge


@nb.njit(cache=True, fastmath=True)
def _bin_spikes_and_Gs(spiketimes, T, tau0, tau1, gr_min, gr_max, diff_gr, dt, p,
                       s_1, s_2, G1sp, G2sp, Gs):

    # bin the continuous-time spikes into frames, then apply the iir filter to
    # get the calcium trace... dump spike amplitudes into the right bin, filter gives the decay
    s_1.fill(0.0)
    s_2.fill(0.0)
    for st in spiketimes:
        idx = int(np.ceil(st / dt)) - 1
        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1
        offset = st - dt * (idx + 1)
        if p > 1:
            s_1[idx] += np.exp(offset / tau0)
        s_2[idx] += np.exp(offset / tau1)
    if p > 1:
        _iir_filter(s_1, gr_min, G1sp)
    else:
        G1sp.fill(0.0)
    _iir_filter(s_2, gr_max, G2sp)
    for i in range(T):
        Gs[i] = (-G1sp[i] + G2sp[i]) / diff_gr


@nb.njit(cache=True, fastmath=True)
def _bin_s1(spiketimes, T, tau0, dt, s_1):

    s_1.fill(0.0)
    for st in spiketimes:
        idx = int(np.ceil(st / dt)) - 1
        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1
        s_1[idx] += np.exp((st - dt * (idx + 1)) / tau0)


@nb.njit(cache=True, fastmath=True)
def _bin_s2(spiketimes, T, tau1, dt, s_2):

    s_2.fill(0.0)
    for st in spiketimes:
        idx = int(np.ceil(st / dt)) - 1
        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1
        s_2[idx] += np.exp((st - dt * (idx + 1)) / tau1)


@nb.njit(cache=True, fastmath=True)
def _posterior_update(Gs, ge, Y, isanY, sg2, ld_scale, mu):

    # gaussian linear regression for [amplitude, baseline, initial_calcium].
    # builds ATA and ATy from the design matrix [Gs, 1, ge] then solves the
    # normal equations with a small ridge prior (ld_scale) to keep it well-conditioned
    ATA = np.zeros((3, 3))
    ATy = np.zeros(3)
    for i in range(len(Y)):
        if isanY[i]:
            g0 = Gs[i]
            g2 = ge[i]
            ATA[0, 0] += g0 * g0
            ATA[0, 1] += g0
            ATA[0, 2] += g0 * g2
            ATA[1, 1] += 1.0
            ATA[1, 2] += g2
            ATA[2, 2] += g2 * g2
            ATy[0] += g0 * Y[i]
            ATy[1] += Y[i]
            ATy[2] += g2 * Y[i]
    ATA[1, 0] = ATA[0, 1]
    ATA[2, 0] = ATA[0, 2]
    ATA[2, 1] = ATA[1, 2]
    A = ATA / sg2
    A[0, 0] += ld_scale
    A[1, 1] += ld_scale
    A[2, 2] += ld_scale
    L = np.linalg.inv(A)
    rhs = ATy / sg2 + ld_scale * mu
    mu_post = np.linalg.solve(A, rhs)
    return L, mu_post


@nb.njit(cache=True, fastmath=True)
def _residual_sse(Y, Gs, ge, A_val, b_val, C_in_val, isanY):

    sse = 0.0
    n_valid = 0
    for i in range(len(Y)):
        if isanY[i]:
            r = Y[i] - A_val * Gs[i] - b_val - C_in_val * ge[i]
            sse += r * r
            n_valid += 1
    return sse, n_valid


@nb.njit(cache=True, fastmath=True)
def _logC_nb(Y, Gs, ge, A_val, b_val, C_in_val, isanY):

    sse = 0.0
    for i in range(len(Y)):
        if isanY[i]:
            r = Y[i] - A_val * Gs[i] - b_val - C_in_val * ge[i]
            sse += r * r
    return -sse


@nb.njit(cache=True, fastmath=True)
def _build_ef_nb(tau_cur, diff_gr_cur, t_arr, T, p, prec):

    # precompute the exponential kernel tails used for incremental likelihood updates.
    # truncate at the point where the kernel drops below prec * peak to save time
    # when looping over spike proposal windows in spike_operations
    ef_d_full = np.exp(-t_arr / tau_cur[1])   # float64

    if p > 1:
        t_max  = (tau_cur[0] * tau_cur[1]) / (tau_cur[1] - tau_cur[0]) \
                 * np.log(tau_cur[1] / tau_cur[0])
        h_max_ = np.exp(-t_max / tau_cur[1]) - np.exp(-t_max / tau_cur[0])
        ef_h_  = -np.exp(-t_arr / tau_cur[0])   # float64

        e_supp = T
        for k in range(T + 1):
            if (ef_d_full[k] - ef_h_[k]) < prec * h_max_:
                e_supp = k
                break
        e_supp = min(e_supp, T)

        ef_h_out = (ef_h_[:e_supp] / diff_gr_cur).astype(np.float32)
        ef_d_out = (ef_d_full[:e_supp] / diff_gr_cur).astype(np.float32)

    else:
        h_max_ = 1.0

        e_supp = T
        for k in range(T + 1):
            if ef_d_full[k] < prec * h_max_:
                e_supp = k
                break
        e_supp = min(e_supp, T)

        ef_h_out = np.zeros(2, dtype=np.float32)
        ef_d_out = (ef_d_full[:e_supp] / diff_gr_cur).astype(np.float32)

    ef_nh_out = np.cumsum(ef_h_out ** 2).astype(np.float32)
    ef_nd_out = np.cumsum(ef_d_out ** 2).astype(np.float32)
    return ef_h_out, ef_d_out, ef_nh_out, ef_nd_out, h_max_


@nb.njit(cache=True, fastmath=True)
def _mcmc_kernel_nb(
    Y, isanY, T,
    spiketimes_0,
    lam_0, A_0, b_0, C_in_0, sg_0,
    tau_0, gr_0, diff_gr_0, ge_0,
    ef_h_0, ef_d_0, ef_nh_0, ef_nd_0,
    lb_arr, mu_prior, ld_scale, A_lb, sp_scale,
    N_total, B, marg_flag, gam_flag, gam_step, p,
    add_move,
    std_move, tau1_std, tau2_std,
    tau_min, tau_min_decay, tau_max,
    lam_scale, prec, sn_mad,
    auto_stop, max_sweeps, min_sweeps,
    burn_tol, conv_tol, check_every,
    Ns,
    t_arr,
    con_lam,
):

    dt = 1.0

    ns   = np.zeros(N_total, dtype=np.float64)
    lam  = np.zeros(N_total, dtype=np.float64)
    Am   = np.zeros(N_total, dtype=np.float64)
    Gam  = np.zeros((N_total, 2), dtype=np.float64)
    Cb   = np.zeros(N_total, dtype=np.float64)
    Cin  = np.zeros(N_total, dtype=np.float64)
    SG   = np.zeros(N_total, dtype=np.float64)
    mub  = np.zeros(2, dtype=np.float64)
    Sigb = np.zeros((2, 2), dtype=np.float64)

    ss = typed.List.empty_list(types.float64[::1])

    spiketimes_ = spiketimes_0.copy()
    lam_    = lam_0
    A_      = A_0
    b_      = b_0
    C_in    = C_in_0
    sg      = sg_0
    tau     = tau_0.copy()
    gr      = gr_0.copy()
    diff_gr = diff_gr_0
    ge      = ge_0.copy()

    ef_h  = ef_h_0.copy()
    ef_d  = ef_d_0.copy()
    ef_nh = ef_nh_0.copy()
    ef_nd = ef_nd_0.copy()

    mu = mu_prior.copy()

    s_1_buf  = np.zeros(T, dtype=np.float32)
    s_2_buf  = np.zeros(T, dtype=np.float32)
    G1sp_buf = np.zeros(T, dtype=np.float32)
    G2sp_buf = np.zeros(T, dtype=np.float32)
    Gs_buf   = np.zeros(T, dtype=np.float32)

    prior_n     = 50.0
    alpha_prior = prior_n / 2.0
    beta_prior  = (prior_n / 2.0) * max(sn_mad, 1e-6) ** 2

    burn_in_done = False
    B_final  = B
    stop_idx = N_total

    for i in range(N_total):

        if gam_flag:
            Gam[i, 0] = tau[0]
            Gam[i, 1] = tau[1]

        if i % 2 == 0:
            _bin_spikes_and_Gs(
                spiketimes_, T,
                float(tau[0]), float(tau[1]),
                float(np.min(gr)), float(np.max(gr)),
                float(diff_gr), dt, p,
                s_1_buf, s_2_buf, G1sp_buf, G2sp_buf, Gs_buf,
            )

        curr_calcium = (A_ * Gs_buf).astype(np.float32)
        Ym_f32 = (Y - b_ - ge * C_in).astype(np.float32)

        spiketimes_buf, n_spikes_out, new_calcium, _ = get_next_spikes(
            spiketimes_, len(spiketimes_), curr_calcium, Ym_f32,
            ef_h, ef_d, ef_nh, ef_nd,
            tau, sg ** 2, float(lam_) * lam_scale,
            float(std_move), add_move, dt, float(A_)
        )

        spiketimes = spiketimes_buf[:n_spikes_out].copy()

        # if it ended up with spikes in more than half the frames something went wrong.
        # clear spike train and bump up the amplitude floor to push it back toward reality
        if len(spiketimes) > T * 0.5:
            spiketimes = np.empty(0, dtype=np.float64)
            A_ = max(A_, A_lb * 2.0)

        mask_lo = spiketimes < 0.0
        mask_hi = spiketimes > T * dt
        spiketimes[mask_lo] = -spiketimes[mask_lo]
        spiketimes[mask_hi] = 2.0 * T * dt - spiketimes[mask_hi]
        spiketimes_ = spiketimes

        # recover the normalized calcium shape (Gs) from the updated calcium trace.
        # Gs_buf is reused as the design matrix column for the regression step below
        if A_ > 1e-9:
            for _k in range(T):
                Gs_buf[_k] = np.float32(new_calcium[_k] / A_)
        else:
            Gs_buf.fill(0.0)


        ss.append(spiketimes_.copy())
        nsp    = len(spiketimes_)
        ns[i]  = nsp
        if not con_lam:
            lam_   = max(nsp / (T * dt), 0.01)
        lam[i] = lam_


        cov_mat, mu_post = _posterior_update(
            Gs_buf, ge, Y, isanY, sg ** 2, ld_scale, mu,
        )


        if not marg_flag:
            x_in = np.array([A_, b_, C_in])
            for d in range(3):
                if x_in[d] <= lb_arr[d]:
                    x_in[d] = (1.0 + 0.1 * np.sign(lb_arr[d])) * lb_arr[d] + 1e-5

            # sample the next [amplitude, baseline, initial_calcium] using exact HMC
            # in the truncated gaussian defined by the lower bounds (lb_arr)
            if np.any(np.isnan(cov_mat)):
                Am[i]  = A_
                Cb[i]  = b_
                Cin[i] = C_in
            else:
                temp, _ = HMC_exact2(
                    np.eye(3), -lb_arr.reshape(-1, 1),
                    cov_mat, mu_post.reshape(-1, 1),
                    True, Ns, x_in.reshape(-1, 1),
                )
                if temp is None:
                    Am[i]  = A_
                    Cb[i]  = b_
                    Cin[i] = C_in
                else:
                    Am[i]  = temp[0, -1]
                    Cb[i]  = temp[1, -1]
                    Cin[i] = temp[2, -1]

            A_, b_, C_in = Am[i], Cb[i], Cin[i]

            # sample noise std from its inverse-gamma posterior (conjugate to gaussian likelihood).
            # sample precision (1/variance) from a gamma, then take the reciprocal
            sse, n_valid = _residual_sse(Y, Gs_buf, ge, A_, b_, C_in, isanY)
            shape_param  = alpha_prior + n_valid / 2.0
            scale_param  = 1.0 / (beta_prior + sse / 2.0)
            sg    = 1.0 / np.sqrt(np.random.gamma(shape_param, scale_param))
            SG[i] = sg

        else:

            repeat = True
            cnt    = 0
            while repeat:
                A_ = mu_post[0] + np.sqrt(cov_mat[0, 0]) * np.random.randn()
                repeat = A_ < 0
                cnt += 1
                if cnt > 100:
                    A_ = max(mu_post[0], 1e-6 * sp_scale)
                    if A_ < 0:
                        A_ = 1e-6
                    repeat = False
            Am[i] = A_
            b_    = mu_post[1] + np.sqrt(cov_mat[1, 1]) * np.random.randn()
            C_in  = mu_post[2] + np.sqrt(cov_mat[2, 2]) * np.random.randn()
            if i >= B:
                mub[0]     += mu_post[1]
                mub[1]     += mu_post[2]
                Sigb[0, 0] += cov_mat[1, 1]
                Sigb[1, 1] += cov_mat[2, 2]

        # metropolis updates for the time constants (every gam_step iterations after burn-in)
        if gam_flag and (i - B) % gam_step == 0:

            logC = _logC_nb(Y, Gs_buf, ge, A_, b_, C_in, isanY)

            # propose new rise time constant tau[0], constrained to stay below tau[1]
            if p >= 2:
                tau_  = tau.copy()
                lc = 0
                tau_temp = tau_[0] + tau1_std * np.random.randn()
                while tau_temp > (tau[1] - 0.1) or tau_temp < max(tau_min, 0.1):
                    tau_temp = tau_[0] + tau1_std * np.random.randn()
                    lc += 1
                    if lc > 100:
                        tau_temp = tau_[0]
                        break
                tau_[0] = tau_temp
                gr_ = np.exp(-dt / tau_)

                s_1_tmp  = np.zeros(T, dtype=np.float32)
                G1sp_tmp = np.zeros(T, dtype=np.float32)
                G2sp_tmp = np.zeros(T, dtype=np.float32)
                _bin_s1(spiketimes_, T, float(tau_[0]), dt, s_1_tmp)
                _iir_filter(s_1_tmp,  float(np.min(gr_)), G1sp_tmp)
                _iir_filter(s_2_buf,  float(np.max(gr_)), G2sp_tmp)
                d_gr_ = float(gr_[1] - gr_[0])
                Gs_tmp = np.empty(T, dtype=np.float32)
                for _k in range(T):
                    Gs_tmp[_k] = (-G1sp_tmp[_k] + G2sp_tmp[_k]) / d_gr_

                logC_ = _logC_nb(Y, Gs_tmp, ge, A_, b_, C_in, isanY)
                ratio = 1.1 if logC_ > logC \
                        else np.exp((logC_ - logC) / (2.0 * sg ** 2))

                if np.random.rand() < ratio:
                    tau     = tau_
                    gr      = gr_
                    diff_gr = d_gr_
                    Gs_buf[:]  = Gs_tmp
                    s_1_buf[:] = s_1_tmp
                    logC = logC_


            # propose new decay time constant tau[1], keeping it above tau[0]
            tau_  = tau.copy()
            lc = 0
            tau_temp = tau_[1] + tau2_std * np.random.randn()
            while tau_temp > tau_max or tau_temp < max(tau_[0] + 0.1, tau_min_decay):
                tau_temp = tau_[1] + tau2_std * np.random.randn()
                lc += 1
                if lc > 100:
                    tau_temp = tau_[1]
                    break
            tau_[1] = tau_temp
            gr_      = np.exp(-dt / tau_)
            ge_new   = _compute_ge(float(np.max(gr_)), T)
            d_gr_    = float(gr_[1] - gr_[0])

            s_2_tmp  = np.zeros(T, dtype=np.float32)
            G1sp_d   = np.zeros(T, dtype=np.float32)
            G2sp_tmp = np.zeros(T, dtype=np.float32)
            _bin_s2(spiketimes_, T, float(tau_[1]), dt, s_2_tmp)
            if p > 1:
                _iir_filter(s_1_buf, float(np.min(gr_)), G1sp_d)
            _iir_filter(s_2_tmp, float(np.max(gr_)), G2sp_tmp)
            Gs_tmp = np.empty(T, dtype=np.float32)
            for _k in range(T):
                Gs_tmp[_k] = (-G1sp_d[_k] + G2sp_tmp[_k]) / d_gr_

            logC_ = _logC_nb(Y, Gs_tmp, ge_new, A_, b_, C_in, isanY)
            ratio = 1.1 if logC_ > logC \
                    else np.exp((logC_ - logC) / (2.0 * sg ** 2))

            if np.random.rand() < ratio:
                tau     = tau_
                ge      = ge_new
                gr      = gr_
                diff_gr = d_gr_
                Gs_buf[:]  = Gs_tmp
                s_2_buf[:] = s_2_tmp

            ef_h, ef_d, ef_nh, ef_nd, _ = _build_ef_nb(
                tau, diff_gr, t_arr, T, p, prec,
            )

        # check for convergence compare the mean of the first half vs second half of recent samples.
        # first wwait for burn-in to finish (amplitude stabilizes), then check whether
        # spike count distribution has stabilized (both halves have similar means)
        if auto_stop and i >= check_every and i % check_every == 0:
            if not burn_in_done:
                win = 100
                if i >= B + win:
                    recent = Am[i - win:i]
                    m1 = np.mean(recent[:win // 2])
                    m2 = np.mean(recent[win // 2:])
                    if m2 > 1e-9 and abs(m1 - m2) < burn_tol * m2:
                        burn_in_done = True
                        B_final = i
            else:
                n_samp = i - B_final
                if n_samp >= min_sweeps:
                    cur = ns[B_final:i]
                    mid = n_samp // 2
                    m1  = np.mean(cur[:mid])
                    m2  = np.mean(cur[mid:])
                    denom = m2 if m2 > 1e-9 else 1.0
                    if abs(m1 - m2) < conv_tol * denom:
                        stop_idx = i + 1
                        break

    n_post = max(stop_idx - B_final, 1)
    if marg_flag:
        mub  /= n_post
        Sigb /= n_post ** 2

    return (
        ss, ns, lam, Am, Gam,
        Cb, Cin, SG,
        mub, Sigb,
        B_final, stop_idx,
        tau, ge,
    )


def cont_ca_sampler(Y, params=None):

    Y = np.atleast_1d(Y).flatten().astype(np.float32)
    T = len(Y)
    isanY = ~np.isnan(Y)

    defparams = {
        'g': None, 'sn': None, 'b': None, 'c1': None,
        'c': None, 'sp': None,
        'bas_nonneg': 0,
        'Nsamples': 200, 'B': 75,
        'marg': 0, 'upd_gam': 1, 'gam_step': 1,
        'A_lb': None, 'b_lb': float(np.nanmin(Y)), 'c1_lb': 0,
        'std_move': 3, 'add_move': int(np.ceil(T / 500)),
        'init': None,
        'f': 10, 'p': 2,
        'defg': [0.6, 0.95],
        'TauStd': [0.2, 2],
        'prec': 1e-2,
        'con_lam': True,
        'print_flag': 0,
        'lam_pr': [0.1, 1.0],
        'auto_stop': True,
        'max_sweeps': 2000, 'min_sweeps': 300,
        'burn_tol': 0.005, 'conv_tol': 0.00067,
        'check_every': 50,
        'prob_thresh': 0.85,
        'lam_scale': 0.002,
    }

    if params is None:
        params = defparams
    else:
        for k, v in defparams.items():
            if k not in params:
                params[k] = v

    # skip inference if the trace is noise-dominated (SNR < 2.7).
    # a pure gaussian trace scores ~2.64 by this formula; anything below 2.7
    # means the calcium signal is completely buried in noise and inference is unreliable.
    # SNR = (99th - 8th percentile) / MAD-based noise std.
    valid_Y = Y[isanY]
    _sn_mad = (float(np.median(np.abs(np.diff(valid_Y)))) / 0.6745
               if len(valid_Y) > 1 else 1e-4)
    _peak   = float(np.percentile(valid_Y, 99)) if len(valid_Y) > 0 else 0.0
    _base   = float(np.percentile(valid_Y,  8)) if len(valid_Y) > 0 else 0.0
    _snr    = (_peak - _base) / (_sn_mad + 1e-9)

    if not params.get('skip_snr', False) and _snr < 2.7:
        _defg    = np.array(params['defg'])
        _tau_def = -1.0 / np.log(_defg)
        return {
            'ns':     np.array([0]),
            'ss':     [np.array([])],
            'ld':     np.array([0.0]),
            'Am':     np.array([0.0]),
            'g':      _tau_def.reshape(1, 2),
            'Cb': np.array([float(np.nanmean(Y))]),
            'Cin': np.array([0.0]),
            'sn2': np.array([_sn_mad ** 2]),
            'sn_mad': _sn_mad,
            'params': {'f': params['f'], 'g': _defg.tolist()},
        }

    dt = 1.0
    marg_flag = int(params['marg'])
    gam_flag = int(params['upd_gam'])
    gam_step = int(params['gam_step'])
    std_move = float(params['std_move'])
    add_move = int(params['add_move'])
    prec = float(params['prec'])
    lam_scale = float(params['lam_scale'])
    con_lam = int(bool(params['con_lam']))
    Ns = 15

    if params['g'] is None:
        p = int(params['p'])
    else:
        p = len(np.atleast_1d(params['g']))

    if params['init'] is None:
        params['init'] = get_init_sample(Y, params)
    SAM = params['init']

    g = np.atleast_1d(SAM['g']).flatten()
    if len(g) == 1 and g[0] == 0:
        gr_def = np.array(params['defg'])
        g = -np.poly(gr_def)[1:]
        p = 2

    gr = np.sort(np.roots(np.concatenate(([1.0], -g))))
    if p == 1:
        gr = np.array([0.0, np.max(gr)])

    # if estimated time constants are garbage (complex, negative, or explosive),
    # fall back to the defaults rather than letting the sampler blow up
    if np.any(gr < 0) or np.any(np.iscomplex(gr)) or len(gr) > 2 or np.max(gr) > 0.998:
        gr = np.array(params['defg'])

    gr = np.real(gr).astype(np.float64)

    # if two roots are too close together the double-exp kernel degenerates,
    # so replace with defaults to keep the model identifiable
    if p > 1 and abs(gr[1] - gr[0]) < 1e-4:
        gr = np.array(params['defg'], dtype=np.float64)
        gr = np.sort(gr)

    gr_for_tau = np.where(gr > 0, gr, 1e-300)
    tau     = -dt / np.log(gr_for_tau)
    if p == 1:
        tau[0] = np.inf

    if np.sum(isanY) > 1:
        sn_mad = float(np.median(np.abs(np.diff(Y[isanY]))) / 0.6745)
    else:
        sn_mad = 1e-4

    if params['A_lb'] is None:
        params['A_lb'] = float(SAM['sg'])

    fs = float(params['f'])
    if fs > 1.0:
        tau_min = 0.0; tau_min_decay = 0.0; tau_max = 5.0 * fs
    else:
        tau_min = 0.0; tau_min_decay = 0.0; tau_max = 500.0

    gr = np.where(np.isfinite(tau), np.exp(-dt / tau), 0.0)
    if p == 1:
        gr[0] = 0.0

    tau1_std = max(tau[0] / 100.0 if np.isfinite(tau[0]) else 0.0,
                   float(params['TauStd'][0]))
    tau2_std = min(tau[1] / 5.0,   float(params['TauStd'][1]))

    ge     = _compute_ge(float(np.max(gr)), T)   # float32
    t_arr  = np.arange(T + 1, dtype=np.float64)
    diff_gr = float(gr[1] - gr[0])

    ef_h, ef_d, ef_nh, ef_nd, _ = _build_ef_nb(tau, diff_gr, t_arr, T, p, prec)

    if params.get('T_supp') is not None:
        ts = max(2, min(int(params['T_supp']), T))
        if ts > len(ef_d):
            ef_h, ef_d, ef_nh, ef_nd, _ = _build_ef_nb(tau, diff_gr, t_arr, T, p, 0.0)
        ef_h  = ef_h[:ts]
        ef_d  = ef_d[:ts]
        ef_nh = ef_nh[:ts]
        ef_nd = ef_nd[:ts]

    sg      = float(SAM['sg'])
    A_      = float(SAM['A_']) * diff_gr
    b_      = float(max(float(SAM['b_']), float(np.nanpercentile(Y, 8))))
    C_in    = float(max(min(float(SAM['C_in']), float(Y[0]) - b_), 0.0))
    lam_0   = float(SAM['lam_'])

    sp_scale = 0.1 * (float(np.nanmax(Y)) - float(np.nanmin(Y)))
    ld_scale = 1.0 / max(sp_scale, 1e-9)

    A_lb_raw = float(params['A_lb'])
    if p > 1 and tau[1] > tau[0]:
        # h_max is the peak of the double-exponential kernel, used to rescale the amplitude lower bound
        # since amplitude in the sampler is in units of "peak kernel height", not raw dff
        t_max = (tau[0] * tau[1]) / (tau[1] - tau[0]) * np.log(tau[1] / tau[0])
        h_max = float(np.exp(-t_max / tau[1]) - np.exp(-t_max / tau[0]))
    else:
        h_max = 1.0

    lb_arr = np.array([A_lb_raw / max(h_max, 1e-9) * diff_gr,
                       float(params['b_lb']),
                       float(params['c1_lb'])])
    A_ = max(A_, 1.1 * lb_arr[0])
    mu_prior = np.array([A_, b_, C_in])

    B           = int(params['B'])
    auto_stop   = int(bool(params.get('auto_stop', False)))
    max_sweeps  = int(params.get('max_sweeps', 2000))
    min_sweeps  = int(params.get('min_sweeps', 300))
    burn_tol    = float(params.get('burn_tol', 0.005))
    conv_tol    = float(params.get('conv_tol', 0.00067))
    check_every = int(params.get('check_every', 50))

    N_total = max_sweeps if auto_stop else int(params['Nsamples']) + B

    spiketimes_0 = np.copy(SAM['spiketimes_']).astype(np.float64)

    (ss, ns_arr, lam_arr, Am_arr, Gam_arr,
     Cb_arr, Cin_arr, SG_arr,
     mub, Sigb,
     B_final, stop_idx,
     tau_final, ge_final) = _mcmc_kernel_nb(
        Y, isanY, T,
        spiketimes_0,
        lam_0, A_, b_, C_in, sg,
        tau, gr, diff_gr, ge,
        ef_h, ef_d, ef_nh, ef_nd,
        lb_arr, mu_prior, ld_scale, A_lb_raw, sp_scale,
        N_total, B, marg_flag, gam_flag, gam_step, p,
        add_move,
        std_move, tau1_std, tau2_std,
        tau_min, tau_min_decay, tau_max,
        lam_scale, prec, sn_mad,
        auto_stop, max_sweeps, min_sweeps,
        burn_tol, conv_tol, check_every,
        Ns,
        t_arr,
        con_lam,
    )

    ss_list = list(ss)[B_final:stop_idx]

    SAMPLES = {}
    if marg_flag:
        SAMPLES['Cb']  = [float(mub[0]),  float(np.sqrt(max(Sigb[0, 0], 0.0)))]
        SAMPLES['Cin'] = [float(mub[1]),  float(np.sqrt(max(Sigb[1, 1], 0.0)))]
    else:
        SAMPLES['Cb']  = Cb_arr[B_final:stop_idx]
        SAMPLES['Cin'] = Cin_arr[B_final:stop_idx]
        SAMPLES['sn2'] = SG_arr[B_final:stop_idx] ** 2

    SAMPLES['ns']     = ns_arr[B_final:stop_idx]
    SAMPLES['ss']     = ss_list
    SAMPLES['ld']     = lam_arr[B_final:stop_idx]
    SAMPLES['Am']     = Am_arr[B_final:stop_idx]
    SAMPLES['g']      = (Gam_arr[B_final:stop_idx, :]
                         if gam_flag
                         else np.tile(tau_final, (stop_idx - B_final, 1)))
    SAMPLES['params']  = params['init']
    SAMPLES['sn_mad']  = sn_mad

    return SAMPLES


