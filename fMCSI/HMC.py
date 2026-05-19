# -*- coding: utf-8 -*-
"""
Exact Hamiltonian Monte Carlo for truncated Gaussians.

Written Feb 2026, DMM
"""


import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def HMC_exact2(F, g, M, mu_r, cov, L, initial_X):

    m = g.shape[0]
    if F.shape[0] != m:
        return None, None

    # whiten the space by cholesky-factoring the covariance matrix M.
    # in the whitened coordinates the covariance is identity, which means
    # hamiltonian dynamics become simple circular motion (no coupling between dims)
    if cov:
        mu = mu_r
        g  = g + F @ mu
        R  = np.linalg.cholesky(M).T
        F  = F @ R.T
        initial_X = np.linalg.solve(R.T, initial_X - mu)
    else:
        r    = mu_r
        R    = np.linalg.cholesky(M).T
        mu   = np.linalg.solve(R, np.linalg.solve(R.T, r))
        g    = g + F @ mu
        F    = np.linalg.solve(R.T, F.T).T
        initial_X = R @ (initial_X - mu)

    d            = initial_X.shape[0]
    bounce_count = 0
    # nearzero is used to avoid re-hitting the same constraint boundary on the very next step
    nearzero     = 10000 * np.finfo(np.float64).eps

    F = np.ascontiguousarray(F)
    initial_X = np.ascontiguousarray(initial_X)

    c = F @ initial_X + g
    if np.any(c < 0):
        return None, None

    F2     = np.sum(F * F, axis=1)
    Ft     = F.T
    last_X = initial_X.copy()
    Xs     = np.zeros((d, L))
    Xs[:, 0:1] = initial_X

    V0 = np.zeros((d, 1))
    X  = np.zeros((d, 1))
    V  = np.zeros((d, 1))
    fa = np.zeros((m, 1))
    fb = np.zeros((m, 1))

    i = 1
    outer_iter = 0
    while i < L:
        outer_iter += 1

        # if we're stuck bouncing around and never accepting new samples,
        # just repeat the last valid sample for the remaning slots and bail out
        if outer_iter > L * 100:
            for k in range(i, L):
                Xs[:, k] = last_X.flatten()
            break

        stop   = False
        j      = -1
        
        for k in range(d):
            V0[k, 0] = np.random.randn()
        
        X[:] = last_X[:]
        T_time = np.pi / 2
        tt     = 0.0
        step_iter = 0

        while True:
            step_iter += 1
            if step_iter > 2000:
                stop = True
                break

            a  = V0
            b  = X
            
            for r in range(m):
                val = 0.0
                for c in range(d):
                    val += F[r, c] * a[c, 0]
                fa[r, 0] = val
            for r in range(m):
                val = 0.0
                for c in range(d):
                    val += F[r, c] * b[c, 0]
                fb[r, 0] = val

            # for each constraint F[r]*x >= -g[r], the trajectory x(t) = a*sin(t) + b*cos(t)
            # means the constraint value traces a sinusoid with amplitude U and phase phi.
            # need to find the t values where this sinusoid crosses zero (hits the boundary)
            U   = np.sqrt(fa ** 2 + fb ** 2)
            phi = np.arctan2(-fa, fb)

            # constraints where |g/U| > 1 are never active along this arc (never hit the wall)
            g_over_U = g / U
            pn       = (np.abs(g_over_U) <= 1).flatten()

            if np.any(pn):
                inds = np.where(pn)[0]
                
                phn  = phi.flatten()[pn]
                gou_pn = g_over_U.flatten()[pn]

                t1 = -phn + np.arccos(np.clip(-gou_pn, -1.0, 1.0))
                t1[t1 < 0] += 2 * np.pi

                t2 = -t1 - 2 * phn
                t2[t2 < 0] += 2 * np.pi
                t2[t2 < 0] += 2 * np.pi

                if j >= 0:
                    if pn[j]:
                        indj = np.where(inds == j)[0][0]
                        tt1  = t1[indj]
                        if abs(tt1) < nearzero or abs(tt1 - 2 * np.pi) < nearzero:
                            t1[indj] = np.inf
                        else:
                            tt2 = t2[indj]
                            if abs(tt2) < nearzero or abs(tt2 - 2 * np.pi) < nearzero:
                                t2[indj] = np.inf

                mt1   = np.min(t1)
                ind1  = np.argmin(t1)
                mt2   = np.min(t2)
                ind2  = np.argmin(t2)

                if mt1 < mt2:
                    mt    = mt1
                    m_ind = ind1
                else:
                    mt    = mt2
                    m_ind = ind2

                j = inds[m_ind]
            else:
                mt = T_time

            tt += mt
            if tt >= T_time:
                mt   = mt - (tt - T_time)
                stop = True

            sin_mt = np.sin(mt)
            cos_mt = np.cos(mt)
            for k in range(d):
                val_a = a[k, 0]
                val_b = b[k, 0]
                X[k, 0] = val_a * sin_mt + val_b * cos_mt
                V[k, 0] = val_a * cos_mt - val_b * sin_mt

            if stop:
                break

            # reflect the velocity off the j-th constraint wall.
            # subtract the component of velocity
            # normal to the wall (qj is the projection onto the normal direction)
            dot_val = 0.0
            for k in range(d):
                dot_val += F[j, k] * V[k, 0]
            qj = dot_val / F2[j]

            for k in range(d):
                V0[k, 0] = V[k, 0] - 2 * qj * Ft[k, j]
            bounce_count += 1

        valid = True
        for r in range(m):
            val = 0.0
            for c in range(d):
                val += F[r, c] * X[c, 0]
            if val + g[r, 0] <= 0:
                valid = False
                break
        
        if valid:
            Xs[:, i:i + 1] = X
            last_X = X
            i += 1

    # transform samples back from whitened coordinates to the original space
    if cov:
        Xs = R.T @ Xs + mu
    else:
        Xs = np.linalg.solve(R, Xs) + mu

    return Xs, bounce_count
