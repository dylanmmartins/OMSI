# -*- coding: utf-8 -*-
"""
OMSI/HMC.py

Exact Hamiltonian Monte Carlo sampler for truncated multivariate Gaussians.

Functions
---------
HMC_exact2
    Draw L samples from a Gaussian truncated by linear inequality constraints.


DMM, Feb 2026
"""

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def HMC_exact2(F, g, M, mu_r, cov, L, initial_X):
    """ Draw L samples from a Gaussian truncated by linear inequality constraints.

    Implements exact HMC for the distribution p(x) proportional to
    N(x; mu, M) * I(F*x + g >= 0). Dynamics are exact sinusoidal arcs;
    boundaries are constraint walls that the momentum reflects off of.

    Parameters
    ----------
    F : np.ndarray, shape (m, d)
        Constraint matrix. Each row is one linear constraint.
    g : np.ndarray, shape (m, 1)
        Constraint offsets. Constraint i is satisfied when F[i]*x + g[i] >= 0.
    M : np.ndarray, shape (d, d)
        Covariance matrix (if cov=True) or precision matrix (if cov=False).
    mu_r : np.ndarray, shape (d, 1)
        Prior mean (if cov=True) or precision-weighted mean (if cov=False).
    cov : bool
        True if M is the covariance; False if M is the precision.
    L : int
        Number of samples to draw.
    initial_X : np.ndarray, shape (d, 1)
        Starting point inside the feasible region.

    Returns
    -------
    Xs : np.ndarray, shape (d, L)
        L samples from the truncated Gaussian, or None if infeasible.
    bounce_count : int
        Total number of constraint-wall reflections across all trajectories.
    """

    m = g.shape[0]
    if F.shape[0] != m:
        return None, None

    # Whiten the space by Cholesky-factoring M so Hamiltonian dynamics become
    # simple circular motion (no coupling between dimensions in whitened coords).
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

    # nearzero: avoid re-hitting the same wall immediately after a reflection.
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

        # If stuck bouncing and never accepting new samples, fill remaining
        # slots with the last valid sample and bail out.
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

            # For each constraint F[r]*x >= -g[r], the trajectory
            # x(t) = a*sin(t) + b*cos(t) traces a sinusoid. Find where
            # each sinusoid crosses zero (hits the constraint boundary).
            U   = np.sqrt(fa ** 2 + fb ** 2)
            phi = np.arctan2(-fa, fb)

            # Constraints where |g/U| > 1 are never active along this arc.
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

            # Reflect velocity off the j-th constraint wall: subtract
            # the component normal to the wall (qj is the projection).
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

    # Transform samples back from whitened coordinates to original space.
    if cov:
        Xs = R.T @ Xs + mu
    else:
        Xs = np.linalg.solve(R, Xs) + mu

    return Xs, bounce_count
