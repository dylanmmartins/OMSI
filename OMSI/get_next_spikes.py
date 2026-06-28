# -*- coding: utf-8 -*-
"""
OMSI/get_next_spikes.py

Propose and accept or reject edits to the spike train for one MCMC sweep.

Functions
---------
get_next_spikes
    Run one sweep of Metropolis-Hastings moves over the continuous-time spike train.


DMM, Feb 2026
"""

import numpy as np
from numba import njit

from .spike_operations import replace_spike, add_spike, remove_spike


@njit(cache=True, fastmath=True)
def get_next_spikes(curr_spikes, n_spikes, curr_calcium, calcium_signal,
                    ef_h, ef_d, ef_nh, ef_nd,
                    tau, calcium_noise_var, lam_val,
                    proposal_std, add_move, dt, A):
    """ Run one sweep of Metropolis-Hastings moves over the continuous-time spike train.

    Two move types per sweep. Time-shift moves nudge each existing spike to a nearby
    time drawn from a Gaussian. Birth-death moves propose adding a spike at a random
    time and removing a random existing one. Both accept or reject via Metropolis ratio
    of calcium log-likelihood under proposed vs. current spike train.

    Parameters
    ----------
    curr_spikes : np.ndarray
        Pre-allocated array holding the current spike times, in seconds.
    n_spikes : int
        Number of active spikes currently stored in curr_spikes.
    curr_calcium : np.ndarray
        Modeled calcium trace corresponding to curr_spikes.
    calcium_signal : np.ndarray
        Observed dF/F trace. NaN entries are excluded from the likelihood.
    ef_h : np.ndarray
        Exponential filter kernel for the fast-rise component.
    ef_d : np.ndarray
        Exponential filter kernel for the slow-decay component.
    ef_nh : np.ndarray
        Cumulative squared norm of ef_h, precomputed to avoid recomputing per spike.
    ef_nd : np.ndarray
        Cumulative squared norm of ef_d, precomputed to avoid recomputing per spike.
    tau : array-like
        Two-element array [tau_rise, tau_decay] in seconds.
    calcium_noise_var : float
        Variance of the calcium observation noise.
    lam_val : float
        Poisson prior rate in spikes per frame, used in the birth-death acceptance ratio.
    proposal_std : float
        Standard deviation for time-shift proposals, in frames.
    add_move : int
        Number of birth-death pairs to attempt per sweep.
    dt : float
        Frame duration in seconds (1 / frame_rate).
    A : float
        Spike amplitude scaling factor.

    Returns
    -------
    si : np.ndarray
        Updated spike time array.
    n_spikes : int
        Updated spike count.
    new_calcium : np.ndarray
        Updated modeled calcium trace.
    moves : np.ndarray
        3 x 2 array of [accepted, proposed] counts for time-shift, birth, and death moves.
    """

    T = len(calcium_signal)

    # Mask over valid frames -- NaNs are dropped rather than imputed so they
    # don't pull the acceptance ratio.
    ff = ~np.isnan(calcium_signal)

    # si is an alias into curr_spikes, not a copy -- numba edits it in place.
    si = curr_spikes
    new_calcium = curr_calcium

    # Unnormalized log-likelihood: -sum of squared residuals over valid frames.
    # Normalization constant is shared across proposals so it drops out of MH ratio.
    logC = -np.sum((new_calcium[ff] - calcium_signal[ff]) ** 2)

    # Tracks [n_accepted, n_proposed] for each move type.
    time_moves = np.array([0, 0])
    add_moves  = np.array([0, 0])
    drop_moves = np.array([0, 0])

    # Time-shift moves: nudge each spike to a nearby time, accept or reject via Metropolis.
    for ni in range(n_spikes):
        tmpi = si[ni]
        tmpi_ = si[ni] + proposal_std * np.random.randn()

        # Reflect proposals outside [0, T] back into bounds.
        if tmpi_ < 0:
            tmpi_ = -tmpi_
        elif tmpi_ > T:
            tmpi_ = 2 * T - tmpi_

        _, delta_ll = replace_spike(
            si, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
            tau, calcium_signal, tmpi, ni, tmpi_, dt, A, check_only=True
        )
        logC_ = logC + delta_ll

        ratio = np.exp((logC_ - logC) / calcium_noise_var)

        if np.random.rand() < ratio:

            new_calcium, logC = replace_spike(
                si, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
                tau, calcium_signal, tmpi, ni, tmpi_, dt, A, check_only=False
            )
            time_moves[0] += 1

        time_moves[1] += 1

    # Birth-death moves: propose adding a spike at a random time and removing a random
    # existing one. Pairing them keeps chain reversible. add_move sets pairs per sweep.
    for ii in range(add_move):

        tmpi = T * dt * np.random.rand()

        _, _, _, delta_ll = add_spike(
            si, n_spikes, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
            tau, calcium_signal, tmpi, dt, A, check_only=True
        )
        logC_ = logC + delta_ll

        # fprob: density of proposing this time, uniform over [0, T*dt].
        # rprob: prob of selecting this spike for removal in reverse move.
        # MH ratio includes Poisson prior on spike count via lam_val.
        fprob = 1.0 / (T * dt)
        rprob = 1.0 / (n_spikes + 1)

        ratio = np.exp((logC_ - logC) / (2 * calcium_noise_var)) * (rprob / fprob) * lam_val

        if np.random.rand() < ratio:

            si, n_spikes, new_calcium, logC_ = add_spike(
                si, n_spikes, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
                tau, calcium_signal, tmpi, dt, A, check_only=False
            )

            logC = logC_
            add_moves[0] += 1
        add_moves[1] += 1

        if n_spikes > 0:
            tmpi_idx = np.random.randint(0, n_spikes)
            tmpi_val = si[tmpi_idx]

            _, _, delta_ll = remove_spike(
                si, n_spikes, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
                tau, calcium_signal, tmpi_val, tmpi_idx, dt, A, check_only=True
            )

            logC_ = logC + delta_ll

            # fprob and rprob flip roles vs. birth: rprob is density of proposing
            # this time in reverse (birth) direction, fprob is prob of picking this spike.
            rprob = 1.0 / (T * dt)
            fprob = 1.0 / n_spikes

            ratio = (np.exp((logC_ - logC) / (2 * calcium_noise_var))
                     * (rprob / fprob) * (1.0 / lam_val))

            if np.random.rand() < ratio:

                n_spikes, new_calcium, logC_ = remove_spike(
                    si, n_spikes, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
                    tau, calcium_signal, tmpi_val, tmpi_idx, dt, A, check_only=False
                )
                logC = logC_
                drop_moves[0] += 1

            drop_moves[1] += 1

    moves = np.vstack((time_moves, add_moves, drop_moves))

    return si, n_spikes, new_calcium, moves
