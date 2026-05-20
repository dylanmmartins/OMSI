# -*- coding: utf-8 -*-
"""
Get the next set of spike times given the current spike train and observed
fluorescence trace, using a pre-allocated array for spikes.

Written Feb 2026, DMM
"""

import numpy as np
from numba import njit

from .spike_operations import replace_spike, add_spike, remove_spike


@njit(cache=True, fastmath=True)
def get_next_spikes(curr_spikes, n_spikes, curr_calcium, calcium_signal,
                    ef_h, ef_d, ef_nh, ef_nd,
                    tau, calcium_noise_var, lam_val,
                    proposal_std, add_move, dt, A):

    T = len(calcium_signal)
    ff = ~np.isnan(calcium_signal)

    si = curr_spikes
    new_calcium = curr_calcium
    logC = -np.sum((new_calcium[ff] - calcium_signal[ff]) ** 2)

    time_moves = np.array([0, 0])
    add_moves  = np.array([0, 0])
    drop_moves = np.array([0, 0])

    # time-shift moves: propose moving each spike to a nearby time, accept/reject via metropolis
    for ni in range(n_spikes):
        tmpi = si[ni]
        tmpi_ = si[ni] + proposal_std * np.random.randn()

        # reflect proposals outside [0, T] back into bounds
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

    # birth-death moves: propose adding a spike at a uniform random time, then
    # propose removing a random existing spike. these two are paired so the chain is reversible.
    # add_move controls how many birth-death pairs to attempt per sweep
    for ii in range(add_move):

        tmpi = T * dt * np.random.rand()

        _, _, _, delta_ll = add_spike(
            si, n_spikes, new_calcium, logC, ef_h, ef_d, ef_nh, ef_nd,
            tau, calcium_signal, tmpi, dt, A, check_only=True
        )
        logC_ = logC + delta_ll

        # fprob is the probability of proposing this exact time (uniform over [0, T])
        # rprob is the probability of proposing to remove it in the reverse move
        # the ratio includes the poisson prior on spike count via lam_val
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
