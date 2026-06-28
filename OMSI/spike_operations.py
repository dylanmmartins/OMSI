# -*- coding: utf-8 -*-
"""
OMSI/spike_operations.py

Incremental spike train edits for adding, removing, and replacing spikes.

Functions
---------
add_spike
    Add a spike to the spike train and update the calcium trace.
remove_spike
    Remove a spike from the spike train and update the calcium trace.
replace_spike
    Move a spike to a new time in a single pass.


DMM, Feb 2026
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def add_spike(spike_train, n_spikes, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd,
              tau, obs_calcium, time_to_add, dt, A, check_only=False):
    """ Add a spike to the spike train and update the calcium trace.

    In check_only mode, returns the change in log-likelihood without modifying
    anything -- used for the Metropolis accept/reject step before committing.
    In commit mode, grows the spike array if needed and updates the calcium trace.

    Parameters
    ----------
    spike_train : np.ndarray
        Pre-allocated spike time array.
    n_spikes : int
        Current number of spikes in spike_train.
    old_calcium : np.ndarray
        Current modeled calcium trace.
    old_ll : float
        Current log-likelihood.
    ef_h : np.ndarray
        Exponential filter kernel for fast-rise component.
    ef_d : np.ndarray
        Exponential filter kernel for slow-decay component.
    ef_nh : np.ndarray
        Cumulative squared norm of ef_h.
    ef_nd : np.ndarray
        Cumulative squared norm of ef_d.
    tau : array-like
        [tau_rise, tau_decay] in seconds.
    obs_calcium : np.ndarray
        Observed calcium trace. NaN entries excluded from likelihood.
    time_to_add : float
        Proposed spike time in frames.
    dt : float
        Frame duration in seconds.
    A : float
        Spike amplitude.
    check_only : bool
        If True, compute delta_ll only without modifying spike_train or old_calcium.

    Returns
    -------
    spike_train : np.ndarray
        Updated spike train (unchanged if check_only).
    n_spikes : int
        Updated spike count (unchanged if check_only).
    old_calcium : np.ndarray
        Updated calcium trace (unchanged if check_only).
    delta_ll or new_ll : float
        Change in log-likelihood if check_only, else the new absolute log-likelihood.
    """

    tau_h, tau_d = tau[0], tau[1]

    # wk_h and wk_d: starting amplitudes of each exponential component at this spike.
    # Sub-frame offset (time_to_add - dt*ceil(...)) shifts the starting amplitude slightly.
    wk_h = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_h)
    wk_d = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_d)
    t_floor = int(np.floor(time_to_add))

    if check_only:

        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor

        dot_h = 0.0
        for k in range(tmp_len_h):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                dot_h += res_val * (wk_h * ef_h[k])

        delta_ll_h = 2.0 * dot_h - (wk_h ** 2 * ef_nh[tmp_len_h - 1])

        end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
        tmp_len_d = end_idx_d - t_floor

        dot_d = 0.0
        for k in range(tmp_len_d):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                if k < tmp_len_h:
                    res_val -= wk_h * ef_h[k]
                dot_d += res_val * (wk_d * ef_d[k])

        delta_ll_d = 2.0 * dot_d - (wk_d ** 2 * ef_nd[tmp_len_d - 1])

        # delta_ll split across both exponential components and summed.
        # The 2*dot term captures how well the new spike explains the current
        # residual; the squared term accounts for self-overlap of the new kernel.
        total_delta_ll = delta_ll_h + delta_ll_d

        return spike_train, n_spikes, old_calcium, total_delta_ll

    # Grow the spike array if out of space.
    if n_spikes == len(spike_train):  # reallocate if full
        new_capacity = max(len(spike_train) * 2, 1)
        new_spike_train = np.empty(new_capacity, dtype=spike_train.dtype)
        new_spike_train[:n_spikes] = spike_train[:n_spikes]
        spike_train = new_spike_train

    spike_train[n_spikes] = time_to_add
    new_n_spikes = n_spikes + 1

    end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
    tmp_len_d = end_idx_d - t_floor

    obstemp = obs_calcium[t_floor:end_idx_d]
    old_ca_tmp = old_calcium[t_floor:end_idx_d]

    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        wef_h = (wk_h * ef_h[:tmp_len_h]).astype(np.float32)
        new_ca_tmp[:tmp_len_h] += wef_h

    wef_d = (wk_d * ef_d[:tmp_len_d]).astype(np.float32)
    new_ca_tmp += wef_d

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)

    new_ll = old_ll - sq_err_new + sq_err_old

    old_calcium[t_floor:end_idx_d] = new_ca_tmp

    return spike_train, new_n_spikes, old_calcium, new_ll


@njit(cache=True, fastmath=True)
def remove_spike(spike_train, n_spikes, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd,
                 tau, obs_calcium, time_to_remove, indx, dt, A, check_only=False):
    """ Remove a spike from the spike train and update the calcium trace.

    In check_only mode, returns the change in log-likelihood without modifying
    anything. In commit mode, uses swap-and-pop deletion (order doesn't matter
    since spikes are an unordered set) and subtracts the spike's kernel from
    the calcium trace.

    Parameters
    ----------
    spike_train : np.ndarray
        Current spike time array.
    n_spikes : int
        Current number of spikes.
    old_calcium : np.ndarray
        Current modeled calcium trace.
    old_ll : float
        Current log-likelihood.
    ef_h : np.ndarray
        Exponential filter kernel for fast-rise component.
    ef_d : np.ndarray
        Exponential filter kernel for slow-decay component.
    ef_nh : np.ndarray
        Cumulative squared norm of ef_h.
    ef_nd : np.ndarray
        Cumulative squared norm of ef_d.
    tau : array-like
        [tau_rise, tau_decay] in seconds.
    obs_calcium : np.ndarray
        Observed calcium trace.
    time_to_remove : float
        Spike time to remove, in frames.
    indx : int
        Index of the spike in spike_train.
    dt : float
        Frame duration in seconds.
    A : float
        Spike amplitude.
    check_only : bool
        If True, compute delta_ll only without modifying spike_train or old_calcium.

    Returns
    -------
    n_spikes : int
        Updated spike count (or old count if check_only).
    old_calcium : np.ndarray
        Updated calcium trace (unchanged if check_only).
    delta_ll or new_ll : float
        Change in log-likelihood if check_only, else new absolute log-likelihood.
    """

    tau_h, tau_d = tau[0], tau[1]

    wk_h = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_h)
    wk_d = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_d)
    t_floor = int(np.floor(time_to_remove))

    if check_only:

        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor

        dot_h = 0.0
        for k in range(tmp_len_h):
            obs_val = obs_calcium[t_floor + k]
            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                dot_h += res_val * (wk_h * ef_h[k])

        delta_ll_h = -2.0 * dot_h - (wk_h ** 2 * ef_nh[tmp_len_h - 1])

        end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
        tmp_len_d = end_idx_d - t_floor

        dot_d = 0.0
        for k in range(tmp_len_d):

            obs_val = obs_calcium[t_floor + k]

            if not np.isnan(obs_val):
                res_val = obs_val - old_calcium[t_floor + k]
                if k < tmp_len_h:
                    res_val += wk_h * ef_h[k]
                dot_d += res_val * (wk_d * ef_d[k])

        delta_ll_d = -2.0 * dot_d - (wk_d ** 2 * ef_nd[tmp_len_d - 1])

        return n_spikes, old_calcium, delta_ll_h + delta_ll_d

    # Swap-and-pop deletion: replace the target with the last spike, then
    # decrement count. Order doesn't matter -- spikes are an unordered set.
    spike_train[indx] = spike_train[n_spikes - 1]
    new_n_spikes = n_spikes - 1

    end_idx_d = min(len(ef_d) + t_floor, len(old_calcium))
    tmp_len_d = end_idx_d - t_floor

    obstemp = obs_calcium[t_floor:end_idx_d]
    old_ca_tmp = old_calcium[t_floor:end_idx_d]

    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + t_floor, len(old_calcium))
        tmp_len_h = end_idx_h - t_floor
        wef_h = (wk_h * ef_h[:tmp_len_h]).astype(np.float32)
        new_ca_tmp[:tmp_len_h] -= wef_h

    wef_d = (wk_d * ef_d[:tmp_len_d]).astype(np.float32)
    new_ca_tmp -= wef_d

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)
    new_ll = old_ll - sq_err_new + sq_err_old

    old_calcium[t_floor:end_idx_d] = new_ca_tmp

    return new_n_spikes, old_calcium, new_ll


@njit(cache=True, fastmath=True)
def replace_spike(spike_train, old_calcium, old_ll, ef_h, ef_d, ef_nh, ef_nd, tau,
                  obs_calcium, time_to_remove, indx, time_to_add, dt, A, check_only=False):
    """ Move a spike from time_to_remove to time_to_add in a single pass.

    Cheaper than calling remove then add because only the affected calcium
    window is touched once. Handles overlapping windows between old and new
    spike positions by shifting the exponential kernel appropriately.

    Parameters
    ----------
    spike_train : np.ndarray
        Current spike time array.
    old_calcium : np.ndarray
        Current modeled calcium trace.
    old_ll : float
        Current log-likelihood.
    ef_h : np.ndarray
        Exponential filter kernel for fast-rise component.
    ef_d : np.ndarray
        Exponential filter kernel for slow-decay component.
    ef_nh : np.ndarray
        Cumulative squared norm of ef_h.
    ef_nd : np.ndarray
        Cumulative squared norm of ef_d.
    tau : array-like
        [tau_rise, tau_decay] in seconds.
    obs_calcium : np.ndarray
        Observed calcium trace.
    time_to_remove : float
        Current spike time in frames.
    indx : int
        Index of the spike in spike_train.
    time_to_add : float
        Proposed new spike time in frames.
    dt : float
        Frame duration in seconds.
    A : float
        Spike amplitude.
    check_only : bool
        If True, return delta_ll only without modifying spike_train or old_calcium.

    Returns
    -------
    old_calcium : np.ndarray
        Updated calcium trace (unchanged if check_only).
    delta_ll or new_ll : float
        Change in log-likelihood if check_only, else new absolute log-likelihood.
    """

    tau_h, tau_d = tau[0], tau[1]

    wk_hr = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_h)
    wk_dr = A * np.exp((time_to_remove - dt * np.ceil(time_to_remove / dt)) / tau_d)
    t_rem_floor = int(np.floor(time_to_remove))

    wk_ha = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_h)
    wk_da = A * np.exp((time_to_add - dt * np.ceil(time_to_add / dt)) / tau_d)
    t_add_floor = int(np.floor(time_to_add))

    # Start window at whichever spike comes first in time.
    min_t = int(np.floor(min(time_to_remove, time_to_add)))

    end_idx_d = min(len(ef_d) + min_t, len(old_calcium))

    obstemp = obs_calcium[min_t:end_idx_d]
    old_ca_tmp = old_calcium[min_t:end_idx_d]

    new_ca_tmp = old_ca_tmp.copy()

    if np.any(ef_h):
        end_idx_h = min(len(ef_h) + min_t, len(old_calcium))
        tmp_len_h = end_idx_h - min_t

        if t_rem_floor == t_add_floor:
            wef_h = (wk_hr - wk_ha) * ef_h[:tmp_len_h]
        elif t_rem_floor > t_add_floor:
            diff_t = t_rem_floor - t_add_floor
            pad_len = min(tmp_len_h, diff_t)
            ef_len = max(0, tmp_len_h - pad_len)

            shifted_ef = np.empty(tmp_len_h, dtype=ef_h.dtype)
            shifted_ef[:pad_len] = 0.0
            shifted_ef[pad_len:] = ef_h[:ef_len]
            wef_h = wk_hr * shifted_ef - wk_ha * ef_h[:tmp_len_h]
        else:
            diff_t = t_add_floor - t_rem_floor
            pad_len = min(tmp_len_h, diff_t)
            ef_len = max(0, tmp_len_h - pad_len)

            shifted_ef = np.empty(tmp_len_h, dtype=ef_h.dtype)
            shifted_ef[:pad_len] = 0.0
            shifted_ef[pad_len:] = ef_h[:ef_len]
            wef_h = wk_hr * ef_h[:tmp_len_h] - wk_ha * shifted_ef

        new_ca_tmp[:tmp_len_h] -= wef_h.astype(np.float32)

    tmp_len_d = end_idx_d - min_t

    if t_rem_floor == t_add_floor:
        wef_d = (wk_dr - wk_da) * ef_d[:tmp_len_d]
    elif t_rem_floor > t_add_floor:
        diff_t = t_rem_floor - t_add_floor
        pad_len = min(tmp_len_d, diff_t)
        ef_len = max(0, tmp_len_d - pad_len)

        shifted_ef = np.empty(tmp_len_d, dtype=ef_d.dtype)
        shifted_ef[:pad_len] = 0.0
        shifted_ef[pad_len:] = ef_d[:ef_len]
        wef_d = wk_dr * shifted_ef - wk_da * ef_d[:tmp_len_d]
    else:
        diff_t = t_add_floor - t_rem_floor
        pad_len = min(tmp_len_d, diff_t)
        ef_len = max(0, tmp_len_d - pad_len)

        shifted_ef = np.empty(tmp_len_d, dtype=ef_d.dtype)
        shifted_ef[:pad_len] = 0.0
        shifted_ef[pad_len:] = ef_d[:ef_len]
        wef_d = wk_dr * ef_d[:tmp_len_d] - wk_da * shifted_ef

    new_ca_tmp -= wef_d.astype(np.float32)

    sq_err_new = np.sum((new_ca_tmp - obstemp)[~np.isnan(new_ca_tmp - obstemp)] ** 2)
    sq_err_old = np.sum((old_ca_tmp - obstemp)[~np.isnan(old_ca_tmp - obstemp)] ** 2)

    delta_ll = -sq_err_new + sq_err_old

    if check_only:
        return old_calcium, delta_ll
    else:
        old_calcium[min_t:end_idx_d] = new_ca_tmp
        spike_train[indx] = time_to_add
        return old_calcium, old_ll + delta_ll
