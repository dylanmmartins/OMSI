# -*- coding: utf-8 -*-
"""
OMSI/extract_outputs.py

Extract spike times, calcium trace, and probability trace from sampler output.

Functions
---------
extract_outputs
    Unpack a results dict into spikes, calcium trace, and probability trace.


DMM, Feb 2026
"""

import numpy as np


def extract_outputs(res):
    """ Unpack a results dict into spikes, calcium trace, and probability trace.

    Parameters
    ----------
    res : dict
        Output of cont_ca_sampler. Must contain 'ss' (list of spike time arrays)
        and at least one of 'C_est', 'Cb', or 'y' for the calcium trace.

    Returns
    -------
    all_spikes : np.ndarray
        Spike times from last posterior sample, used as point estimate.
    model_traces : np.ndarray
        Calcium trace, interpolated to n_frames if stored at a different length.
    all_probs : np.ndarray
        Per-frame spike probability: fraction of posterior samples with a spike
        in each frame.
    """

    samples = res['ss']
    n_post = np.size(samples, 0)
    n_frames = np.size(samples, 1)

    # Count how often each frame got a spike across all posterior samples,
    # then normalize by sample count to get per-frame probability.
    prob_trace = np.zeros(n_frames)
    for st in samples:
        if len(st) > 0:
            idx = np.round(st).astype(int)
            idx = idx[(idx >= 0) & (idx < n_frames)]
            np.add.at(prob_trace, idx, 1)

    all_probs = prob_trace / max(1, n_post)

    # Last posterior sample as point estimate for spike times.
    all_spikes = samples[-1]

    if 'C_est' in res:
        temp_trace = np.atleast_1d(res['C_est']).flatten()
    elif 'Cb' in res:
        temp_trace = np.atleast_1d(res['Cb']).flatten()
    else:
        temp_trace = np.atleast_1d(y).flatten()

    # If stored trace has wrong length, interp onto frame grid.
    if len(temp_trace) != n_frames:
        old_x = np.linspace(0, n_frames - 1, len(temp_trace))
        new_x = np.arange(n_frames)
        model_traces = np.interp(new_x, old_x, temp_trace)
    else:
        model_traces = temp_trace

    return all_spikes, model_traces, all_probs
