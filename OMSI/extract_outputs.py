# -*- coding: utf-8 -*-
"""
Extracts outputs from the MCMC sampler results.

Written Feb 2026, DMM
"""

import numpy as np

def extract_outputs(res):

    samples = res['ss']
    n_post = np.size(samples, 0)
    n_frames = np.size(samples, 1)

    # count how often each frame got a spike across all posterior samples to get a probablity trace
    prob_trace = np.zeros(n_frames)
    for st in samples:
        if len(st) > 0:

            idx = np.round(st).astype(int)

            idx = idx[(idx >= 0) & (idx < n_frames)]

            np.add.at(prob_trace, idx, 1)

    all_probs = prob_trace / max(1, n_post)

    # use last posterior sample as the point estimate for spike times
    all_spikes = samples[-1]

    if 'C_est' in res:
        temp_trace = np.atleast_1d(res['C_est']).flatten()
    elif 'Cb' in res:
        temp_trace = np.atleast_1d(res['Cb']).flatten()
    else:
        temp_trace = np.atleast_1d(y).flatten()

    # if the stored trace has the wrong length, interp onto the frame grid
    if len(temp_trace) != n_frames:

        old_x = np.linspace(0, n_frames - 1, len(temp_trace))
        new_x = np.arange(n_frames)

        model_traces = np.interp(new_x, old_x, temp_trace)
    else:
        model_traces = temp_trace

    return all_spikes, model_traces, all_probs


