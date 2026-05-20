# -*- coding: utf-8 -*-
"""
Creates the sparse Toeplitz matrix that models the AR dynamics.

Written Feb 2026, DMM
"""



import numpy as np
import scipy.sparse as sps

def make_G_matrix(T, g, segment_lengths=None):

    g = np.atleast_1d(g).flatten()

    if len(g) == 1 and g[0] < 0:
        g = np.array([0.0])

    p = len(g)

    # each row of G encodes: calcium[t] - g1*calcium[t-1] - g2*calcium[t-2] = spikes[t]
    # so the diagonals are [-g2, -g1, 1] at offsets [-2, -1, 0]
    offsets = np.arange(-p, 1)
    vals = np.append(-np.flip(g), 1.0)

    G = sps.diags(vals, offsets, shape=(T, T), format='lil')

    if segment_lengths is not None:
        # zero out the entries that would couple the last frame of one segment
        # to the first frame of the next, so ar dynamics dont leak across boundaries
        segment_lengths = np.atleast_1d(segment_lengths).flatten()
        sl = np.concatenate(([0], np.cumsum(segment_lengths)))

        for i in range(len(sl) - 1):

            row_idx = int(sl[i])
            col_idx = int(sl[i+1] - 1)

            if row_idx < T and col_idx < T:
                G[row_idx, col_idx] = 0.0

    return G.tocsr()

