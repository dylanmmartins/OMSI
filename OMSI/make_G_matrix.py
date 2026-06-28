# -*- coding: utf-8 -*-
"""
OMSI/make_G_matrix.py

Builds the sparse AR Toeplitz matrix used to model calcium dynamics.

Functions
---------
make_G_matrix
    Create sparse Toeplitz G matrix encoding AR(p) dynamics.


DMM, Feb 2026
"""

import numpy as np
import scipy.sparse as sps


def make_G_matrix(T, g, segment_lengths=None):
    """ Create sparse Toeplitz G matrix encoding AR(p) dynamics.

    Each row encodes calcium[t] - g1*calcium[t-1] - g2*calcium[t-2] = spikes[t],
    so the diagonals are [-g2, -g1, 1] at offsets [-2, -1, 0] for AR(2).

    Parameters
    ----------
    T : int
        Number of time frames.
    g : array-like
        AR coefficients [g1, ...] or [g1, g2] for AR(1)/AR(2).
    segment_lengths : array-like or None
        If provided, zero out coupling entries between segments so AR dynamics
        don't leak across recording boundaries.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse G matrix of shape (T, T).
    """

    g = np.atleast_1d(g).flatten()

    if len(g) == 1 and g[0] < 0:
        g = np.array([0.0])

    p = len(g)

    offsets = np.arange(-p, 1)
    vals = np.append(-np.flip(g), 1.0)

    G = sps.diags(vals, offsets, shape=(T, T), format='lil')

    if segment_lengths is not None:
        # Zero out entries that would couple the last frame of one segment to
        # the first frame of the next -- AR dynamics shouldn't leak across boundaries.
        segment_lengths = np.atleast_1d(segment_lengths).flatten()
        sl = np.concatenate(([0], np.cumsum(segment_lengths)))

        for i in range(len(sl) - 1):

            row_idx = int(sl[i])
            col_idx = int(sl[i+1] - 1)

            if row_idx < T and col_idx < T:
                G[row_idx, col_idx] = 0.0

    return G.tocsr()
