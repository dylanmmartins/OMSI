# -*- coding: utf-8 -*-
"""
fMCSI -- fast Markov chain Monte Carlo spike inference for calcium imaging data.

Continuous-time Markov chain Monte Carlo (MCMC) algorithm for inferring spike
times from dF/F traces recorded with calcium indicators.

If running from Suite2P:

    import fMCSI
    results = fMCSI.deconv_from_suite2p(
        '/path/to/suite2p/output'
    )

If running from CaImAn:

    import fMCSI
    results = fMCSI.deconv_from_caiman(
        '/path/to/caiman/results'
    )

If running on a numpy array of fluorescence (F) and
neuropil fluorescence (Fneu):

    import fMCSI
    results = fMCSI.deconv_from_array(
        f=F, fneu=Fneu, hz=30.0, outdir='/path/to/save'
    )

If running on a numpy array of dF/F (dff):

    import fMCSI
    results = fMCSI.deconv_from_array(
        dff=dFF
    )

If running from command line:

    python -m fMCSI.main --suite2p -dir /data/session -hz 30
    python -m fMCSI.main --caiman  -dir /data/session -hz 30 --mat
    python -m fMCSI.main --array   -dir /data/session -hz 30

    and add the flags

        -dir /path/to/data        for the data directory (required)
        -hz 30.0                  for the sample rate in Hz
        --outdir /path/to/save    to specify an output directory (defaults to input dir)
        --mat                     to save a .mat file in addition to the npz file
        --f-corr 0.7              to specify the neuropil correction coefficient (suite2p only, default: 0.7)
        --plane 0 1               to specify plane index/indices to process (suite2p only)
        --all-rois                to process all ROIs, including those not classified as cells by suite2p (suite2p only, default: False)

Each function returns a dict with keys:

    Ca_trace    (n_cells, n_frames)  - MCMC-reconstructed calcium signal
    prob_trace  (n_cells, n_frames)  - per-frame spike-probability trace
    spikes      (n_cells,) object    - per-cell spike times in seconds
    spike_train (n_cells, n_frames)  - binary, frame-resolved spike train

When `outdir` is provided, results are written into a numpy npz file,
"spike_inference.npz" containing the above arrays. The npz file  will
also contain n_spikes (number of spikes per cell), dFF (the input dF/F array),
and hz (the sampling rate). Pass `save_mat=True` to also write a .mat file.

Written Feb 2026, DMM
"""

__version__ = '1.0.0'

__all__ = [

    'deconv_from_suite2p',
    'deconv_from_caiman',
    'deconv_from_array',
    'deconv',

    'detect_spikes_from_probs',

    'spikes_to_calcium',
    'compute_accuracy_strict',
    'compute_cosmic',
    'compute_kurtosis',
]

from .deconv import (
    deconv_from_suite2p,
    deconv_from_caiman,
    deconv_from_array,
    deconv
)

from .helpers import (
    compute_accuracy_strict,
    compute_cosmic,
    compute_kurtosis,
    spikes_to_calcium,
    detect_spikes_from_probs,
)
