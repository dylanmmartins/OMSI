# Optimized Markov chain Monte Carlo spike inference

Optimized continuous-time Markov chain Monte Carlo (MCMC) algorithm for inferring spike times from dF/F traces recorded with calcium indicators. On standard lab hardware, this method can analyze a 20-minute 30 Hz recording with 500 cells in ~5 minutes, compared to 2.5+ hours with existing Matlab implementations of MCMC spike inference.

## Installation

Requires Python 3.9 or later.

### Install with conda

```bash
git clone https://github.com/dylanmmartins/OMSI.git
cd OMSI
conda env create -f environment.yml
conda activate spikeinf
pip install -e .
```

### Install with pip only

```bash
git clone https://github.com/dylanmmartins/OMSI.git
cd OMSI
pip install -r requirements.txt
pip install -e .
```

Both options install the package in editable mode so that updates pulled with `git pull` take effect immediately without reinstalling.

## Usage from scripts

If running from Suite2P:
```
import OMSI
results = OMSI.deconv_from_suite2p(
    '/path/to/suite2p/output'
)
```
If running from CaImAn:
```
import OMSI
results = OMSI.deconv_from_caiman(
    '/path/to/caiman/results'
)
```
If running on a numpy array of fluorescence (`F`) and
neuropil fluorescence (`Fneu`):
```
import OMSI
results = OMSI.deconv_from_array(
    f=F, fneu=Fneu, hz=30.0, outdir='/path/to/save'
)
```
To maximize accuracy, if you have access to the `F` and `Fneu` arrays, you should always use those when computing spike times from an array. If you only have dFF, you can call it running on a numpy array of dF/F (`dFF`):
```
import OMSI
results = OMSI.deconv_from_array(
    dff=dFF
)
```
## Usage from command line
```
python -m OMSI.deconv --suite2p -dir /data/session
python -m OMSI.deconv --caiman  -dir /data/session
python -m OMSI.deconv --array   -dir /data/session -hz 30
```
and add the optional flags
```
-dir /path/to/data        for the data directory (required)
-hz 30.0                  for the sample rate in Hz
--outdir /path/to/save    to specify an output directory (defaults to input dir)
--mat                     to save a .mat file in addition to the npz file
--f-corr 0.7              to specify the neuropil correction coefficient (suite2p only, default: 0.7)
--plane 0 1               to specify plane index/indices to process (suite2p only)
--all-rois                to process all ROIs, including those not classified as cells by suite2p (suite2p only, default: False)
```
You don't need to use the `-Hz` flag if you are running from Suite2p or CaImAn because the sample rate will be identified in the metadata saved out by that method.

## Resulting data

Each function returns a dict with keys:
```
Ca_trace    (n_cells, n_frames)  - MCMC-reconstructed calcium signal
prob_trace  (n_cells, n_frames)  - per-frame spike-probability trace
spikes      (n_cells,) object    - per-cell spike times in seconds
spike_train (n_cells, n_frames)  - binary, frame-resolved spike train
```

When `outdir` is provided, results are written into a numpy npz file,
"spike_inference.npz" containing the above arrays. The npz file  will
also contain n_spikes (number of spikes per cell), dFF (the input dF/F array),
and hz (the sampling rate). Pass `save_mat=True` to also write a .mat file.

