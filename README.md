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
`spike_inference.npz` (or `spike_inference_planeN.npz` per plane, for
multi-plane suite2p sessions), with keys:
```
dFF          (n_cells, n_frames)    - dF/F input used for inference
Ca_trace     (n_cells, n_frames)    - MCMC-reconstructed calcium signal
prob_trace   (n_cells, n_frames)    - per-frame spike-probability trace
spike_train  (n_cells, n_frames)    - uint8, frame-resolved binary spike train
spike_times  (n_cells, max_spikes)  - spike times in seconds, NaN-padded
n_spikes     (n_cells,)             - number of detected spikes per cell
hz           scalar                 - frame rate used during inference
```
Note that `spike_times` is a NaN-padded 2-D array, not the variable-length
object array (`spikes`) returned in the in-memory dict above. Pass
`save_mat=True` to also write a `.mat` file with the same fields (`spike_times`
becomes a MATLAB cell array).

## Setting parameters

`deconv_from_array`, `deconv_from_suite2p`, and `deconv_from_caiman` all
accept an optional `params` dict to override the sampler's default
settings:

```python
import OMSI

params = {
    'spike_method': 'last',  # 'last' | 'map' | 'prob'
    'lam_scale':    0.002,   # sparsity prior scale
}
results = OMSI.deconv_from_array(
    dff=dFF, hz=30.0, params=params
)
```

Any key left out of `params` falls back to its default below. The frame
rate (`f`) is always taken from `hz` (or read automatically from
suite2p/CaImAn metadata) and should not be set inside `params` — it is
overwritten by `hz` regardless.

How the per-cell MCMC posterior samples are turned into a single reported
spike train.

| Parameter | Default | Options | Description |
|---|---|---|---|
| `spike_method` | `'last'` | `'last'`, `'map'`, `'prob'` | `'last'`: return the final post-burn-in posterior sample directly, mirroring CaImAn's `cont_ca_sampler` (recommended default). `'map'`: score every posterior sample against the posterior-mean probability trace (penalizing spike counts above the expected count) and keep the best-scoring sample. `'prob'` (or any other value): smooth the per-frame spike-probability trace and detect peaks, separating "real" spikes from noise peaks with Otsu thresholding. |
| `f` | — (sampler falls back to `10` if omitted) | float, Hz | Imaging frame rate. Set via the `hz` argument (or auto-read from suite2p/CaImAn metadata) — `deconv_from_array`/`deconv_from_suite2p`/`deconv_from_caiman` always overwrite `params['f']` with `hz`, so it doesn't need to be (and shouldn't be) set inside `params` directly. |
| `p` | `2` | `1`, `2` | Order of the autoregressive calcium kernel: `1` is a single decaying exponential (rise time ignored), `2` models both rise and decay. Ignored if `g` is given. |
| `g` | `None` | array-like of length `p`, or `None` | Explicit AR coefficients. `None` auto-estimates them per cell from the trace. |
| `defg` | `[0.6, 0.95]` | `[g_rise, g_decay]` | Fallback rise/decay poles used when the per-cell time-constant estimate is unstable (complex, negative, or explosive). |
| `TauStd` | `[0.2, 2]` | `[tau1_std, tau2_std]` (frames) | Standard deviations bounding the rise/decay time-constant MCMC proposal moves. |
| `upd_gam` | `1` | `0`, `1` | `1`: re-estimate (sample) the AR time constants during MCMC. `0`: hold them fixed at their initial estimate. Automatically disabled for fast indicators (decay < 0.6 s) where re-estimation is unstable. |
| `gam_step` | `1` | int | Number of sweeps between attempted time-constant updates (only relevant if `upd_gam=1`). |
| `sn` | `None` | float, or `None` | Noise standard deviation of the trace. `None` auto-estimates it (MAD of first differences, or PSD-based, depending on indicator speed). |
| `b` | `None` | float, or `None` | Baseline fluorescence offset. `None` auto-estimates it as the 8th percentile of the trace. |
| `bas_nonneg` | `0` | `0`, `1` | `1` constrains the estimated baseline to be non-negative. |
| `c1` | `None` | float, or `None` | Initial calcium concentration at t=0. `None` auto-estimates it. |
| `A_lb` | `None` | float, or `None` | Lower bound on spike amplitude during sampling. `None` uses the estimated noise level. |
| `b_lb` | `min(Y)` | float | Lower bound on the baseline during sampling. |
| `c1_lb` | `0` | float | Lower bound on `c1` during sampling. |
| `init` | `None` | dict, list of dicts, or `None` | Pre-computed initial sample (as returned by `get_init_sample`). `None` computes it automatically per cell. A list is split per-cell automatically when passed to `deconv()`. |
| `init_method` | `'nnls'` | `'nnls'`, `'foopsi'` | Algorithm used to build the initial spike-train guess. `'nnls'`: block coordinate-descent non-negative least-squares deconvolution. `'foopsi'`: L-BFGS-B regularized deconvolution (closer to the original FOOPSI algorithm). |
| `c`, `sp` | `None` | array, or `None` | Explicit initial calcium trace / spike train. Only used if `c`, `b`, `c1`, `g`, `sn`, and `sp` are *all* provided, in which case auto-initialization is skipped entirely. |
| `auto_stop` | `True` | `True`, `False` | `True` (recommended): use the automatic convergence-based stopping rule. `False`: run a fixed schedule of `Nsamples` + `B` sweeps. |
| `Nsamples` | `200` | int | Number of post-burn-in posterior samples to collect. Only used when `auto_stop=False`. |
| `B` | `75` | int | Number of burn-in sweeps. Only used when `auto_stop=False`. |
| `max_sweeps` | `2000` | int | Upper bound on total sweeps when `auto_stop=True`. |
| `min_sweeps` | `300` | int | Lower bound on total sweeps when `auto_stop=True`. |
| `burn_tol` | `0.005` | float | Convergence tolerance for ending burn-in. |
| `conv_tol` | `0.00067` | float | Convergence tolerance for ending sampling once the posterior estimate stabilizes. |
| `check_every` | `50` | int | How often (in sweeps) convergence is checked. |
| `std_move` | `3` | float | Standard deviation of the proposal distribution for spike-time add/move MCMC moves. |
| `add_move` | `ceil(T/500)` | int | Number of spike add/remove moves attempted per sweep. |
| `marg` | `0` | `0`, `1` | `0`: explicitly sample the spike amplitude/baseline/`c1` via Metropolis-Hastings each sweep (more exact, slower). `1`: analytically marginalize out the amplitude (faster). |
| `prec` | `1e-2` | float | Precision threshold used to auto-determine the effective length of the calcium kernel's support window when `T_supp` isn't set explicitly. |
| `T_supp` | `None` | int, or `None` | Explicit kernel support window length, in frames, overriding the automatic precision-based calculation. Useful for speeding up inference on slow indicators whose kernel otherwise spans many frames. |
| `con_lam` | `True` | `True`, `False` | `True`: keep the firing-rate prior fixed at its initial estimate across sweeps. `False`: re-estimate it empirically from each sweep's spike count. |
| `lam_scale` | `0.002` | float | Scale of the Poisson sparsity prior on firing rate. Smaller values favor sparser (fewer) inferred spikes. |
| `skip_snr` | `False` | `True`, `False` | `False` (default): skip inference and return zero spikes for traces whose SNR is below 2.0 (noise-dominated). `True`: always run inference regardless of SNR. |
| `print_flag` | `0` | `0`, `1` | Verbosity of the sampler's internal progress prints. |
