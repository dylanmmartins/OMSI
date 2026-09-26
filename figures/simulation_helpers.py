# -*- coding: utf-8 -*-
"""
figures/simulation_helpers.py

Helper functions for simulating populations of neurons.

Functions
---------
estimate_real_properties
    Extract SNR, kurtosis, and firing rate statistics from real Suite2p data.
generate_synthetic_data
    Generate synthetic calcium traces and ground-truth spike trains.


DMM, March 2026
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import OMSI

np.random.seed(3)


def estimate_real_properties(suite2p_dir):
    """ Extract SNR, kurtosis, and firing rate statistics from Suite2p data.

    Parameters
    ----------
    suite2p_dir : str
        Path to the Suite2p output directory.

    Returns
    -------
    snrs : np.ndarray
        Signal-to-noise ratios for each cell.
    kurtosis : np.ndarray
        Kurtosis values for each cell.
    rates : np.ndarray
        Estimated firing rates in Hz for each cell.
    """
    F = np.load(os.path.join(suite2p_dir, 'F.npy'))
    Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'))
    iscell = np.load(os.path.join(suite2p_dir, 'iscell.npy'))
    ops = np.load(os.path.join(suite2p_dir, 'ops.npy'), allow_pickle=True).item()
    fs_real = ops['fs']

    good_mask = iscell[:, 0] == 1
    F = F[good_mask]
    Fneu = Fneu[good_mask]

    F_corr = F - 0.7 * Fneu
    baselines = np.percentile(F_corr, 20, axis=1, keepdims=True)
    dff = (F_corr - baselines) / np.maximum(baselines, 1.0)

    n_cells_real, n_frames_real = dff.shape
    duration_real = n_frames_real / fs_real

    diff = np.diff(dff, axis=1)
    sigma = np.median(np.abs(diff), axis=1) / (0.6745 * np.sqrt(2))
    sigma = np.maximum(sigma, 1e-9)

    signal_peak = np.percentile(dff, 98, axis=1)
    snrs = signal_peak / sigma
    kurtosis = OMSI.compute_kurtosis(dff)

    rates = []
    for i in range(n_cells_real):
        thresh = 4.0 * sigma[i]
        peaks, _ = find_peaks(dff[i], height=thresh, distance=int(0.1*fs_real))
        rates.append(len(peaks) / duration_real)

    return snrs, kurtosis, np.array(rates)



def generate_synthetic_data(
        n_cells=400,
        fs=30.0,
        duration=20.0,
        tau=1.2,
        snr=None,
        use_real_data=False,
        target_kurtosis_range=(4.0, 50.0),
        suite2p_dir=None,
        amp_cv=0.0,
        tau_cv=0.0,
        drift_sd=0.0,
        drift_timescale=60.0,
        return_params=False
    ):
    """ Generate synthetic calcium traces and ground-truth spike trains.

    Parameters
    ----------
    n_cells : int, optional
        Number of cells to simulate.
    fs : float, optional
        Sampling rate in Hz.
    duration : float, optional
        Recording duration in seconds.
    tau : float, optional
        Calcium decay time constant in seconds.
    snr : float or array-like, optional
        Target signal-to-noise ratio. If None, kurtosis-based noise is used.
    use_real_data : bool, optional
        If True, draw firing rate and SNR distributions from real data.
    target_kurtosis_range : tuple, optional
        (min, max) kurtosis range for synthetic noise calibration.
    suite2p_dir : str, optional
        Suite2p directory, required when use_real_data is True.
    amp_cv : float, optional
        Coefficient of variation of per-spike amplitude, drawn lognormal with
        mean 1 (default 0: every spike has amplitude 1).
    tau_cv : float, optional
        Coefficient of variation of per-cell decay time constant, lognormal
        around tau (default 0: every cell uses tau).
    drift_sd : float, optional
        SD of a slow baseline drift, in units of a single-spike peak
        (default 0: baseline fixed at 0). Added after the noise level is set,
        so SNR keeps its meaning.
    drift_timescale : float, optional
        Timescale of the drift in seconds (Gaussian smoothing SD of white noise).
    return_params : bool, optional
        If True, also return a dict of the per-cell simulation parameters.

    Returns
    -------
    noisy_traces : np.ndarray
        Noisy dF/F traces, shape (n_cells, n_frames).
    true_spike_times : list of np.ndarray
        Ground-truth spike times in seconds for each cell.
    clean_traces : np.ndarray
        Noise-free calcium traces, shape (n_cells, n_frames).
    t : np.ndarray
        Time vector in seconds.
    firing_rates : np.ndarray
        Simulated firing rates in Hz for each cell.
    gen_kurtosis : np.ndarray
        Kurtosis of each noisy trace.
    params : dict
        Only if return_params: 'tau' (per-cell decay constants, s) and
        'drift_sd' (realized SD of each cell's drift).
    """
    n_frames = int(fs * duration)
    t = np.arange(n_frames) / fs

    upsample = 10
    n_high = n_frames * upsample
    fs_high = fs * upsample

    firing_rates = None

    if use_real_data:
        print('Estimating simulation parameters from real data in {}...'.format(suite2p_dir))
        real_snrs, real_kurtosis, real_rates = estimate_real_properties(suite2p_dir)
        if real_snrs is not None and real_rates is not None and real_kurtosis is not None:

            valid_mask = (real_snrs > 0) & (np.isfinite(real_snrs)) & (np.isfinite(real_rates))
            real_snrs = real_snrs[valid_mask]
            real_kurtosis = real_kurtosis[valid_mask]
            real_rates = real_rates[valid_mask]

            if len(real_snrs) > 0:

                idx_samples = np.random.choice(len(real_snrs), size=n_cells, replace=True)
                if snr is None:
                    snr = real_snrs[idx_samples]
                firing_rates = real_rates[idx_samples]
                print('  Using Real Data Props: Mean SNR={:.2f}, Mean Rate={:.2f}Hz'.format(np.mean(snr), np.mean(firing_rates)))
            else:
                print("  Warning: No valid properties extracted from real data. Using defaults.")
        else:
            print("  Warning: Failed to load real data. Using defaults.")

    if firing_rates is None:

        firing_rates = np.random.lognormal(mean=np.log(0.2), sigma=1.0, size=n_cells)
        firing_rates = np.clip(firing_rates, 0.01, 4.0)

    p_spike = firing_rates[:, None] / fs_high
    spikes_high = (np.random.rand(n_cells, n_high) < p_spike).astype(float)

    n_bursty = int(n_cells * 0.15)
    if n_bursty > 0:
        print('  Making {} cells bursty (adding spikes)...'.format(n_bursty))
        bursty_indices = np.random.choice(np.arange(n_cells), size=n_bursty, replace=False)
        for idx in bursty_indices:

            spike_locs = np.where(spikes_high[idx])[0]
            for t in spike_locs:

                if np.random.rand() < 0.6:

                    n_extra = np.random.randint(2, 6)
                    for k in range(1, n_extra + 1):

                        t_new = t + k * 2
                        if t_new < n_high:
                            spikes_high[idx, t_new] = 1.0

    true_spike_times = []

    print('Generating simulated data for {} cells...'.format(n_cells))
    for i in range(n_cells):
        true_spike_times.append(np.where(spikes_high[i])[0] / fs_high)

    if amp_cv > 0:
        # Lognormal with mean 1 and the requested CV; spike train entries become amplitudes.
        s2 = np.log(1.0 + amp_cv ** 2)
        amps = np.random.lognormal(mean=-s2 / 2.0, sigma=np.sqrt(s2), size=spikes_high.shape)
        spikes_high = spikes_high * amps

    dummy_snr = np.full(n_cells, 1000.0)
    if tau_cv > 0:
        s2 = np.log(1.0 + tau_cv ** 2)
        cell_tau = tau * np.random.lognormal(mean=-s2 / 2.0, sigma=np.sqrt(s2), size=n_cells)
        clean_traces = np.vstack([
            OMSI.spikes_to_calcium(spikes_high[i:i + 1], fs_high, fs, cell_tau[i],
                                   dummy_snr[i:i + 1])[1]
            for i in range(n_cells)])
    else:
        cell_tau = np.full(n_cells, float(tau))
        _, clean_traces = OMSI.spikes_to_calcium(spikes_high, fs_high, fs, tau, dummy_snr)

    noisy_traces = np.zeros_like(clean_traces)

    min_k, max_k = target_kurtosis_range

    scale = (max_k - min_k) / 3.0
    target_kurtosis = min_k + np.random.exponential(scale=scale, size=n_cells)
    target_kurtosis = np.clip(target_kurtosis, min_k, max_k)

    actual_snrs = []

    for i in range(n_cells):
        trace = clean_traces[i]
        trace_centered = trace - np.mean(trace)

        m2 = np.mean(trace_centered**2)
        m4 = np.mean(trace_centered**4)

        peak_signal = np.percentile(trace, 99) - np.percentile(trace, 1)

        if m2 < 1e-9:

            sigma = 1.0
            noisy_traces[i] = np.random.normal(0, sigma, size=len(trace))
            actual_snrs.append(0.0)
            continue

        if snr is not None:

            target_snr_val = snr[i] if isinstance(snr, (list, np.ndarray)) else snr
            sigma = peak_signal / target_snr_val
            noise = np.random.normal(0, sigma, size=len(trace))
            noisy_traces[i] = trace + noise
            actual_snrs.append(target_snr_val)
            continue

        k_clean = (m4 / (m2**2)) - 3.0

        if k_clean < target_kurtosis[i]:
            sigma = np.sqrt(m2) / 20.0
        else:

            k_tgt = min(target_kurtosis[i], k_clean * 0.99)
            k_tgt = max(k_tgt, 0.1)

            v = m2 * (np.sqrt(k_clean / k_tgt) - 1)
            sigma = np.sqrt(v)

        noise = np.random.normal(0, sigma, size=len(trace))
        noisy_traces[i] = trace + noise
        actual_snrs.append(peak_signal / sigma if sigma > 1e-9 else 100.0)

    drift_real = np.zeros(n_cells)
    if drift_sd > 0:
        # Slow baseline drift: smoothed white noise scaled to drift_sd single-spike
        # peaks. Added after the noise level was set from the spike-only trace.
        for i in range(n_cells):
            d = gaussian_filter1d(np.random.normal(size=n_frames), drift_timescale * fs,
                                  mode='reflect')
            d = (d - d.mean()) / (d.std() + 1e-12) * drift_sd
            clean_traces[i] += d
            noisy_traces[i] += d
            drift_real[i] = d.std()

    gen_kurtosis = OMSI.compute_kurtosis(noisy_traces)

    if return_params:
        return (noisy_traces, true_spike_times, clean_traces, t, firing_rates, gen_kurtosis,
                {'tau': cell_tau, 'drift_sd': drift_real})
    return noisy_traces, true_spike_times, clean_traces, t, firing_rates, gen_kurtosis
