# -*- coding: utf-8 -*-
"""
figures/run_pnev_MCMC.py

Run MCMC spike inference following Pnevmatikakis et al. via MATLAB subprocess.

Functions
---------
run_matlab_pnevMCMC
    Run MCMC inference on dF/F traces by calling MATLAB as a subprocess.


DMM, March 2026
"""


import os
import time
import platform
import subprocess
import numpy as np
import scipy.io
from pathlib import Path



def run_matlab_pnevMCMC(dff, fs=30.0, tau=0.5, n_sweeps=1000, true_spikes=None, sparsity_scale=0.001):
    """ Run MCMC spike inference via MATLAB subprocess.

    Parameters
    ----------
    dff : np.ndarray
        dF/F traces, shape (n_cells, n_frames) or (n_frames,).
    fs : float, optional
        Sampling rate in Hz.
    tau : float, optional
        Calcium indicator decay time constant in seconds.
    n_sweeps : int or str, optional
        Number of MCMC sweeps, or 'auto' to use 500.
    true_spikes : list of np.ndarray, optional
        Ground-truth spike times (unused, reserved for future use).
    sparsity_scale : float, optional
        Sparsity prior scale parameter.

    Returns
    -------
    final_spikes : list of np.ndarray
        Inferred spike times in seconds for each cell.
    model_traces : np.ndarray
        Reconstructed calcium traces, shape (n_cells, n_frames).
    all_probs : np.ndarray
        Posterior spike probability traces, shape (n_cells, n_frames).
    sweeps_per_cell : np.ndarray
        Number of MCMC sweeps run for each cell.
    """
    if dff.ndim == 1:
        dff = dff[np.newaxis, :]
    n_cells, n_frames = dff.shape

    if n_sweeps == 'auto':
        n_sweeps_val = 500
    else:
        n_sweeps_val = int(n_sweeps)

    input_mat = 'mcmc_input.mat'
    output_mat = 'mcmc_output.mat'
    wrapper_script = 'run_pnev_wrapper.m'

    scipy.io.savemat(input_mat, {
        'dff': dff.astype(np.float64),
        'fs': float(fs),
        'tau': float(tau),
        'n_sweeps': n_sweeps_val,
        'sparsity_scale': float(sparsity_scale)
    })


    matlab_code = f"""
    cvx_clear
    cvx_setup

    try
        addpath(genpath(pwd));
        if ispc
            addpath(genpath('C:\\Users\\dmartins\\Documents\\MATLAB\\cvx'));
            addpath(genpath('C:\\Users\\dmartins\\Documents\\Github\\spike-inference\\matlab'));
        elseif isunix
            addpath(genpath('/home/dylan/Documents/MATLAB/cvx'));
            addpath(genpath('/home/dylan/Documents/Github/CaImAn-MATLAB'));
        end

        load('{input_mat}');
        [n_cells, n_frames] = size(dff);

        params.Nsamples = n_sweeps;
        params.B = floor(n_sweeps / 2);
        params.p = 2;
        params.f = fs;

        all_spikes = cell(n_cells, 1);
        all_probs = zeros(n_cells, n_frames);
        model_traces = zeros(n_cells, n_frames);

        set(0, 'DefaultFigureVisible', 'off');
        fprintf('Running MCMC on %d cells...\\n', n_cells);

        for i = 1:n_cells

            y = double(dff(i, :))';

            try
                res = cont_ca_sampler(y, params);

                samples = res.ss;
                n_post = length(samples);

                prob_trace = zeros(1, n_frames);
                for s = 1:n_post
                    st = samples{{s}};
                    if ~isempty(st)
                        idx = round(st);
                        idx = idx(idx >= 1 & idx <= n_frames);
                        prob_trace(idx) = prob_trace(idx) + 1;
                    end
                end
                all_probs(i, :) = prob_trace / max(1, n_post);

                if ~isempty(samples)
                    all_spikes{{i}} = samples{{end}};
                end

                temp_trace = make_mean_sample(res, y);
                model_traces(i, :) = temp_trace(:)'; % Ensure row vector

            catch ME
                fprintf('Error on cell %d: %s\\n', i, ME.message);
            end
        end

        save('{output_mat}', 'all_spikes', 'all_probs', 'model_traces');
        exit(0);

    catch ME
        fprintf('Global MATLAB Error: %s\\n', ME.message);
        exit(1);
    end
    """

    with open(wrapper_script, 'w') as f:
        f.write(matlab_code)

    print('Calling MATLAB subprocess (sweeps={})...'.format(n_sweeps_val))
    t0 = time.time()

    if platform.system() == "Windows":
        matlab_exe = Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")
    elif platform.system() == "Linux":
        matlab_exe = '/usr/local/MATLAB/R2025b/bin/matlab'

    if not os.path.exists(matlab_exe):
        matlab_exe = 'matlab'

    cmd = f"\"{matlab_exe}\" -batch \"run_pnev_wrapper\""

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("MATLAB execution failed. Please ensure 'matlab' is in your PATH and 'cont_ca_sampler' is available.")

        return [np.array([]) for _ in range(n_cells)], np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells)

    print('MATLAB finished in {:.2f}s.'.format(time.time() - t0))

    if not os.path.exists(output_mat):
        print("Error: Output MAT file not found.")
        return [np.array([]) for _ in range(n_cells)], np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells)

    res = scipy.io.loadmat(output_mat)

    all_spikes_raw = res['all_spikes']
    all_probs = res['all_probs']
    model_traces = res['model_traces']

    final_spikes = []
    for i in range(n_cells):

        spks = all_spikes_raw[i][0]
        if spks.size > 0:

            times = spks.flatten() / fs
            final_spikes.append(times)
        else:
            final_spikes.append(np.array([]))

    sweeps_per_cell = np.full(n_cells, n_sweeps_val, dtype=np.int32)

    return final_spikes, model_traces, all_probs, sweeps_per_cell
