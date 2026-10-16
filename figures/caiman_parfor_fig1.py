# -*- coding: utf-8 -*-
"""
figures/caiman_parfor_fig1.py

CaImAn MCMC on figure 1's population with cells spread across a parfor pool.

October 2026
"""

import argparse
import glob
import os
import subprocess
import tempfile
import time

import numpy as np
import scipy.io

import OMSI
import run_pnev_MCMC as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, 'data', 'fig1_10hz')
_SOURCE_NPZ = 'fixed_benchmark_MATLAB.npz'
_OUTPUT_NPZ = 'fixed_benchmark_MATLAB_parfor.npz'

_PARFOR_TEMPLATE = r"""
try
    cd('__WORK_DIR__');

    cvx_root = '__CVX_ROOT__';
    caiman_root = '__CAIMAN_ROOT__';
    if ~isempty(cvx_root) && exist(cvx_root, 'dir') == 7
        addpath(genpath(cvx_root));
    end
    if ~isempty(caiman_root) && exist(caiman_root, 'dir') == 7
        addpath(genpath(caiman_root));
    end

    % Statistics Toolbox stand-ins, appended so a real installation wins.
    addpath('__SHIM_DIR__', '-end');

    if exist('cvx_setup', 'file') == 2 && isempty(which('cvx_begin'))
        try
            cvx_setup;
        catch setup_err
            fprintf('cvx_setup failed: %s\n', setup_err.message);
        end
    end
    if isempty(which('cont_ca_sampler'))
        error('cont_ca_sampler not found on the MATLAB path. Set CAIMAN_MATLAB_HOME.');
    end
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('Parallel Computing Toolbox is not licensed; parfor would run serially.');
    end

    load('__INPUT_MAT__');
    [n_cells, n_frames] = size(dff);

    params.Nsamples = n_sweeps;
    params.B = floor(n_sweeps / 2);
    params.p = 2;
    params.f = fs;

    all_spikes = cell(n_cells, 1);
    all_probs = zeros(n_cells, n_frames);
    model_traces = zeros(n_cells, n_frames);
    cell_times = nan(n_cells, 1);

    set(0, 'DefaultFigureVisible', 'off');

    % Fresh pool so start-up is timed and the worker count is the one asked for.
    % The client path set above is passed on to the workers.
    delete(gcp('nocreate'));
    t_pool = tic;
    if n_workers > 0
        pool = parpool('Processes', n_workers);
    else
        pool = parpool('Processes');
    end
    pool_time = toc(t_pool);
    worker_threads = fetchOutputs(parfevalOnAll(pool, @maxNumCompThreads, 1));
    fprintf('POOL workers=%d start=%.2fs threads/worker=%s\n', ...
            pool.NumWorkers, pool_time, mat2str(worker_threads(:)'));

    % cont_ca_sampler initialises with constrained_foopsi, whose default
    % method is CVX, as in the serial run. Make sure every worker sees the
    % same CVX and solver as the client instead of failing cell by cell.
    worker_cvx = fetchOutputs(parfevalOnAll(pool, @() which('cvx_begin'), 1), 'UniformOutput', false);
    if any(cellfun(@isempty, worker_cvx))
        error('cvx_begin not on the path of every worker.');
    end
    fprintf('CVX client: %s  solver=%s\n', which('cvx_begin'), cvx_solver);
    worker_solver = fetchOutputs(parfevalOnAll(pool, @cvx_solver, 1), 'UniformOutput', false);
    fprintf('CVX workers: %s  solvers=%s\n', worker_cvx{1}, strjoin(unique(worker_solver), ','));

    fprintf('Running MCMC on %d cells with parfor...\n', n_cells);
    t_loop = tic;
    parfor i = 1:n_cells

        y = double(dff(i, :))';

        try
            t_cell = tic;
            res = cont_ca_sampler(y, params);
            cell_times(i) = toc(t_cell);

            samples = res.ss;
            n_post = length(samples);

            prob_trace = zeros(1, n_frames);
            for s = 1:n_post
                st = samples{s};
                if ~isempty(st)
                    idx = round(st);
                    idx = idx(idx >= 1 & idx <= n_frames);
                    prob_trace(idx) = prob_trace(idx) + 1;
                end
            end
            all_probs(i, :) = prob_trace / max(1, n_post);

            if ~isempty(samples)
                all_spikes{i} = samples{end};
            end

            temp_trace = make_mean_sample(res, y);
            model_traces(i, :) = temp_trace(:)';

        catch ME
            fprintf('Error on cell %d: %s\n', i, ME.message);
        end
    end
    loop_time = toc(t_loop);
    n_workers_used = pool.NumWorkers;
    fprintf('TIMING parfor=%.2fs sum_per_cell=%.2fs failed_cells=%d\n', ...
            loop_time, sum(cell_times, 'omitnan'), sum(isnan(cell_times)));

    delete(pool);
    save('__OUTPUT_MAT__', 'all_spikes', 'all_probs', 'model_traces', 'cell_times', ...
         'pool_time', 'loop_time', 'n_workers_used', 'worker_threads');
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\n', ME.message);
    exit(1);
end
"""


def _has_parallel_toolbox(exe):

    root = os.path.dirname(os.path.dirname(os.path.abspath(exe)))
    return os.path.isdir(os.path.join(root, 'toolbox', 'parallel'))


def _find_matlab_with_pct(explicit=None):

    exe = R.find_matlab(explicit)
    if exe is None or explicit or _has_parallel_toolbox(exe):
        return exe
    candidates = glob.glob(r'C:\Program Files\MATLAB\R*\bin\matlab.exe') + \
                 glob.glob('/usr/local/MATLAB/R*/bin/matlab') + \
                 glob.glob('/Applications/MATLAB_R*.app/bin/matlab')
    candidates = sorted((c for c in candidates if _has_parallel_toolbox(c)),
                        key=R._release_sort_key, reverse=True)
    if candidates:
        print('{} has no Parallel Computing Toolbox; using {} instead.'.format(exe, candidates[0]))
        return candidates[0]
    print('Warning: no MATLAB with the Parallel Computing Toolbox found; using {}.'.format(exe))
    return exe


def _write_wrapper(path, work_dir, input_mat, output_mat, cvx_root, caiman_root, shim_dir):


    def _posix(p):
        return p.replace('\\', '/').replace("'", "''")

    code = _PARFOR_TEMPLATE
    for key, val in [('__WORK_DIR__', work_dir), ('__INPUT_MAT__', input_mat),
                     ('__OUTPUT_MAT__', output_mat), ('__CVX_ROOT__', cvx_root),
                     ('__CAIMAN_ROOT__', caiman_root), ('__SHIM_DIR__', shim_dir)]:
        code = code.replace(key, _posix(val))
    with open(path, 'w') as fh:
        fh.write(code)


def run(data_dir=_DEFAULT_DATA_DIR, n_workers=0, n_cells=None, n_sweeps=500,
        matlab_exe=None):

    src = np.load(os.path.join(data_dir, _SOURCE_NPZ), allow_pickle=True)
    dff = np.asarray(src['noisy_traces'], dtype=np.float64)
    true_spikes = list(src['true_spikes'])
    fs = float(src['f'])
    serial_time = float(src['tradmat_time'])
    serial_cells = dff.shape[0]
    if n_cells is not None:
        dff = dff[:n_cells]
        true_spikes = true_spikes[:n_cells]
    n_cells = dff.shape[0]
    print('Loaded {} cells x {} frames at {:g} Hz from {}.'.format(
        n_cells, dff.shape[1], fs, os.path.join(data_dir, _SOURCE_NPZ)))

    exe = _find_matlab_with_pct(matlab_exe)
    if exe is None:
        raise RuntimeError('MATLAB not found. Put it on PATH or set MATLAB_EXECUTABLE.')
    cvx_root = R.find_toolbox(R._CVX_ENV, 'cvx')
    caiman_root = R.find_toolbox(R._CAIMAN_ENV, 'CaImAn-MATLAB',
                                 extra_candidates=[
                                     os.path.join(os.path.expanduser('~'), 'Documents',
                                                  'Github', 'spike-inference', 'matlab'),
                                 ])
    print('MATLAB executable: {}'.format(exe))
    print('  cvx: {}'.format(cvx_root or 'not found'))
    print('  CaImAn-MATLAB: {}'.format(caiman_root or 'not found'))

    work_dir = os.path.join(data_dir, 'caiman_parfor')
    os.makedirs(work_dir, exist_ok=True)
    input_mat = os.path.join(work_dir, 'mcmc_input.mat')
    output_mat = os.path.join(work_dir, 'mcmc_output.mat')
    script_stem = 'run_pnev_parfor_wrapper'

    scipy.io.savemat(input_mat, {'dff': dff, 'fs': fs, 'n_sweeps': int(n_sweeps),
                                 'n_workers': int(n_workers)})
    shim_dir = R._write_shims(os.path.join(tempfile.gettempdir(), 'omsi_matlab_shims'))
    _write_wrapper(os.path.join(work_dir, script_stem + '.m'), work_dir, input_mat,
                   output_mat, cvx_root, caiman_root, shim_dir)
    if os.path.exists(output_mat):
        os.remove(output_mat)

    print('Calling MATLAB (parfor, sweeps={}, workers={})...'.format(
        n_sweeps, n_workers or 'default'))
    t0 = time.time()
    subprocess.run([exe] + R._matlab_invocations(script_stem, exe)[0], cwd=work_dir, check=True)
    total_time = time.time() - t0

    res = scipy.io.loadmat(output_mat)
    spikes = []
    for i in range(n_cells):
        spks = res['all_spikes'][i][0]
        spikes.append(np.clip(spks.flatten() - 1.0, 0.0, None) / fs if spks.size > 0
                      else np.array([]))
    cell_times = np.asarray(res['cell_times'], dtype=float).ravel()
    pool_time = float(res['pool_time'])
    loop_time = float(res['loop_time'])
    n_used = int(res['n_workers_used'])

    prec, rec, f1 = OMSI.compute_accuracy_strict(true_spikes, spikes)

    print('\n' + '=' * 60)
    print('CaImAn MCMC with parfor: {} cells, {} workers'.format(n_cells, n_used))
    print('=' * 60)
    print('  Total wall time (Python, incl. MATLAB start-up): {:9.1f} s  ({:.2f} min)'.format(
        total_time, total_time / 60))
    print('  Pool start-up:                                   {:9.1f} s'.format(pool_time))
    print('  parfor loop:                                     {:9.1f} s'.format(loop_time))
    print('  Sum of per-cell sampler time:                    {:9.1f} s'.format(np.nansum(cell_times)))
    print('  Time per cell (total / cells):                   {:9.3f} s'.format(total_time / n_cells))
    print('  Failed cells:                                    {:9d}'.format(int(np.isnan(cell_times).sum())))
    serial_scaled = serial_time * n_cells / serial_cells
    print('  Figure 1 serial run (scaled to {} cells):       {:9.1f} s  -> speed-up {:.2f}x'.format(
        n_cells, serial_scaled, serial_scaled / total_time))
    print('  Precision {:.3f} ± {:.3f}   Recall {:.3f} ± {:.3f}  (median ± MAD)'.format(
        np.nanmedian(prec), np.nanmedian(np.abs(prec - np.nanmedian(prec))),
        np.nanmedian(rec),  np.nanmedian(np.abs(rec - np.nanmedian(rec)))))

    out = os.path.join(data_dir, _OUTPUT_NPZ)
    np.savez(out, tradmat_spikes=np.array(spikes, dtype=object),
             tradmat_traces=res['model_traces'], tradmat_probs=res['all_probs'],
             tradmat_time=total_time, tradmat_times_per_cell=cell_times,
             tradmat_precision=prec, tradmat_recall=rec, tradmat_F1=f1,
             pool_time=pool_time, loop_time=loop_time, n_workers=n_used, n_cells=n_cells)
    print('Saved to {}.'.format(out))


def main():

    parser = argparse.ArgumentParser(
        description='CaImAn MCMC with parfor on figure 1\'s population')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Figure 1 results directory (default: data/fig1_10hz)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parfor pool size (default 0: MATLAB default, one per physical core)')
    parser.add_argument('--n-cells', type=int, default=None,
                        help='Only run the first N cells (quick check)')
    parser.add_argument('--sweeps', type=int, default=500,
                        help='MCMC sweeps per cell (default 500, as in figure 1)')
    parser.add_argument('--matlab', default=None,
                        help='MATLAB executable (default: newest with the Parallel Computing Toolbox)')
    args = parser.parse_args()
    run(data_dir=args.data_dir, n_workers=args.workers, n_cells=args.n_cells,
        n_sweeps=args.sweeps, matlab_exe=args.matlab)


if __name__ == '__main__':
    main()
