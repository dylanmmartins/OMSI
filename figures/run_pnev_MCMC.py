# -*- coding: utf-8 -*-
"""
figures/run_pnev_MCMC.py

Run MCMC spike inference following Pnevmatikakis et al. via MATLAB subprocess.

MATLAB location and toolbox paths are discovered at runtime rather than
hardcoded, so the same call works on Linux, macOS, and Windows across MATLAB
releases. Override any of it with the MATLAB_EXECUTABLE, CVX_HOME, and
CAIMAN_MATLAB_HOME environment variables.

Functions
---------
find_matlab
    Locate a MATLAB executable, newest release first.
find_toolbox
    Locate a MATLAB toolbox directory from env var or common install paths.
_matlab_invocations
    Argument lists to try, modern batch mode first.
_write_shims
    Write base-MATLAB stand-ins for the Statistics Toolbox functions used.
_write_wrapper
    Fill the MATLAB wrapper template and write it next to the input file.
run_matlab_pnevMCMC
    Run MCMC inference on dF/F traces by calling MATLAB as a subprocess.


DMM, March 2026
"""


import glob
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import scipy.io

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_MATLAB_ENV = 'MATLAB_EXECUTABLE'
_CVX_ENV    = 'CVX_HOME'
_CAIMAN_ENV = 'CAIMAN_MATLAB_HOME'


_WRAPPER_TEMPLATE = """
try
    __MAX_COMP_THREADS_LINE__
    cd('__WORK_DIR__');
    addpath(genpath(pwd));

    % Toolbox roots resolved on the Python side; empty when not found.
    cvx_root = '__CVX_ROOT__';
    caiman_root = '__CAIMAN_ROOT__';

    if ~isempty(cvx_root) && exist(cvx_root, 'dir') == 7
        addpath(genpath(cvx_root));
    end
    if ~isempty(caiman_root) && exist(caiman_root, 'dir') == 7
        addpath(genpath(caiman_root));
    end

    % Statistics Toolbox stand-ins, appended so a real installation wins.
    % Kept outside the genpath tree above so it cannot jump the queue.
    addpath('__SHIM_DIR__', '-end');
    shimmed = {};
    for k = {'range', 'prctile', 'gamrnd', 'normrnd'}
        w = which(k{1});
        if ~isempty(strfind(w, '__SHIM_DIR__'))
            shimmed{end+1} = k{1};
        end
    end
    if ~isempty(shimmed)
        fprintf('Statistics Toolbox not available for: %s -- using built-in stand-ins.\\n', ...
                strjoin(shimmed, ', '));
    end

    % cvx_setup is slow and only needed once per session, so run it only when
    % cvx is present but not yet initialised.
    if exist('cvx_setup', 'file') == 2 && isempty(which('cvx_begin'))
        try
            cvx_setup;
        catch setup_err
            fprintf('cvx_setup failed: %s\\n', setup_err.message);
        end
    end

    if isempty(which('cont_ca_sampler'))
        error('cont_ca_sampler not found on the MATLAB path. Set CAIMAN_MATLAB_HOME.');
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
            fprintf('Error on cell %d: %s\\n', i, ME.message);
        end
    end

    save('__OUTPUT_MAT__', 'all_spikes', 'all_probs', 'model_traces');
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\\n', ME.message);
    exit(1);
end
"""


def _release_sort_key(path):

    match = re.search(r'R(\d{4})([ab])', path)
    if not match:
        return (0, '')
    return (int(match.group(1)), match.group(2))


def find_matlab(explicit=None):

    if explicit and os.path.exists(explicit):
        return explicit

    from_env = os.environ.get(_MATLAB_ENV, '').strip()
    if from_env and os.path.exists(from_env):
        return from_env

    on_path = shutil.which('matlab')
    if on_path:
        return on_path

    system = platform.system()
    if system == 'Windows':
        patterns = [
            r'C:\Program Files\MATLAB\R*\bin\matlab.exe',
            r'C:\Program Files (x86)\MATLAB\R*\bin\matlab.exe',
        ]
    elif system == 'Darwin':
        patterns = ['/Applications/MATLAB_R*.app/bin/matlab']
    else:
        patterns = [
            '/usr/local/MATLAB/R*/bin/matlab',
            '/opt/MATLAB/R*/bin/matlab',
            os.path.expanduser('~/MATLAB/R*/bin/matlab'),
        ]

    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))

    if not found:
        return None

    found.sort(key=_release_sort_key, reverse=True)
    return found[0]


def find_toolbox(env_var, folder_name, extra_candidates=()):

    from_env = os.environ.get(env_var, '').strip()
    if from_env and os.path.isdir(from_env):
        return os.path.abspath(from_env)

    home = os.path.expanduser('~')

    parents = [
        os.path.join(home, 'Documents', 'MATLAB'),
        os.path.join(home, 'Documents'),
        os.path.join(home, 'Documents', 'Github'),
        os.path.join(home, 'Documents', 'GitHub'),
        os.path.join(home, 'OneDrive', 'Documents'),
        os.path.join(home, 'OneDrive', 'Documents', 'MATLAB'),
        os.path.join(home, 'Desktop'),
        os.path.join(home, 'MATLAB'),
        os.path.join(home, 'Github'),
        os.path.join(home, 'GitHub'),
        home,
        os.path.dirname(_THIS_DIR),
        os.path.dirname(os.path.dirname(_THIS_DIR)),
        '/opt',
        '/usr/local',
    ]

    candidates = [os.path.join(p, folder_name) for p in parents]
    candidates += list(extra_candidates)

    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return ''


def _matlab_invocations(script_stem, exe=None):

    batch = ['-batch', script_stem]

    match = re.search(r'R(\d{4})([ab])', exe or '')
    if match and (int(match.group(1)), match.group(2)) >= (2019, 'a'):
        return [batch]

    legacy = ("try, {}; catch err, fprintf('%s\\n', err.message); "
              "exit(1); end, exit(0)".format(script_stem))

    if platform.system() == 'Windows':
        return [batch, ['-wait', '-nosplash', '-nodesktop', '-r', legacy]]
    return [batch, ['-nodisplay', '-nosplash', '-nodesktop', '-r', legacy]]


_SHIMS = {

    'range': """function r = range(x, dim)
%RANGE Sample range, max minus min. Stand-in for the Statistics Toolbox.
if nargin < 2
    if isvector(x)
        r = max(x(:)) - min(x(:));
        return;
    end
    dim = find(size(x) ~= 1, 1);
    if isempty(dim)
        dim = 1;
    end
end
r = max(x, [], dim) - min(x, [], dim);
end
""",

    'normrnd': """function r = normrnd(mu, sigma, varargin)
%NORMRND Normal random numbers. Stand-in for the Statistics Toolbox.
if isempty(varargin)
    sz = size(mu .* sigma);
elseif numel(varargin) == 1
    sz = varargin{1};
    if isscalar(sz)
        sz = [sz sz];
    end
else
    sz = cell2mat(varargin);
end
r = mu + sigma .* randn(sz);
end
""",

    'prctile': """function y = prctile(x, p, dim)
%PRCTILE Percentiles by linear interpolation on midpoint positions.
%   Stand-in for the Statistics Toolbox. NaNs are ignored, as there.
if nargin < 3
    if isvector(x)
        x = x(:);
    end
    dim = 1;
end
if dim ~= 1
    perm = 1:max(ndims(x), dim);
    perm([1 dim]) = perm([dim 1]);
    y = permute(prctile(permute(x, perm), p, 1), perm);
    return;
end

sz = size(x);
cols = reshape(x, sz(1), []);
p = p(:);
y = zeros(numel(p), size(cols, 2));

for k = 1:size(cols, 2)
    col = sort(cols(~isnan(cols(:, k)), k));
    n = numel(col);
    if n == 0
        y(:, k) = NaN;
    elseif n == 1
        y(:, k) = col;
    else
        q = 100 * ((1:n)' - 0.5) / n;
        y(:, k) = interp1(q, col, p, 'linear');
        y(p <= q(1), k) = col(1);
        y(p >= q(end), k) = col(end);
    end
end

y = reshape(y, [numel(p), sz(2:end)]);
if numel(p) == 1 && numel(sz) == 2 && sz(2) == 1
    y = y(1);
end
end
""",

    'gamrnd': """function r = gamrnd(a, b, varargin)
%GAMRND Gamma random numbers, shape a and scale b, by Marsaglia-Tsang.
%   Stand-in for the Statistics Toolbox. Draws differ from that version
%   for a given seed, so use it for timing, not for reproducing samples.
if isempty(varargin)
    sz = size(a .* b);
elseif numel(varargin) == 1
    sz = varargin{1};
    if isscalar(sz)
        sz = [sz sz];
    end
else
    sz = cell2mat(varargin);
end

a = a .* ones(sz);
b = b .* ones(sz);
r = zeros(sz);
for i = 1:numel(r)
    r(i) = local_gamma(a(i)) * b(i);
end
end

function g = local_gamma(a)
if a < 1
    g = local_gamma(1 + a) * rand() ^ (1 / a);
    return;
end
d = a - 1/3;
c = 1 / sqrt(9 * d);
while true
    x = randn();
    v = (1 + c * x) ^ 3;
    if v <= 0
        continue;
    end
    u = rand();
    if u < 1 - 0.0331 * x ^ 4
        g = d * v;
        return;
    end
    if log(u) < 0.5 * x ^ 2 + d * (1 - v + log(v))
        g = d * v;
        return;
    end
end
end
""",
}


def _write_shims(shim_dir):

    os.makedirs(shim_dir, exist_ok=True)
    for name, body in _SHIMS.items():
        path = os.path.join(shim_dir, name + '.m')
        with open(path, 'w') as fh:
            fh.write(body)
    return shim_dir


def _write_wrapper(path, work_dir, input_mat, output_mat, cvx_root, caiman_root,
                   shim_dir, max_comp_threads=None):

    def _posix(p):
        """Normalise a path for embedding in a MATLAB string literal."""
        return p.replace('\\', '/').replace("'", "''")

    max_threads_line = ('maxNumCompThreads({});'.format(int(max_comp_threads))
                        if max_comp_threads else '% max_comp_threads not set -- using MATLAB default')

    code = _WRAPPER_TEMPLATE
    code = code.replace('__MAX_COMP_THREADS_LINE__', max_threads_line)
    code = code.replace('__WORK_DIR__',    _posix(work_dir))
    code = code.replace('__INPUT_MAT__',   _posix(input_mat))
    code = code.replace('__OUTPUT_MAT__',  _posix(output_mat))
    code = code.replace('__CVX_ROOT__',    _posix(cvx_root))
    code = code.replace('__CAIMAN_ROOT__', _posix(caiman_root))
    code = code.replace('__SHIM_DIR__',    _posix(shim_dir))

    with open(path, 'w') as fh:
        fh.write(code)


def run_matlab_pnevMCMC(dff, fs=30.0, tau=0.5, n_sweeps=1000, true_spikes=None,
                        sparsity_scale=0.001, work_dir=None, matlab_exe=None,
                        verbose=True, max_comp_threads=None):

    if dff.ndim == 1:
        dff = dff[np.newaxis, :]
    n_cells, n_frames = dff.shape

    n_sweeps_val = 500 if n_sweeps == 'auto' else int(n_sweeps)

    work_dir = os.path.abspath(work_dir or _THIS_DIR)
    os.makedirs(work_dir, exist_ok=True)

    input_mat  = os.path.join(work_dir, 'mcmc_input.mat')
    output_mat = os.path.join(work_dir, 'mcmc_output.mat')
    script_stem = 'run_pnev_wrapper'
    wrapper_script = os.path.join(work_dir, script_stem + '.m')

    def _empty():
        """Null result in the shape callers expect, used on any failure."""
        return ([np.array([]) for _ in range(n_cells)],
                np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells))

    exe = find_matlab(matlab_exe)
    if exe is None:
        print('MATLAB not found. Put it on PATH or set {}.'.format(_MATLAB_ENV))
        return _empty()

    cvx_root = find_toolbox(_CVX_ENV, 'cvx')
    caiman_root = find_toolbox(_CAIMAN_ENV, 'CaImAn-MATLAB',
                               extra_candidates=[
                                   os.path.join(os.path.expanduser('~'),
                                                'Documents', 'Github',
                                                'spike-inference', 'matlab'),
                               ])

    if verbose:
        print('MATLAB executable: {}'.format(exe))
        print('  cvx: {}'.format(cvx_root or 'not found'))
        print('  CaImAn-MATLAB: {}'.format(caiman_root or 'not found -- '
                                           'set {}'.format(_CAIMAN_ENV)))

    scipy.io.savemat(input_mat, {
        'dff': dff.astype(np.float64),
        'fs': float(fs),
        'tau': float(tau),
        'n_sweeps': n_sweeps_val,
        'sparsity_scale': float(sparsity_scale),
    })

    shim_dir = _write_shims(os.path.join(tempfile.gettempdir(),
                                         'omsi_matlab_shims'))

    _write_wrapper(wrapper_script, work_dir, input_mat, output_mat,
                   cvx_root, caiman_root, shim_dir, max_comp_threads=max_comp_threads)

    # Stale output from an earlier run would otherwise be read back as if it
    # were this run's result.
    if os.path.exists(output_mat):
        os.remove(output_mat)

    if verbose:
        print('Calling MATLAB subprocess (sweeps={})...'.format(n_sweeps_val))
    t0 = time.time()

    ran = False
    forms = _matlab_invocations(script_stem, exe)
    for i, args in enumerate(forms):
        last = (i == len(forms) - 1)
        try:
            subprocess.run([exe] + args, cwd=work_dir, check=True)
            ran = True
            break
        except subprocess.CalledProcessError as exc:
            if last:
                print('MATLAB exited with status {}. The error it printed above '
                      'is the real cause.'.format(exc.returncode))
            elif verbose:
                print('MATLAB call {} failed (exit {}) -- trying next form...'.format(
                    args[0], exc.returncode))
        except OSError as exc:
            print('Could not launch MATLAB: {}'.format(exc))
            return _empty()

    if not ran:
        print('MATLAB execution failed. Check that cont_ca_sampler is on the '
              'MATLAB path, or set {}.'.format(_CAIMAN_ENV))
        return _empty()

    if verbose:
        print('MATLAB finished in {:.2f}s.'.format(time.time() - t0))

    if not os.path.exists(output_mat):
        print('Error: output MAT file not found at {}.'.format(output_mat))
        return _empty()

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
