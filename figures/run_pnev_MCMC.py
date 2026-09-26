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
_write_rise_fix
    Write a copy of the installed cont_ca_sampler.m with the rise-time fix.
_init_cells
    Per-cell init dicts as a MATLAB cell array of structs.
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

# Environment variables that override discovery, checked before anything else.
_MATLAB_ENV = 'MATLAB_EXECUTABLE'
_CVX_ENV    = 'CVX_HOME'
_CAIMAN_ENV = 'CAIMAN_MATLAB_HOME'

# MATLAB wrapper. Placeholders are substituted by plain string replacement so
# the MATLAB brace syntax does not have to be escaped.
_WRAPPER_TEMPLATE = """
try
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

    % Optional copy of cont_ca_sampler with the rise-time fix. Added last so it is
    % first on the path; empty unless the caller asked for it.
    fix_root = '__FIX_ROOT__';
    if ~isempty(fix_root)
        addpath(fix_root);
        fprintf('Using cont_ca_sampler with rise-time fix: %s\\n', which('cont_ca_sampler'));
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
    if burn_in >= 0
        params.B = burn_in;
    end
    params.p = 2;
    params.f = fs;

    all_spikes = cell(n_cells, 1);
    all_ss = cell(n_cells, 1);
    all_init = cell(n_cells, 1);
    all_probs = zeros(n_cells, n_frames);
    model_traces = zeros(n_cells, n_frames);
    cell_times = nan(n_cells, 1);

    set(0, 'DefaultFigureVisible', 'off');
    fprintf('Running MCMC on %d cells...\\n', n_cells);

    for i = 1:n_cells

        y = double(dff(i, :))';

        try
            if has_inits
                params.init = inits{i};
            elseif has_shifts
                % CaImAn's own start (the call cont_ca_sampler makes when no
                % init is given), with every spike moved by init_shifts(i) frames.
                p0 = params;
                p0.c = []; p0.b = []; p0.c1 = []; p0.g = []; p0.sn = []; p0.sp = [];
                p0.bas_nonneg = 0;
                SAM = get_initial_sample(y, p0);
                SAM.spiketimes_ = min(max(SAM.spiketimes_ + init_shifts(i), 1e-3), n_frames);
                params.init = SAM;
                all_init{i} = SAM.spiketimes_;
            end
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
            if save_samples
                all_ss{i} = samples;
            end

            temp_trace = make_mean_sample(res, y);
            model_traces(i, :) = temp_trace(:)';

        catch ME
            fprintf('Error on cell %d: %s\\n', i, ME.message);
        end
    end

    if save_samples
        save('__OUTPUT_MAT__', 'all_spikes', 'all_probs', 'model_traces', 'all_ss', 'all_init');
    else
        save('__OUTPUT_MAT__', 'all_spikes', 'all_probs', 'model_traces');
    end
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\\n', ME.message);
    exit(1);
end
"""


def _release_sort_key(path):
    """ Sort key that orders MATLAB install paths newest release first.

    Parameters
    ----------
    path : str
        Path containing a release tag such as R2025b.

    Returns
    -------
    tuple
        (year, update letter) descending-friendly key, (0, '') when unparsed.
    """

    match = re.search(r'R(\d{4})([ab])', path)
    if not match:
        return (0, '')
    return (int(match.group(1)), match.group(2))


def find_matlab(explicit=None):
    """ Locate a MATLAB executable, newest release first.

    Order: explicit argument, MATLAB_EXECUTABLE, PATH, then the standard
    install directory for the current OS.

    Parameters
    ----------
    explicit : str, optional
        Path supplied by the caller. Returned as-is when it exists.

    Returns
    -------
    str or None
        Path to a MATLAB executable, or None when none was found.
    """

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
    """ Locate a MATLAB toolbox directory from env var or common install paths.

    Parameters
    ----------
    env_var : str
        Environment variable checked first.
    folder_name : str
        Directory name to look for, e.g. 'cvx'.
    extra_candidates : iterable of str, optional
        Additional absolute paths to check before the generic ones.

    Returns
    -------
    str
        Absolute path to the toolbox, or empty string when not found.
    """

    from_env = os.environ.get(env_var, '').strip()
    if from_env and os.path.isdir(from_env):
        return os.path.abspath(from_env)

    home = os.path.expanduser('~')

    # Parent folders people actually keep toolboxes in. OneDrive-redirected
    # Documents and the Desktop are both common on Windows.
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
    """ Argument lists to try, modern batch mode first.

    -batch arrived in R2019a. Older releases need -r with an explicit exit,
    and Windows needs -wait so the call blocks. When the release is readable
    from the executable path and is R2019a or later, only -batch is offered:
    a non-zero exit then means the MATLAB script itself failed, and retrying
    with -r would just open a second session that hangs.

    Parameters
    ----------
    script_stem : str
        Wrapper script name without the .m extension.
    exe : str, optional
        MATLAB executable path, used to read the release tag.

    Returns
    -------
    list of list of str
        Argument lists to append to the executable, in order of preference.
    """

    batch = ['-batch', script_stem]

    # Known release of R2019a or later: -batch is supported, so never retry.
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
    """ Write base-MATLAB stand-ins for the Statistics Toolbox functions used.

    cont_ca_sampler needs range, prctile, gamrnd, and normrnd, all of which
    ship with the Statistics and Machine Learning Toolbox. The wrapper adds
    this directory to the end of the path, so a real toolbox installation
    still wins and these are used only when it is missing or unlicensed.

    Parameters
    ----------
    shim_dir : str
        Directory to write the .m files into. Created if absent.

    Returns
    -------
    str
        The directory written to.
    """

    os.makedirs(shim_dir, exist_ok=True)
    for name, body in _SHIMS.items():
        path = os.path.join(shim_dir, name + '.m')
        with open(path, 'w') as fh:
            fh.write(body)
    return shim_dir


def _write_wrapper(path, work_dir, input_mat, output_mat, cvx_root, caiman_root,
                   shim_dir, fix_root=''):
    """ Fill the MATLAB wrapper template and write it next to the input file.

    Forward slashes are used throughout: MATLAB accepts them on Windows too,
    and they avoid escape trouble inside single-quoted MATLAB strings.

    Parameters
    ----------
    path : str
        Destination .m file.
    work_dir : str
        Directory MATLAB changes into before running.
    input_mat : str
        Input .mat filename.
    output_mat : str
        Output .mat filename.
    cvx_root : str
        cvx directory, or empty string.
    caiman_root : str
        CaImAn-MATLAB directory, or empty string.
    shim_dir : str
        Directory holding the Statistics Toolbox stand-ins.
    fix_root : str, optional
        Directory holding the rise-time-fixed cont_ca_sampler, or empty string.
    """

    def _posix(p):
        """Normalise a path for embedding in a MATLAB string literal."""
        return p.replace('\\', '/').replace("'", "''")

    code = _WRAPPER_TEMPLATE
    code = code.replace('__WORK_DIR__',    _posix(work_dir))
    code = code.replace('__INPUT_MAT__',   _posix(input_mat))
    code = code.replace('__OUTPUT_MAT__',  _posix(output_mat))
    code = code.replace('__CVX_ROOT__',    _posix(cvx_root))
    code = code.replace('__CAIMAN_ROOT__', _posix(caiman_root))
    code = code.replace('__SHIM_DIR__',    _posix(shim_dir))
    code = code.replace('__FIX_ROOT__',    _posix(fix_root))

    with open(path, 'w') as fh:
        fh.write(code)


# Rise-time Metropolis step in cont_ca_sampler scores the proposal with the current
# calcium shape Gs instead of the proposed Gs_, so every rise-time proposal is
# accepted. The decay step a few lines later uses Gs_ correctly.
_RISE_BUG_LINE = 'logC_ = -norm(E*(Y(:)-A_*Gs-b_-C_in*ge))^2;'
_RISE_FIX_LINE = 'logC_ = -norm(E*(Y(:)-A_*Gs_-b_-C_in*ge))^2;'


def _write_rise_fix(caiman_root, out_dir):
    """ Write a copy of the installed cont_ca_sampler.m with the rise-time fix.

    The installed CaImAn is only read, never modified; the copy goes to out_dir.

    Parameters
    ----------
    caiman_root : str
        CaImAn-MATLAB directory.
    out_dir : str
        Directory for the patched copy. Created if absent.

    Returns
    -------
    str
        out_dir.
    """

    hits = glob.glob(os.path.join(caiman_root, '**', 'cont_ca_sampler.m'), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError('Expected one cont_ca_sampler.m under {}, found {}.'.format(
            caiman_root, len(hits)))
    with open(hits[0]) as fh:
        code = fh.read()
    if code.count(_RISE_BUG_LINE) != 1:
        raise ValueError('Rise-time line not found exactly once in {} -- '
                         'CaImAn version differs, fix not applied.'.format(hits[0]))
    code = code.replace(_RISE_BUG_LINE, _RISE_FIX_LINE + '  % rise-time fix (fMCSI)')

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'cont_ca_sampler.m'), 'w') as fh:
        fh.write(code)
    return out_dir


def _init_cells(inits, n_cells):
    """ Per-cell init dicts as a MATLAB cell array of structs (empty when None). """

    arr = np.empty((n_cells, 1), dtype=object)
    for i in range(n_cells):
        if inits is None:
            arr[i, 0] = np.zeros((0, 0))
            continue
        d = inits[i]
        arr[i, 0] = {k: (np.asarray(d[k], dtype=np.float64).reshape(-1, 1)
                         if k in ('spiketimes_', 'g') else float(d[k]))
                     for k in ('lam_', 'spiketimes_', 'A_', 'b_', 'C_in', 'sg', 'g')}
    return arr


def run_matlab_pnevMCMC(dff, fs=30.0, tau=0.5, n_sweeps=1000, true_spikes=None,
                        sparsity_scale=0.001, work_dir=None, matlab_exe=None,
                        verbose=True, return_samples=False, inits=None, burn_in=None,
                        init_shifts=None, fix_rise_bug=False):
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
    work_dir : str, optional
        Directory for the .mat and wrapper files. Defaults to this file's
        directory, so results do not depend on where Python was launched.
    matlab_exe : str, optional
        MATLAB executable. Discovered automatically when omitted.
    verbose : bool, optional
        Print progress and discovery details.
    return_samples : bool, optional
        Also return every post-burn-in posterior sample.
    inits : list of dict, optional
        Per-cell starting sample passed as cont_ca_sampler's params.init. Keys:
        lam_, spiketimes_ (1-based frame units), A_, b_, C_in, sg, g.
    burn_in : int, optional
        Burn-in sweeps (params.B). Default is half of n_sweeps; 0 keeps every sweep.
    init_shifts : array-like, optional
        Per-cell shift in frames applied to every spike of CaImAn's own starting
        sample. Ignored when inits is given.
    fix_rise_bug : bool, optional
        Run a copy of cont_ca_sampler whose rise-time Metropolis step scores the
        proposed calcium shape (Gs_) instead of the current one (Gs). The
        installed CaImAn is not modified.
    return_cell_times : bool, optional
        If True, also return each cell's cont_ca_sampler time in seconds.

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
    samples : list of list of np.ndarray
        Only when return_samples is True: per cell, each posterior sample's spike
        times in 0-based frame units.
    init_spikes : list of np.ndarray
        Only when return_samples is True and init_shifts is given: per cell, the
        shifted starting spike times in 0-based frame units.
    cell_times : np.ndarray
        Only if return_cell_times: seconds spent in cont_ca_sampler per cell,
        timed inside MATLAB (NaN for cells that errored or on failure).
    """

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
        out = ([np.array([]) for _ in range(n_cells)],
               np.zeros_like(dff), np.zeros_like(dff), np.zeros(n_cells))
        if not return_samples:
            return out
        out = out + ([[] for _ in range(n_cells)],)
        if init_shifts is not None and inits is None:
            out = out + ([np.array([]) for _ in range(n_cells)],)
        return out + (np.full(n_cells, np.nan),) if return_cell_times else out

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
        'save_samples': bool(return_samples),
        'burn_in': -1 if burn_in is None else int(burn_in),
        'has_inits': inits is not None,
        'has_shifts': init_shifts is not None and inits is None,
        'init_shifts': np.zeros(n_cells) if init_shifts is None
                       else np.asarray(init_shifts, dtype=np.float64).ravel(),
        'inits': _init_cells(inits, n_cells),
    })

    # Shims live outside work_dir so the wrapper's genpath(pwd) cannot pull
    # them in ahead of a real Statistics Toolbox.
    shim_dir = _write_shims(os.path.join(tempfile.gettempdir(),
                                         'omsi_matlab_shims'))

    fix_root = ''
    if fix_rise_bug:
        if not caiman_root:
            print('CaImAn-MATLAB not found -- cannot apply the rise-time fix.')
            return _empty()
        fix_root = _write_rise_fix(caiman_root, os.path.join(tempfile.gettempdir(),
                                                             'omsi_caiman_rise_fix'))

    _write_wrapper(wrapper_script, work_dir, input_mat, output_mat,
                   cvx_root, caiman_root, shim_dir, fix_root)

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

            # cont_ca_sampler reports times in 1-based frame units: a spike at
            # continuous time t first shows in sample ceil(t), and sample k is
            # frame k-1 in Python. Shift by one frame so times match the
            # 0-based frame clock used for ground truth.
            times = np.clip(spks.flatten() - 1.0, 0.0, None) / fs
            final_spikes.append(times)
        else:
            final_spikes.append(np.array([]))

    sweeps_per_cell = np.full(n_cells, n_sweeps_val, dtype=np.int32)

    if not return_samples:
        return final_spikes, model_traces, all_probs, sweeps_per_cell

    # Same 1-based to 0-based frame shift as the called spikes.
    samples = []
    for i in range(n_cells):
        cell_ss = np.atleast_1d(res['all_ss'][i][0]).ravel() \
            if 'all_ss' in res and res['all_ss'][i][0].size > 0 else []
        samples.append([np.asarray(st, dtype=np.float64).ravel() - 1.0
                        for st in cell_ss])
    if init_shifts is None or inits is not None:
        return final_spikes, model_traces, all_probs, sweeps_per_cell, samples

    init_spikes = [np.asarray(res['all_init'][i][0], dtype=np.float64).ravel() - 1.0
                   for i in range(n_cells)]
    return final_spikes, model_traces, all_probs, sweeps_per_cell, samples, init_spikes
    if return_cell_times:
        cell_times = (np.asarray(res['cell_times'], dtype=float).ravel()
                      if 'cell_times' in res else np.full(n_cells, np.nan))
        return final_spikes, model_traces, all_probs, sweeps_per_cell, cell_times
    return final_spikes, model_traces, all_probs, sweeps_per_cell
