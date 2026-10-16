# -*- coding: utf-8 -*-
"""
figures/diagnose_rj_mcmc_bug.py

DMM, September 2026
"""

import argparse
import glob
import os
import subprocess
import tempfile
import time

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from run_pnev_MCMC import find_matlab, find_toolbox, _write_shims, _CVX_ENV, _CAIMAN_ENV

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, 'data', 'rj_mcmc_diag')

_REC_NPZ  = 'rj_diag_recovery.npz'
_STAT_NPZ = 'rj_diag_stationary.npz'
_SPLIT_NPZ = 'rj_diag_split.npz'
_CONS_NPZ  = 'rj_diag_consensus.npz'
_ABL_NPZ   = 'rj_diag_ablation.npz'
_CNT_NPZ   = 'rj_diag_counts.npz'
_TIMING_DIR = os.path.join(_HERE, 'data', 'timing')

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7
mpl.rcParams['axes.titlesize']  = 7
mpl.rcParams['axes.labelsize']  = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7

FS    = 30.0
TAU_D = 0.6
SNR   = 6.0
RATE  = 1.0

COL_BUG   = '#E7B415'
COL_FIX   = '#A8323E'
COL_NOISE = '#8C8C8C'

REC_RISES_MS = (10.0, 50.0, 150.0)
REC_CELLS    = 12
REC_DURATION = 60.0
REC_SWEEPS   = 6000
REC_BURN     = 2000

STAT_RISE_MS     = 50.0
STAT_DURATION    = 30.0
STAT_SWEEPS      = 50000
STAT_BURN        = 5000
STAT_START_FRACS = (0.1, 0.35, 0.6, 0.85)

BAD_MOVE = 3.0

SPLIT_N_BAD   = 8
SPLIT_N_GOOD  = 4
SPLIT_SWEEPS  = 750
SPLIT_BURN    = 250
SPLIT_ISO     = 1.0
SPLIT_WIN     = (-0.5, 1.0)

CONS_METHODS = {
    'OMSI':     ('OMSI',                       '#808080', '-'),
    'CAIMAN':   ('CaImAn',                     '#E7B415', '-'),
    'FIX_LAST': ('CaImAn fix, last sample',    '#A8323E', '--'),
    'FIX_CONS': ('CaImAn fix, OMSI consensus', '#A8323E', '-'),
    'THR_LAST': ('CaImAn fix + init thr, last sample',    '#2BB5B0', '--'),
    'THR_CONS': ('CaImAn fix + init thr, OMSI consensus', '#2BB5B0', '-'),
}
CONS_PLOT = {
    'OMSI':     ('OMSI',                  '#808080', '-'),
    'CAIMAN':   ('CaImAn',                '#E7B415', '-'),
    'FIX_LAST': ('CaImAn fix',            '#A8323E', '-'),
    'THR_LAST': ('CaImAn fix + init thr', '#2BB5B0', '-'),
}
CNT_VARIANTS = {
    'CAIMAN':   {'fix': False},
    'FIX_LAST': {'fix': True},
    'THR_LAST': {'fix': True, 'thr': True},
}

CNT_XMAX = 450
CNT_PLOT = ('CAIMAN', 'THR_LAST')
CNT_N_CELLS   = 4
CNT_SNR_RANGE = (1.5, 3.0)
CAIMAN_INIT_THR = 0.15
CONS_INIT_THR = 0.70
ABL_N_EXTRA = 28
ABL_LAM_SCALE = 0.002

ABL_CONDS = {
    'base':      ('fixed CaImAn',            {}),
    'init':      ('OMSI init',               {'init': True}),
    'lam_fixed': ('rate held fixed',         {'lam_fixed': True}),
    'lam_scale': ('rate x {:g}'.format(ABL_LAM_SCALE), {'lam_scale': ABL_LAM_SCALE}),
    'alb':       ('amp. floor = noise SD',   {'alb_noise': True}),
    'thr10':     ('FOOPSI init, thr 0.10',   {'init_thr': 0.10}),
    'thr30':     ('FOOPSI init, thr 0.30',   {'init_thr': 0.30}),
    'thr50':     ('FOOPSI init, thr 0.50',   {'init_thr': 0.50}),
    'thr60':     ('FOOPSI init, thr 0.60',   {'init_thr': 0.60}),
    'thr70':     ('FOOPSI init, thr 0.70',   {'init_thr': 0.70}),
    'thr80':     ('FOOPSI init, thr 0.80',   {'init_thr': 0.80}),
    'all':       ('all four',                {'init': True, 'lam_fixed': True,
                                              'lam_scale': ABL_LAM_SCALE,
                                              'alb_noise': True}),
}
ABL_SPLIT_MIN = 2

_INIT_PATCHES = [
    ('function SAM = get_initial_sample(Y,params)',
     'function SAM = get_initial_sample_diag(Y,params)\n'
     '% Instrumented copy written by fMCSI figures/diagnose_rj_mcmc_bug.py.'),
    ('s_in = sp>0.15*max(sp);',
     'thr = 0.15; if isfield(params,\'init_thr\'); thr = params.init_thr; end\n'
     's_in = sp>thr*max(sp);'),
]

_RISE_BUG_LINE = 'logC_ = -norm(E*(Y(:)-A_*Gs-b_-C_in*ge))^2;'

_DIAG_PATCHES = [
    ('function SAMPLES = cont_ca_sampler(Y,params)',
     'function SAMPLES = cont_ca_sampler_diag(Y,params)\n'
     '% Instrumented copy written by fMCSI figures/diagnose_rj_mcmc_bug.py.'),

    ('tauMoves = [0 0];',
     'tauMoves = [0 0];\n'
     'fix_rise = isfield(params,\'fix_rise\') && params.fix_rise;\n'
     'lam_fixed = isfield(params,\'lam_fixed\') && params.lam_fixed;\n'
     'lam_scale_dg = 1; if isfield(params,\'lam_scale\'); lam_scale_dg = params.lam_scale; end\n'
     'DG.B = B; DG.tau0 = tau(:)\'; DG.tau1_std = tau1_std; DG.fix_rise = fix_rise;\n'
     'DG.rise_ratio = nan(N,1); DG.rise_acc = false(N,1);\n'
     'DG.rise_dssr = nan(N,1); DG.rise_sg = nan(N,1);\n'
     'DG.decay_ratio = nan(N,1); DG.decay_acc = false(N,1);\n'
     'DG.ssr_pre = nan(N,1); DG.ssr = nan(N,1); DG.tau = nan(N,2); DG.ns = zeros(N,1);'),

    (_RISE_BUG_LINE,
     'logC_bug = -norm(E*(Y(:)-A_*Gs-b_-C_in*ge))^2;\n'
     '                logC_fix = -norm(E*(Y(:)-A_*Gs_-b_-C_in*ge))^2;\n'
     '                if fix_rise; logC_ = logC_fix; else; logC_ = logC_bug; end\n'
     '                DG.ssr_pre(i) = -logC; DG.rise_dssr(i) = logC - logC_fix;\n'
     '                DG.rise_sg(i) = sg;'),

    ('ratio = exp((logC_-logC)/(2*sg^2))*prior_ratio;',
     'ratio = exp((logC_-logC)/(2*sg^2))*prior_ratio;\n'
     '                DG.rise_ratio(i) = ratio;'),

    ('if rand < ratio %accept',
     'DG.rise_acc(i) = rand < ratio;\n'
     '                if DG.rise_acc(i) %accept'),

    ('ratio = exp((1./(2*sg^2)).*(logC_-logC))*prior_ratio;',
     'ratio = exp((1./(2*sg^2)).*(logC_-logC))*prior_ratio;\n'
     '            DG.decay_ratio(i) = ratio;'),

    ('if rand<ratio %accept',
     'DG.decay_acc(i) = rand<ratio;\n'
     '            if DG.decay_acc(i) %accept'),

    ('if params.print_flag && mod(i,100)==0',
     'DG.tau(i,:) = tau(:)\'; DG.ns(i) = nsp;\n'
     '    if ~marg_flag; DG.ssr(i) = norm(E*(Y(:)-A_*Gs-b_-C_in*ge))^2; end\n'
     '    if params.print_flag && mod(i,100)==0'),

    ('rate = @(t) lambda_rate(t,lam_);',
     'rate = @(t) lambda_rate(t,lam_*lam_scale_dg);'),

    ('lam_ = lam(i);',
     'if ~lam_fixed; lam_ = lam(i); end'),

    ('SAMPLES.params = params.init;',
     'SAMPLES.params = params.init;\nSAMPLES.diag = DG;'),
]

_WRAPPER_TEMPLATE = """
try
    addpath(genpath('__CAIMAN_ROOT__'));
    cvx_root = '__CVX_ROOT__';
    if ~isempty(cvx_root) && exist(cvx_root, 'dir') == 7
        addpath(genpath(cvx_root));
    end
    addpath('__DIAG_DIR__');
    addpath('__SHIM_DIR__', '-end');
    if exist('cvx_setup', 'file') == 2 && isempty(which('cvx_begin'))
        try
            cvx_setup;
        catch setup_err
            fprintf('cvx_setup failed: %s\\n', setup_err.message);
        end
    end
    set(0, 'DefaultFigureVisible', 'off');

    load('__INPUT_MAT__');
    n_jobs = numel(Ys);
    out = cell(n_jobs, 1);

    for j = 1:n_jobs
        t_job = tic;
        y = double(Ys{j}(:));
        rng(seeds(j));

        params = struct();
        params.p = 2;
        params.f = fs;
        params.B = burn(j);
        params.Nsamples = n_sweeps(j) - burn(j);
        params.fix_rise = logical(fixes(j));

        p0 = params;
        p0.c = []; p0.b = []; p0.c1 = []; p0.g = []; p0.sn = []; p0.sp = [];
        p0.bas_nonneg = 0;
        p0.init_thr = init_thr(j);
        SAM = get_initial_sample_diag(y, p0);
        if all(isfinite(init_tau(j, :)))
            pl = poly(exp(-1 ./ init_tau(j, :)));
            SAM.g = -pl(2:end)';
        end
        if has_init(j)
            SAM.spiketimes_ = double(init_sp{j}(:));
            SAM.A_ = init_sc(j, 1); SAM.b_ = init_sc(j, 2); SAM.C_in = init_sc(j, 3);
            SAM.sg = init_sc(j, 4); SAM.lam_ = init_sc(j, 5);
            gi = init_g(j, :); SAM.g = gi(isfinite(gi))';
        end
        params.init = SAM;
        params.lam_fixed = logical(lam_fixed(j));
        params.lam_scale = lam_scale(j);
        if alb_noise(j); params.A_lb = SAM.sg; end

        try
            res = cont_ca_sampler_diag(y, params);
            out{j} = res.diag;
            out{j}.n_init = numel(SAM.spiketimes_);
            out{j}.ss_last = res.ss{end};
            out{j}.Am = res.Am;
            if keep_ss(j); out{j}.ss = res.ss; end
        catch ME
            fprintf('Job %d failed: %s\\n', job_ids(j), ME.message);
            out{j} = struct('failed', 1);
        end
        fprintf('Job %d done in %.1fs.\\n', job_ids(j), toc(t_job));
    end

    save('__OUTPUT_MAT__', 'out', 'job_ids');
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\\n', ME.message);
    exit(1);
end
"""

_DIAG_FIELDS = ('rise_ratio', 'rise_acc', 'rise_dssr', 'rise_sg', 'decay_ratio',
                'decay_acc', 'ssr_pre', 'ssr', 'tau', 'ns', 'ss_last', 'Am')


def _simulate(n_cells, rise, decay, duration, rate, snr, seed):

    rng = np.random.default_rng(seed)
    n_frames = int(FS * duration)
    t = np.arange(n_frames) / FS

    t_pk = rise * decay / (decay - rise) * np.log(decay / rise)
    h_max = np.exp(-t_pk / decay) - np.exp(-t_pk / rise)

    noisy = np.zeros((n_cells, n_frames))
    spikes = []
    for i in range(n_cells):
        st = np.sort(rng.uniform(0.0, duration, rng.poisson(rate * duration)))
        dt = t[None, :] - st[:, None]
        k = np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / decay)
                     - np.exp(-np.clip(dt, 0, None) / rise), 0.0)
        noisy[i] = k.sum(axis=0) / h_max + rng.normal(0.0, 1.0 / snr, n_frames)
        spikes.append(st)
    return noisy, spikes


def _find_sampler(caiman_root):

    hits = sorted(glob.glob(os.path.join(caiman_root, '**', 'cont_ca_sampler.m'),
                            recursive=True))
    if not hits:
        raise FileNotFoundError('No cont_ca_sampler.m under {}.'.format(caiman_root))
    pref = [h for h in hits if os.path.join('deconvolution', 'MCMC') in h]
    path = (pref or hits)[0]
    if len(hits) > 1:
        print('Found {} copies of cont_ca_sampler.m -- using {}.'.format(len(hits), path))
    return path


def _write_diag_sampler(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    init_src = os.path.join(os.path.dirname(src), 'utilities', 'get_initial_sample.m')
    for path, patches, name in ((src, _DIAG_PATCHES, 'cont_ca_sampler_diag.m'),
                                (init_src, _INIT_PATCHES, 'get_initial_sample_diag.m')):
        with open(path) as fh:
            code = fh.read()
        for anchor, new in patches:
            if code.count(anchor) != 1:
                raise ValueError('Anchor not found exactly once in {} -- CaImAn version '
                                 'differs: {}'.format(path, anchor))
            code = code.replace(anchor, new)
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(code)
    return out_dir


def _write_wrapper(path, input_mat, output_mat, caiman_root, cvx_root, diag_dir,
                   shim_dir):

    def _posix(p):
        """Normalise a path for a MATLAB string literal."""
        return p.replace('\\', '/').replace("'", "''")

    code = _WRAPPER_TEMPLATE
    for key, val in (('__INPUT_MAT__', input_mat), ('__OUTPUT_MAT__', output_mat),
                     ('__CAIMAN_ROOT__', caiman_root), ('__CVX_ROOT__', cvx_root),
                     ('__DIAG_DIR__', diag_dir), ('__SHIM_DIR__', shim_dir)):
        code = code.replace(key, _posix(val))
    with open(path, 'w') as fh:
        fh.write(code)


def _run_jobs(jobs, work_dir, n_workers, post=None):

    exe = find_matlab()
    if exe is None:
        raise RuntimeError('MATLAB not found. Put it on PATH or set MATLAB_EXECUTABLE.')
    caiman_root = find_toolbox(_CAIMAN_ENV, 'CaImAn-MATLAB')
    if not caiman_root:
        raise RuntimeError('CaImAn-MATLAB not found -- set {}.'.format(_CAIMAN_ENV))
    cvx_root = find_toolbox(_CVX_ENV, 'cvx')

    os.makedirs(work_dir, exist_ok=True)
    diag_dir = _write_diag_sampler(_find_sampler(caiman_root),
                                   os.path.join(work_dir, 'sampler'))
    shim_dir = _write_shims(os.path.join(tempfile.gettempdir(), 'omsi_matlab_shims'))

    n_workers = max(1, min(n_workers, len(jobs)))
    procs = []
    for w in range(n_workers):
        ids = np.arange(w, len(jobs), n_workers)
        stem = 'rj_diag_worker_{}'.format(w)
        input_mat = os.path.join(work_dir, stem + '_in.mat')
        output_mat = os.path.join(work_dir, stem + '_out.mat')
        if os.path.exists(output_mat):
            os.remove(output_mat)

        ys = np.empty((len(ids), 1), dtype=object)
        for k, j in enumerate(ids):
            ys[k, 0] = np.asarray(jobs[j]['y'], dtype=np.float64).reshape(-1, 1)
        init_tau = np.array([jobs[j]['init_tau'] if jobs[j]['init_tau'] is not None
                             else (np.nan, np.nan) for j in ids], dtype=np.float64)
        init_sp = np.empty((len(ids), 1), dtype=object)
        init_sc = np.full((len(ids), 5), np.nan)
        init_g = np.full((len(ids), 2), np.nan)
        for k, j in enumerate(ids):
            ini = jobs[j].get('init')
            init_sp[k, 0] = np.zeros((0, 1))
            if ini is None:
                continue
            init_sp[k, 0] = np.asarray(ini['spiketimes_'], dtype=np.float64).reshape(-1, 1)
            init_sc[k] = [ini[key] for key in ('A_', 'b_', 'C_in', 'sg', 'lam_')]
            g = np.atleast_1d(np.asarray(ini['g'], dtype=np.float64)).ravel()[:2]
            init_g[k, :len(g)] = g
        scipy.io.savemat(input_mat, {
            'Ys': ys, 'fs': FS,
            'fixes': np.array([float(jobs[j]['fix']) for j in ids]),
            'n_sweeps': np.array([float(jobs[j]['n_sweeps']) for j in ids]),
            'burn': np.array([float(jobs[j]['burn']) for j in ids]),
            'seeds': np.array([float(jobs[j]['seed']) for j in ids]),
            'init_tau': init_tau.reshape(-1, 2),
            'keep_ss': np.array([float(jobs[j].get('keep_ss', False)) for j in ids]),
            'has_init': np.array([float(jobs[j].get('init') is not None) for j in ids]),
            'init_sp': init_sp, 'init_sc': init_sc, 'init_g': init_g,
            'lam_fixed': np.array([float(jobs[j].get('lam_fixed', False)) for j in ids]),
            'lam_scale': np.array([float(jobs[j].get('lam_scale', 1.0)) for j in ids]),
            'alb_noise': np.array([float(jobs[j].get('alb_noise', False)) for j in ids]),
            'init_thr': np.array([float(jobs[j].get('init_thr', 0.15)) for j in ids]),
            'job_ids': ids.astype(np.float64),
        })
        _write_wrapper(os.path.join(work_dir, stem + '.m'), input_mat, output_mat,
                       caiman_root, cvx_root, diag_dir, shim_dir)

        log = open(os.path.join(work_dir, stem + '.log'), 'w')
        procs.append((subprocess.Popen([exe, '-singleCompThread', '-batch', stem],
                                       cwd=work_dir, stdout=log,
                                       stderr=subprocess.STDOUT), log, output_mat))

    print('Running {} jobs on {} MATLAB sessions (logs in {})...'.format(
        len(jobs), n_workers, work_dir))
    t0 = time.time()
    results = [None] * len(jobs)
    for w, (proc, log, output_mat) in enumerate(procs):
        proc.wait()
        log.close()
        if proc.returncode != 0 or not os.path.exists(output_mat):
            print('Worker {} failed (exit {}) -- see its log.'.format(w, proc.returncode))
            continue

        res = scipy.io.loadmat(output_mat, squeeze_me=True, struct_as_record=False)
        outs = np.atleast_1d(res['out'])
        for j, d in zip(np.atleast_1d(res['job_ids']).astype(int), outs):
            if not hasattr(d, 'rise_ratio'):
                continue
            rec = {k: np.asarray(getattr(d, k), dtype=np.float64) for k in _DIAG_FIELDS}
            rec['tau'] = rec['tau'].reshape(-1, 2)
            rec['B'] = int(d.B)
            rec['tau0'] = np.asarray(d.tau0, dtype=np.float64).ravel()
            rec['tau1_std'] = float(d.tau1_std)
            rec['n_init'] = float(getattr(d, 'n_init', np.nan))
            if hasattr(d, 'ss'):
                rec['ss'] = [np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
                             for x in np.atleast_1d(d.ss)]
            results[j] = post(j, rec) if post is not None else rec

    n_bad = sum(r is None for r in results)
    print('MATLAB finished in {:.1f} min -- {} of {} jobs failed.'.format(
        (time.time() - t0) / 60.0, n_bad, len(jobs)))
    
    return results


def _save(path, meta, results):

    res = np.empty(len(results), dtype=object)
    for i, r in enumerate(results):
        res[i] = r
    np.savez(path, results=res, **{k: np.asarray(v) for k, v in meta.items()})
    print('Saved to {}.'.format(path))


def _run_recovery(data_dir, n_workers):

    jobs, meta = [], {'true_rise_ms': [], 'fix': [], 'cell': []}
    for r, rise_ms in enumerate(REC_RISES_MS):
        noisy, _ = _simulate(REC_CELLS, rise_ms / 1000.0, TAU_D, REC_DURATION,
                             RATE, SNR, seed=100 + r)
        for fix in (False, True):
            for c in range(REC_CELLS):
                jobs.append({'y': noisy[c], 'fix': fix, 'n_sweeps': REC_SWEEPS,
                             'burn': REC_BURN, 'seed': 1000 * r + c, 'init_tau': None})
                meta['true_rise_ms'].append(rise_ms)
                meta['fix'].append(fix)
                meta['cell'].append(c)

    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers)
    _save(os.path.join(data_dir, _REC_NPZ), meta, results)


def _run_stationary(data_dir, n_workers):

    noisy, _ = _simulate(1, STAT_RISE_MS / 1000.0, TAU_D, STAT_DURATION, RATE, SNR,
                         seed=200)
    noise = np.random.default_rng(201).normal(0.0, 1.0 / SNR, noisy.shape[1])
    tau_d_fr = TAU_D * FS

    jobs, meta = [], {'trace': [], 'fix': [], 'chain': []}
    for trace, y in (('data', noisy[0]), ('noise', noise)):
        for fix in (False, True):
            for c, frac in enumerate(STAT_START_FRACS):
                jobs.append({'y': y, 'fix': fix, 'n_sweeps': STAT_SWEEPS,
                             'burn': STAT_BURN, 'seed': 5000 + c,
                             'init_tau': (frac * tau_d_fr, tau_d_fr)})
                meta['trace'].append(trace)
                meta['fix'].append(fix)
                meta['chain'].append(c)

    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers)
    _save(os.path.join(data_dir, _STAT_NPZ), meta, results)


def _run_split(data_dir, n_workers, timing_dir=_TIMING_DIR):

    import timing_calibration as tc

    pop = np.load(os.path.join(timing_dir, tc._POP_NPZ), allow_pickle=True)
    fix = np.load(os.path.join(timing_dir, tc._FIX_NPZ), allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    called = [np.asarray(x, dtype=np.float64) for x in fix['MATLAB_FIX_spikes']]
    prec = np.array([tc._prf_by_cell(t, x, tc.MATCH_TOL)[0] for t, x in zip(true, called)])
    bad = np.where(prec < 0.6)[0][:SPLIT_N_BAD]
    good = np.where(prec > 0.8)[0][:SPLIT_N_GOOD]
    cells = np.concatenate([bad, good])

    noisy, regen = tc._simulate(len(true), tc.FS, float(pop['duration'][0]), tc.POP_SEED)
    if not all(np.allclose(true[c], regen[c]) for c in cells):
        raise ValueError('Regenerated population differs from {}.'.format(timing_dir))

    jobs, meta = [], {'cell': [], 'fix': [], 'bad': []}
    for c in cells:
        for fx in (False, True):
            jobs.append({'y': noisy[c], 'fix': fx, 'n_sweeps': SPLIT_SWEEPS,
                         'burn': SPLIT_BURN, 'seed': int(c), 'init_tau': None})
            meta['cell'].append(c)
            meta['fix'].append(fx)
            meta['bad'].append(c in bad)

    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers)
    true_arr = np.empty(len(cells), dtype=object)
    for k, c in enumerate(cells):
        true_arr[k] = true[c]
    meta.update({'cells': cells, 'true_spikes': true_arr, 'traces': noisy[cells],
                 'fs': tc.FS})
    _save(os.path.join(data_dir, _SPLIT_NPZ), meta, results)

def _calls(r, fs):
    return np.sort(np.clip(np.atleast_1d(r['ss_last']).astype(float) - 1.0, 0, None) / fs)

def _isolated(t):
    gap_pre = np.diff(np.r_[-np.inf, t])
    gap_post = np.diff(np.r_[t, np.inf])
    return t[(gap_pre > SPLIT_ISO) & (gap_post > SPLIT_ISO)]


def plot_split_figure(data_dir=_DEFAULT_DATA_DIR):

    loaded = _load(os.path.join(data_dir, _SPLIT_NPZ))
    if loaded is None:
        raise FileNotFoundError('No split results in {}. Run --stages split.'.format(data_dir))
    meta, res = loaded
    fs = float(meta['fs'])
    cells = list(meta['cells'])
    versions = (('bug', 'CaImAn', COL_BUG, False), ('fix', 'rise-time fix', COL_FIX, True))

    def _job(c, fx):

        for r, cc, f in zip(res, meta['cell'], meta['fix']):
            if cc == c and bool(f) == fx:
                return r
        return None

    bad = [c for c, b in zip(meta['cell'][::2], meta['bad'][::2]) if b]

    fig = plt.figure(figsize=(7.5, 2.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2.0, 1.0, 1.0], wspace=0.4)
    ax_tr = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_ns = fig.add_subplot(gs[0, 2])

    def _ratio(c, fx):
        return len(_calls(_job(c, fx), fs)) / max(1, len(meta['true_spikes'][cells.index(c)]))

    med = np.median([_ratio(c, True) for c in bad])
    pool = [c for c in bad if abs(_ratio(c, False) - 1.0) < 0.1] or bad
    ex = min(pool, key=lambda c: abs(_ratio(c, True) - med))
    k = cells.index(ex)
    t_true = np.asarray(meta['true_spikes'][k], dtype=float)
    iso = _isolated(t_true)
    span = 6.0
    starts = iso - 0.5
    t0 = starts[np.argmax([np.sum((iso >= a) & (iso < a + span)) for a in starts])]
    y = meta['traces'][k]
    t = np.arange(len(y)) / fs
    m = (t >= t0) & (t < t0 + span)
    ax_tr.plot(t[m] - t0, y[m], color='k', lw=0.6)
    lo, hi = np.min(y[m]), np.max(y[m])
    step = 0.2 * (hi - lo)
    rows = [('true', t_true, 'k')] + [(lab, _calls(_job(ex, fx), fs), col)
                                      for _, lab, col, fx in versions]
    for i, (lab, times, col) in enumerate(rows):
        yy = lo - (i + 1) * step
        sel = np.sort(times[(times >= t0) & (times < t0 + span)] - t0)

        clusters = np.split(sel, np.where(np.diff(sel) > 0.05)[0] + 1) if len(sel) else []
        for cl in clusters:
            for j in range(len(cl)):
                base = yy - 0.4 * step + j * 0.28 * step
                ax_tr.vlines(np.mean(cl), base, base + 0.22 * step, color=col, lw=1.0)
        ax_tr.text(span, yy, lab, color=col, va='center', ha='right')
    ax_tr.set_xlim(0, span)
    ax_tr.set_xlabel('time (s)')
    ax_tr.set_ylabel('dF/F')
    ax_tr.set_title('example cell: {} true, {} fixed-CaImAn calls'.format(
        len(t_true), len(_calls(_job(ex, True), fs))))

    cats = np.arange(5)
    width = 0.38
    for j, (_, lab, col, fx) in enumerate(versions):
        counts = []
        for c in bad:
            kk = cells.index(c)
            calls = _calls(_job(c, fx), fs)
            for x in _isolated(np.asarray(meta['true_spikes'][kk], dtype=float)):
                counts.append(np.sum((calls >= x + SPLIT_WIN[0]) & (calls < x + SPLIT_WIN[1])))
        frac = np.bincount(np.minimum(counts, 4), minlength=5) / max(1, len(counts))
        ax_n.bar(cats + (j - 0.5) * width, frac, width=width, color=col, alpha=0.5, lw=0,
                 label=lab)
    ax_n.set_xticks(cats)
    ax_n.set_xticklabels(['0', '1', '2', '3', '4+'])
    ax_n.set_xlabel('calls per isolated true spike')
    ax_n.set_ylabel('frac true spikes')
    ax_n.set_title('over-called cells (n={})'.format(len(bad)))
    ax_n.legend(frameon=False, loc='upper right')

    for _, lab, col, fx in versions:
        for i, c in enumerate(bad):
            r = _job(c, fx)
            n_true = max(1, len(meta['true_spikes'][cells.index(c)]))
            ax_ns.plot(np.arange(len(r['ns'])), r['ns'] / n_true, color=col, lw=0.7,
                       alpha=0.8, label=lab if i == 0 else None)
            
    ax_ns.axhline(1.0, color='k', ls='--', lw=0.6)
    ax_ns.set_yscale('log')
    ax_ns.set_xlabel('sweep')
    ax_ns.set_ylabel('spike count / true count')
    ax_ns.set_title('pruning from CaImAn init')
    ax_ns.legend(frameon=False, loc='upper right')

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'rj_mcmc_split.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def _run_consensus(data_dir, n_workers, timing_dir=_TIMING_DIR):

    import timing_calibration as tc
    from OMSI.deconv import spikes_from_samples

    pop_path = os.path.join(timing_dir, tc._POP_NPZ)
    if not os.path.exists(pop_path):
        raise FileNotFoundError('No population at {}. Run timing_calibration.py '
                                '--mode test --stages population first.'.format(pop_path))
    pop = np.load(pop_path, allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    fs = tc.FS

    noisy, regen = tc._simulate(len(true), fs, float(pop['duration'][0]), tc.POP_SEED)
    if not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(true, regen)):
        raise ValueError('Regenerated population differs from {}.'.format(timing_dir))
    n_frames = noisy.shape[1]

    def _call(_j, r):
        ss = r.pop('ss')
        r['prob'], r['cons_s'] = spikes_from_samples(ss, n_frames, fs, spike_method='map')
        r['last_s'] = _calls(r, fs)
        r['n_samples'] = len(ss)
        return r

    jobs = [{'y': noisy[c], 'fix': True, 'n_sweeps': SPLIT_SWEEPS, 'burn': SPLIT_BURN,
             'seed': c, 'init_tau': None, 'keep_ss': True} for c in range(len(true))]
    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers, post=_call)

    ok = [c for c, r in enumerate(results) if r is not None]
    t_ok = [true[c] for c in ok]
    called = {
        'OMSI':     [pop['OMSI_spikes'][c] for c in ok],
        'FIX_LAST': [results[c]['last_s'] for c in ok],
        'FIX_CONS': [results[c]['cons_s'] for c in ok],
    }
    if 'MATLAB_spikes' in pop.files:
        called['CAIMAN'] = [pop['MATLAB_spikes'][c] for c in ok]

    out = {'cells': np.array(ok), 'fs': fs, 'tol_grid': tc.TOL_GRID,
           'n_true': np.array([len(t) for t in t_ok]),
           'n_samples': np.array([results[c]['n_samples'] for c in ok]),
           'FIX_probs': np.array([results[c]['prob'] for c in ok], dtype=np.float32)}
    for key, spk in called.items():
        spk = [np.asarray(p, dtype=np.float64).ravel() for p in spk]
        print('{}: timing analysis...'.format(key))
        res = tc._timing_analysis(t_ok, spk)
        for k in ('err_s', 'fb', 'prec', 'rec'):
            out['{}_{}'.format(key, k)] = res[k]
        arr = np.empty(len(spk), dtype=object)
        arr[:] = spk
        out['{}_spikes'.format(key)] = arr

    path = os.path.join(data_dir, _CONS_NPZ)
    np.savez(path, **out)
    print('Saved to {} -- {} of {} cells.'.format(path, len(ok), len(true)))


def _run_consensus_thr(data_dir, n_workers, timing_dir=_TIMING_DIR):

    import timing_calibration as tc
    from OMSI.deconv import spikes_from_samples

    path = os.path.join(data_dir, _CONS_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError('No consensus results in {}. Run --stages consensus '
                                'first.'.format(data_dir))
    out = dict(np.load(path, allow_pickle=True))
    pop = np.load(os.path.join(timing_dir, tc._POP_NPZ), allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    fs = tc.FS
    noisy, regen = tc._simulate(len(true), fs, float(pop['duration'][0]), tc.POP_SEED)
    if not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(true, regen)):
        raise ValueError('Regenerated population differs from {}.'.format(timing_dir))
    n_frames = noisy.shape[1]

    def _call(_j, r):
        ss = r.pop('ss')
        _, r['cons_s'] = spikes_from_samples(ss, n_frames, fs, spike_method='map')
        r['last_s'] = _calls(r, fs)
        return r

    cells = [int(c) for c in out['cells']]
    jobs = [{'y': noisy[c], 'fix': True, 'n_sweeps': SPLIT_SWEEPS, 'burn': SPLIT_BURN,
             'seed': c, 'init_tau': None, 'keep_ss': True, 'init_thr': CONS_INIT_THR}
            for c in cells]
    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers, post=_call)

    keep = [k for k, r in enumerate(results) if r is not None]
    if len(keep) < len(cells):
        print('{} chains failed -- dropping those cells from every row.'.format(
            len(cells) - len(keep)))
    spikes = {key: [out['{}_spikes'.format(key)][k] for k in keep]
              for key in CONS_METHODS if '{}_spikes'.format(key) in out}
    spikes['THR_LAST'] = [results[k]['last_s'] for k in keep]
    spikes['THR_CONS'] = [results[k]['cons_s'] for k in keep]
    t_ok = [true[cells[k]] for k in keep]

    for key in ('n_samples', 'FIX_probs'):
        out[key] = out[key][keep]
    out['cells'] = np.array([cells[k] for k in keep])
    out['n_true'] = np.array([len(t) for t in t_ok])
    out['init_thr'] = CONS_INIT_THR
    for key, spk in spikes.items():
        spk = [np.asarray(p, dtype=np.float64).ravel() for p in spk]
        print('{}: timing analysis...'.format(key))
        res = tc._timing_analysis(t_ok, spk)
        for k in ('err_s', 'fb', 'prec', 'rec'):
            out['{}_{}'.format(key, k)] = res[k]
        arr = np.empty(len(spk), dtype=object)
        arr[:] = spk
        out['{}_spikes'.format(key)] = arr

    np.savez(path, **out)
    print('Saved to {} -- {} of {} cells.'.format(path, len(keep), len(cells)))


def _run_counts(data_dir, n_workers, timing_dir=_TIMING_DIR):

    import timing_calibration as tc

    pop = np.load(os.path.join(timing_dir, tc._POP_NPZ), allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    noisy, regen = tc._simulate(len(true), tc.FS, float(pop['duration'][0]), tc.POP_SEED)
    if not all(len(a) == len(b) and np.allclose(a, b) for a, b in zip(true, regen)):
        raise ValueError('Regenerated population differs from {}.'.format(timing_dir))

    sn = np.median(np.abs(np.diff(noisy, axis=1)), axis=1) / 0.6745 / np.sqrt(2.0)
    snr = 1.0 / sn
    burst_isi = 2.0 / (10.0 * tc.FS)
    bursty = np.array([np.sum(np.abs(np.diff(np.sort(t)) - burst_isi) < 1e-9)
                       >= tc.BURSTY_MIN_PAIRS for t in true])
    ok = (~bursty) & (snr >= CNT_SNR_RANGE[0]) & (snr <= CNT_SNR_RANGE[1])
    picked = np.where(ok)[0][np.argsort(snr[ok])][:CNT_N_CELLS].astype(int)

    jobs, keys, cells = [], [], []
    for key, flags in CNT_VARIANTS.items():
        for c in picked:
            job = {'y': noisy[c], 'fix': flags['fix'], 'n_sweeps': SPLIT_SWEEPS,
                   'burn': SPLIT_BURN, 'seed': c, 'init_tau': None}
            if flags.get('thr'):
                job['init_thr'] = CONS_INIT_THR
            jobs.append(job)
            keys.append(key)
            cells.append(c)

    def _keep_ns(_j, r):
        return {'ns': r['ns'], 'n_init': r['n_init']}

    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers, post=_keep_ns)
    n_true = np.array([max(1, len(true[c])) for c in picked], dtype=float)
    out = {'cells': picked, 'n_true': n_true, 'init_thr': CONS_INIT_THR,
           'burn': SPLIT_BURN, 'snr': snr[picked]}
    for key in CNT_VARIANTS:
        ratio = np.full((len(picked), SPLIT_SWEEPS), np.nan)
        init = np.full(len(picked), np.nan)
        for r, k, c in zip(results, keys, cells):
            if k == key and r is not None:
                i = int(np.where(picked == c)[0][0])
                ratio[i, :len(r['ns'])] = r['ns'][:SPLIT_SWEEPS] / n_true[i]
                init[i] = r['n_init']
        out['{}_ratio'.format(key)] = ratio.astype(np.float32)
        out['{}_n_init'.format(key)] = init
    path = os.path.join(data_dir, _CNT_NPZ)
    np.savez(path, **out)
    print('Saved to {}.'.format(path))


def plot_consensus_figure(data_dir=_DEFAULT_DATA_DIR):

    import timing_calibration as tc

    path = os.path.join(data_dir, _CONS_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError('No consensus results in {}. Run --stages consensus.'.format(
            data_dir))
    d = np.load(path, allow_pickle=True)
    fs = float(d['fs'])
    tol_ms = d['tol_grid'] * 1e3
    present = [(k, v) for k, v in CONS_PLOT.items() if '{}_fb'.format(k) in d.files]

    cnt_path = os.path.join(data_dir, _CNT_NPZ)
    cnt = np.load(cnt_path) if os.path.exists(cnt_path) else None
    has_abl = os.path.exists(os.path.join(data_dir, _ABL_NPZ))
    n_top = 3 if has_abl else 2
    width = 7.5 if has_abl else 5.6
    if cnt is None:
        fig = plt.figure(figsize=(width, 2.3))
        outer = gridspec.GridSpec(1, 1, figure=fig)
    else:
        fig = plt.figure(figsize=(width, 6.0))
        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.75, 1.6], hspace=0.35,
                                  top=0.93)
    top = gridspec.GridSpecFromSubplotSpec(1, n_top, subplot_spec=outer[0], wspace=0.45)
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])

    if has_abl:
        ax_t = fig.add_subplot(top[2])
        m, _ = _ablation_metrics(data_dir)
        pts = [(CAIMAN_INIT_THR, 'base')] + [(f['init_thr'], k) for k, (_, f)
                                             in ABL_CONDS.items() if 'init_thr' in f]
        pts = [(x, k) for x, k in pts if np.any(np.isfinite(m[k]['fb']))]
        pts = sorted(pts)
        xs = np.array([x for x, _ in pts])
        fb = np.array([m[k]['fb'] for _, k in pts])
        q25, med, q75 = np.nanpercentile(fb, [25, 50, 75], axis=1)
        color = CONS_PLOT['THR_LAST'][1]
        ax_t.fill_between(xs, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax_t.plot(xs, med, '-', color=color, lw=1.0)
        ax_t.axvline(CAIMAN_INIT_THR, color='k', ls=':', lw=0.8, alpha=0.7,
                     label='CaImAn default')
        ax_t.axvline(CONS_INIT_THR, color='k', ls='--', lw=0.7, alpha=0.6,
                     label='chosen')
        ax_t.set_xlim(0.05, 0.85)
        ax_t.legend(frameon=False, loc='lower right', fontsize=6)
        ax_t.set_ylim(0, 1.02)
        ax_t.set_xlabel('FOOPSI init threshold')
        ax_t.set_ylabel('$F_\\beta$ (100 ms)')

    if cnt is not None:
        bottom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.3,
                                                  hspace=0.45)
        n_true = cnt['n_true']
        for i, k in enumerate(range(min(4, len(n_true)))):
            ax = fig.add_subplot(bottom[i // 2, i % 2])
            top_n = n_true[k]
            for key in CNT_PLOT:
                label, color, ls = CONS_PLOT[key]
                n = cnt['{}_ratio'.format(key)][k] * n_true[k]

                n0 = cnt['{}_n_init'.format(key)][k] if '{}_n_init'.format(key) in cnt.files \
                    else np.nan
                n = np.r_[n0, n]
                ax.plot(np.arange(len(n)), n, ls, color=color, lw=0.8)
                top_n = max(top_n, np.nanmax(n[:CNT_XMAX]))
            ax.axhline(n_true[k], color='k', ls='--', lw=0.6, alpha=0.6)
            ax.axvline(float(cnt['burn']), color='0.5', ls=':', lw=0.7,
                       label='end of burn-in')
            ax.text(0.97, 0.95, 'N={:.0f}'.format(n_true[k]), transform=ax.transAxes,
                    ha='right', va='top')

            ax.set_ylim(0, 1.25 * top_n)
            ax.set_xlim(0, min(len(n) - 1, CNT_XMAX))
            ax.set_title('cell {}'.format(i + 1))
            if i // 2 == 1:
                ax.set_xlabel('sweep')
            if i % 2 == 0:
                ax.set_ylabel('spike count')
            if i == 0:
                ax.legend(frameon=False, loc='upper left', fontsize=6)

    step = 1000.0 / (fs * 10.0)
    bins = np.arange(-100.0 - step / 2, 100.0 + step, step)
    centers = 0.5 * (bins[:-1] + bins[1:])
    handles = []
    for key, (label, color, ls) in present:
        dens, _ = np.histogram(d['{}_err_s'.format(key)] * 1e3, bins=bins, density=True)
        h, = ax_a.plot(centers, dens, ls, color=color, lw=1.0, label=label)
        handles.append(h)
    ax_a.axvspan(-500.0 / fs, 500.0 / fs, color='0.5', alpha=0.10, linewidth=0)
    ax_a.axvline(0, color='k', ls='--', lw=0.7, alpha=0.6)
    ax_a.set_xlim(-75, 75)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel('timing error (ms)')
    ax_a.set_ylabel('matched spikes (density, 1/ms)')

    for key, (label, color, ls) in present:
        fb = d['{}_fb'.format(key)]
        med = np.nanmedian(fb, axis=1)
        mad = tc._mad(fb, axis=1)
        ax_b.fill_between(tol_ms, np.clip(med - mad, 0, 1), np.clip(med + mad, 0, 1),
                          color=color, alpha=0.15, linewidth=0)
        ax_b.plot(tol_ms, med, ls, color=color, lw=1.0)
    ax_b.axvline(1000.0 / fs, color='k', ls='--', lw=0.7, alpha=0.6,
                 label='1 frame ({:.0f} ms)'.format(1000.0 / fs))
    ax_b.legend(frameon=False, loc='lower right', fontsize=6)
    ax_b.set_xlim(0, tol_ms.max())
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel('coincidence window (ms)')
    ax_b.set_ylabel('$F_\\beta$')

    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 0.98))

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'rj_mcmc_consensus.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)

def print_consensus_stats(data_dir=_DEFAULT_DATA_DIR):

    d = np.load(os.path.join(data_dir, _CONS_NPZ), allow_pickle=True)
    tol_ms = d['tol_grid'] * 1e3
    n_true = int(np.sum(d['n_true']))
    print('Consensus: {} cells, {} true spikes, {:.0f} samples per fixed chain.'.format(
        len(d['cells']), n_true, np.median(d['n_samples'])))
    print('  {:<28} {:>8} {:>9} {:>9}'.format('method', 'called', 'med |e|', '90% |e|'))
    for key, (label, _, _) in CONS_METHODS.items():
        if '{}_fb'.format(key) not in d.files:
            continue
        err = np.abs(d['{}_err_s'.format(key)]) * 1e3
        n_called = sum(len(p) for p in d['{}_spikes'.format(key)])
        print('  {:<28} {:>7.2f}x {:>7.1f}ms {:>7.1f}ms'.format(
            label, n_called / max(1, n_true), np.median(err), np.percentile(err, 90)))

    print('  Median F_beta by coincidence window:')
    print('  {:<28}'.format('method') + ''.join('{:>7.0f}'.format(t) for t in tol_ms)
          + '   (ms)')
    for key, (label, _, _) in CONS_METHODS.items():
        if '{}_fb'.format(key) not in d.files:
            continue
        med = np.nanmedian(d['{}_fb'.format(key)], axis=1)
        print('  {:<28}'.format(label) + ''.join('{:>7.3f}'.format(v) for v in med))


def _split_cells(timing_dir):

    import timing_calibration as tc

    pop = np.load(os.path.join(timing_dir, tc._POP_NPZ), allow_pickle=True)
    fix = np.load(os.path.join(timing_dir, tc._FIX_NPZ), allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    called = [np.asarray(x, dtype=np.float64) for x in fix['MATLAB_FIX_spikes']]
    prec = np.array([tc._prf_by_cell(t, x, tc.MATCH_TOL)[0] for t, x in zip(true, called)])
    return np.where(prec < 0.6)[0][:SPLIT_N_BAD], np.where(prec > 0.8)[0][:SPLIT_N_GOOD]


def _run_ablation(data_dir, n_workers, timing_dir=_TIMING_DIR):

    import timing_calibration as tc
    from OMSI.get_init_sample import get_init_sample

    pop = np.load(os.path.join(timing_dir, tc._POP_NPZ), allow_pickle=True)
    true = [np.asarray(t, dtype=np.float64) for t in pop['true_spikes']]
    bad, good = _split_cells(timing_dir)
    rest = np.setdiff1d(np.arange(len(true)), np.concatenate([bad, good]))
    extra = np.sort(np.random.default_rng(7).choice(rest, ABL_N_EXTRA, replace=False))
    cells = np.concatenate([bad, good, extra])
    group = ['bad'] * len(bad) + ['good'] * len(good) + ['random'] * len(extra)

    noisy, regen = tc._simulate(len(true), tc.FS, float(pop['duration'][0]), tc.POP_SEED)
    if not all(np.allclose(true[c], regen[c]) for c in cells):
        raise ValueError('Regenerated population differs from {}.'.format(timing_dir))

    print('Building OMSI inits for {} cells...'.format(len(cells)))
    inits = {}
    for c in cells:
        np.random.seed(int(c))
        sam = get_init_sample(np.asarray(noisy[c], dtype=np.float64), {'f': tc.FS, 'p': 2})
        inits[c] = {k: sam[k] for k in ('spiketimes_', 'A_', 'b_', 'C_in', 'sg', 'lam_', 'g')}

    # Reuse saved chains for the same cells; run only conditions not saved yet.
    path = os.path.join(data_dir, _ABL_NPZ)
    prev = _load(path)
    done = {}
    if prev is not None and np.array_equal(prev[0]['cells'], cells):
        # Failed chains (None) are rerun.
        done = {(int(c), str(k)): r for r, c, k in
                zip(prev[1], prev[0]['cell'], prev[0]['cond']) if r is not None}
        print('Reusing {} saved chains.'.format(len(done)))

    jobs, meta = [], {'cell': [], 'cond': []}
    for c in cells:
        for key, (_, flags) in ABL_CONDS.items():
            if (int(c), key) in done:
                continue
            job = {'y': noisy[c], 'fix': True, 'n_sweeps': SPLIT_SWEEPS,
                   'burn': SPLIT_BURN, 'seed': int(c), 'init_tau': None}
            job.update({k: v for k, v in flags.items() if k != 'init'})
            if flags.get('init'):
                job['init'] = inits[c]
            jobs.append(job)
            meta['cell'].append(c)
            meta['cond'].append(key)

    results = _run_jobs(jobs, os.path.join(data_dir, 'matlab_work'), n_workers) if jobs else []
    for (c, key), r in done.items():
        meta['cell'].append(c)
        meta['cond'].append(key)
        results.append(r)
    true_arr = np.empty(len(cells), dtype=object)
    for k, c in enumerate(cells):
        true_arr[k] = true[c]
    meta.update({'cells': cells, 'group': np.array(group), 'true_spikes': true_arr,
                 'fs': tc.FS,
                 'omsi_init_n': np.array([len(inits[c]['spiketimes_']) for c in cells])})
    _save(path, meta, results)


def _ablation_metrics(data_dir=_DEFAULT_DATA_DIR):

    import timing_calibration as tc

    loaded = _load(os.path.join(data_dir, _ABL_NPZ))
    if loaded is None:
        raise FileNotFoundError('No ablation results in {}. Run --stages ablation.'.format(
            data_dir))
    meta, res = loaded
    fs = float(meta['fs'])
    cells = list(meta['cells'])
    b2 = tc.BETA ** 2
    keys = ('ratio', 'split', 'fb', 'prec', 'rec', 'rise_ms', 'Am')
    m = {cond: {k: np.full(len(cells), np.nan) for k in keys} for cond in ABL_CONDS}
    for r, c, cond in zip(res, meta['cell'], meta['cond']):
        if r is None:
            continue
        k = cells.index(c)
        t = np.asarray(meta['true_spikes'][k], dtype=float)
        calls = _calls(r, fs)
        iso = _isolated(t)
        n_per = [np.sum((calls >= x + SPLIT_WIN[0]) & (calls < x + SPLIT_WIN[1])) for x in iso]
        p, rc = tc._prf_by_cell(t, calls, tc.MATCH_TOL)
        mm = m[str(cond)]
        mm['ratio'][k] = len(calls) / max(1, len(t))
        mm['split'][k] = np.mean(np.array(n_per) >= ABL_SPLIT_MIN) if n_per else np.nan
        mm['prec'][k], mm['rec'][k] = p, rc
        mm['fb'][k] = (1 + b2) * p * rc / (b2 * p + rc) if (b2 * p + rc) > 0 else 0.0
        mm['rise_ms'][k] = np.median(r['tau'][-200:, 0]) * 1000.0 / fs
        mm['Am'][k] = np.mean(np.atleast_1d(r['Am'])[-200:])
    return m, np.asarray(meta['group']).astype(str)


def plot_ablation_figure(data_dir=_DEFAULT_DATA_DIR):

    m, group = _ablation_metrics(data_dir)
    conds = list(ABL_CONDS)
    labels = [ABL_CONDS[c][0] for c in conds]
    bad = group == 'bad'
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    panels = (('ratio', 'calls / true spikes', True),
              ('split', 'frac isolated spikes split', False),
              ('fb', '$F_\\beta$ (100 ms)', False))
    for ax, (key, ylabel, log) in zip(axes, panels):
        for xi, c in enumerate(conds):
            v = m[c][key]
            jit = rng.uniform(-0.15, 0.15, len(v))
            ax.scatter(xi + jit[~bad], v[~bad], s=6, color='0.65', lw=0)
            ax.scatter(xi + jit[bad], v[bad], s=8, color='k', lw=0)
            ax.plot([xi - 0.3, xi + 0.3], [np.nanmedian(v)] * 2, color=COL_FIX, lw=1.2)
        if log:
            ax.set_yscale('log')
            ax.axhline(1.0, color='k', ls='--', lw=0.6)
        else:
            ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=40, ha='right')
        ax.set_ylabel(ylabel)
    axes[0].scatter([], [], s=8, color='k', lw=0, label='over-called (split stage)')
    axes[0].scatter([], [], s=6, color='0.65', lw=0, label='other cells')
    axes[0].plot([], [], color=COL_FIX, lw=1.2, label='median')
    axes[0].legend(frameon=False, loc='upper right', fontsize=6)
    fig.tight_layout()

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'rj_mcmc_ablation.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def print_ablation_stats(data_dir=_DEFAULT_DATA_DIR):

    m, group = _ablation_metrics(data_dir)
    for name, sel in (('Over-called split-stage cells', group == 'bad'),
                      ('All cells', np.ones(len(group), dtype=bool))):
        print('{} (n={}), medians:'.format(name, int(sel.sum())))
        print('  {:<24} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6}'.format(
            'condition', 'calls/n', 'split', 'F_beta', 'prec', 'rec', 'rise ms', 'Am'))
        for c, (label, _) in ABL_CONDS.items():
            v = {k: np.nanmedian(m[c][k][sel]) for k in m[c]}
            print('  {:<24} {:>6.2f}x {:>7.2f} {:>7.3f} {:>7.3f} {:>7.3f} {:>8.1f} {:>6.3f}'.format(
                label, v['ratio'], v['split'], v['fb'], v['prec'], v['rec'], v['rise_ms'],
                v['Am']))


def run_test(data_dir, stages, n_workers):

    os.makedirs(data_dir, exist_ok=True)
    if 'recovery' in stages:
        print('Recovery: {} true rise times x {} cells x bug/fix, {} sweeps...'.format(
            len(REC_RISES_MS), REC_CELLS, REC_SWEEPS))
        _run_recovery(data_dir, n_workers)
    if 'stationary' in stages:
        print('Stationary: {} chains x data/noise x bug/fix, {} sweeps...'.format(
            len(STAT_START_FRACS), STAT_SWEEPS))
        _run_stationary(data_dir, n_workers)
    if 'split' in stages:
        print('Split: timing-population cells x bug/fix, {} sweeps...'.format(SPLIT_SWEEPS))
        _run_split(data_dir, n_workers)
    if 'consensus' in stages:
        print('Consensus: timing population, fixed CaImAn, {} sweeps...'.format(SPLIT_SWEEPS))
        _run_consensus(data_dir, n_workers)
    if 'counts' in stages:
        print('Counts: CaImAn variants x {} cells, {} sweeps...'.format(
            CNT_N_CELLS, SPLIT_SWEEPS))
        _run_counts(data_dir, n_workers)
    if 'consensus_thr' in stages:
        print('Consensus with init thr {:g}: timing population, fixed CaImAn...'.format(
            CONS_INIT_THR))
        _run_consensus_thr(data_dir, n_workers)
    if 'ablation' in stages:
        print('Ablation: {} conditions x split + {} random cells, {} sweeps...'.format(
            len(ABL_CONDS), ABL_N_EXTRA, SPLIT_SWEEPS))
        _run_ablation(data_dir, n_workers)


def _load(path):
    if not os.path.exists(path):
        return None
    f = np.load(path, allow_pickle=True)
    meta = {k: f[k] for k in f.files if k != 'results'}
    return meta, list(f['results'])


def _post(r, key):
    return r[key][r['B']:]


def _ks_uniform(tau):
    u = np.sort(tau[:, 0] / tau[:, 1])
    u = u[np.isfinite(u)]
    if len(u) == 0:
        return np.nan
    n = len(u)
    return max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n))


def _summary(rec, stat):
    s = {}
    if rec is not None:
        meta, res = rec
        for fix, name in ((False, 'bug'), (True, 'fix')):
            rs = [r for r, f in zip(res, meta['fix']) if r is not None and bool(f) == fix]
            if not rs:
                continue
            rr = np.concatenate([_post(r, 'rise_ratio') for r in rs])
            dr = np.concatenate([_post(r, 'decay_ratio') for r in rs])
            ra = np.concatenate([_post(r, 'rise_acc') for r in rs]).astype(bool)
            da = np.concatenate([_post(r, 'decay_acc') for r in rs]).astype(bool)
            dz = np.concatenate([_post(r, 'rise_dssr') / (2 * _post(r, 'rise_sg') ** 2)
                                 for r in rs])
            s['rise_ratio', name] = rr
            s['decay_ratio', name] = dr
            s['rise_one', name] = np.mean(np.abs(rr[np.isfinite(rr)] - 1.0) < 1e-9)
            s['rise_acc', name] = np.mean(ra)
            s['decay_acc', name] = np.mean(da)
            s['dz_acc', name] = dz[ra & np.isfinite(dz)]
            s['bad_frac', name] = np.mean(s['dz_acc', name] > BAD_MOVE)
            s['n_acc', name] = int(np.sum(ra))

            post_ms, init_ms, true_ms = [], [], []
            for r, f, t in zip(res, meta['fix'], meta['true_rise_ms']):
                if r is None or bool(f) != fix:
                    continue
                post_ms.append(np.nanmedian(_post(r, 'tau')[:, 0]) * 1000.0 / FS)
                init_ms.append(r['tau0'][0] * 1000.0 / FS)
                true_ms.append(float(t))
            s['post_ms', name] = np.array(post_ms)
            s['init_ms', name] = np.array(init_ms)
            s['true_ms', name] = np.array(true_ms)


    if stat is not None:
        meta, res = stat
        for fix, name in ((False, 'bug'), (True, 'fix')):
            for trace in ('data', 'noise'):
                rs = [r for r, f, t in zip(res, meta['fix'], meta['trace'])
                      if r is not None and bool(f) == fix and t == trace]
                if not rs:
                    continue
                tau = np.concatenate([_post(r, 'tau') for r in rs]) * 1000.0 / FS
                s['stat_tau', name, trace] = tau
                s['stat_ks', name, trace] = _ks_uniform(tau)
                s['stat_traj', name, trace] = [r['tau'][:, 0] * 1000.0 / FS for r in rs]
                s['stat_rise_acc', name, trace] = np.mean(
                    np.concatenate([_post(r, 'rise_acc') for r in rs]))
    return s


def plot_figure(data_dir=_DEFAULT_DATA_DIR):
    rec = _load(os.path.join(data_dir, _REC_NPZ))
    stat = _load(os.path.join(data_dir, _STAT_NPZ))
    if rec is None and stat is None:
        raise FileNotFoundError('No results in {}. Run --mode test first.'.format(data_dir))
    s = _summary(rec, stat)

    fig = plt.figure(figsize=(7.5, 4.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.45, hspace=0.6)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1:])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])

    versions = (('bug', 'CaImAn', COL_BUG), ('fix', 'rise-time fix', COL_FIX))
    hist_order = versions[::-1]

    def _hist(ax, x, bins, w, col, label):

        ax.hist(x, bins=bins, weights=w, color=col, histtype='stepfilled', alpha=0.2, lw=0)
        ax.hist(x, bins=bins, weights=w, color=col, histtype='step', lw=1.0, label=label)

    bins = np.linspace(0, 1, 21)
    for ax, key, title in ((ax_a, 'rise_ratio', 'rise-time step'),
                           (ax_b, 'decay_ratio', 'decay-time step')):
        for name, label, col in hist_order:
            if (key, name) not in s:
                continue
            p = np.minimum(1.0, s[key, name])
            p = p[np.isfinite(p)]
            acc = s[key.replace('ratio', 'acc'), name]
            w = np.full(len(p), 1.0 / len(p))
            _hist(ax, p, bins, w, col, '{} ({:.0f}% acc.)'.format(label, 100 * acc))
        ax.set_xlabel('MH acceptance prob.')
        ax.set_ylabel('frac sweeps')
        ax.set_title(title)

        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], frameon=False, loc='upper center')

    for name, label, col in versions:
        for k, tr in enumerate(s.get(('stat_traj', name, 'data'), [])):
            ax_d.plot(np.arange(len(tr)), tr, color=col, lw=0.7, alpha=0.8,
                      label=label if k == 0 else None)
    if ('stat_traj', 'bug', 'data') in s:
        ax_d.axhline(STAT_RISE_MS, color='k', ls='--', lw=0.6)
        ax_d.axvline(STAT_BURN, color=COL_NOISE, ls=':', lw=0.6)
        ax_d.set_xlim(0, 3 * STAT_BURN)
        ax_d.set_xlabel('sweep')
        ax_d.set_ylabel('rise time (ms)')
        ax_d.set_title('long chains, true {:g} ms'.format(STAT_RISE_MS))
        ax_d.legend(frameon=False, loc='upper right')

    if ('dz_acc', 'bug') in s:
        allz = np.concatenate([s['dz_acc', n] for n, _, _ in versions if ('dz_acc', n) in s])
        top = max(10.0, np.nanmax(np.abs(allz)))
        pos = np.logspace(0, np.log10(top) + 0.1, 16)
        edges = np.concatenate([-pos[::-1], np.linspace(-1, 1, 9)[1:-1], pos])
        for name, label, col in hist_order:
            z = s['dz_acc', name]
            w = np.full(len(z), 1.0 / max(1, len(z)))
            _hist(ax_g, z, edges, w, col, '{} ({:.0f}% > {:g})'.format(
                label, 100 * s['bad_frac', name], BAD_MOVE))
        ax_g.set_xscale('symlog', linthresh=1.0)
        ticks = [v for v in (-100, -1, 0, 1, 100, 10000) if abs(v) <= top]
        ax_g.set_xticks(ticks)
        ax_g.set_xticklabels(['{:g}'.format(v) for v in ticks])
        ax_g.minorticks_off()

        ax_g.axvline(0, color='k', lw=0.5, ymax=0.7)
        ax_g.axvline(BAD_MOVE, color='k', ls='--', lw=0.6, ymax=0.7)
        ax_g.set_xlabel(r'$-\Delta$LLH (proposed vs current $\tau_{rise}$)')
        ax_g.set_title('accepted rise moves')
        ax_g.set_ylabel('frac accepted')
        ax_g.set_ylim(0, ax_g.get_ylim()[1] * 1.45)
        h, l = ax_g.get_legend_handles_labels()
        ax_g.legend(h[::-1], l[::-1], frameon=False, loc='upper left')

    for sfx in ('png', 'svg'):
        out = os.path.join(data_dir, 'rj_mcmc_diag.{}'.format(sfx))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved to {}.'.format(out))
    plt.close(fig)


def print_stats(data_dir=_DEFAULT_DATA_DIR):

    s = _summary(_load(os.path.join(data_dir, _REC_NPZ)),
                 _load(os.path.join(data_dir, _STAT_NPZ)))
    for name in ('bug', 'fix'):
        if ('rise_acc', name) not in s:
            continue
        print('{}:'.format(name))
        print('  Rise ratio == 1: {:.1f}% of sweeps. Rise accepted: {:.1f}%. '
              'Decay accepted: {:.1f}%.'.format(100 * s['rise_one', name],
                                                100 * s['rise_acc', name],
                                                100 * s['decay_acc', name]))
        print('  Accepted rise moves with dSSR/2sg^2 > {:g}: {:.1f}% of {}.'.format(
            BAD_MOVE, 100 * s['bad_frac', name], s['n_acc', name]))
        for t in REC_RISES_MS:
            m = s['true_ms', name] == t
            print('  True rise {:>5g} ms: posterior median {:6.1f} ms '
                  '(IQR {:.1f}-{:.1f}), init {:6.1f} ms.'.format(
                      t, np.median(s['post_ms', name][m]),
                      *np.percentile(s['post_ms', name][m], [25, 75]),
                      np.median(s['init_ms', name][m])))
    for name in ('bug', 'fix'):
        for trace in ('data', 'noise'):
            if ('stat_ks', name, trace) not in s:
                continue
            print('Stationary {} on {}: KS(tau_r/tau_d vs U) = {:.3f}, rise accepted '
                  '{:.1f}%, median tau_r {:.1f} ms.'.format(
                      name, trace, s['stat_ks', name, trace],
                      100 * s['stat_rise_acc', name, trace],
                      np.nanmedian(s['stat_tau', name, trace][:, 0])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Diagnostics for the rise-time MH bug in CaImAn cont_ca_sampler'
    )
    parser.add_argument('--mode', required=True, choices=['test', 'plot', 'stats'])
    parser.add_argument('--stages', nargs='+', default=['recovery', 'stationary'],
                        choices=['recovery', 'stationary', 'split', 'consensus',
                                 'consensus_thr', 'counts', 'ablation'],
                        help='Stages to run (test mode)')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for reading/writing results')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1),
                        help='Parallel MATLAB sessions')
    args = parser.parse_args()

    if args.mode == 'test':
        run_test(args.data_dir, args.stages, args.workers)
        if set(args.stages) & {'recovery', 'stationary'}:
            plot_figure(args.data_dir)
            print_stats(args.data_dir)
        if 'split' in args.stages:
            plot_split_figure(args.data_dir)
        if set(args.stages) & {'consensus', 'consensus_thr', 'counts'}:
            plot_consensus_figure(args.data_dir)
            print_consensus_stats(args.data_dir)
        if 'ablation' in args.stages:
            plot_ablation_figure(args.data_dir)
            print_ablation_stats(args.data_dir)
    elif args.mode == 'plot':
        if any(os.path.exists(os.path.join(args.data_dir, f)) for f in (_REC_NPZ, _STAT_NPZ)):
            plot_figure(args.data_dir)
        if os.path.exists(os.path.join(args.data_dir, _SPLIT_NPZ)):
            plot_split_figure(args.data_dir)
        if os.path.exists(os.path.join(args.data_dir, _CONS_NPZ)):
            plot_consensus_figure(args.data_dir)
        if os.path.exists(os.path.join(args.data_dir, _ABL_NPZ)):
            plot_ablation_figure(args.data_dir)
    elif args.mode == 'stats':
        print_stats(args.data_dir)
        if os.path.exists(os.path.join(args.data_dir, _CONS_NPZ)):
            print_consensus_stats(args.data_dir)
        if os.path.exists(os.path.join(args.data_dir, _ABL_NPZ)):
            print_ablation_stats(args.data_dir)
