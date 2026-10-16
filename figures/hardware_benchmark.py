# -*- coding: utf-8 -*-
"""
figures/hardware_benchmark.py

DMM, September 2026
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from simulation_helpers import generate_synthetic_data
from OMSI._win_perf import no_power_throttling

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

METHOD_COLORS = {
    'OMSI':        '#4C72B0',
    'CaImAn MCMC': '#DD8452',
    'OASIS':       '#55A868',
    'CASCADE_GPU': '#8172B3',
    'CASCADE_CPU': '#B39DDB',
}

METHOD_ORDER = ['OMSI', 'CaImAn MCMC', 'OASIS', 'CASCADE_CPU', 'CASCADE_GPU']

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'hardware'
)

FS   = 30.0
TAU  = 1.2
SNR  = 5.0
SEED = 1234

CELL_SWEEP     = [50, 200]
CELL_DURATION  = 300.0
DUR_SWEEP      = [1200.0]
DUR_N_CELLS    = 100

QUICK_CELL_SWEEP    = [20, 50]
QUICK_CELL_DURATION = 120.0
QUICK_DUR_SWEEP     = [300.0]
QUICK_DUR_N_CELLS   = 20

GEN_CHUNK = 100

MATLAB_MAX_CELLS = 25
MATLAB_SWEEPS    = 500

CASCADE_ENV = os.environ.get('CASCADE_ENV', 'cascade')


def _gpu_info():

    empty = {'name': None, 'memory_mb': None, 'compute_cap': None,
             'clock_max_mhz': None}

    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,compute_cap,clocks.max.sm',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return empty

    if out.returncode != 0:
        return empty

    rows = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not rows:
        return empty

    parts = [p.strip() for p in rows[0].split(',')]
    if len(parts) < 4:
        return empty

    def _num(text):
        try:
            return float(text)
        except ValueError:
            return None

    return {
        'name': parts[0],
        'memory_mb': _num(parts[1]),
        'compute_cap': parts[2],
        'clock_max_mhz': _num(parts[3]),
    }


def _conda_launcher():

    candidates = [os.environ.get('CONDA_EXE', '').strip()]

    candidates.append(shutil.which('conda'))

    home = os.path.expanduser('~')
    if platform.system() == 'Windows':
        for root in (os.path.join(home, 'anaconda3'),
                     os.path.join(home, 'miniconda3'),
                     os.path.join(home, 'miniforge3'),
                     r'C:\ProgramData\anaconda3',
                     r'C:\ProgramData\miniconda3'):
            candidates.append(os.path.join(root, 'Scripts', 'conda.exe'))
            candidates.append(os.path.join(root, 'condabin', 'conda.bat'))
    else:
        for root in (os.path.join(home, 'anaconda3'),
                     os.path.join(home, 'miniconda3'),
                     os.path.join(home, 'miniforge3'),
                     '/opt/conda'):
            candidates.append(os.path.join(root, 'bin', 'conda'))

    for path in candidates:
        if path and os.path.exists(path):
            if path.lower().endswith(('.bat', '.cmd')):
                return [os.environ.get('COMSPEC', 'cmd.exe'), '/c', path]
            return [path]

    return None


def _matlab_release():

    try:
        from run_pnev_MCMC import find_matlab
        exe = find_matlab()
    except Exception:
        return None, None

    if not exe:
        return None, None

    match = re.search(r'R(\d{4}[ab])', exe)
    return (match.group(1) if match else 'unknown'), exe


def _machine_info():

    gpu = _gpu_info()
    matlab_release, matlab_exe = _matlab_release()

    info = {
        'hostname':      socket.gethostname(),
        'platform':      platform.platform(),
        'processor':     platform.processor() or platform.machine(),
        'logical_cores': os.cpu_count(),
        'physical_cores': None,
        'cpu_freq_max_mhz': None,
        'ram_gb':        None,
        'gpu':           gpu['name'],
        'gpu_memory_mb': gpu['memory_mb'],
        'gpu_compute_cap': gpu['compute_cap'],
        'gpu_clock_max_mhz': gpu['clock_max_mhz'],
        'matlab_release': matlab_release,
        'matlab_exe':    matlab_exe,
        'python':        sys.version.split()[0],
        'numpy':         np.__version__,
        'ray':           None,
        'timestamp':     datetime.now().isoformat(timespec='seconds'),
    }

    try:
        import psutil
        info['physical_cores'] = psutil.cpu_count(logical=False)
        info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        freq = psutil.cpu_freq()
        if freq is not None and freq.max:
            info['cpu_freq_max_mhz'] = round(freq.max, 0)
    except Exception:
        pass

    try:
        import ray
        info['ray'] = ray.__version__
    except Exception:
        pass

    try:
        with open('/proc/cpuinfo', 'r') as fh:
            for line in fh:
                if line.startswith('model name'):
                    info['processor'] = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass

    return info


def _ensure_ray_dir(ray_dir=None):

    from OMSI import _config

    config = _config._load()
    existing = config.get('ray_dir', '').strip()
    if ray_dir is None and existing:
        return existing

    path = ray_dir or os.path.join(tempfile.gettempdir(), 'omsi_ray')
    os.makedirs(path, exist_ok=True)
    config['ray_dir'] = path
    _config._save(config)
    print('Ray scratch directory: {}.'.format(path))
    return path


def _generate(n_cells, duration, seed=SEED):

    chunks_dff, chunks_spk = [], []
    done = 0
    chunk_idx = 0

    while done < n_cells:
        n_this = min(GEN_CHUNK, n_cells - done)
        np.random.seed(seed + chunk_idx)
        dff, spikes, _, _, _, _ = generate_synthetic_data(
            n_cells=n_this, fs=FS, duration=duration, tau=TAU, snr=SNR
        )
        chunks_dff.append(np.asarray(dff, dtype=np.float32))
        chunks_spk.extend(spikes)
        done += n_this
        chunk_idx += 1

    return np.concatenate(chunks_dff, axis=0), chunks_spk


def _oasis_spikes_from_s(s, sigma, fs, height=1.0):

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(s, height=height * sigma)
    return peaks / fs


def _run_oasis(dff, fs, tau):

    try:
        from oasis.functions import deconvolve
    except Exception as exc:
        print('    OASIS unavailable: {}.'.format(exc))
        return None

    n_cells = dff.shape[0]
    sigmas = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / (0.6745 * np.sqrt(2))
    sigmas = np.maximum(sigmas, 1e-9)
    g = np.exp(-1.0 / (fs * tau))

    t0 = time.perf_counter()
    n_spikes = 0
    for i in range(n_cells):
        _, s, _, _, _ = deconvolve(dff[i].astype(np.float64), g=(g,),
                                   sn=sigmas[i], penalty=1)
        n_spikes += len(_oasis_spikes_from_s(s, sigmas[i], fs))
    wall = time.perf_counter() - t0

    print('    OASIS: {:.1f}s'.format(wall))
    return {'wall_s': wall, 'cpu_s': None, 'n_spikes': int(n_spikes), 'f1': None}


def _run_cascade(dff, fs, data_dir, prefix, device):

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'run_cascade_subprocess.py')
    input_path  = os.path.join(data_dir, '{}_input.npz'.format(prefix))
    output_path = os.path.join(data_dir, '{}_output.npz'.format(prefix))

    launcher = _conda_launcher()
    if launcher is None:
        print('    CASCADE ({}) skipped: conda not found. Set CONDA_EXE or put '
              'conda on PATH.'.format(device.upper()))
        return None

    np.savez(input_path, dff=dff.astype(np.float32), fs=np.float32(fs))

    cmd = launcher + ['run', '-n', CASCADE_ENV, 'python', script,
                      '--mode', 'inference', '--input', input_path,
                      '--output', output_path, '--device', device]

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or '').strip().splitlines()
        print('    CASCADE ({}) failed (exit {}).'.format(
            device.upper(), exc.returncode))
        for line in tail[-6:]:
            print('      {}'.format(line))
        return None
    except OSError as exc:
        print('    CASCADE ({}) could not launch conda at {}: {}.'.format(
            device.upper(), launcher[0], exc))
        return None
    wall_total = time.perf_counter() - t0

    try:
        result = np.load(output_path, allow_pickle=True)
        inference_s = float(result['cascade_time'])
        n_spikes = int(sum(len(s) for s in result['cascade_spikes']))
    except Exception as exc:
        print('    CASCADE ({}) output unreadable: {}.'.format(device.upper(), exc))
        return None

    for path in (input_path, output_path):
        try:
            os.remove(path)
        except OSError:
            pass

    print('    CASCADE ({}): {:.1f}s inference, {:.1f}s including startup'.format(
        device.upper(), inference_s, wall_total))
    return {'wall_s': inference_s, 'cpu_s': None, 'n_spikes': n_spikes,
            'f1': None, 'wall_with_startup_s': wall_total}


def _run_matlab(dff, fs, tau, data_dir, max_cells=MATLAB_MAX_CELLS):

    from run_pnev_MCMC import run_matlab_pnevMCMC

    n_run = dff.shape[0] if not max_cells else min(int(max_cells), dff.shape[0])
    subset = dff[:n_run]

    t0 = time.perf_counter()
    try:
        spikes, _, _, _ = run_matlab_pnevMCMC(
            subset, fs=fs, tau=tau, n_sweeps=MATLAB_SWEEPS,
            work_dir=data_dir, verbose=True,
        )
    except Exception as exc:
        print('    CaImAn MCMC failed: {}.'.format(exc))
        return None
    wall = time.perf_counter() - t0

    n_spikes = int(sum(len(s) for s in spikes))

    if n_spikes == 0:
        print('    CaImAn MCMC returned no spikes -- treating as unavailable.')
        return None

    print('    CaImAn MCMC: {:.1f}s on {} of {} cells'.format(
        wall, n_run, dff.shape[0]))

    return {'wall_s': wall, 'cpu_s': None, 'n_spikes': n_spikes, 'f1': None,
            'n_cells_run': n_run, 'sweeps': MATLAB_SWEEPS}


def _run_omsi(dff, true_spikes, fs):

    import OMSI

    t0 = time.perf_counter()
    try:
        res = OMSI.deconv(
            dff, params={'f': fs, 'p': 2, 'auto_stop': True},
            true_spikes=true_spikes, benchmark=True,
        )
    except Exception as exc:
        print('    OMSI failed: {}.'.format(exc))
        return None
    wall = time.perf_counter() - t0

    cpu_s = float(np.sum(res['optim_times_per_cell']))
    f1 = res['optim_F1']
    f1_med = float(np.median(f1)) if f1 is not None else None
    n_spikes = int(sum(len(s) for s in res['optim_spikes']))

    print('    OMSI: {:.1f}s wall, {:.1f}s CPU, speedup {:.1f}x, median F1 {:.3f}'.format(
        wall, cpu_s, cpu_s / max(wall, 1e-9), f1_med if f1_med is not None else float('nan')))

    return {
        'wall_s': wall,
        'cpu_s': cpu_s,
        'parallel_speedup': cpu_s / max(wall, 1e-9),
        'n_spikes': n_spikes,
        'f1': f1_med,
        'median_sweeps': float(np.median(res['optim_nsamples'])),
    }


def _conditions(quick=False):

    if quick:
        cell_sweep, cell_dur = QUICK_CELL_SWEEP, QUICK_CELL_DURATION
        dur_sweep, dur_cells = QUICK_DUR_SWEEP, QUICK_DUR_N_CELLS
    else:
        cell_sweep, cell_dur = CELL_SWEEP, CELL_DURATION
        dur_sweep, dur_cells = DUR_SWEEP, DUR_N_CELLS

    conds = [{'sweep': 'cells', 'n_cells': n, 'duration': cell_dur,
              'matlab_ref': False}
             for n in cell_sweep]
    conds += [{'sweep': 'duration', 'n_cells': dur_cells, 'duration': d,
               'matlab_ref': False}
              for d in dur_sweep]

    ref = min(conds, key=lambda c: c['n_cells'] * c['duration'])
    ref['matlab_ref'] = True

    return conds


def run_benchmark(data_dir=_DEFAULT_DATA_DIR, quick=False, skip=(), ray_dir=None,
                  matlab_max_cells=MATLAB_MAX_CELLS):

    os.makedirs(data_dir, exist_ok=True)
    skip = set(s.lower() for s in skip)

    info = _machine_info()
    print('\nMachine')
    for key in ('hostname', 'processor', 'logical_cores', 'physical_cores',
                'cpu_freq_max_mhz', 'ram_gb', 'gpu', 'gpu_memory_mb',
                'matlab_release', 'python', 'numpy', 'ray'):
        print('  {:18s} {}'.format(key, info[key]))

    if 'omsi' not in skip:
        _ensure_ray_dir(ray_dir)

    conds = _conditions(quick)
    total_cell_seconds = sum(c['n_cells'] * c['duration'] for c in conds)
    print('\nWorkload: {} conditions, {:.0f} cell-seconds total.'.format(
        len(conds), total_cell_seconds))
    if quick:
        print('Quick mode -- results are NOT comparable across machines.')

    records = []
    t_start = time.perf_counter()

    for cond in conds:
        n_cells, duration = cond['n_cells'], cond['duration']
        print('\n{} sweep: {} cells, {:.0f}s at {:.0f} Hz...'.format(
            cond['sweep'], n_cells, duration, FS))

        t0 = time.perf_counter()
        dff, true_spikes = _generate(n_cells, duration)
        gen_s = time.perf_counter() - t0
        print('    Generated in {:.1f}s ({} frames per cell).'.format(
            gen_s, dff.shape[1]))

        base = dict(cond)
        base.update({'fs': FS, 'tau': TAU, 'snr': SNR,
                     'n_frames': int(dff.shape[1]), 'generation_s': gen_s})

        methods = []
        if 'omsi' not in skip:
            methods.append(('OMSI', lambda: _run_omsi(dff, true_spikes, FS)))
        if 'oasis' not in skip:
            methods.append(('OASIS', lambda: _run_oasis(dff, FS, TAU)))
        if 'cascade_cpu' not in skip:
            methods.append(('CASCADE_CPU', lambda: _run_cascade(
                dff, FS, data_dir, 'hwbench_{}c_{:.0f}s_cpu'.format(n_cells, duration), 'cpu')))
        if 'cascade_gpu' not in skip and info['gpu'] is not None:
            methods.append(('CASCADE_GPU', lambda: _run_cascade(
                dff, FS, data_dir, 'hwbench_{}c_{:.0f}s_gpu'.format(n_cells, duration), 'gpu')))
        if 'matlab' not in skip and cond.get('matlab_ref'):
            methods.append(('CaImAn MCMC', lambda: _run_matlab(
                dff, FS, TAU, data_dir, matlab_max_cells)))

        for name, fn in methods:
            rec = fn()
            if rec is None:
                continue
            row = dict(base)
            row['method'] = name
            row.update(rec)
            row['cells_measured'] = int(rec.get('n_cells_run', n_cells))
            records.append(row)

        del dff, true_spikes

    elapsed = time.perf_counter() - t_start
    print('\nFinished in {:.1f} min.'.format(elapsed / 60.0))

    out = {
        'machine': info,
        'config': {
            'quick': quick, 'fs': FS, 'tau': TAU, 'snr': SNR, 'seed': SEED,
            'cell_sweep': QUICK_CELL_SWEEP if quick else CELL_SWEEP,
            'cell_duration': QUICK_CELL_DURATION if quick else CELL_DURATION,
            'dur_sweep': QUICK_DUR_SWEEP if quick else DUR_SWEEP,
            'dur_n_cells': QUICK_DUR_N_CELLS if quick else DUR_N_CELLS,
            'matlab_max_cells': matlab_max_cells,
            'matlab_sweeps': MATLAB_SWEEPS,
            'total_elapsed_s': elapsed,
        },
        'records': records,
    }

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'quick_' if quick else ''
    out_path = os.path.join(
        data_dir, 'hardware_{}{}_{}.json'.format(tag, info['hostname'], stamp))
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Saved results to {}.'.format(out_path))


def _load_runs(data_dir):

    runs = []
    for path in sorted(glob.glob(os.path.join(data_dir, 'hardware_*.json'))):
        try:
            with open(path, 'r') as fh:
                runs.append(json.load(fh))
        except Exception as exc:
            print('Skipping {}: {}.'.format(os.path.basename(path), exc))

    runs.sort(key=lambda r: r['machine']['hostname'])
    return runs


def _throughput(record):

    cells = record.get('cells_measured', record['n_cells'])
    return cells * record['duration'] / max(record['wall_s'], 1e-9)


def report(data_dir=_DEFAULT_DATA_DIR):

    runs = _load_runs(data_dir)
    if not runs:
        print('No results in {} -- run with --mode run first.'.format(data_dir))
        return

    print('\nMACHINES')
    cols = '  {:14s} {:>6s} {:>6s} {:>7s} {:>7s}  {:22s} {:>8s}  {:34s}'
    print(cols.format('machine', 'cores', 'logic', 'GHz', 'RAM GB',
                      'GPU', 'MATLAB', 'CPU'))
    for run in runs:
        m = run['machine']
        ghz = '{:.1f}'.format(m['cpu_freq_max_mhz'] / 1000.0) \
              if m.get('cpu_freq_max_mhz') else '?'
        print(cols.format(
            m['hostname'][:14],
            str(m['physical_cores'] or '?'), str(m['logical_cores'] or '?'),
            ghz, str(m['ram_gb'] or '?'),
            (m['gpu'] or 'none')[:22],
            str(m.get('matlab_release') or 'none'),
            m['processor'][:34]))
        if run['config'].get('quick'):
            print('  {:14s} QUICK workload -- not comparable with full runs.'.format(''))

    hosts = [r['machine']['hostname'] for r in runs]
    print('\nWALL-CLOCK SECONDS' + (' (n = cells run, when a subset)' if any(
        rec.get('cells_measured', rec['n_cells']) != rec['n_cells']
        for run in runs for rec in run['records']) else ''))

    head = '  {:12s} {:>6s} {:>7s}'.format('method', 'cells', 'dur_s')
    head += ''.join('{:>14s}'.format(h[:13]) for h in hosts)
    print(head)

    conds = sorted({(r['method'], r['n_cells'], r['duration'])
                    for run in runs for r in run['records']},
                   key=lambda k: (METHOD_ORDER.index(k[0])
                                  if k[0] in METHOD_ORDER else 99, k[1], k[2]))

    for method, n_cells, duration in conds:
        line = '  {:12s} {:6d} {:7.0f}'.format(method, n_cells, duration)
        for run in runs:
            cell = ''
            for r in run['records']:
                if (r['method'], r['n_cells'], r['duration']) == (method, n_cells, duration):
                    measured = r.get('cells_measured', r['n_cells'])
                    cell = '{:.1f}'.format(r['wall_s'])
                    if measured != r['n_cells']:
                        cell += ' (n={})'.format(measured)
            line += '{:>14s}'.format(cell or '-')
        print(line)

    print('\nTHROUGHPUT, cell-seconds of recording per wall second (median over conditions)')
    head = '  {:14s}'.format('machine')
    head += ''.join('{:>14s}'.format(m) for m in METHOD_ORDER)
    print(head)
    for run in runs:
        by_method = {}
        for r in run['records']:
            by_method.setdefault(r['method'], []).append(_throughput(r))
        line = '  {:14s}'.format(run['machine']['hostname'][:14])
        for method in METHOD_ORDER:
            vals = by_method.get(method)
            line += '{:>14s}'.format('{:.0f}'.format(float(np.median(vals)))
                                     if vals else '-')
        print(line)

    print('\nOMSI PARALLEL SPEEDUP and CORRECTNESS')
    print('  {:14s} {:>10s} {:>8s} {:>10s}'.format(
        'machine', 'speedup', 'cores', 'median F1'))
    for run in runs:
        recs = [r for r in run['records'] if r['method'] == 'OMSI']
        if not recs:
            continue
        sp = [r['parallel_speedup'] for r in recs if r.get('parallel_speedup')]
        f1 = [r['f1'] for r in recs if r.get('f1') is not None]
        print('  {:14s} {:>10s} {:>8s} {:>10s}'.format(
            run['machine']['hostname'][:14],
            '{:.1f}x'.format(float(np.median(sp))) if sp else '-',
            str(run['machine']['logical_cores'] or '?'),
            '{:.3f}'.format(float(np.median(f1))) if f1 else '-'))
    print('\n  F1 should match across machines; a difference means a numerical '
          'or version problem, not a hardware one.\n')


def plot_figure(data_dir=_DEFAULT_DATA_DIR):

    runs = _load_runs(data_dir)
    if not runs:
        print('No results in {} -- run with --mode run first.'.format(data_dir))
        return

    shared = None
    for run in runs:
        conds = {(r['n_cells'], r['duration']) for r in run['records']}
        shared = conds if shared is None else (shared & conds)
    if not shared:
        print('No condition is shared by all machines -- nothing to plot.')
        return

    def _n_methods(cond):
        return len({r['method'] for run in runs for r in run['records']
                    if (r['n_cells'], r['duration']) == cond})

    n_cells, duration = max(shared, key=lambda c: (_n_methods(c), c[0] * c[1]))

    methods = [m for m in METHOD_ORDER
               if any(r['method'] == m and (r['n_cells'], r['duration']) == (n_cells, duration)
                      for run in runs for r in run['records'])]
    hosts = [r['machine']['hostname'] for r in runs]

    fig, ax = plt.subplots(figsize=(max(2.6, 0.55 * len(hosts) * len(methods)), 2.6),
                           dpi=300)

    width = 0.8 / max(len(methods), 1)
    xpos = np.arange(len(hosts))

    for j, method in enumerate(methods):
        vals, notes = [], []
        for run in runs:
            hit = [r for r in run['records']
                   if r['method'] == method
                   and (r['n_cells'], r['duration']) == (n_cells, duration)]
            vals.append(hit[0]['wall_s'] if hit else np.nan)
            measured = hit[0].get('cells_measured', n_cells) if hit else n_cells
            notes.append(measured != n_cells)

        offs = xpos - 0.4 + width * (j + 0.5)
        bars = ax.bar(offs, vals, width=width * 0.92,
                      color=METHOD_COLORS.get(method, '#888888'), label=method)

        for bar, note in zip(bars, notes):
            if note and np.isfinite(bar.get_height()):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                        '*', ha='center', va='bottom', fontsize=6)

    ax.set_yscale('log')
    ax.set_xticks(xpos)
    ax.set_xticklabels([h[:14] for h in hosts], rotation=20, ha='right')
    ax.set_ylabel('wall-clock seconds')
    ax.set_title('{} cells, {:.0f} s at {:.0f} Hz'.format(n_cells, duration, FS),
                 fontsize=7, pad=16)

    finite = [r['wall_s'] for run in runs for r in run['records']
              if (r['n_cells'], r['duration']) == (n_cells, duration)
              and np.isfinite(r['wall_s']) and r['wall_s'] > 0]
    if finite:
        ax.set_ylim(min(finite) / 4.0, max(finite) * 3.0)

    ax.legend(frameon=False, fontsize=5.5, ncol=min(len(methods), 5),
              loc='lower center', bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.2, handlelength=1.2)

    if any(r.get('cells_measured', r['n_cells']) != r['n_cells']
           for run in runs for r in run['records']):
        ax.annotate('* run on a subset of cells', xy=(0.99, 0.02),
                    xycoords='axes fraction', ha='right', fontsize=5)

    fig.tight_layout()
    for ext in ('png', 'svg'):
        out = os.path.join(data_dir, 'hardware_benchmark.{}'.format(ext))
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print('Saved figure to {}.'.format(out))
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description='Cross-machine timing benchmark for OMSI, OASIS, and CASCADE.'
    )
    parser.add_argument('--mode', default='run',
                        choices=['run', 'report', 'plot', 'all'],
                        help='Run the benchmark, print tables, draw the figure, or all.')
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory for result JSONs.')
    parser.add_argument('--quick', action='store_true',
                        help='Small smoke-test workload, not comparable across machines.')
    parser.add_argument('--skip', default='',
                        help='Comma-separated methods to skip: '
                             'omsi, oasis, cascade_cpu, cascade_gpu, matlab.')
    parser.add_argument('--matlab-max-cells', type=int, default=MATLAB_MAX_CELLS,
                        help='Cells per condition given to CaImAn MCMC. '
                             '0 runs every cell and takes many hours.')
    parser.add_argument('--ray-dir', default=None,
                        help='Ray scratch directory. Defaults to the system temp dir.')
    args = parser.parse_args()

    skip = [s.strip() for s in args.skip.split(',') if s.strip()]

    if args.mode in ('run', 'all'):
        with no_power_throttling(verbose=True):
            run_benchmark(args.data_dir, quick=args.quick, skip=skip,
                          ray_dir=args.ray_dir, matlab_max_cells=args.matlab_max_cells)
    if args.mode in ('report', 'all'):
        report(args.data_dir)
    if args.mode in ('plot', 'all'):
        plot_figure(args.data_dir)


if __name__ == '__main__':
    main()
