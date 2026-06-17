#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

def _dff_kurtosis(fluo):
    """Excess kurtosis computed on a dF/F-normalised trace (8th-pct baseline)."""
    f = np.asarray(fluo, dtype=np.float64)
    b = float(np.percentile(f, 8))
    if abs(b) < 1.0:
        b = 1.0
    dff = (f - b) / abs(b)
    m = np.mean(dff)
    s = np.std(dff)
    if s < 1e-9:
        return 0.0
    return float(np.mean(((dff - m) / s) ** 4) - 3.0)


_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'fig4')
_DEFAULT_OUT_DIR  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'figS4')

BETA = 0.5

_TARGET_SENSORS = ['GCaMP6f', 'GCaMP6s', 'GCaMP8f', 'GCaMP8m']

_METHODS = {
    'fmcsi':       {'label': 'OMSI',   'color': '#4C72B0'},
    'matlab':      {'label': 'MATLAB',  'color': '#DD8452'},
    'oasis':       {'label': 'OASIS',   'color': '#55A868'},
    'cascade_loo': {'label': 'CASCADE', 'color': '#8172B3'},
}
_METHOD_ORDER = ['fmcsi', 'matlab', 'oasis', 'cascade_loo']
_METHOD_GRID  = [('fmcsi', 'matlab'), ('oasis', 'cascade_loo')]


def _sensor_colors():

    cmap = plt.get_cmap('hsv')
    n = len(_TARGET_SENSORS)
    return {s: cmap(i / n) for i, s in enumerate(_TARGET_SENSORS)}


def _get_sensor(ds_folder):
    s = ds_folder.lower()
    for keyword, label in [
        ('gcaMP8s', 'GCaMP8s'), ('gcaMP8m', 'GCaMP8m'), ('gcaMP8f', 'GCaMP8f'),
        ('gcaMP7f', 'GCaMP7f'), ('gcaMP6s', 'GCaMP6s'), ('gcaMP6f', 'GCaMP6f'),
    ]:
        if keyword.lower() in s:
            return label
    return None


def _load_records(path):
    d    = np.load(path, allow_pickle=True)
    keys = list(d.files)
    if not keys:
        return []
    n = len(d[keys[0]])
    records = []
    for i in range(n):
        row = {}
        for k in keys:
            v = d[k][i]
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            elif hasattr(v, 'item'):
                v = v.item()
            row[k] = v
        records.append(row)
    return records


def _fbeta(precision, recall):

    p  = np.asarray(precision, dtype=float)
    r  = np.asarray(recall,    dtype=float)
    b2 = BETA ** 2
    denom = b2 * p + r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def _load_all_records(data_dir):

    all_records = {}
    for method_key in _METHOD_ORDER:
        path = os.path.join(data_dir, f'ground_truth_results_{method_key}.npz')
        if not os.path.exists(path):
            continue
        recs = _load_records(path)
        for r in recs:
            r['method'] = method_key
            r['sensor'] = _get_sensor(r['dataset'])
            r['fbeta']  = float(_fbeta(r.get('precision_window_oto', 0.0),
                                        r.get('recall_window_oto',    0.0)))
        all_records[method_key] = recs
        print(f'  Loaded {len(recs):5d} records for {method_key}')

        traces_dir  = os.path.join(data_dir, f'ground_truth_traces_{method_key}')
        ds_counters = {}
        npz_cache   = {}
        for r in recs:
            ds = r['dataset']
            ci = ds_counters.get(ds, 0)
            ds_counters[ds] = ci + 1
            tp = os.path.join(traces_dir, f'{ds}_traces.npz')
            if tp not in npz_cache:
                try:
                    npz_cache[tp] = np.load(tp, allow_pickle=False)
                except Exception:
                    npz_cache[tp] = None
            npz = npz_cache.get(tp)
            if npz is not None:
                try:
                    r['kurtosis'] = _dff_kurtosis(npz[f'dff_{ci}'])
                except Exception:
                    pass

    return all_records


def _plot_method_panel(ax, records, method_key, sensor_colors):

    for sensor in _TARGET_SENSORS:
        recs  = [r for r in records if r.get('sensor') == sensor]
        if len(recs) < 2:
            continue
        kurts = np.array([float(r['kurtosis']) for r in recs], dtype=float)
        fbs   = np.array([r['fbeta']           for r in recs], dtype=float)
        mask  = np.isfinite(kurts) & np.isfinite(fbs)
        kurts = kurts[mask]
        fbs   = fbs[mask]
        if len(kurts) < 2:
            continue

        mk  = np.mean(kurts);  sk = np.std(kurts)
        mf  = np.mean(fbs);    sf = np.std(fbs)
        col = sensor_colors[sensor]

        ax.plot([mk - sk, mk + sk], [mf, mf], '-', color=col, lw=1.4)
        ax.plot([mk, mk], [mf - sf, mf + sf], '-', color=col, lw=1.4)
        ax.plot(mk, mf, 'o', color=col, ms=3.5, zorder=5)

    xlim  = ax.get_xlim()
    x_ref = np.array([0.0, min(xlim[1], 1.0)])
    ax.set_xlabel('Kurtosis', fontsize=6)
    ax.set_ylabel(r'$F_\beta$ score', fontsize=6)
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis='both', labelsize=5.5)
    ax.set_title(_METHODS[method_key]['label'], fontsize=7, pad=3)


def plot_figure(data_dir=_DEFAULT_DATA_DIR, out_dir=_DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    all_records = _load_all_records(data_dir)
    if not all_records:
        print(f'No results found in {data_dir}. Run figure4.py --mode test first.')
        return

    sensor_colors = _sensor_colors()

    all_flat = [r for recs in all_records.values() for r in recs]
    all_kurts = np.array(
        [float(r['kurtosis']) for r in all_flat
         if r.get('sensor') in _TARGET_SENSORS and np.isfinite(float(r['kurtosis']))],
        dtype=float)

    fig = plt.figure(figsize=(5.5, 7.0), dpi=200)
    gs_outer = gridspec.GridSpec(3, 2, figure=fig,
                                 height_ratios=[0.7, 1.0, 1.0],
                                 hspace=0.45, wspace=0.40)

    ax_hist = fig.add_subplot(gs_outer[0, :])
    if len(all_kurts) > 0:
        ax_hist.hist(all_kurts, bins=30, color='#888888', alpha=0.75, edgecolor='none')
    ax_hist.set_xlabel('Kurtosis', fontsize=6)
    ax_hist.set_ylabel('# cells', fontsize=6)
    ax_hist.set_title('Kurtosis distribution (all cells, target sensors)', fontsize=7, pad=3)
    ax_hist.tick_params(axis='both', labelsize=5.5)

    k_lo = max(0.0, np.percentile(all_kurts, 1))  if len(all_kurts) else 0.0
    k_hi = 100

    for row_i, (mk_left, mk_right) in enumerate(_METHOD_GRID):
        for col_i, mk in enumerate([mk_left, mk_right]):
            ax = fig.add_subplot(gs_outer[row_i + 1, col_i])
            ax.set_xlim([k_lo, k_hi])
            if mk in all_records:
                _plot_method_panel(ax, all_records[mk], mk, sensor_colors)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6)
                ax.set_title(_METHODS[mk]['label'], fontsize=7, pad=3)

    legend_handles = [
        Line2D([0], [0], color=sensor_colors[s], lw=2, label=s)
        for s in _TARGET_SENSORS
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=4, fontsize=5.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    for ext in ('png', 'svg'):
        out = os.path.join(out_dir, f'figureS4.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Figure S4 — kurtosis vs F_beta score per method'
    )
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='figure4 data directory (default: data/fig4)')
    parser.add_argument('--out-dir', default=_DEFAULT_OUT_DIR,
                        help='output directory for saved figure (default: data/figS4)')
    args = parser.parse_args()
    plot_figure(data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
