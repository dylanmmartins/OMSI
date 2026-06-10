#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

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


def _patch_records_kurtosis(all_records, data_dir):
    """Replace stored kurtosis scalars with dF/F kurtosis from trace files."""
    for method_key, recs in all_records.items():
        traces_dir = os.path.join(data_dir, f'ground_truth_traces_{method_key}')
        if not os.path.isdir(traces_dir):
            continue
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


_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'fig4')
_DEFAULT_OUT_DIR  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'figS2')

_METHODS = {
    'fmcsi':       {'label': 'OMSI',   'color': '#4C72B0'},
    'matlab':      {'label': 'CaImAn',  'color': '#DD8452'},
    'oasis':       {'label': 'OASIS',   'color': '#55A868'},
    'cascade_loo': {'label': 'CASCADE', 'color': '#8172B3'},
}
_METHOD_ORDER  = ['fmcsi', 'matlab', 'oasis', 'cascade_loo']
_TRACE_METHODS = ['fmcsi', 'matlab', 'oasis', 'cascade_loo']

SENSORS = ['GCaMP8m', 'GCaMP8f']

RASTER_WINDOW   = 30.0
ROWS_PER_SENSOR = 3
N_SENSORS       = len(SENSORS)


def _get_sensor(ds_folder):
    s = ds_folder.lower()
    for keyword, label in [
        ('gcaMP8s', 'GCaMP8s'), ('gcaMP8m', 'GCaMP8m'), ('gcaMP8f', 'GCaMP8f'),
        ('gcaMP7f', 'GCaMP7f'), ('gcaMP6s', 'GCaMP6s'), ('gcaMP6f', 'GCaMP6f'),
        ('gcaMP5k', 'GCaMP5k'), ('jgeco',   'jGECO'),   ('xcaMP',   'XCaMP'),
        ('jrcamp',  'jRCaMP'),  ('rcamp',   'R-CaMP'),  ('ogb',     'OGB1'),
        ('cal520',  'Cal520'),
    ]:
        if keyword.lower() in s:
            return label
    return 'Other'


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


def _traces_dir(data_dir, method_key):
    return os.path.join(data_dir, f'ground_truth_traces_{method_key}')


def _load_all_records(data_dir):
    all_records = {}
    for method_key in _METHOD_ORDER:
        path = os.path.join(data_dir, f'ground_truth_results_{method_key}.npz')
        if not os.path.exists(path):
            continue
        recs = _load_records(path)
        for r in recs:
            r['method'] = method_key
        all_records[method_key] = recs
        print(f'  Loaded {len(recs):5d} records for {method_key}')
    return all_records


def _best_window(raw, fs, true_spk, pred_spks_list,
                 window=30.0, target_spikes=10):
    block        = max(1, int(window * fs))
    n            = len(raw)
    best_t0, best_score = 0.0, -np.inf
    for t in range(0, n - block + 1, block):
        t0       = t / fs
        t1       = t0 + window
        true_win = true_spk[(true_spk >= t0) & (true_spk < t1)]
        n_true   = len(true_win)
        spike_sc = float(np.exp(-0.5 * ((n_true - target_spikes) / 8.0) ** 2))
        recalls  = []
        for pred in pred_spks_list:
            if n_true == 0 or len(pred) == 0:
                continue
            det  = pred[(pred >= t0 - 0.1) & (pred < t1 + 0.1)]
            hits = sum(1 for ts in true_win if np.any(np.abs(det - ts) <= 0.1))
            recalls.append(hits / n_true)
        rec_sc = float(np.mean(recalls)) if recalls else 0.0
        score  = (spike_sc + rec_sc) / 2.0
        if score > best_score:
            best_score = score
            best_t0    = t0
    return best_t0


def _select_sensor_cells(data_dir, all_records, sensor):

    fmcsi_recs = [r for r in all_records.get('fmcsi', [])
                  if _get_sensor(r['dataset']) == sensor]
    if not fmcsi_recs:
        return []

    fmcsi_sorted = sorted(fmcsi_recs, key=lambda r: float(r.get('cosmic', 0.0)))
    n            = len(fmcsi_sorted)
    target_pcts  = [0.60, 0.40, 0.20]
    selected     = []

    for pct in target_pcts:
        idx = int(round(pct * (n - 1)))
        idx = max(0, min(n - 1, idx))
        rec = fmcsi_sorted[idx]
        ds  = rec['dataset']

        ds_fmcsi_recs = [r for r in all_records.get('fmcsi', [])
                         if r['dataset'] == ds]
        local_idx = ds_fmcsi_recs.index(rec)

        dff = None; true_spikes = None; fs = None; kurtosis = None
        pred_spikes = {}
        for mk in _TRACE_METHODS:
            tp = os.path.join(_traces_dir(data_dir, mk), f'{ds}_traces.npz')
            if not os.path.exists(tp):
                continue
            try:
                npz = np.load(tp, allow_pickle=False)
                if local_idx >= int(npz['n_cells']):
                    continue
                pred_spikes[mk] = np.asarray(
                    npz[f'pred_spikes_{local_idx}'], dtype=np.float64)
                if dff is None:
                    dff         = npz[f'dff_{local_idx}']
                    true_spikes = np.asarray(
                        npz[f'true_spikes_{local_idx}'], dtype=np.float64)
                    true_spikes = true_spikes[np.isfinite(true_spikes)]
                    fs          = float(npz['fs'])
                    kurtosis    = _dff_kurtosis(dff)
            except Exception as exc:
                print(f'    Warning loading {tp}: {exc}')

        if dff is None or fs is None:
            continue

        t_start = _best_window(
            dff, fs, true_spikes, list(pred_spikes.values()),
            window=RASTER_WINDOW)

        cosmic_by_method = {}
        for mk in _METHOD_ORDER:
            ds_mk_recs = [r for r in all_records.get(mk, [])
                          if r['dataset'] == ds]
            if local_idx < len(ds_mk_recs):
                cosmic_by_method[mk] = float(
                    ds_mk_recs[local_idx].get('cosmic', np.nan))
            else:
                cosmic_by_method[mk] = np.nan

        selected.append({
            'dataset':         ds,
            'local_idx':       local_idx,
            'percentile':      pct,
            'cosmic':          float(rec.get('cosmic', np.nan)),
            'cosmic_by_method': cosmic_by_method,
            'kurtosis':        kurtosis,
            'dff':             dff,
            'true_spikes':     true_spikes,
            'pred_spikes':     pred_spikes,
            'fs':              fs,
            't_start':         t_start,
        })

    return selected


def _plot_single_raster(ax, cell, window=30.0,
                        show_xlabels=True, show_method_labels=True,
                        cell_number=None):

    rr  = 0.85
    th  = 2.0
    pad = 0.2
    gap = 0.25


    row_specs = []
    for mk in ['oasis', 'matlab', 'cascade_loo', 'fmcsi']:
        if mk in cell['pred_spikes']:
            row_specs.append((_METHODS[mk]['label'], mk, _METHODS[mk]['color']))
    row_specs.append(('Ground Truth', None, '#111111'))
    n_rows  = len(row_specs)
    label_x = -3.8
    t0      = cell['t_start']

    for row_i, (row_name, mk, color) in enumerate(row_specs):
        y_lo  = row_i * rr + 0.05
        y_hi  = row_i * rr + rr * 0.85
        y_mid = row_i * rr + rr * 0.45
        spk   = (cell['true_spikes'] if mk is None
                 else cell['pred_spikes'].get(mk, np.array([])))
        spk    = np.atleast_1d(np.asarray(spk, dtype=float))
        in_win = spk[(spk >= t0) & (spk <= t0 + window)] - t0
        if len(in_win):
            ax.vlines(in_win, y_lo, y_hi, color=color, lw=0.6, alpha=0.9)

        ax.text(label_x, y_mid, row_name,
                va='center', ha='right', color=color, fontsize=4.5)

    trace_y0 = n_rows * rr + pad
    raw      = cell['dff']
    fs       = cell['fs']
    t_arr    = np.arange(len(raw)) / fs
    mask     = (t_arr >= t0) & (t_arr <= t0 + window)
    t_pl     = t_arr[mask] - t0
    r_pl     = raw[mask]
    lo, hi   = np.nanmin(r_pl), np.nanmax(r_pl)
    r_norm   = ((r_pl - lo) / (hi - lo) * th + trace_y0
                if hi > lo else np.full_like(r_pl, trace_y0 + th / 2))
    ax.plot(t_pl, r_norm, color='k', lw=0.7, alpha=0.85)
    ax.text(label_x, trace_y0 + th / 2, 'ΔF/F',
            va='center', ha='right', color='k', fontsize=4.5)
    if cell_number is not None:
        ax.text((label_x + 0) / 2, trace_y0 + th / 2, str(cell_number),
                va='center', ha='center', color='k', fontsize=5.5,
                fontweight='bold')

    total_h = n_rows * rr + pad + th + gap
    ax.set_xlim(label_x - 0.5, window + 2.5)
    ax.set_ylim(-gap / 2, total_h)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

    if show_xlabels:
        step = 10 if window >= 20 else 5
        ax.set_xticks(np.arange(0, window + 1, step))
        ax.set_xticklabels(
            [f'{int(t)}' for t in np.arange(0, window + 1, step)], fontsize=5)
        ax.set_xlabel('Time (s)', fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_xlabel('')

    ann_text = (f'kurt = {cell["kurtosis"]:.1f}\n')
    ax.text(window + 0.3, total_h * 0.55, ann_text,
            va='center', ha='left', fontsize=4.5, linespacing=1.4,
            color='#333333')


def _plot_cosmic_violins(ax, all_records, sensor, selected_cells):

    method_positions  = {}
    x_pos             = 0
    method_labels_out = []
    all_vals_list     = []

    for mk in _METHOD_ORDER:
        recs = [r for r in all_records.get(mk, [])
                if _get_sensor(r['dataset']) == sensor]
        vals = np.array([float(r.get('cosmic', np.nan)) for r in recs],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        all_vals_list.append(vals)
        method_positions[mk] = x_pos
        method_labels_out.append((x_pos, _METHODS[mk]['label']))
        parts = ax.violinplot([vals], positions=[x_pos],
                              showmedians=True, widths=0.65)
        col = _METHODS[mk]['color']
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(0.72)
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color('k')
                parts[key].set_linewidth(0.8)
        x_pos += 1

    for cell in selected_cells:
        for mk, pos in method_positions.items():
            cv = cell['cosmic_by_method'].get(mk, np.nan)
            if np.isfinite(cv):
                ax.plot(pos, cv, '*', color='red',
                        markersize=5, zorder=6, markeredgewidth=0.3,
                        markeredgecolor='darkred')

    if method_labels_out:
        xs, lbls = zip(*method_labels_out)
        ax.set_xticks(list(xs))


    max_val = max(np.max(v) for v in all_vals_list) if all_vals_list else 1.0
    ax.set_ylim(0, 1.1 * max_val)
    ax.set_ylabel('CosMIC', fontsize=6)
    ax.tick_params(axis='y', labelsize=5.5)
    ax.set_title(sensor, fontsize=6.5, fontweight='bold', pad=3)


def _plot_kurtosis_hist(ax, all_records, sensor, selected_cells):

    recs  = [r for r in all_records.get('fmcsi', [])
             if _get_sensor(r['dataset']) == sensor]
    kurts = np.array([float(r.get('kurtosis', np.nan)) for r in recs],
                     dtype=float)
    kurts = kurts[np.isfinite(kurts)]

    if len(kurts) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=6)
        return

    ax.hist(kurts, bins=np.linspace(0,80,16), color='#999999', alpha=0.70, edgecolor='none')

    for cell in selected_cells:
        ax.axvline(cell['kurtosis'], color='red', lw=0.8, ls='--', alpha=0.8)

    ax.set_xlabel('Kurtosis', fontsize=5.5)
    ax.set_ylabel('# cells', fontsize=5.5)
    ax.tick_params(axis='both', labelsize=5.5)



def plot_figure(data_dir=_DEFAULT_DATA_DIR, out_dir=_DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    print('Loading figure4 results...')
    all_records = _load_all_records(data_dir)
    _patch_records_kurtosis(all_records, data_dir)
    if not all_records:
        print(f'No results found in {data_dir}. '
              f'Run figure4.py --mode test first.')
        return
    print(f'Loaded {sum(len(v) for v in all_records.values())} total records '
          f'across {len(all_records)} methods.\n')

    fig = plt.figure(figsize=(6.5, 5), dpi=200)

    outer_gs = gridspec.GridSpec(
        N_SENSORS, 1,
        figure=fig,
        hspace=0.45,
    )

    for si, sensor in enumerate(SENSORS):
        print(f'[{sensor}]  Selecting example cells...')
        selected = _select_sensor_cells(data_dir, all_records, sensor)

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            ROWS_PER_SENSOR, 3,
            subplot_spec=outer_gs[si],
            width_ratios=[1, 1, 1],
            hspace=0.10,
            wspace=0.75,
        )

        if not selected:
            print(f'  No cells found for {sensor} — skipping block.')
            for ri in range(ROWS_PER_SENSOR):
                fig.add_subplot(inner_gs[ri, 0:2]).axis('off')
            fig.add_subplot(inner_gs[0:2, 2]).axis('off')
            fig.add_subplot(inner_gs[2, 2]).axis('off')
            continue

        cosmic_str = ', '.join(f'{c["cosmic"]:.2f}' for c in selected)
        print(f'  Selected {len(selected)} cells (CosMIC: {cosmic_str})')

        for ci, cell in enumerate(selected):
            ax = fig.add_subplot(inner_gs[ci, 0:2])
            show_xlabels      = (ci == len(selected) - 1)
            show_method_labels = (ci == 0)
            cell_number = si * ROWS_PER_SENSOR + ci + 1
            _plot_single_raster(
                ax, cell, window=RASTER_WINDOW,
                show_xlabels=show_xlabels,
                show_method_labels=show_method_labels,
                cell_number=cell_number,
            )
            if ci == 0:
                ax.set_title(sensor, fontsize=7, fontweight='bold',
                             loc='left', pad=3)

        for ci in range(len(selected), ROWS_PER_SENSOR):
            fig.add_subplot(inner_gs[ci, 0:2]).axis('off')

        ax_vio = fig.add_subplot(inner_gs[0:2, 2])
        _plot_cosmic_violins(ax_vio, all_records, sensor, selected)

        ax_hist = fig.add_subplot(inner_gs[2, 2])
        _plot_kurtosis_hist(ax_hist, all_records, sensor, selected)

    legend_handles = [
        plt.Line2D([0], [0], color=_METHODS[m]['color'], marker='.', linestyle='-',
                   label=_METHODS[m]['label'])
        for m in ['fmcsi', 'matlab', 'oasis', 'cascade_loo']
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7)

    for ext in ('png', 'svg'):
        out = os.path.join(out_dir, f'figureS2.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'\nSaved -> {out}')
    plt.close(fig)


def main():
    
    parser = argparse.ArgumentParser(
        description='Figure S2 — per-sensor CosMIC distributions and example cells'
    )
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='figure4 data directory (default: data/fig4)')
    parser.add_argument('--out-dir', default=_DEFAULT_OUT_DIR,
                        help='output directory for saved figure (default: data/figS2)')
    args = parser.parse_args()
    plot_figure(data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == '__main__':
    
    main()
