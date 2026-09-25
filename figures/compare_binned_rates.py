# -*- coding: utf-8 -*-
"""
figures/compare_binned_rates.py

Compare continuous-time inference outputs against a ground-truth spike rate
binned to match the native imaging sample rate, for the CASCADE 7.5 Hz vs
30 Hz comparison (figure 2B) and the Allen Institute benchmark (figure 3).

Model colors and ordering mirror figure2.py (COLORS, _CASCADE_CMP_COLOR_7P5,
_CASCADE_CMP_COLOR_30) and figure3.py (model_colors, _MODEL_ORDER) so panels
read consistently across the whole figures/ directory.

Functions
---------
_bin_ground_truth
    Bin ground-truth spike times into per-frame counts at a given sample rate.
_corr_and_residual
    Compute Pearson correlation and RMSE residual between two rate traces.
_mad
    Compute the median absolute deviation, ignoring NaNs.
_load_cascade_samplerate_group
    Load true spikes and continuous CASCADE output for one sample rate.
compare_cascade_samplerate
    Compute per-cell correlation and residual for CASCADE at 7.5 Hz vs 30 Hz.
_plot_cascade_samplerate_comparison
    Plot side-by-side correlation and residual violins for the CASCADE comparison.
normalize_label
    Strip leading run-label prefix from a dataset name.
clean_label
    Reformat raw CASCADE-style labels into a canonical short label.
get_zoom_for_label
    Return zoom level string for a dataset label.
_load_records
    Load a list of result dicts from a compressed NPZ file.
_build_cascade_lookup
    Build a label-to-filepath lookup for CASCADE trace NPZ files.
_build_omsi_traces_lookup
    Build a label-to-filepath lookup for newer OMSI traces NPZ files.
_load_allen_group_traces
    Load per-cell continuous rate traces and ground truth for every Allen group.
compare_allen_binned_rates
    Compute per-cell correlation and residual for OASIS, CASCADE, and OMSI.
_plot_binned_violin
    Plot a violin per model for one metric/zoom panel.
_plot_allen_binned_grid
    Plot a 2x2 grid of correlation/residual violins split by zoom level.
main
    Run both comparisons and save the resulting figures.


DMM, July 2026
"""

import argparse
import glob as _glob
import os
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

_DEFAULT_FIG2_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig2')
_DEFAULT_FIG3_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig3')

_EXCLUDED_DATASETS = {'DS29-GCaMP7f-m-V1', 'DS32-GCaMP8s-m-V1', 'DS28-XCaMPgf-m-V1'}

COLORS = {
    'OMSI':   '#4C72B0',
    'OASIS':   '#55A868',
    'CASCADE': '#8172B3',
}
_MODEL_ORDER = ['OMSI', 'OASIS', 'CASCADE']

_CASCADE_CMP_COLOR_7P5 = 'tab:red'
_CASCADE_CMP_COLOR_30  = 'tab:cyan'


def _bin_ground_truth(spike_times, fs, n_frames):
    """
    Bin ground-truth spike times into per-frame counts at a given sample rate.

    Parameters
    ----------
    spike_times : array-like
        Ground-truth spike times in seconds.
    fs : float
        Sampling rate in Hz to bin to (matches the native imaging frame rate).
    n_frames : int
        Number of frame bins to produce.

    Returns
    -------
    ndarray
        Spike count per frame, length n_frames.
    """
    edges = np.arange(n_frames + 1) / fs
    counts, _ = np.histogram(np.asarray(spike_times, dtype=float), bins=edges)
    return counts.astype(np.float64)


def _corr_and_residual(binned_gt, pred):
    """
    Compute Pearson correlation and RMSE residual between two rate traces.

    CASCADE pads its output with NaN at the start/end of each trace (edge
    frames it can't infer from), so NaN frames are dropped from both traces
    before computing either statistic.

    Parameters
    ----------
    binned_gt : ndarray
        Binned ground-truth spike rate.
    pred : ndarray
        Continuous-time predicted rate (unthresholded model output).

    Returns
    -------
    corr : float
        Pearson correlation coefficient, NaN if either trace is constant.
    residual : float
        Root-mean-square error between the two traces.
    """
    n = min(len(binned_gt), len(pred))
    a = binned_gt[:n]
    b = np.asarray(pred[:n], dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    residual = float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else np.nan
    return corr, residual


def _mad(x, axis=None):
    """ Compute the median absolute deviation, ignoring NaNs.

    Parameters
    ----------
    x : array-like
        Input values.
    axis : int or None, optional
        Axis along which to compute; None flattens the input.

    Returns
    -------
    float or np.ndarray
        Median of |x - median(x)| along axis.
    """
    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _load_cascade_samplerate_group(data_dir, fs):
    """
    Load true spikes and continuous CASCADE output for one sample rate.

    Parameters
    ----------
    data_dir : str
        Directory containing figure 2B cascade sample-rate output files.
    fs : float
        Sampling rate in Hz (7.5 or 30.0).

    Returns
    -------
    true_spikes : list of ndarray or None
        Ground-truth spike times in seconds, one array per cell.
    cascade_probs : ndarray or None, shape (n_cells, n_frames)
        Continuous per-frame CASCADE spike-probability output.
    """
    ts_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_true_spikes.npz')
    if not os.path.exists(ts_path):
        return None, None
    ts_data     = np.load(ts_path, allow_pickle=True)
    true_spikes = list(ts_data['true_spikes'])

    for dev in ('gpu', 'cpu'):
        out_path = os.path.join(data_dir, f'cascade_samplerate_{fs}hz_{dev}_output.npz')
        if os.path.exists(out_path):
            out_data = np.load(out_path, allow_pickle=True)
            return true_spikes, np.asarray(out_data['cascade_probs'])
    return true_spikes, None


def compare_cascade_samplerate(data_dir):
    """
    Compute per-cell correlation and residual for CASCADE at 7.5 Hz vs 30 Hz.

    Ground-truth spikes are binned to the same frame rate as the CASCADE
    output before comparison, so no thresholding of the CASCADE output is
    involved.

    Parameters
    ----------
    data_dir : str
        Directory containing figure 2B cascade sample-rate output files.

    Returns
    -------
    dict
        Keys 'corr_7', 'corr_30', 'resid_7', 'resid_30' mapping to per-cell
        ndarrays.
    """
    results = {}
    for fs, suffix in [(7.5, '7'), (30.0, '30')]:
        true_spikes, cascade_probs = _load_cascade_samplerate_group(data_dir, fs)
        if true_spikes is None or cascade_probs is None:
            print('  No CASCADE output found for {} Hz.'.format(fs))
            results[f'corr_{suffix}']  = np.array([])
            results[f'resid_{suffix}'] = np.array([])
            continue

        n_cells, n_frames = cascade_probs.shape
        corrs  = np.full(n_cells, np.nan)
        resids = np.full(n_cells, np.nan)
        for i in range(n_cells):
            binned_gt   = _bin_ground_truth(true_spikes[i], fs, n_frames)
            corrs[i], resids[i] = _corr_and_residual(binned_gt, cascade_probs[i])
        results[f'corr_{suffix}']  = corrs
        results[f'resid_{suffix}'] = resids
        print('  {} Hz: corr={:.3f} ± {:.3f}  resid={:.4f} ± {:.4f}'.format(
            fs, np.nanmedian(corrs), _mad(corrs), np.nanmedian(resids), _mad(resids)))
    return results


def _plot_cascade_samplerate_comparison(ax_corr, ax_resid, results):
    """
    Plot side-by-side correlation and residual violins for the CASCADE comparison.

    Styled identically to figure2.py's _plot_cascade_comparison (same violin
    widths, alpha, and colors), split into two panels instead of one.

    Parameters
    ----------
    ax_corr : matplotlib.axes.Axes
        Axes for the correlation panel.
    ax_resid : matplotlib.axes.Axes
        Axes for the residual panel.
    results : dict
        Output of compare_cascade_samplerate.
    """
    def _draw(ax, d7, d30, ylabel):
        """Draw one 7.5 Hz vs 30 Hz violin panel onto ax."""
        all_datasets = [(d7[np.isfinite(d7)],   _CASCADE_CMP_COLOR_7P5),
                        (d30[np.isfinite(d30)], _CASCADE_CMP_COLOR_30)]
        pos      = [p for p, (d, _) in zip([1, 2], all_datasets) if len(d) > 0]
        datasets = [(d, c) for d, c in all_datasets if len(d) > 0]
        if not datasets:
            ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7)
            return
        parts = ax.violinplot([d for d, _ in datasets], positions=pos,
                              showmedians=True, widths=0.65)
        for pc, (_, col) in zip(parts['bodies'], datasets):
            pc.set_facecolor(col); pc.set_alpha(0.75)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            parts[partname].set_color('k'); parts[partname].set_linewidth(0.8)
        labels = [lbl for lbl, (d, _) in zip(['7.5 Hz', '30 Hz'], all_datasets) if len(d) > 0]
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)

    _draw(ax_corr,  results['corr_7'],  results['corr_30'],  'Pearson correlation')
    _draw(ax_resid, results['resid_7'], results['resid_30'], 'RMSE residual')

    ax_corr.legend(handles=[
        Patch(facecolor=_CASCADE_CMP_COLOR_7P5, alpha=0.75, label='7.5 Hz'),
        Patch(facecolor=_CASCADE_CMP_COLOR_30,  alpha=0.75, label='30 Hz'),
    ], loc='upper right', handlelength=1.0, handleheight=0.8,
       borderpad=0.4, labelspacing=0.2, frameon=False)


def normalize_label(label):
    """
    Strip leading run-label prefix from a dataset name.

    Parameters
    ----------
    label : str
        Raw dataset label string.

    Returns
    -------
    str
        Label with any leading 'prefix_' removed.
    """
    return re.sub(r'^[^_]+_', '', label)


def clean_label(label):
    """
    Reformat raw CASCADE-style labels into a canonical short label.

    Parameters
    ----------
    label : str
        Raw label string, possibly in CASCADE format.

    Returns
    -------
    str
        Reformatted label or the original string if no match.
    """
    m = re.search(r'(.*)_\(\'([^\']+)\',\s*(\d+),\s*(\d+)\)frames', label)
    if m:
        return f"{m.group(2)}_{m.group(1)}_{m.group(4)}frames_{m.group(3)}hz"
    return label


def get_zoom_for_label(label):
    """
    Return zoom level string for a dataset label.

    Parameters
    ----------
    label : str
        Dataset label string.

    Returns
    -------
    str
        'High Zoom', 'Low Zoom', or 'Unknown'.
    """
    if 'lowzoom' in label:
        return 'Low Zoom'
    elif 'highzoom' in label:
        return 'High Zoom'
    return 'Unknown'


def _load_records(path):
    """
    Load a list of result dicts from a compressed NPZ file.

    Parameters
    ----------
    path : str
        Path to an NPZ file saved by figure3.py's _save_records.

    Returns
    -------
    list of dict
        Reconstructed list of result records.
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
    if not keys:
        return []
    n = len(d[keys[0]])
    records = []
    for i in range(n):
        row = {}
        for k in keys:
            v = d[k][i]
            if hasattr(v, 'item'):
                v = v.item()
            row[k] = v
        records.append(row)
    return records


def _build_cascade_lookup(data_dir):
    """
    Build a label-to-filepath lookup for CASCADE trace NPZ files.

    Parameters
    ----------
    data_dir : str
        Directory containing allen_data_results_cascade_*_traces.npz files.

    Returns
    -------
    dict
        Mapping from label string to (traces_path, cell_id_to_row) tuples.
    """
    lookup = {}
    for npz_path in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_cascade_*_traces.npz')):
        name      = os.path.basename(npz_path)
        label_raw = (name.replace('allen_data_results_cascade_', '')
                        .replace('_traces.npz', ''))
        rec_npz_path   = npz_path.replace('_traces.npz', '.npz')
        cell_id_to_row = {}
        if os.path.exists(rec_npz_path):
            try:
                jdata = _load_records(rec_npz_path)
                seen  = {}
                for entry in jdata:
                    cid = entry['cell_id']
                    if cid not in seen:
                        seen[cid] = len(seen)
                cell_id_to_row = seen
            except Exception:
                pass
        lookup[label_raw] = (npz_path, cell_id_to_row)
    return lookup


def _build_omsi_traces_lookup(data_dir):
    """
    Build a label-to-filepath lookup for newer OMSI traces NPZ files.

    Parameters
    ----------
    data_dir : str
        Directory containing allen_data_results_omsi_*_traces.npz files.

    Returns
    -------
    dict
        Mapping from label string to OMSI traces NPZ file path.
    """
    lookup = {}
    for fpath in _glob.glob(
            os.path.join(data_dir, 'allen_data_results_omsi_*_traces.npz')):
        name  = os.path.basename(fpath)
        orig  = name.replace('allen_data_results_omsi_', '').replace('_traces.npz', '')
        group = os.path.join(data_dir, f'allen_data_results_{orig}_traces.npz')
        if not os.path.exists(group) or \
                os.path.getmtime(fpath) > os.path.getmtime(group):
            lookup[orig] = fpath
    return lookup


def _load_allen_group_traces(data_dir):
    """
    Load per-cell continuous rate traces and ground truth for every Allen group.

    Parameters
    ----------
    data_dir : str
        Directory containing figure 3 Allen benchmark output files.

    Returns
    -------
    list of dict
        One entry per cell with keys 'label', 'zoom', 'fs', 'true_spikes',
        'oasis_prob', 'omsi_prob', 'cascade_prob' (cascade_prob is None if
        no CASCADE output was found for that cell).
    """
    cascade_lookup      = _build_cascade_lookup(data_dir)
    omsi_traces_lookup = _build_omsi_traces_lookup(data_dir)

    cells = []
    for fpath in sorted(_glob.glob(
            os.path.join(data_dir, 'allen_data_results_*_traces.npz'))):
        basename = os.path.basename(fpath)
        if 'allen_data_results_omsi_'   in basename: continue
        if 'allen_data_results_cascade_' in basename: continue

        orig = basename.replace('allen_data_results_', '').replace('_traces.npz', '')
        if any(orig.startswith(ds) for ds in _EXCLUDED_DATASETS):
            continue
        label = clean_label(normalize_label(orig))
        zoom  = get_zoom_for_label(label)
        if zoom == 'Unknown':
            continue

        try:
            d = np.load(fpath, allow_pickle=True)
        except Exception as exc:
            print('  Warning: could not load {}: {}.'.format(fpath, exc))
            continue
        if 'true_spikes' not in d:
            continue

        df = None
        if orig in omsi_traces_lookup:
            try:
                df = np.load(omsi_traces_lookup[orig], allow_pickle=True)
            except Exception:
                df = None

        my_probs    = df['my_probs']    if (df is not None and 'my_probs'    in df) \
            else (d['my_probs']    if 'my_probs'    in d else None)
        oasis_probs = df['oasis_probs'] if (df is not None and 'oasis_probs' in df) \
            else (d['oasis_probs'] if 'oasis_probs' in d else None)
        if my_probs is None or oasis_probs is None:
            continue

        true_spikes_arr = list(d['true_spikes'])
        fs = float(d['fs']) if 'fs' in d else 30.0

        cas_probs_all, cas_cell_id_to_row = None, {}
        if orig in cascade_lookup:
            cas_fpath, cas_cell_id_to_row = cascade_lookup[orig]
            try:
                dc = np.load(cas_fpath, allow_pickle=True)
                if 'cascade_probs' in dc:
                    cas_probs_all = dc['cascade_probs']
            except Exception:
                cas_probs_all = None

        n = min(my_probs.shape[0], oasis_probs.shape[0], len(true_spikes_arr))
        for i in range(n):
            spk = np.atleast_1d(np.asarray(true_spikes_arr[i], dtype=float))

            cascade_prob = None
            cas_row = cas_cell_id_to_row.get(i, -1)
            if cas_probs_all is not None and 0 <= cas_row < cas_probs_all.shape[0]:
                cascade_prob = cas_probs_all[cas_row]

            cells.append({
                'label': label, 'zoom': zoom, 'fs': fs,
                'true_spikes': spk,
                'oasis_prob':  oasis_probs[i],
                'omsi_prob':  my_probs[i],
                'cascade_prob': cascade_prob,
            })
    return cells


def compare_allen_binned_rates(data_dir):
    """
    Compute per-cell correlation and residual for OASIS, CASCADE, and OMSI.

    Ground-truth spikes are binned to match the native 2P imaging sample
    rate of each dataset group before comparison. CaImAn MCMC is skipped.

    Parameters
    ----------
    data_dir : str
        Directory containing figure 3 Allen benchmark output files.

    Returns
    -------
    dict
        Nested dict results[zoom][model] -> {'corr': ndarray, 'resid': ndarray},
        for zoom in ('High Zoom', 'Low Zoom') and model in _MODEL_ORDER.
    """
    cells = _load_allen_group_traces(data_dir)
    print('  Loaded {} cells across all Allen dataset groups.'.format(len(cells)))

    results = {
        zoom: {model: {'corr': [], 'resid': []} for model in _MODEL_ORDER}
        for zoom in ('High Zoom', 'Low Zoom')
    }
    for c in cells:
        if c['zoom'] not in results:
            continue
        n_frames  = len(c['omsi_prob'])
        binned_gt = _bin_ground_truth(c['true_spikes'], c['fs'], n_frames)

        for model, prob in [('OMSI',   c['omsi_prob']),
                            ('OASIS',   c['oasis_prob']),
                            ('CASCADE', c['cascade_prob'])]:
            if prob is None:
                continue
            corr, resid = _corr_and_residual(binned_gt, np.asarray(prob))
            if np.isfinite(corr):
                results[c['zoom']][model]['corr'].append(corr)
            if np.isfinite(resid):
                results[c['zoom']][model]['resid'].append(resid)

    for zoom in results:
        for model in results[zoom]:
            for metric in ('corr', 'resid'):
                vals = np.array(results[zoom][model][metric])
                results[zoom][model][metric] = vals
                if len(vals) > 0:
                    print('  {} / {} / {}: n={}  median={:.3f} ± {:.3f}'.format(
                        zoom, model, metric, len(vals), np.median(vals), _mad(vals)))
    return results


def _plot_binned_violin(ax, data_by_model, ylabel, ylim_bottom=None):
    """
    Plot a violin per model for one metric/zoom panel.

    Styled identically to figure3.py's _plot_fbeta_violin (same violin
    widths, alpha, and per-model colors).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    data_by_model : dict
        Mapping model name to per-cell ndarray of metric values.
    ylabel : str
        Y-axis label.
    ylim_bottom : float or None, optional
        If given, fixes the bottom of the y-axis (top autoscales).
    """
    positions, violin_data, violin_colors, tick_labels = [], [], [], []
    for i, model in enumerate(_MODEL_ORDER):
        vals = np.asarray(data_by_model.get(model, np.array([])))
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            positions.append(i)
            violin_data.append(vals)
            violin_colors.append(COLORS.get(model, 'k'))
            tick_labels.append(model)
    if violin_data:
        parts = ax.violinplot(violin_data, positions=positions,
                              showmedians=True, widths=0.65)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            parts[pn].set_color('k'); parts[pn].set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)


def _plot_allen_binned_grid(results, out_dir):
    """
    Plot a 2x2 grid of correlation/residual violins split by zoom level.

    Top row is Pearson correlation (high zoom, low zoom); bottom row is
    RMSE residual (high zoom, low zoom).

    Parameters
    ----------
    results : dict
        Output of compare_allen_binned_rates.
    out_dir : str
        Directory to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(5, 6), dpi=300)

    for col, zoom in enumerate(['High Zoom', 'Low Zoom']):
        _plot_binned_violin(
            axes[0, col], {m: results[zoom][m]['corr'] for m in _MODEL_ORDER},
            'Pearson correlation', ylim_bottom=0.0)
        axes[0, col].set_title('high zoom' if zoom == 'High Zoom' else 'low zoom')

        _plot_binned_violin(
            axes[1, col], {m: results[zoom][m]['resid'] for m in _MODEL_ORDER},
            'RMSE residual', ylim_bottom=0.0)

    for row in (0, 1):
        row_top = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        axes[row, 0].set_ylim(0.0, row_top)
        axes[row, 1].set_ylim(0.0, row_top)

    legend_handles = [
        plt.Line2D([0], [0], color=COLORS['OMSI'],   marker='.', linestyle='-', label='OMSI'),
        plt.Line2D([0], [0], color=COLORS['OASIS'],   marker='.', linestyle='-', label='OASIS'),
        plt.Line2D([0], [0], color=COLORS['CASCADE'], marker='.', linestyle='-', label='CASCADE'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=7)

    fig.tight_layout()
    for sfx in ('png', 'svg'):
        out_path = os.path.join(out_dir, f'compare_binned_rates_allen.{sfx}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print('Saved: {}.'.format(out_path))
    plt.close(fig)


def main(fig2_data_dir=_DEFAULT_FIG2_DATA_DIR, fig3_data_dir=_DEFAULT_FIG3_DATA_DIR,
        out_dir=None):
    """
    Run both comparisons and save the resulting figures.

    Parameters
    ----------
    fig2_data_dir : str, optional
        Directory containing figure 2B cascade sample-rate output files.
    fig3_data_dir : str, optional
        Directory containing figure 3 Allen benchmark output files.
    out_dir : str or None, optional
        Directory to save figures. Defaults to fig2_data_dir / fig3_data_dir
        for each respective figure.
    """
    print('=== CASCADE 7.5 Hz vs 30 Hz binned-rate comparison (figure 2B) ===')
    cascade_results = compare_cascade_samplerate(fig2_data_dir)

    fig, (ax_corr, ax_resid) = plt.subplots(1, 2, figsize=(4.5, 3), dpi=300)
    _plot_cascade_samplerate_comparison(ax_corr, ax_resid, cascade_results)
    fig.tight_layout()
    out2 = out_dir if out_dir else fig2_data_dir
    os.makedirs(out2, exist_ok=True)
    for sfx in ('png', 'svg'):
        out_path = os.path.join(out2, f'compare_binned_rates_cascade.{sfx}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print('Saved: {}.'.format(out_path))
    plt.close(fig)

    print('\n=== Allen dataset binned-rate comparison (figure 3) ===')
    allen_results = compare_allen_binned_rates(fig3_data_dir)
    out3 = out_dir if out_dir else fig3_data_dir
    os.makedirs(out3, exist_ok=True)
    _plot_allen_binned_grid(allen_results, out3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compare continuous-time inference outputs against a binned '
                    'ground-truth spike rate for figures 2B and 3.'
    )
    parser.add_argument('--fig2-data-dir', default=_DEFAULT_FIG2_DATA_DIR,
                        help='Directory containing figure 2B cascade sample-rate output files')
    parser.add_argument('--fig3-data-dir', default=_DEFAULT_FIG3_DATA_DIR,
                        help='Directory containing figure 3 Allen benchmark output files')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save figures (defaults to each source data dir)')
    args = parser.parse_args()

    main(fig2_data_dir=args.fig2_data_dir, fig3_data_dir=args.fig3_data_dir,
        out_dir=args.out_dir)
