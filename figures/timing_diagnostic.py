# -*- coding: utf-8 -*-
"""
figures/timing_diagnostic.py

Spike-timing-error diagnostic for the figure 1 fixed benchmark.

For each method, every inferred spike is paired with its nearest
ground-truth spike, giving a signed timing offset (inferred - nearest
true) in milliseconds. Pooling these offsets across all cells and
histogramming them yields a peri-event ("PSTH-style") plot of how
predicted spikes fall relative to true spikes -- the continuous-time
analogue of a spike-train cross-correlogram, but referenced to the
single nearest true spike rather than every true spike in the window.
The unsigned version of the same offsets gives, for each method, the
timing precision reported as the 95th-percentile absolute offset: the
window (in ms) that contains 95% of inferred spikes relative to the
nearest ground-truth spike.

Functions
---------
_load_method_spikes
    Load ground-truth and inferred spikes for one method's NPZ file.
_nearest_offsets_ms
    Compute signed nearest-neighbor timing offsets, in ms, pooled across cells.
plot_joes_diagnostic
    Load all four methods and render the four-panel diagnostic figure.

To run:
    $ python timing_diagnostic.py --data-dir /path/to/results

DMM, July 2026
"""

import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size']    = 7

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fig1')

# (display name, source NPZ file, spike-array key inside that NPZ)
_METHODS = [
    ('OMSI',    'fixed_benchmark_fMCSI.npz',       'optim_spikes'),
    ('CaImAn',  'fixed_benchmark_MATLAB.npz',      'tradmat_spikes'),
    ('CASCADE', 'fixed_benchmark_CASCADE_GPU.npz', 'cascade_spikes'),
    ('OASIS',   'fixed_benchmark_OASIS.npz',       'oasis_spikes'),
]

COLORS = {
    'OMSI':    '#4C72B0',
    'CaImAn':  '#DD8452',
    'CASCADE': '#8172B3',
    'OASIS':   '#55A868',
}

WINDOW_MS = 200.0   # half-width of the displayed peri-event histogram
BIN_MS    = 5.0     # histogram bin width


def _load_method_spikes(data_dir, npz_name, spike_key):
    """ Load ground-truth and inferred spikes for one method's NPZ file.

    Parameters
    ----------
    data_dir : str
        Directory containing the figure 1 NPZ result files.
    npz_name : str
        Filename of the method's NPZ file.
    spike_key : str
        Key of the inferred-spike array inside the NPZ file.

    Returns
    -------
    true_spikes : list of np.ndarray
        Per-cell ground-truth spike times in seconds.
    pred_spikes : list of np.ndarray
        Per-cell inferred spike times in seconds.
    """
    path = os.path.join(data_dir, npz_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run figure1.py --mode test first.')
    d = np.load(path, allow_pickle=True)
    true_spikes = list(d['true_spikes'])
    pred_spikes = list(d[spike_key])
    return true_spikes, pred_spikes


def _nearest_offsets_ms(true_spikes, pred_spikes):
    """ Compute signed nearest-neighbor timing offsets, in ms, pooled across cells.

    For every inferred spike, finds the closest ground-truth spike in the
    same cell and records (inferred - true) in milliseconds. Cells with no
    ground-truth spikes or no inferred spikes are skipped.

    Parameters
    ----------
    true_spikes : list of array-like
        Per-cell ground-truth spike times in seconds.
    pred_spikes : list of array-like
        Per-cell inferred spike times in seconds.

    Returns
    -------
    np.ndarray
        Signed nearest-neighbor offsets in milliseconds, pooled across all
        cells and all inferred spikes.
    """
    offsets_ms = []
    for t_spk, p_spk in zip(true_spikes, pred_spikes):
        t_spk = np.sort(np.atleast_1d(np.asarray(t_spk, dtype=float)))
        p_spk = np.atleast_1d(np.asarray(p_spk, dtype=float))
        t_spk = t_spk[np.isfinite(t_spk)]
        p_spk = p_spk[np.isfinite(p_spk)]
        if len(t_spk) == 0 or len(p_spk) == 0:
            continue

        idx    = np.searchsorted(t_spk, p_spk)
        idx_lo = np.clip(idx - 1, 0, len(t_spk) - 1)
        idx_hi = np.clip(idx,     0, len(t_spk) - 1)
        d_lo   = p_spk - t_spk[idx_lo]
        d_hi   = p_spk - t_spk[idx_hi]
        use_hi = np.abs(d_hi) < np.abs(d_lo)
        nearest_signed = np.where(use_hi, d_hi, d_lo)
        offsets_ms.append(nearest_signed * 1000.0)

    return np.concatenate(offsets_ms) if offsets_ms else np.array([])


def plot_joes_diagnostic(data_dir=_DEFAULT_DATA_DIR, out_dir=None,
                         window_ms=WINDOW_MS, bin_ms=BIN_MS):
    """ Load all four methods and render the four-panel diagnostic figure.

    One panel per method (OMSI, CaImAn, CASCADE, OASIS), each showing a
    peri-event histogram of inferred-spike offsets relative to the nearest
    ground-truth spike, annotated with the 95th-percentile absolute offset.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the figure 1 NPZ result files.
    out_dir : str or None, optional
        Directory to save the figure. Defaults to `data_dir`.
    window_ms : float, optional
        Half-width of the displayed histogram, in ms.
    bin_ms : float, optional
        Histogram bin width, in ms.

    Returns
    -------
    dict
        Mapping method name to its pooled offset array (ms) and 95th
        percentile absolute offset (ms).
    """
    out_dir = out_dir if out_dir else data_dir
    bins = np.arange(-window_ms, window_ms + bin_ms, bin_ms)

    results = {}
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.4), dpi=300, sharey=True)

    for ax, (name, npz_name, spike_key) in zip(axes, _METHODS):
        print(f'Loading {name}...')
        true_spikes, pred_spikes = _load_method_spikes(data_dir, npz_name, spike_key)
        offsets_ms = _nearest_offsets_ms(true_spikes, pred_spikes)
        p95_ms = float(np.percentile(np.abs(offsets_ms), 95)) if len(offsets_ms) > 0 else np.nan
        results[name] = {'offsets_ms': offsets_ms, 'p95_ms': p95_ms}
        print(f'  n={len(offsets_ms)} inferred spikes, 95th pct |offset| = {p95_ms:.1f} ms')

        in_window = offsets_ms[np.abs(offsets_ms) <= window_ms]
        ax.hist(in_window, bins=bins, color=COLORS[name], alpha=0.85, edgecolor='none')
        ax.axvline(0.0, color='k', lw=0.6, ls='--')
        ax.set_title(name, color=COLORS[name], fontsize=8)
        ax.set_xlabel('Offset (ms)\ninferred - nearest true')
        ax.text(0.97, 0.95, '95%: {:.0f} ms'.format(p95_ms),
                transform=ax.transAxes, ha='right', va='top', fontsize=6)

    axes[0].set_ylabel('# inferred spikes')
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'svg'):
        out_path = os.path.join(out_dir, f'joes_diagnostic.{ext}')
        fig.savefig(out_path, bbox_inches='tight')
        print('Saved to {}.'.format(out_path))
    plt.close(fig)

    print('\n{:<10} {:>20}'.format('Method', '95th pct |offset| (ms)'))
    print('-' * 32)
    for name, *_ in _METHODS:
        print('{:<10} {:>20.1f}'.format(name, results[name]['p95_ms']))

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Diagnostic: timing offset of inferred spikes relative to nearest ground-truth spike.'
    )
    parser.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                        help='Directory containing figure 1 NPZ result files')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save the figure (defaults to --data-dir)')
    parser.add_argument('--window-ms', type=float, default=WINDOW_MS,
                        help='Half-width of the displayed histogram, in ms')
    parser.add_argument('--bin-ms', type=float, default=BIN_MS,
                        help='Histogram bin width, in ms')
    args = parser.parse_args()

    plot_joes_diagnostic(data_dir=args.data_dir, out_dir=args.out_dir,
                         window_ms=args.window_ms, bin_ms=args.bin_ms)
