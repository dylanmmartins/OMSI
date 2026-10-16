"""
Time CaImAn MCMC on the first few cells of figure 2's 120-min timing dataset.

DMM, October 2026
"""

import argparse
import time

import numpy as np

import figure2
from run_pnev_MCMC import run_matlab_pnevMCMC
from simulation_helpers import generate_synthetic_data


def probe(duration, n_probe, fs=30.0, tau=0.5, n_cells=100):

    np.random.seed(figure2._timing_seed('Duration_Scaling', duration, 0))
    dff, _, _, _, _, _ = generate_synthetic_data(n_cells=n_cells, fs=fs,
                                                 duration=duration, tau=tau)
    dff = dff[:n_probe]
    t0 = time.time()
    _, _, _, sweeps, cell_times = run_matlab_pnevMCMC(
        dff, fs=fs, tau=tau, n_sweeps='auto', return_cell_times=True)
    wall = time.time() - t0
    figure2._check_matlab_ok(sweeps)
    print('{:.0f} s ({} frames): per cell {}  | wall {:.0f} s for {} cells'.format(
        duration, dff.shape[1], np.round(cell_times, 1), wall, n_probe))
    return cell_times, wall


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cells', type=int, default=3, help='Cells to run at each duration')
    parser.add_argument('--skip-3600', action='store_true',
                        help='Only run the 7200 s cells')
    args = parser.parse_args()

    out = {}
    for d in ([7200] if args.skip_3600 else [3600, 7200]):
        out[d] = probe(d, args.cells)

    print('\nEstimated 100-cell CaImAn time (100 x median wall time per cell):')
    for d, (ct, wall) in out.items():
        print('  {:.0f} s: sampler {:.0f} s/cell, wall {:.0f} s/cell -> ~{:.1f} h'.format(
            d, np.nanmedian(ct), wall / len(ct), 100 * wall / len(ct) / 3600))
