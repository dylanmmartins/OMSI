# -*- coding: utf-8 -*-
"""
figures/stats_helpers.py

Significance tests shared by the figure scripts.

Functions
---------
signed_rank
    Two-sided Wilcoxon signed-rank test on paired samples.
rank_sum
    Two-sided Mann-Whitney U test on independent samples.
fmt_test
    Format a test result as a fixed-width table cell.
print_test_header
    Print the column header used by print_test_row.
print_test_row
    Print one comparison as a table row.

DMM, September 2026
"""

import warnings

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu


def _stars(p):
    """ Return the significance marker for a p-value. """
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def signed_rank(x, y):
    """ Two-sided Wilcoxon signed-rank test on paired samples.

    Pairs where either value is non-finite are dropped before testing.

    Parameters
    ----------
    x, y : array-like
        Paired samples of equal length (x[i] is paired with y[i]).

    Returns
    -------
    dict
        'test' ('signed-rank'), 'n' (number of pairs used), 'stat' (W),
        'p', 'med_x', 'med_y', and 'med_diff' (median of x - y). stat and p are
        NaN when fewer than two pairs remain or every difference is zero.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError('signed_rank needs paired samples of equal length '
                         '({} vs {}).'.format(len(x), len(y)))
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = {'test': 'signed-rank', 'n': int(len(x)), 'stat': np.nan, 'p': np.nan,
           'med_x': np.nan, 'med_y': np.nan, 'med_diff': np.nan}
    if len(x) == 0:
        return out
    out['med_x'], out['med_y'] = float(np.median(x)), float(np.median(y))
    out['med_diff'] = float(np.median(x - y))
    if len(x) < 2 or np.all(x == y):
        return out
    with warnings.catch_warnings():
        # Many tied (zero) differences trigger a normal-approximation warning;
        # the p-value is still reported.
        warnings.simplefilter('ignore', UserWarning)
        res = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    out['stat'], out['p'] = float(res.statistic), float(res.pvalue)
    return out


def rank_sum(x, y):
    """ Two-sided Mann-Whitney U test on independent samples.

    Parameters
    ----------
    x, y : array-like
        Independent samples; non-finite values are dropped.

    Returns
    -------
    dict
        Same keys as signed_rank, with 'test' set to 'rank-sum', 'n' given as
        'n_x/n_y', 'stat' the U statistic, and 'med_diff' the difference of medians.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    out = {'test': 'rank-sum', 'n': '{}/{}'.format(len(x), len(y)), 'stat': np.nan,
           'p': np.nan, 'med_x': np.nan, 'med_y': np.nan, 'med_diff': np.nan}
    if len(x) == 0 or len(y) == 0:
        return out
    out['med_x'], out['med_y'] = float(np.median(x)), float(np.median(y))
    out['med_diff'] = out['med_x'] - out['med_y']
    res = mannwhitneyu(x, y, alternative='two-sided')
    out['stat'], out['p'] = float(res.statistic), float(res.pvalue)
    return out


def fmt_test(res):
    """ Format a test result as 'p=... (stars)'. """
    if not np.isfinite(res['p']):
        return 'p=n/a'
    return 'p={:.2e} {}'.format(res['p'], _stars(res['p']))


_ROW_FMT = '  {:<34} {:<26} {:>7} {:>10} {:>10} {:>10} {:>11} {:>20}'


def print_test_header(label='Comparison'):
    """ Print the column header used by print_test_row. """
    header = _ROW_FMT.format(label, 'Pair', 'n', 'med A', 'med B',
                             'med A-B', 'stat', 'p')
    print(header)
    print('  ' + '-' * (len(header) - 2))


def print_test_row(label, name_a, name_b, res):
    """ Print one comparison as a table row.

    Parameters
    ----------
    label : str
        What was compared (panel, metric, x value).
    name_a, name_b : str
        Names of the two groups; A is the first sample passed to the test.
    res : dict
        Result from signed_rank or rank_sum.
    """
    def _f(v):
        return 'n/a' if not np.isfinite(v) else '{:.4g}'.format(v)
    print(_ROW_FMT.format(label, '{} vs {}'.format(name_a, name_b), str(res['n']),
                          _f(res['med_x']), _f(res['med_y']), _f(res['med_diff']),
                          _f(res['stat']), fmt_test(res)))
