"""
Regression tests: the default code path must reproduce the thesis tables.

Reference values are Table 2 (N = 64) and Table 3 (N = 32) of
optimal_mixing_thesis_report.pdf, sine-bump data, F = 1, last-2/3 fit.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mixing as mx  # noqa: E402
import convergence as cv  # noqa: E402

A = np.arange(0.5, 15 / 16 + 1e-9, 1 / 16)

# (t_stop, ‖θ(t_stop)‖_{H⁻¹}, r) at N = 64, Table 2
TABLE2 = [(1.30, 0.6058, 0.4400), (1.60, 0.5831, 0.3858), (1.95, 0.5658, 0.3342),
          (2.30, 0.5627, 0.2863), (2.70, 0.5642, 0.2436), (3.00, 0.5862, 0.2074),
          (3.25, 0.6220, 0.1741), (3.20, 0.6920, 0.1423)]
# (t_stop, τ) at N = 32, Table 3
TABLE3 = [(0.65, 2.83), (0.75, 3.23), (1.00, 3.49), (1.20, 3.97),
          (1.45, 4.51), (1.75, 5.23), (1.90, 6.44), (2.00, 8.29)]


def sweep(N):
    ops = mx.build_operators(N)
    return [mx.run_simulation(a, mx.idata_sin, ops) for a in A]


def rate(res):
    return cv.fit_decay(res['t'], res['norm_hm1'], cv.original_window(res['t']))['r']


@pytest.fixture(scope='module')
def sweep32():
    return sweep(32)


@pytest.fixture(scope='module')
def sweep64():
    return sweep(64)


def test_table3_n32(sweep32):
    for res, (t_stop, tau) in zip(sweep32, TABLE3):
        assert np.isclose(res['t'][-1], t_stop)
        assert round(1 / rate(res), 2) == tau
    alpha = cv.fit_power_law(A, [rate(r) for r in sweep32])['alpha']
    assert round(alpha, 2) == 1.61


def test_table2_n64(sweep64):
    for res, (t_stop, hm1, r) in zip(sweep64, TABLE2):
        assert np.isclose(res['t'][-1], t_stop)
        assert round(res['norm_hm1'][-1], 4) == hm1
        assert round(rate(res), 4) == r
    alpha = cv.fit_power_law(A, [rate(r) for r in sweep64])['alpha']
    assert round(alpha, 3) == 1.777


def test_replot_norms_agrees_with_fit_helpers(sweep32):
    import matplotlib
    matplotlib.use('Agg')
    _, slopes = mx.replot_norms(sweep32, A)
    assert np.allclose(-slopes, [rate(r) for r in sweep32], rtol=0, atol=0)


def test_runs_are_deterministic():
    ops = mx.build_operators(32)
    r1 = mx.run_simulation(0.75, mx.idata_sin, ops)
    r2 = mx.run_simulation(0.75, mx.idata_sin, ops)
    for k in r1:
        assert np.array_equal(r1[k], r2[k])


def test_save_load_round_trip(tmp_path):
    ops = mx.build_operators(32)
    results = [mx.run_simulation(a, mx.idata_sin, ops, t_eval=np.arange(0, 0.55, 0.05))
               for a in (0.5, 0.75)]
    path = str(tmp_path / 'run')
    mx.save_results(results, [0.5, 0.75], path)
    loaded, a_range = mx.load_results(path)
    assert np.array_equal(a_range, [0.5, 0.75])
    for r0, r1 in zip(results, loaded):
        for k in r0:
            assert np.array_equal(r0[k], r1[k])

