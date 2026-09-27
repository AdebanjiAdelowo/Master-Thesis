"""Tests for the optional diffusion term  ∂_t θ + u·∇θ = κΔθ."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mixing as mx  # noqa: E402


def state(th_hat):
    return np.concatenate([th_hat.real.ravel(), th_hat.imag.ravel()])


def unpack(y, N):
    return (y[:N * N] + 1j * y[N * N:]).reshape(N, N)


def test_kappa_zero_is_bitwise_the_thesis_rhs():
    ops = mx.build_operators(32)
    _, h = mx.initial_data(0.5, mx.idata_sin, ops)
    y = state(h)
    assert np.array_equal(mx.make_convection_hat(ops)(0.0, y),
                          mx.make_convection_hat(ops, kappa=0.0)(0.0, y))


def test_diffusion_term_is_exact_spectral_laplacian():
    N, kappa = 32, 3e-3
    ops = mx.build_operators(N)
    _, h = mx.initial_data(0.625, mx.idata_sin, ops)
    d0 = unpack(mx.make_convection_hat(ops)(0.0, state(h)), N)
    d1 = unpack(mx.make_convection_hat(ops, kappa=kappa)(0.0, state(h)), N)
    k = np.arange(N)
    k = k - N * (k > N // 2)
    k2 = k[:, None]**2 + k[None, :]**2
    # d1 − d0 cancels O(|d0|) numbers, so compare at roundoff relative to |d0|
    assert np.max(np.abs((d1 - d0) - (-4 * np.pi**2 * kappa * k2 * h))) < 1e-12 * np.max(np.abs(d0))


def test_variance_budget_with_dealiasing():
    """Dealiased Galerkin scheme: d/dt ‖θ‖²_{L²} = −2κ‖∇θ‖²_{L²} exactly."""
    N, kappa = 48, 2e-3
    ops = mx.build_operators(N)
    _, h = mx.initial_data(0.75, mx.idata_sin, ops, dealias=True)
    d = unpack(mx.make_convection_hat(ops, kappa=kappa, dealias=True)(0.0, state(h)), N)
    rate = 2 * np.real(np.vdot(h, d)) / N**4
    grad2 = np.sum(-ops['LAP'] * np.abs(h)**2) / N**4
    assert np.isclose(rate, -2 * kappa * grad2, rtol=1e-10)


def test_lp_stopping_rule_refused_with_diffusion():
    ops = mx.build_operators(16)
    with pytest.raises(ValueError):
        mx.integrate(0.5, mx.idata_sin, ops, kappa=1e-3)
    with pytest.raises(ValueError):
        mx.integrate(0.5, mx.idata_sin, ops, kappa=-1.0, stop_on_res_loss=False)


def test_diffusive_run_dissipates_variance_and_is_close_to_inviscid_for_small_kappa():
    ops = mx.build_operators(32)
    t = np.linspace(0, 0.5, 6)
    s0, _, _, _ = mx.integrate(0.5, mx.idata_sin, ops, t_eval=t)
    s1, _, _, _ = mx.integrate(0.5, mx.idata_sin, ops, t_eval=t, kappa=1e-5, stop_on_res_loss=False)
    l2 = [np.linalg.norm(s1.y[:, i]) for i in range(len(t))]
    assert all(np.diff(l2) < 0)
    rel = np.linalg.norm(s1.y[:, -1] - s0.y[:, -1]) / np.linalg.norm(s0.y[:, -1])
    assert 0 < rel < 1e-2
