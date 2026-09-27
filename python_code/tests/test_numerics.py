"""
Verification tests for the numerical building blocks in mixing.py and
convergence.py.  These are code-verification checks (the discrete operators
do what they are documented to do, and converge as expected on problems with
known answers); they are not validation against physical experiment.
"""

import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mixing as mx  # noqa: E402
import convergence as cv  # noqa: E402


def smooth_field(ops):
    """exp(sin 2πx cos 2πy): smooth, periodic, not band-limited."""
    xx, yy = ops['xx'], ops['yy']
    return np.exp(np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy))


def smooth_field_dx(ops):
    xx, yy = ops['xx'], ops['yy']
    return (2 * np.pi * np.cos(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
            * smooth_field(ops))


def to_phys(f_hat):
    return np.real(np.fft.ifft2(f_hat))


# ---------------------------------------------------------------------------
# FFT conventions
# ---------------------------------------------------------------------------

def test_fft_round_trip_and_parseval():
    rng = np.random.default_rng(0)
    N = 32
    f = rng.standard_normal((N, N))
    f_hat = np.fft.fft2(f)
    assert np.allclose(to_phys(f_hat), f, atol=1e-13)
    # ‖f‖_{L²} on [0,1]² = sqrt(Σ f² dx²) = ‖f̂‖₂ / N²
    assert np.isclose(np.linalg.norm(f) / N, np.linalg.norm(f_hat) / N**2, rtol=1e-13)


def test_wavenumber_layout_and_nyquist():
    N = 16
    ops = mx.build_operators(N)
    k = cv.wavenumbers(N)
    assert k.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, -7, -6, -5, -4, -3, -2, -1]
    assert ops['DEL_X'][0, N // 2] == 0 and ops['DEL_Y'][N // 2, 0] == 0
    assert np.allclose(ops['DEL_X'][0, 1:N // 2], 2j * np.pi * k[1:N // 2])


def test_grid_orientation():
    ops = mx.build_operators(8)
    assert np.allclose(ops['xx'][0], np.arange(8) / 8)      # x varies along columns
    assert np.allclose(ops['yy'][:, 0], np.arange(8) / 8)   # y varies along rows


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------

def test_spectral_derivative_converges_spectrally():
    errs = []
    for N in (8, 16, 32, 64):
        ops = mx.build_operators(N)
        df = to_phys(ops['DEL_X'] * np.fft.fft2(smooth_field(ops)))
        errs.append(np.max(np.abs(df - smooth_field_dx(ops))))
    assert errs[-1] < 1e-11
    # faster than any power: each doubling gains far more than a fixed factor
    assert errs[1] / errs[2] > 1e3 and errs[0] / errs[1] > 10


def test_derivative_of_trig_polynomial_is_exact():
    ops = mx.build_operators(32)
    xx, yy = ops['xx'], ops['yy']
    f = np.sin(2 * np.pi * 3 * xx) * np.cos(2 * np.pi * 5 * yy)
    dfy = to_phys(ops['DEL_Y'] * np.fft.fft2(f))
    assert np.allclose(dfy, -10 * np.pi * np.sin(6 * np.pi * xx) * np.sin(10 * np.pi * yy), atol=1e-11)


def test_inverse_laplacian_and_zero_mode():
    N = 64
    ops = mx.build_operators(N)
    f = smooth_field(ops)
    f_hat = np.fft.fft2(f)
    assert ops['LAP_INV'][0, 0] == 0
    # Δ(Δ⁻¹ f) = f − mean(f)
    lap = ops['DEL_X']**2 + ops['DEL_Y']**2   # first-derivative symbols: Nyquist dropped
    g = to_phys(ops['LAP'] * ops['LAP_INV'] * f_hat)
    assert np.allclose(g, f - f.mean(), atol=1e-12)
    # Δ⁻¹ of a constant is zero; Δ⁻¹ of a mean-zero mode is exact
    assert np.allclose(to_phys(ops['LAP_INV'] * np.fft.fft2(np.ones((N, N)))), 0)
    xx, yy = ops['xx'], ops['yy']
    m = np.cos(2 * np.pi * (2 * xx + 3 * yy))
    assert np.allclose(to_phys(ops['LAP_INV'] * np.fft.fft2(m)), -m / (4 * np.pi**2 * 13), atol=1e-14)
    assert lap.shape == (N, N)


def test_lambda_inv_is_inverse_gradient_symbol():
    ops = mx.build_operators(16)
    k = cv.wavenumbers(16)
    kk = np.sqrt(k[:, None]**2 + k[None, :]**2)
    expected = np.where(kk > 0, 1 / (2 * np.pi * np.where(kk > 0, kk, 1)), 0)
    assert np.allclose(ops['LAMBDA_INV'], expected)


def random_vector_hat(N, seed, kmax=6):
    rng = np.random.default_rng(seed)
    k = np.abs(cv.wavenumbers(N))
    band = (k[:, None] <= kmax) & (k[None, :] <= kmax)
    return tuple(np.fft.fft2(to_phys(np.fft.fft2(rng.standard_normal((N, N))) * band)) for _ in range(2))


def test_leray_projection_divergence_free_idempotent_kills_gradients():
    N = 32
    ops = mx.build_operators(N)
    gx, gy = random_vector_hat(N, 1)
    px, py = mx.leray_project(gx, gy, ops)
    div = ops['DEL_X'] * px + ops['DEL_Y'] * py
    assert np.max(np.abs(div)) < 1e-10 * np.max(np.abs(gx))
    qx, qy = mx.leray_project(px, py, ops)
    assert np.allclose(qx, px) and np.allclose(qy, py)
    phi = np.fft.fft2(smooth_field(ops))
    gx2, gy2 = mx.leray_project(ops['DEL_X'] * phi, ops['DEL_Y'] * phi, ops)
    assert np.max(np.abs(gx2)) < 1e-9 * np.max(np.abs(ops['DEL_X'] * phi))


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

def test_hm1_norm_single_mode():
    N = 32
    ops = mx.build_operators(N)
    th = np.cos(2 * np.pi * (2 * ops['xx'] + 3 * ops['yy']))
    exact = np.sqrt(0.5) / (2 * np.pi * np.sqrt(13))
    th_hat = np.fft.fft2(th)
    assert np.isclose(cv.hm1_norm(th_hat), exact, rtol=1e-12)
    assert np.isclose(np.linalg.norm(ops['LAMBDA_INV'] * th_hat) / N**2, exact, rtol=1e-12)


def test_lp_norm_formulas():
    N = 32
    ops = mx.build_operators(N)
    dx = ops['dx']
    th = np.sin(2 * np.pi * ops['xx']) * np.sin(2 * np.pi * ops['yy'])
    # ∫ sin⁴ = 3/8, ∫ sin⁸ = 35/128 over a period (trapezoid rule is exact here)
    assert np.isclose(np.linalg.norm(th.ravel()) * dx, 0.5)
    assert np.isclose(np.linalg.norm(th.ravel(), 4) * np.sqrt(dx), (3 / 8)**0.5)
    assert np.isclose(np.linalg.norm(th.ravel(), 8) * dx**0.25, (35 / 128)**0.25)
    # the resolution-check event uses the same formulas: for an L²-normalised
    # field with matching reference norms it returns exactly −tol
    th = th / 0.5
    l4 = np.linalg.norm(th.ravel(), 4) * np.sqrt(dx)
    l8 = np.linalg.norm(th.ravel(), 8) * dx**0.25
    ev = mx.make_res_check(N, dx, l4, l8, tol=1e-3)
    y = np.fft.fft2(th).ravel()
    assert np.isclose(ev(0.0, np.concatenate([y.real, y.imag])), -1e-3, atol=1e-12)
    ev = mx.make_res_check(N, dx, l4 * 1.01, l8, tol=1e-3)   # L⁴ off by ~1%
    assert ev(0.0, np.concatenate([y.real, y.imag])) > 0


def test_shell_spectrum_sums_to_l2():
    ops = mx.build_operators(32)
    f = smooth_field(ops)
    E = cv.shell_spectrum(np.fft.fft2(f))
    assert np.isclose(E.sum(), np.mean(f**2), rtol=1e-12)


# ---------------------------------------------------------------------------
# Initial data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [mx.idata_sin, mx.idata_diag, mx.idata_strip, mx.idata_trigpoly])
@pytest.mark.parametrize('N', [32, 64])
def test_initial_data_unit_l2(fn, N):
    ops = mx.build_operators(N)
    for a in (0.5, 0.75, 0.9375):
        th = fn(a, ops)
        assert np.isclose(np.linalg.norm(th) * ops['dx'], 1.0, rtol=1e-12)


def test_idata_sin_support_is_centred_square_of_side_a():
    N, a = 64, 0.75
    ops = mx.build_operators(N)
    th = mx.idata_sin(a, ops)
    rows, cols = np.nonzero(np.abs(th) > 0)
    s = int(np.floor(N * (1 - a) / 2))
    assert rows.min() >= s and rows.max() < s + int(a * N)
    assert cols.min() >= s and cols.max() < s + int(a * N)


def test_idata_is_the_same_function_on_nested_grids():
    """For a on the 1/16 lattice the N and 2N data sample the same function."""
    for a in np.arange(0.5, 15 / 16 + 1e-9, 1 / 16):
        th1 = mx.idata_sin(a, mx.build_operators(32))
        th2 = mx.idata_sin(a, mx.build_operators(64))
        ratio = th2[::2, ::2][np.abs(th1) > 1e-8] / th1[np.abs(th1) > 1e-8]
        assert np.allclose(ratio, ratio[0], rtol=1e-12)     # same shape
        assert abs(ratio[0] - 1) < 1e-2                      # normalisations agree to O(dx²)


# ---------------------------------------------------------------------------
# LTD velocity
# ---------------------------------------------------------------------------

def ltd_u(theta_hat, ops, F=1.0):
    vx, vy, _, _ = mx.ltd_velocity_hat(theta_hat, ops)
    nv = mx.gradient_l2_norm(vx, vy, ops)
    return F * to_phys(vx) / nv, F * to_phys(vy) / nv, vx, vy


def hm1_rate(theta_hat, dtheta_hat, ops):
    """d/dt ‖θ‖²_{H⁻¹} for a given θ̂_t."""
    L = ops['LAMBDA_INV']
    return 2 * np.real(np.sum(np.conj(L * theta_hat) * (L * dtheta_hat))) / ops['N']**4


def test_ltd_velocity_divergence_free_and_enstrophy_normalised():
    N, F = 64, 1.7
    ops = mx.build_operators(N)
    th_hat = np.fft.fft2(mx.idata_sin(0.5, ops))
    ux, uy, vx, vy = ltd_u(th_hat, ops, F)
    div = np.abs(ops['DEL_X'] * vx + ops['DEL_Y'] * vy)
    nyq = np.zeros((N, N), bool)
    nyq[N // 2, :] = nyq[:, N // 2] = True
    assert div[~nyq].max() < 1e-12 * np.max(np.abs(vx))
    # Known, inherited from the MATLAB code: on the Nyquist lines the first-
    # derivative symbol is zeroed but LAP_INV is not, so P is not an exact
    # projection there.  The affected content is tiny for resolved data.
    e = np.abs(vx)**2 + np.abs(vy)**2
    assert e[nyq].sum() / e.sum() < 1e-7
    ux_hat, uy_hat = np.fft.fft2(ux), np.fft.fft2(uy)
    assert np.isclose(mx.gradient_l2_norm(ux_hat, uy_hat, ops), F, rtol=1e-10)
    # physical-space check of the same quantity
    grads = [to_phys(D * u) for D in (ops['DEL_X'], ops['DEL_Y']) for u in (ux_hat, uy_hat)]
    assert np.isclose(np.sqrt(sum(np.sum(g**2) for g in grads)) * ops['dx'], F, rtol=1e-10)


@pytest.mark.parametrize('N, bound', [(32, 2e-6), (64, 1e-8), (128, 5e-11)])
def test_nyquist_line_content_of_ltd_velocity_is_small_and_decreasing(N, bound):
    """Quantifies the inherited Nyquist-line inconsistency documented in ERRATA.md
    (velocity energy on k = N/2 lines: about 1e-6 at N=32, 1e-11 at N=128, a = 0.5)."""
    ops = mx.build_operators(N)
    vx, vy, _, _ = mx.ltd_velocity_hat(np.fft.fft2(mx.idata_sin(0.5, ops)), ops)
    nyq = np.zeros((N, N), bool)
    nyq[N // 2, :] = nyq[:, N // 2] = True
    e = np.abs(vx)**2 + np.abs(vy)**2
    assert e[nyq].sum() / e.sum() < bound


def test_ltd_velocity_is_steepest_descent_among_divergence_free_fields():
    """The LTD field minimises d/dt‖θ‖²_{H⁻¹} over div-free u with ‖∇u‖ = F."""
    N, F = 32, 1.0
    ops = mx.build_operators(N)
    th_hat = np.fft.fft2(mx.idata_sin(0.625, ops))
    ux, uy, _, _ = ltd_u(th_hat, ops, F)
    r_ltd = hm1_rate(th_hat, mx.advection_hat(ux, uy, th_hat, ops['DEL_X'], ops['DEL_Y']), ops)
    assert r_ltd < 0
    rng = np.random.default_rng(3)
    for _ in range(20):
        psi = np.fft.fft2(rng.standard_normal((N, N))) * (np.abs(ops['LAP_INV']) * 40)**2
        wx, wy = ops['DEL_Y'] * psi, -ops['DEL_X'] * psi       # div-free
        s = F / mx.gradient_l2_norm(wx, wy, ops)
        r = hm1_rate(th_hat, mx.advection_hat(s * to_phys(wx), s * to_phys(wy), th_hat,
                                              ops['DEL_X'], ops['DEL_Y']), ops)
        assert r >= r_ltd - 1e-12


# ---------------------------------------------------------------------------
# Transport with a prescribed velocity (exact solutions)
# ---------------------------------------------------------------------------

def solve_prescribed(ops, ux, uy, th0, t_end, rtol=1e-10, atol=1e-12):
    N = ops['N']
    n = N * N

    def rhs(t, y):
        th_hat = (y[:n] + 1j * y[n:]).reshape(N, N)
        d = mx.advection_hat(ux, uy, th_hat, ops['DEL_X'], ops['DEL_Y'])
        return np.concatenate([d.real.ravel(), d.imag.ravel()])

    y0 = np.fft.fft2(th0).ravel()
    sol = solve_ivp(rhs, (0, t_end), np.concatenate([y0.real, y0.imag]),
                    method='RK45', rtol=rtol, atol=atol)
    y = sol.y[:, -1]
    return to_phys((y[:n] + 1j * y[n:]).reshape(N, N))


def test_uniform_translation_matches_exact_solution():
    N = 32
    ops = mx.build_operators(N)
    xx, yy = ops['xx'], ops['yy']
    U, V, t = 0.3, -0.2, 0.7
    f = lambda x, y: np.exp(np.sin(2 * np.pi * x) + 0.5 * np.cos(2 * np.pi * y))
    th = solve_prescribed(ops, U * np.ones((N, N)), V * np.ones((N, N)), f(xx, yy), t)
    assert np.max(np.abs(th - f(xx - U * t, yy - V * t))) < 1e-7


def test_steady_shear_converges_to_exact_solution():
    """u = (U sin 2πy, 0): θ(x,y,t) = θ₀(x − U t sin 2πy, y)."""
    U, t = 0.25, 1.0
    f = lambda x, y: np.cos(2 * np.pi * x) * (1 + 0.5 * np.sin(2 * np.pi * y))
    errs = []
    for N in (16, 32, 64):
        ops = mx.build_operators(N)
        xx, yy = ops['xx'], ops['yy']
        th = solve_prescribed(ops, U * np.sin(2 * np.pi * yy), np.zeros((N, N)), f(xx, yy), t)
        errs.append(np.max(np.abs(th - f(xx - U * t * np.sin(2 * np.pi * yy), yy))))
    assert errs[-1] < 1e-7
    assert errs[0] > errs[1] > errs[2]


# ---------------------------------------------------------------------------
# Dealiasing
# ---------------------------------------------------------------------------

def test_dealias_mask_is_two_thirds_rule():
    N = 64
    ops = mx.build_operators(N)
    k = np.abs(cv.wavenumbers(N))
    kept = k[ops['DEALIAS'][0] > 0]
    assert kept.max() == 21 and 2 * kept.max() - N < -kept.max()   # 2K − N < −K ⇔ K < N/3


def test_dealiased_rhs_is_band_limited_and_conserves_l2():
    N = 32
    ops = mx.build_operators(N)
    th0, th0_hat = mx.initial_data(0.5, mx.idata_sin, ops, dealias=True)
    assert np.isclose(np.linalg.norm(th0) * ops['dx'], 1.0)
    assert np.all(th0_hat[ops['DEALIAS'] == 0] == 0) or \
        np.max(np.abs(th0_hat[ops['DEALIAS'] == 0])) < 1e-10 * np.max(np.abs(th0_hat))
    rhs = mx.make_convection_hat(ops, dealias=True)
    y = rhs(0.0, np.concatenate([th0_hat.real.ravel(), th0_hat.imag.ravel()]))
    d = (y[:N * N] + 1j * y[N * N:]).reshape(N, N)
    assert np.all(d[ops['DEALIAS'] == 0] == 0)
    # Galerkin truncation with exact products: d/dt ‖θ‖² = 2 Re⟨θ̂, θ̂_t⟩ = 0
    rate = np.real(np.vdot(th0_hat, d)) / np.linalg.norm(d) / np.linalg.norm(th0_hat)
    assert abs(rate) < 1e-12
    # the un-dealiased scheme does not have this property exactly
    rhs0 = mx.make_convection_hat(ops)
    _, h0 = mx.initial_data(0.5, mx.idata_sin, ops)
    y0 = rhs0(0.0, np.concatenate([h0.real.ravel(), h0.imag.ravel()]))
    d0 = (y0[:N * N] + 1j * y0[N * N:]).reshape(N, N)
    assert abs(np.real(np.vdot(h0, d0)) / np.linalg.norm(d0) / np.linalg.norm(h0)) > 1e-8


# ---------------------------------------------------------------------------
# Grid transfer
# ---------------------------------------------------------------------------

def test_upsampling_reproduces_band_limited_function_including_nyquist():
    Nc, Nf = 16, 64
    f = lambda x, y: (np.cos(2 * np.pi * 8 * x) + np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 7 * y)
                      + 0.3 * np.cos(2 * np.pi * 8 * y) * np.cos(2 * np.pi * 8 * x))
    oc, of = mx.build_operators(Nc), mx.build_operators(Nf)
    c = cv.upsample_coeffs(cv.coeffs(f(oc['xx'], oc['yy'])), Nf)
    assert np.allclose(np.real(np.fft.ifft2(c * Nf**2)), f(of['xx'], of['yy']), atol=1e-12)
    assert cv.relative_l2_difference(f(oc['xx'], oc['yy']), f(of['xx'], of['yy'])) < 1e-13


def test_upsampling_preserves_coarse_grid_values():
    rng = np.random.default_rng(5)
    th = rng.standard_normal((16, 16))
    fine = np.real(np.fft.ifft2(cv.upsample_coeffs(cv.coeffs(th), 48) * 48**2))
    assert np.allclose(fine[::3, ::3], th, atol=1e-12)


def test_unrepresentable_fraction_is_a_lower_bound():
    rng = np.random.default_rng(7)
    ops = mx.build_operators(64)
    thf = smooth_field(ops) + 0.05 * rng.standard_normal((64, 64))
    for _ in range(5):
        thc = rng.standard_normal((32, 32)) * 0.1 + smooth_field(mx.build_operators(32))
        assert cv.unrepresentable_fraction(thf, 32) <= cv.relative_l2_difference(thc, thf) + 1e-15


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def test_fit_line_and_power_law_recover_exact_parameters():
    x = np.linspace(0, 2, 11)
    f = cv.fit_line(x, 3 - 0.4 * x)
    assert np.isclose(f['slope'], -0.4) and np.isclose(f['intercept'], 3) and f['se_slope'] < 1e-12
    a = np.arange(0.5, 1.0, 1 / 16)
    p = cv.fit_power_law(a, 0.2 * a**-1.3)
    assert np.isclose(p['alpha'], 1.3) and np.isclose(p['C'], 0.2)
    assert np.allclose(p['local_alpha'], 1.3)


def test_original_window_matches_replot_figs():
    t = np.arange(0, 1.30 + 1e-9, 0.05)       # 27 samples
    sl = cv.original_window(t)
    assert sl.start == 9 and np.isclose(t[sl][0], 0.45)


def test_observed_order_on_model_sequence():
    v = [1 + 2.0**(-2 * k) for k in range(5)]
    o = cv.observed_order(v)
    assert np.allclose(o['orders'], 2.0)
    assert np.isclose(cv.richardson(v[-2], v[-1], 2.0), 1.0)


def test_richardson_assessment_refuses_non_monotone_sequences():
    ok = cv.richardson_assessment([1 + 2.0**(-2 * k) for k in range(4)])
    assert ok['justified'] and np.isclose(ok['extrapolated'], 1.0)
    bad = cv.richardson_assessment([1.61, 1.78, 1.66, 1.48, 1.40])
    assert not bad['justified'] and 'change sign' in bad['reason']
    assert not cv.richardson_assessment([1.0, 1.1])['justified']
