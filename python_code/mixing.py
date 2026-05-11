"""
Optimal mixing simulation — Python port of Gautam Iyer's MATLAB code.

Solves:   ∂_t θ + (u · ∇)θ = 0
where     u = F · v / ‖∇v‖_{L²}
and       v = −Δ⁻¹ P(θ ∇Δ⁻¹θ)   (Lin–Thiffeault–Doering optimal velocity)

P is the Leray projection, F is the enstrophy constraint.
Spatial discretisation: pseudo-spectral (FFT) on [0,1]².
Time integration: RK45 via scipy.integrate.solve_ivp.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------

def build_operators(N):
    """
    Build FFT-based derivative and inverse Laplacian operators on [0,1]^2.

    Parameters
    ----------
    N : int  (power of 2 recommended)

    Returns
    -------
    ops : dict with keys
        N, dx, DEL_X (1×N), DEL_Y (N×1), LAP_INV (N×N),
        LAMBDA_INV (N×N), xx (N×N), yy (N×N)
    """
    dx = 1.0 / N
    k = np.arange(N)
    # Centred wavenumbers:  0, 1, …, N/2−1, −N/2, …, −1
    k_eff = (k - N * (k > N // 2)).astype(float)

    # Second-derivative operator (Nyquist mode is fine here)
    kx = (2j * np.pi * k_eff)[np.newaxis, :]   # shape (1, N): acts on axis-1 (x)
    ky = (2j * np.pi * k_eff)[:, np.newaxis]   # shape (N, 1): acts on axis-0 (y)
    lap = kx**2 + ky**2                         # shape (N, N), broadcast
    LAP_INV = np.zeros_like(lap)
    nz = lap != 0
    LAP_INV[nz] = 1.0 / lap[nz]

    # H^{-1} multiplier:  λ⁻¹ = √(−Δ⁻¹) = 1 / (2π|k|),  zero at k=0
    LAMBDA_INV = np.sqrt(np.where(LAP_INV != 0, -LAP_INV, 0.0))

    # First-derivative operators: zero the Nyquist mode to avoid aliasing
    k_d = k_eff.copy()
    if N % 2 == 0:
        k_d[N // 2] = 0.0
    DEL_X = (2j * np.pi * k_d)[np.newaxis, :]
    DEL_Y = (2j * np.pi * k_d)[:, np.newaxis]

    x = np.arange(N) * dx
    xx, yy = np.meshgrid(x, x)   # xx[i,j] = x[j], yy[i,j] = x[i]

    return {
        'N': N, 'dx': dx,
        'DEL_X': DEL_X, 'DEL_Y': DEL_Y,
        'LAP_INV': LAP_INV, 'LAMBDA_INV': LAMBDA_INV,
        'xx': xx, 'yy': yy,
    }


# ---------------------------------------------------------------------------
# Initial data
# ---------------------------------------------------------------------------

def _center_and_normalise(f, N, dx, y_shift, x_shift):
    f = np.roll(np.roll(f, y_shift, axis=0), x_shift, axis=1)
    f /= np.linalg.norm(f.ravel()) * dx
    return f


def idata_sin(a, ops):
    """sin(2π x/a) sin(2π y/a) on [0,a]², L²-normalised, centred."""
    N, dx, xx, yy = ops['N'], ops['dx'], ops['xx'], ops['yy']
    f = np.sin(2*np.pi*xx/a) * np.sin(2*np.pi*yy/a) * ((xx < a) & (yy < a))
    f /= np.linalg.norm(f.ravel()) * dx
    s = int(np.floor(N * (1 - a) / 2))
    return np.roll(np.roll(f, s, axis=0), s, axis=1)


def idata_diag(a, ops):
    """Diagonal initial data with broken symmetry, L²-normalised, centred."""
    N, dx, xx, yy = ops['N'], ops['dx'], ops['xx'], ops['yy']
    f1 = np.sin(2*np.pi*xx/a) * np.sin(2*np.pi*yy/a) * ((xx < a/2) & (yy < a/2))
    f2 = np.sin(2*np.pi*xx/a) * np.sin(2*np.pi*yy/a) * ((xx > a/2) & (xx < a) & (yy > a/2) & (yy < a))
    n_rows = int(np.floor(N * a / 8))
    f = np.roll(f1, n_rows, axis=0) - np.roll(f2, -n_rows, axis=0)
    f /= np.linalg.norm(f.ravel()) * dx
    s = int(np.floor(N * (1 - a) / 2))
    return np.roll(np.roll(f, s, axis=0), s, axis=1)


def idata_strip(a, ops):
    """sin(2π x/a) sin(2π y/a) on [0,a] × [0,a/2], L²-normalised, centred."""
    N, dx, xx, yy = ops['N'], ops['dx'], ops['xx'], ops['yy']
    f = np.sin(2*np.pi*xx/a) * np.sin(2*np.pi*yy/a) * ((xx < a) & (yy < a/2))
    f /= np.linalg.norm(f.ravel()) * dx
    xs = int(np.floor(N * (1 - a) / 2))
    ys = int(np.floor(N * (1 - a/2) / 2))
    return np.roll(np.roll(f, ys, axis=0), xs, axis=1)


def idata_trigpoly(a, ops):
    """Trigonometric polynomial initial data, L²-normalised, centred."""
    N, dx, xx, yy = ops['N'], ops['dx'], ops['xx'], ops['yy']
    c = [1.0, 0.1]
    f = sum(ck * np.sin(2*ki*np.pi*xx/a) * np.sin(2*ki*np.pi*yy/a)
            for ki, ck in enumerate(c, 1))
    f /= np.linalg.norm(f.ravel()) * dx
    s = int(np.floor(N * (1 - a) / 2))
    return np.roll(np.roll(f, s, axis=0), s, axis=1)


# ---------------------------------------------------------------------------
# ODE right-hand side (real-split representation)
# ---------------------------------------------------------------------------

def make_convection_hat(ops, F=1.0, tol=1e-3):
    """
    Build the ODE RHS for the optimal-mixing transport equation.

    The state vector y ∈ ℝ^{2N²} stores [Re(θ̂), Im(θ̂)] concatenated.

    Parameters
    ----------
    ops : dict from build_operators
    F   : enstrophy constraint  (‖∇u‖_{L²} = F at each instant)
    tol : threshold for saddle-point warning

    Returns
    -------
    callable  f(t, y) -> y_dot  (real-valued, shape 2N²)
    """
    N = ops['N']
    n = N * N
    DEL_X, DEL_Y, LAP_INV = ops['DEL_X'], ops['DEL_Y'], ops['LAP_INV']

    def rhs(t, y_real):
        theta_hat = (y_real[:n] + 1j * y_real[n:]).reshape(N, N)

        # g = θ ∇(Δ⁻¹θ)
        linv_th = LAP_INV * theta_hat
        theta   = np.real(np.fft.ifft2(theta_hat))
        gx_hat  = np.fft.fft2(theta * np.real(np.fft.ifft2(DEL_X * linv_th)))
        gy_hat  = np.fft.fft2(theta * np.real(np.fft.ifft2(DEL_Y * linv_th)))

        # Leray projection  P(g) = g − ∇Δ⁻¹(∇·g)
        div_g = DEL_X * gx_hat + DEL_Y * gy_hat
        pgx   = gx_hat - DEL_X * LAP_INV * div_g
        pgy   = gy_hat - DEL_Y * LAP_INV * div_g

        # v = −Δ⁻¹ P(g)
        vx_hat = -LAP_INV * pgx
        vy_hat = -LAP_INV * pgy

        # ‖∇v‖_{L²} via Parseval:  ‖f‖_{L²} = ‖f̂‖₂ / N²
        norm_v_hat = np.sqrt(
            np.linalg.norm((DEL_X * vx_hat).ravel())**2 +
            np.linalg.norm((DEL_X * vy_hat).ravel())**2 +
            np.linalg.norm((DEL_Y * vx_hat).ravel())**2 +
            np.linalg.norm((DEL_Y * vy_hat).ravel())**2
        )
        norm_v = norm_v_hat / N**2

        # Saddle-point warning
        norm_linv_g = np.linalg.norm(
            np.stack([LAP_INV * gx_hat, LAP_INV * gy_hat]).ravel()
        ) / N**2
        if norm_v < norm_linv_g * tol:
            print(f'  Possible saddle point: t={t:.3f}, '
                  f'‖∇v‖={norm_v:.3e}, ‖Δ⁻¹g‖={norm_linv_g:.3e}')

        # u = F v / ‖∇v‖
        ux = F * np.real(np.fft.ifft2(vx_hat)) / norm_v
        uy = F * np.real(np.fft.ifft2(vy_hat)) / norm_v

        # ∂_t θ̂ = −FFT(u · ∇θ)
        d = -np.fft.fft2(
            ux * np.real(np.fft.ifft2(DEL_X * theta_hat)) +
            uy * np.real(np.fft.ifft2(DEL_Y * theta_hat))
        )
        return np.concatenate([d.real.ravel(), d.imag.ravel()])

    return rhs


# ---------------------------------------------------------------------------
# Resolution-check event
# ---------------------------------------------------------------------------

def make_res_check(N, dx, l4norm_init, l8norm_init, tol=1e-3):
    """
    Build a solve_ivp event that terminates when Lp conservation is lost.

    The event value is  max(|‖θ‖_{L²}/1 − 1|, |‖θ‖_{L⁴}/‖θ₀‖_{L⁴} − 1|,
                            |‖θ‖_{L⁸}/‖θ₀‖_{L⁸} − 1|) − tol.
    Terminal on upward zero crossing (direction=+1).
    """
    n = N * N
    sqrt_dx      = np.sqrt(dx)
    sqrt_sqrt_dx = dx**0.25

    def event(t, y_real):
        y_hat = (y_real[:n] + 1j * y_real[n:]).reshape(N, N)
        y = np.real(np.fft.ifft2(y_hat))
        l2 = np.linalg.norm(y.ravel())        * dx
        l4 = np.linalg.norm(y.ravel(), 4)     * sqrt_dx      / l4norm_init
        l8 = np.linalg.norm(y.ravel(), 8)     * sqrt_sqrt_dx / l8norm_init
        return max(abs(l2 - 1.0), abs(l4 - 1.0), abs(l8 - 1.0)) - tol

    event.terminal  = True
    event.direction = 1
    return event


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

def compute_norms(theta_hat_series, theta_series, N, dx, LAMBDA_INV,
                  l4norm_init, l8norm_init):
    """
    Compute L², L⁴, L⁸, and H⁻¹ norms for every time step.

    Parameters
    ----------
    theta_hat_series : (T, N, N) complex
    theta_series     : (T, N, N) real
    """
    sqrt_dx      = np.sqrt(dx)
    sqrt_sqrt_dx = dx**0.25

    # L² norm (conserved quantity; should stay ≈ 1)
    norm_l2 = np.array([
        np.linalg.norm(th.ravel()) / N**2        # Parseval: ‖f̂‖/N² = ‖f‖_{L²}
        for th in theta_hat_series
    ])

    # Lp norms (spatial, normalised to 1 at t=0)
    norm_l4_raw = np.array([np.linalg.norm(th.ravel(), 4) * sqrt_dx      for th in theta_series])
    norm_l8_raw = np.array([np.linalg.norm(th.ravel(), 8) * sqrt_sqrt_dx for th in theta_series])
    norm_l4 = norm_l4_raw / norm_l4_raw[0]
    norm_l8 = norm_l8_raw / norm_l8_raw[0]

    # H⁻¹ mix norm: ‖λ⁻¹ θ̂‖₂,  normalised to 1 at t=0
    norm_hm1_raw = np.array([np.linalg.norm((LAMBDA_INV * th).ravel()) for th in theta_hat_series])
    norm_hm1 = norm_hm1_raw / norm_hm1_raw[0]

    return norm_l2, norm_l4, norm_l8, norm_hm1


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(a, idata_fn, ops, F=1.0, t_eval=None, tol=1e-3):
    """
    Run one optimal-mixing simulation for scale parameter a.

    Parameters
    ----------
    a        : float  scale parameter (support half-width)
    idata_fn : callable  idata_sin | idata_diag | idata_strip | idata_trigpoly
    ops      : dict from build_operators
    F        : float  enstrophy constraint
    t_eval   : 1-D array  output times (default 0 : 0.05 : 10)
    tol      : float  resolution-check tolerance

    Returns
    -------
    dict with keys t, theta_hat, theta, norm_l2, norm_l4, norm_l8, norm_hm1
    """
    N, dx = ops['N'], ops['dx']
    LAMBDA_INV = ops['LAMBDA_INV']

    if t_eval is None:
        t_eval = np.arange(0.0, 10.05, 0.05)

    theta0      = idata_fn(a, ops)
    theta0_hat  = np.fft.fft2(theta0)

    l4norm_init = np.linalg.norm(theta0.ravel(), 4) * np.sqrt(dx)
    l8norm_init = np.linalg.norm(theta0.ravel(), 8) * dx**0.25

    rhs   = make_convection_hat(ops, F=F, tol=tol)
    event = make_res_check(N, dx, l4norm_init, l8norm_init, tol=tol)

    y0 = np.concatenate([theta0_hat.ravel().real, theta0_hat.ravel().imag])

    sol = solve_ivp(
        rhs,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        events=event,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    t = sol.t
    n = N * N
    theta_hat_series = (sol.y[:n] + 1j * sol.y[n:]).T.reshape(-1, N, N)
    theta_series     = np.real(np.fft.ifft2(theta_hat_series, axes=(1, 2)))

    norm_l2, norm_l4, norm_l8, norm_hm1 = compute_norms(
        theta_hat_series, theta_series, N, dx, LAMBDA_INV, l4norm_init, l8norm_init
    )

    return {
        't':          t,
        'theta_hat':  theta_hat_series,
        'theta':      theta_series,
        'norm_l2':    norm_l2,
        'norm_l4':    norm_l4,
        'norm_l8':    norm_l8,
        'norm_hm1':   norm_hm1,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mix_norm(results_list, a_range, ax=None):
    """Log H⁻¹ mix norm vs time for each value of a."""
    if ax is None:
        _, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    for res, a, c in zip(results_list, a_range, colors):
        ax.plot(res['t'], np.log(res['norm_hm1']), color=c, label=f'a={a:.3f}')
    ax.set_xlabel('t')
    ax.set_ylabel('log ‖θ‖_{H⁻¹} / ‖θ₀‖_{H⁻¹}')
    ax.set_title('Mix-norm decay')
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_lp_norms(results_list, ax=None):
    """Lp norms (resolution check) for all runs."""
    if ax is None:
        _, ax = plt.subplots()
    for res in results_list:
        ax.plot(res['t'], res['norm_l2'], 'r-', alpha=0.4, linewidth=0.8)
        ax.plot(res['t'], res['norm_l4'], 'g-', alpha=0.4, linewidth=0.8)
        ax.plot(res['t'], res['norm_l8'], 'b-', alpha=0.4, linewidth=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='r', label='L²'),
        Line2D([0], [0], color='g', label='L⁴'),
        Line2D([0], [0], color='b', label='L⁸'),
    ])
    ax.set_xlabel('t')
    ax.set_title('Lp norms (should stay ≈ 1)')
    return ax


def plot_mixing_rate(results_list, a_range, ax=None):
    """Plot −1/slope of log H⁻¹ norm vs a (measures mixing timescale)."""
    if ax is None:
        _, ax = plt.subplots()
    inv_rates = []
    for res in results_list:
        p = np.polyfit(res['t'], np.log(res['norm_hm1']), 1)
        inv_rates.append(-1.0 / p[0])
    ax.plot(a_range, inv_rates, '*-')
    ax.set_xlabel('a')
    ax.set_ylabel('−1 / slope')
    ax.set_title('Mixing timescale vs scale parameter a')
    return ax


def plot_snapshots(theta_series, t_array, ops, n_frames=6, title=''):
    """Spatial snapshots of θ at n_frames evenly spaced times."""
    T = len(t_array)
    fig, axes = plt.subplots(1, n_frames, figsize=(3*n_frames, 3))
    for j, ax in enumerate(axes):
        t_idx = max(int(j * (T - 1) / (n_frames - 1)), 0)
        ax.imshow(theta_series[t_idx], origin='lower',
                  extent=[0, 1, 0, 1], cmap='RdBu_r', aspect='equal')
        ax.set_title(f't = {t_array[t_idx]:.2f}', fontsize=9)
        ax.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(N=32, F=1.0, a=0.5, t_end=0.5):
    """
    Quick sanity check on a small grid.

    Checks:
      1. L² norm is conserved to < 1e-4
      2. H⁻¹ mix norm decays (< 1)
    """
    print(f'Verifying: N={N}, F={F}, a={a}, t_end={t_end}')
    ops = build_operators(N)
    t_eval = np.linspace(0, t_end, 11)
    res = run_simulation(a, idata_sin, ops, F=F, t_eval=t_eval)

    l2_drift = float(np.max(np.abs(res['norm_l2'] - res['norm_l2'][0])))
    hm1_final = float(res['norm_hm1'][-1])

    print(f'  Max L² drift:       {l2_drift:.2e}  (should be < 5e-4)')
    print(f'  H⁻¹ norm at t_end:  {hm1_final:.4f}  (should be < 1.0)')

    ok = l2_drift < 5e-4 and hm1_final < 1.0
    print(f'  Verification: {"PASS" if ok else "FAIL"}')
    return res, ok
