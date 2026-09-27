"""
Diagnostics for the resolution study of the LTD optimal-mixing runs.

Conventions (identical to mixing.py)
------------------------------------
* Grid: x_j = j/N on [0,1), row index = y, column index = x.
* DFT:  θ̂ = fft2(θ) (unnormalised).  The Fourier-series coefficients of the
  trigonometric interpolant are c = θ̂ / N², so by Parseval
  ‖θ‖²_{L²} = Σ_k |c_k|².
* Wavenumbers: k_eff = 0, 1, …, N/2, −N/2+1, …, −1 (Nyquist stored as +N/2).
* H⁻¹ norm: ‖θ‖²_{H⁻¹} = Σ_{k≠0} |c_k|² / (4π²|k|²).

Everything here is post-processing; nothing changes the time integration.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Spectral transfer between grids
# ---------------------------------------------------------------------------

def _upsample_axis(c, M, axis):
    """Zero-pad Fourier coefficients along one axis from N to M ≥ N modes.

    The Nyquist coefficient of an even-N grid (k = N/2, which is also −N/2)
    is split equally between +N/2 and −N/2 on the finer grid.  This is the
    standard trigonometric interpolant of real data: it keeps the result
    real and reproduces the coarse grid values exactly.
    """
    N = c.shape[axis]
    if M == N:
        return c.copy()
    c = np.moveaxis(c, axis, 0)
    out = np.zeros((M,) + c.shape[1:], dtype=complex)
    h = N // 2
    out[:h] = c[:h]                       # k = 0 … N/2−1
    out[M - (N - h - 1):] = c[h + 1:]     # k = −N/2+1 … −1
    if N % 2 == 0:
        out[h] = 0.5 * c[h]               # +N/2
        out[M - h] = 0.5 * c[h]           # −N/2
    else:
        out[h] = c[h]
    return np.moveaxis(out, 0, axis)


def upsample_coeffs(c, M):
    """Coefficients c (N×N, normalised) → M×M coefficients of the same interpolant."""
    return _upsample_axis(_upsample_axis(c, M, 0), M, 1)


def coeffs(theta):
    """Normalised Fourier coefficients c = fft2(θ)/N² of a real grid field."""
    N = theta.shape[0]
    return np.fft.fft2(theta) / N**2


def relative_l2_difference(theta_c, theta_f):
    """
    ‖I θ_c − θ_f‖_{L²} / ‖θ_f‖_{L²}, where I is trigonometric interpolation
    of the coarse field onto the fine grid.  Evaluated exactly in Fourier
    space (Parseval), so no grid-point comparison of different grids occurs.
    """
    cf = coeffs(theta_f)
    cc = upsample_coeffs(coeffs(theta_c), cf.shape[0])
    return np.linalg.norm(cc - cf) / np.linalg.norm(cf)


def unrepresentable_fraction(theta_f, N_coarse):
    """
    ‖(I − Π) θ_f‖ / ‖θ_f‖ where Π keeps modes with max(|k_x|,|k_y|) ≤ N_c/2.

    The interpolant of any N_c-grid field is supported in that band, so this
    is a rigorous lower bound for relative_l2_difference(θ_c, θ_f) whatever
    θ_c is.
    """
    M = theta_f.shape[0]
    c = coeffs(theta_f)
    k = np.abs(wavenumbers(M))
    outside = (k[:, None] > N_coarse // 2) | (k[None, :] > N_coarse // 2)
    return np.sqrt(np.sum(np.abs(c[outside])**2) / np.sum(np.abs(c)**2))


# ---------------------------------------------------------------------------
# Spectra and norms
# ---------------------------------------------------------------------------

def wavenumbers(N):
    k = np.arange(N)
    return (k - N * (k > N // 2)).astype(float)


def shell_spectrum(theta_hat):
    """
    Shell-summed scalar variance spectrum.

    E[m] = Σ_{k : round(|k|) = m} |c_k|²,  m = 0 … ⌊√2 N/2⌉.
    Σ_m E[m] = ‖θ‖²_{L²}.  Shells with m > N/2 are only partially inside the
    square grid and are not comparable across resolutions.
    """
    N = theta_hat.shape[0]
    k = wavenumbers(N)
    kmag = np.rint(np.sqrt(k[:, None]**2 + k[None, :]**2)).astype(int)
    e = np.abs(theta_hat / N**2)**2
    return np.bincount(kmag.ravel(), weights=e.ravel())


def band_fraction(theta_hat, k_cut):
    """Fraction of ‖θ‖²_{L²} in modes with max(|k_x|,|k_y|) > k_cut."""
    N = theta_hat.shape[0]
    k = np.abs(wavenumbers(N))
    outside = (k[:, None] > k_cut) | (k[None, :] > k_cut)
    e = np.abs(theta_hat)**2
    return float(np.sum(e[outside]) / np.sum(e))


def hm1_norm(theta_hat):
    """Physical H⁻¹ norm (not normalised by the initial value)."""
    N = theta_hat.shape[0]
    k = wavenumbers(N)
    k2 = k[:, None]**2 + k[None, :]**2
    w = np.zeros_like(k2)
    w[k2 > 0] = 1.0 / (2 * np.pi * np.sqrt(k2[k2 > 0]))
    return float(np.linalg.norm(w * theta_hat) / N**2)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def original_window(t):
    """Index slice used by replot_figs.m / replot_norms: drop the first ⌊n/3⌋ samples."""
    return slice(int(np.floor(len(t) / 3)), len(t))


def window_slice(t, t0, t1, eps=1e-9):
    """Indices of samples with t0 ≤ t ≤ t1."""
    idx = np.nonzero((t >= t0 - eps) & (t <= t1 + eps))[0]
    if len(idx) == 0:
        return slice(0, 0)
    return slice(idx[0], idx[-1] + 1)


def fit_line(x, y):
    """
    Ordinary least squares y ≈ slope·x + intercept with diagnostics.

    se_slope is the textbook OLS standard error, which assumes independent
    residuals.  lag1_autocorr reports the lag-1 autocorrelation of the
    residuals; values near 1 mean the residuals are a smooth systematic
    curve, the OLS error bar is not a meaningful uncertainty, and the
    departure from a straight line is better read from rms_resid.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    out = {'slope': float(slope), 'intercept': float(intercept), 'n': int(n),
           'x0': float(x[0]), 'x1': float(x[-1]),
           'rms_resid': float(np.sqrt(np.mean(resid**2)))}
    if n > 2:
        s2 = np.sum(resid**2) / (n - 2)
        sxx = np.sum((x - x.mean())**2)
        out['se_slope'] = float(np.sqrt(s2 / sxx))
        out['se_intercept'] = float(np.sqrt(s2 * (1.0 / n + x.mean()**2 / sxx)))
        from scipy.stats import t as student_t
        out['t95'] = float(student_t.ppf(0.975, n - 2))
        rc = resid - resid.mean()
        denom = np.sum(rc**2)
        out['lag1_autocorr'] = float(np.sum(rc[1:] * rc[:-1]) / denom) if denom > 0 else 0.0
    else:
        out['se_slope'] = out['se_intercept'] = out['t95'] = out['lag1_autocorr'] = float('nan')
    return out


def fit_decay(t, norm_hm1, sl):
    """Fit log‖θ‖_{H⁻¹} on a window; returns fit_line output plus r = −slope."""
    f = fit_line(t[sl], np.log(norm_hm1[sl]))
    f['r'] = -f['slope']
    return f


def fit_power_law(a, r):
    """
    Fit r = C a^(−α) by OLS on (log a, log r).

    Returns alpha (so that r ∝ a^(−α)), the fitted log-log slope (= −α,
    the number the thesis quotes), C, standard error and 95% interval of α,
    and the local exponents −Δlog r/Δlog a between neighbouring a values.
    """
    la, lr = np.log(np.asarray(a, float)), np.log(np.asarray(r, float))
    f = fit_line(la, lr)
    alpha = -f['slope']
    return {
        'alpha': alpha,
        'loglog_slope': f['slope'],
        'C': float(np.exp(f['intercept'])),
        'se_alpha': f['se_slope'],
        'ci95': [alpha - f['t95'] * f['se_slope'], alpha + f['t95'] * f['se_slope']],
        'rms_resid_log': f['rms_resid'],
        'n': f['n'],
        'local_alpha': (-np.diff(lr) / np.diff(la)).tolist(),
    }


def local_rate(t, norm_hm1):
    """Instantaneous decay rate −d/dt log‖θ‖_{H⁻¹} (second-order differences)."""
    if len(t) < 3:
        return np.full(len(t), np.nan)
    return -np.gradient(np.log(norm_hm1), t)


def observed_order(values, ratio=2.0):
    """
    Observed convergence order from successive differences of a sequence
    computed at resolutions N, rN, r²N, …

    Returns the differences and p_i = log(|d_i|/|d_{i+1}|)/log(ratio).
    A Richardson estimate is only meaningful when the differences shrink
    monotonically, keep one sign, and the p_i are approximately constant.
    """
    v = np.asarray(values, float)
    d = np.diff(v)
    p = [float(np.log(abs(d[i]) / abs(d[i + 1])) / np.log(ratio))
         if d[i + 1] != 0 else float('nan') for i in range(len(d) - 1)]
    return {'differences': d.tolist(), 'orders': p}


def richardson(v_coarse, v_fine, p, ratio=2.0):
    """Richardson extrapolation of the last two values assuming order p."""
    return v_fine + (v_fine - v_coarse) / (ratio**p - 1.0)


def richardson_assessment(values, ratio=2.0, order_tol=0.25):
    """
    Decide whether Richardson extrapolation of a sequence at N, 2N, 4N, …
    is supported by the data, and apply it only if so.

    Criteria: at least two observed orders; successive differences of one
    sign and decreasing in magnitude; observed orders within ±order_tol of
    their mean.  Returns a dict with 'justified', 'reason' and, when
    justified, the extrapolated value using the mean observed order and
    |extrapolated − finest| as an error estimate.
    """
    o = observed_order(values, ratio)
    d = np.array(o['differences'])
    p = np.array(o['orders'])
    out = {'values': list(map(float, values)), **o}
    if len(p) < 2:
        out.update(justified=False, reason='fewer than three resolutions: no observed order to check')
        return out
    if not (np.all(d > 0) or np.all(d < 0)):
        out.update(justified=False, reason='successive differences change sign: no monotone convergence')
        return out
    if not np.all(np.abs(d[1:]) < np.abs(d[:-1])):
        out.update(justified=False, reason='successive differences do not decrease')
        return out
    if np.max(np.abs(p - p.mean())) > order_tol:
        out.update(justified=False, reason=f'observed orders not consistent (spread > {order_tol})')
        return out
    ext = richardson(values[-2], values[-1], float(p.mean()), ratio)
    out.update(justified=True, reason='monotone convergence with consistent observed order',
               order=float(p.mean()), extrapolated=float(ext), error_estimate=float(abs(ext - values[-1])))
    return out
