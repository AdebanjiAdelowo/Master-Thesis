# mixing.py — Function Reference

This file is a pseudo-spectral simulation of **optimal scalar mixing** on the 2D torus [0,1]².
It solves the transport equation

```
∂_t θ + (u · ∇)θ = 0
```

where the velocity field **u** is chosen at every instant to maximally decrease the H⁻¹ mix-norm of the scalar field θ, subject to an enstrophy constraint. The spatial discretisation uses the FFT, and time integration uses SciPy's RK45 solver.

---

## Spectral Operators

### `build_operators(N)`

Constructs all wavenumber arrays and spectral operators needed for the simulation.

**What it builds:**

| Key | Shape | Meaning |
|-----|-------|---------|
| `DEL_X` | (1, N) | Fourier multiplier for ∂/∂x: `2πi kₓ` |
| `DEL_Y` | (N, 1) | Fourier multiplier for ∂/∂y: `2πi k_y` |
| `LAP_INV` | (N, N) | Fourier multiplier for Δ⁻¹: `1/(kₓ² + k_y²)`, zero at k=0 |
| `LAMBDA_INV` | (N, N) | Fourier multiplier for (-Δ)^{-1/2}: used for the H⁻¹ norm |
| `xx`, `yy` | (N, N) | Physical-space coordinate grids |

The Nyquist mode of the first-derivative operators (`DEL_X`, `DEL_Y`) is zeroed to prevent aliasing errors. The zero mode of `LAP_INV` is left at zero (corresponding to enforcing zero mean on the inverse Laplacian).

---

## Initial Data

All four functions below produce an L²-normalised scalar field centred in the domain, controlled by a scale parameter `a ∈ (0, 1)` which sets the support size.

### `_center_and_normalise(f, N, dx, y_shift, x_shift)`

Internal helper. Shifts a field by `(y_shift, x_shift)` grid points using `np.roll`, then normalises it so that `‖f‖_{L²} = 1`.

---

### `idata_sin(a, ops)`

Generates the initial condition

```
θ(x,y) = sin(2π x/a) sin(2π y/a)
```

supported on the square `[0,a]²`. The field is L²-normalised and shifted so it sits at the centre of [0,1]².

---

### `idata_diag(a, ops)`

Generates a **broken-symmetry diagonal** initial condition. It is constructed from two half-square patches of the sinusoidal mode, with opposite signs and a small row offset applied to each:

- Upper-left quadrant: `+sin·sin` shifted slightly downward.
- Lower-right quadrant: `−sin·sin` shifted slightly upward.

This breaks the diagonal symmetry of the pure sinusoidal case, which can otherwise cause the solver to stall near a saddle point.

---

### `idata_strip(a, ops)`

Generates a sinusoidal field supported on a **half-width strip** `[0,a] × [0, a/2]`:

```
θ(x,y) = sin(2π x/a) sin(2π y/a)   on [0,a] × [0,a/2]
```

Compared to `idata_sin`, this is elongated along the x-direction, giving a different aspect ratio for the initial blob.

---

### `idata_trigpoly(a, ops)`

Generates a **trigonometric polynomial** initial condition — a superposition of two sinusoidal modes:

```
θ = 1.0·sin(2πx/a)sin(2πy/a) + 0.1·sin(4πx/a)sin(4πy/a)
```

The second harmonic (weight 0.1) introduces a small perturbation, breaking perfect sinusoidal symmetry.

---

## ODE Right-Hand Side

### `make_convection_hat(ops, F=1.0, tol=1e-3)`

Factory that constructs and returns the ODE right-hand side function `rhs(t, y_real)` for `solve_ivp`.

**State representation:** The complex Fourier coefficients θ̂ are stored as a real vector by concatenating real and imaginary parts: `y = [Re(θ̂), Im(θ̂)]` ∈ ℝ^{2N²}.

**Algorithm inside `rhs(t, y_real)`:**

1. **Reconstruct θ̂** from the real-split state vector.
2. **Compute g = θ ∇(Δ⁻¹θ)** — the nonlinear term — in physical space, then FFT back.
3. **Apply Leray projection P(g)** to make the velocity divergence-free:
   ```
   P(g) = g − ∇ Δ⁻¹(∇·g)
   ```
4. **Compute v = −Δ⁻¹ P(g)**, the optimal velocity in spectral space.
5. **Normalise v** so that `‖∇v‖_{L²} = F` (the enstrophy constraint). The L² norm is computed via Parseval's theorem.
6. **Saddle-point check:** if `‖∇v‖ < tol · ‖Δ⁻¹g‖`, print a warning — the system may be near a saddle point where the optimal velocity is ill-defined.
7. **Compute the transport derivative** `∂_t θ̂ = −FFT(u · ∇θ)` and return the real-split version.

---

## Resolution Check

### `make_res_check(N, dx, l4norm_init, l8norm_init, tol=1e-3)`

Factory that returns a `solve_ivp` **terminal event** function.

The event fires (and halts integration) when any of the Lp norms drift away from their initial values by more than `tol`:

```
event value = max(|‖θ‖_{L²} − 1|,  |‖θ‖_{L⁴}/‖θ₀‖_{L⁴} − 1|,  |‖θ‖_{L⁸}/‖θ₀‖_{L⁸} − 1|) − tol
```

Since transport by a divergence-free velocity conserves all Lp norms, a significant drift indicates the spectral resolution is no longer sufficient to represent θ accurately (under-resolution / aliasing).

---

## Norm Computation

### `compute_norms(theta_hat_series, theta_series, N, dx, LAMBDA_INV, l4norm_init, l8norm_init)`

Computes four diagnostic norms across the entire time series:

| Norm | Description |
|------|-------------|
| `norm_l2` | L² norm via Parseval's theorem; should stay ≈ 1 (conservation check). |
| `norm_l4` | L⁴ norm in physical space, normalised to 1 at t=0. |
| `norm_l8` | L⁸ norm in physical space, normalised to 1 at t=0. |
| `norm_hm1` | H⁻¹ mix-norm `‖Λ⁻¹ θ̂‖₂`, normalised to 1 at t=0. Measures mixing progress — decay means θ is being mixed to finer scales. |

The L⁴ and L⁸ norms serve as resolution monitors. The H⁻¹ norm is the primary scientific output.

---

## Simulation Runner

### `run_simulation(a, idata_fn, ops, F=1.0, t_eval=None, tol=1e-3)`

Orchestrates a single simulation run:

1. Builds the initial condition using `idata_fn(a, ops)`.
2. Computes initial L⁴ and L⁸ norms (needed to normalise the resolution event).
3. Constructs the RHS via `make_convection_hat` and the event via `make_res_check`.
4. Calls `scipy.integrate.solve_ivp` with RK45, tolerances `rtol=1e-6, atol=1e-8`.
5. Reconstructs θ̂ and θ from the solver output.
6. Computes all norms via `compute_norms`.

**Returns** a dict with keys: `t`, `theta_hat`, `theta`, `norm_l2`, `norm_l4`, `norm_l8`, `norm_hm1`.

---

## Plotting

### `plot_mix_norm(results_list, a_range, ax=None)`

Plots `log ‖θ‖_{H⁻¹}` vs time for each value of `a`, using a Viridis colour map to distinguish runs. A steeper (more negative) slope means faster mixing.

---

### `plot_lp_norms(results_list, ax=None)`

Overlays the L², L⁴, and L⁸ norm time series for all runs on one axes. All curves should stay close to 1; significant drift signals under-resolution.

---

### `plot_mixing_rate(results_list, a_range, ax=None)`

For each run, fits a linear trend to `log ‖θ‖_{H⁻¹}` vs `t` (using the full time series), then plots `−1/slope` vs `a`. The quantity `−1/slope` is the **mixing timescale** — larger means slower mixing.

---

### `replot_norms(results_list, a_range, t_trunc_fraction=1/3, ax=None)`

Refined version of `plot_mixing_rate` that:

1. Skips the first `t_trunc_fraction` of each trajectory (default: first third) to exclude the initial transient.
2. Fits the slope of `log ‖θ‖_{H⁻¹}` over the remaining (steadier) portion.
3. Plots `−1/slope` vs `a` with a linear overlay.
4. If `len(a_range) > 3`, fits a **log-log power law** `slope ∝ a^p` and prints the exponent `p` (expected ≈ −1 from theory).

Returns `(ax, slopes)`.

---

### `plot_snapshots(theta_series, t_array, ops, n_frames=6, title='')`

Displays `n_frames` evenly spaced spatial snapshots of θ(x,y,t) as a row of `imshow` panels with a diverging colormap (RdBu_r). Useful for visual inspection of how the scalar field is being stirred and mixed.

---

## Verification

### `verify(N=32, F=1.0, a=0.5, t_end=0.5)`

Quick sanity check on a small grid. Runs one simulation and checks:

1. **L² drift** < 5×10⁻⁴ — confirms Lp conservation is numerically maintained.
2. **H⁻¹ norm at `t_end`** < 1.0 — confirms mixing is actually occurring.

Prints `PASS` or `FAIL` and returns `(res, ok)`.

---

## Save / Load

### `save_results(results_list, a_range, filename)`

Saves all simulation results to a compressed NumPy `.npz` archive. Each field of each result dict is stored as a named array with the pattern `{key}_{run_index}`.

---

### `load_results(filename)`

Loads an archive saved by `save_results`. Reconstructs the list of result dicts and the `a_range` array. Returns `(results_list, a_range)`.

---

## Entry Point

### `main(N=64, F=1.0, t_eval=None, a_range=None, idata_fn=None, tol=1e-3, save_path=None)`

Full sweep driver (equivalent to `gen_figures.m` in the original MATLAB code):

1. Sets defaults: grid size N=64, enstrophy F=1, time range `t ∈ [0, 10]` in steps of 0.05, `a ∈ {0.5, 0.5625, …, 0.9375}`, initial data = `idata_sin`.
2. Opens live matplotlib figures that update after each run.
3. Loops over `a_range`, calling `run_simulation` for each value and appending results.
4. After all runs, generates figures:
   - **Fig 1** — log H⁻¹ mix-norm vs time (all `a` values).
   - **Fig 2** — Lp norm resolution check.
   - **Fig 3** — Mixing timescale vs `a` (full-series fit).
   - **Fig 3b** — Mixing timescale vs `a` (refined fit, last 2/3 of data).
   - **Snapshot fig** — 6 spatial snapshots for the last run.
5. Optionally saves results and figures to disk if `save_path` is provided.
