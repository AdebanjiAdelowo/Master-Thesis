# Optimal Mixing of Passive Scalars

A Python pseudo-spectral simulation of the **Lin–Thiffeault–Doering optimal
mixing velocity** for passive scalars on the 2-torus $[0,1]^2$.  
Ported from [Gautam Iyer's MATLAB code](https://www.math.cmu.edu/~gautam/research/201208-mix-bounds/).

**Full write-up:** [`optimal_mixing_report.pdf`](optimal_mixing_report.pdf)

---

## Background

The passive scalar $\theta(x,y,t)$ evolves by

$$\partial_t\theta + (u\cdot\nabla)\theta = 0$$

The velocity $u$ is chosen at each instant to minimise the rate of change of
the **$H^{-1}$ mix norm** — the theoretically optimal greedy mixing strategy:

$$v = -\Delta^{-1} P(\theta\,\nabla\Delta^{-1}\theta), \qquad u = F\,\frac{v}{\|\nabla v\|_{L^2}}$$

where $P$ is the Leray projection and $F$ is the enstrophy constraint.

**References**  
- Lin, Thiffeault & Doering (2011) — *Optimal stirring strategies*, J. Fluid Mech.  
- Iyer, Kiselev & Xu (2014) — *Lower bounds on the mix norm*, Nonlinearity  
- Drivas, Elgindi, Iyer & Jeong (2022) — *Anomalous dissipation*, ARMA  

PDFs are in [`references/`](references/).

---

## Repository Layout

```
Master-Thesis/
├── python_code/
│   ├── mixing.py                        # full simulation module
│   ├── requirements.txt                 # dependencies
│   ├── 01_operators_and_idata.ipynb     # notebook: spectral ops + initial data
│   └── 02_rhs_simulation_analysis.ipynb # notebook: RHS, simulation, analysis
├── matlab_code/                         # original MATLAB implementation
├── references/                          # key papers (PDF)
├── figures/                             # generated figures (PDF)
├── Optimal_Mixing_Simulation.ipynb      # high-level demo notebook
├── optimal_mixing_report.tex            # LaTeX report source
├── optimal_mixing_report.pdf            # compiled report (11 pages)
└── .venv/                               # Python virtual environment
```

---

## Setup

**Python 3.12** is required (tested with Anaconda 3.12.2).

```bash
# Clone and enter the project
cd Master-Thesis

# Activate the virtual environment (already created)
source .venv/bin/activate

# Or create it fresh
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_code/requirements.txt
```

---

## How to Run

### Quick verification (~5 seconds)

```bash
cd python_code
python -c "from mixing import verify; verify()"
```

Expected output:
```
Verifying: N=32, F=1.0, a=0.5, t_end=0.5
  Max L² drift:       1.13e-04  (should be < 5e-4)
  H⁻¹ norm at t_end:  0.8598   (should be < 1.0)
  Verification: PASS
```

### Jupyter Notebooks

```bash
cd python_code
jupyter notebook
```

| Notebook | Content |
|---|---|
| `01_operators_and_idata.ipynb` | Spectral operators, wavenumbers, initial data |
| `02_rhs_simulation_analysis.ipynb` | ODE RHS step-by-step, simulation, norm analysis |
| `../Optimal_Mixing_Simulation.ipynb` | High-level demo with all plots |

Select kernel **"Python (Master Thesis)"** when prompted.

### Full sweep

```bash
cd python_code
python mixing.py
```

Runs `main()` with defaults: `N=64`, `F=1`, `a ∈ [0.5, 15/16]` in steps of `1/16`.  
Results are displayed as interactive plots; pass `save_path='run'` to save `.npz` + figures.

### Custom simulation

```python
from mixing import build_operators, idata_sin, run_simulation, plot_snapshots
import numpy as np, matplotlib.pyplot as plt

ops = build_operators(64)
res = run_simulation(0.5, idata_sin, ops, F=1.0, t_eval=np.arange(0, 5.05, 0.05))

print(f"H⁻¹ at t_end: {res['norm_hm1'][-1]:.4f}")
plot_snapshots(res['theta'], res['t'], ops)
plt.show()
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `N` | 64 | Spectral modes per direction (power of 2) |
| `F` | 1.0 | Enstrophy constraint $\|\nabla u\|_{L^2} = F$ |
| `a` | 0.5 | Scale parameter (initial data support size) |
| `tol` | 1e-3 | $L^p$ conservation tolerance (stopping criterion) |

---

## Module Reference (`mixing.py`)

| Function | Description |
|---|---|
| `build_operators(N)` | Build spectral operator dict (DEL_X, LAP_INV, …) |
| `idata_sin(a, ops)` | Sinusoidal initial data on $[0,a]^2$ |
| `idata_diag(a, ops)` | Diagonal antisymmetric initial data |
| `idata_strip(a, ops)` | Strip initial data on $[0,a]\times[0,a/2]$ |
| `idata_trigpoly(a, ops)` | Trigonometric polynomial initial data |
| `make_convection_hat(ops, F)` | Build the ODE right-hand side closure |
| `make_res_check(N, dx, …)` | Build the $L^p$ resolution-check event |
| `run_simulation(a, idata_fn, ops, …)` | Run a single simulation |
| `compute_norms(…)` | Compute $L^2$, $L^4$, $L^8$, $H^{-1}$ norms |
| `plot_mix_norm(results, a_range)` | Plot log mix norm vs time |
| `plot_lp_norms(results)` | Plot $L^p$ conservation check |
| `plot_mixing_rate(results, a_range)` | Plot mixing timescale vs $a$ |
| `plot_snapshots(theta, t, ops)` | Plot spatial snapshots |
| `replot_norms(results, a_range)` | Refined slope fit + power-law analysis |
| `save_results(results, a_range, path)` | Save to compressed `.npz` |
| `load_results(path)` | Load saved results |
| `verify(N, F, a, t_end)` | Quick sanity check |
| `main(…)` | Full sweep driver |

---

## MATLAB Correspondence

| MATLAB file | Python equivalent |
|---|---|
| `gen_figures.m` | `main()` + `run_simulation()` |
| `convection_hat.m` | `make_convection_hat()` |
| `res_check.m` | `make_res_check()` |
| `fn_norm.m` | `compute_norms()` |
| `idata_sin/diag/strip/trigpoly.m` | same names in `mixing.py` |
| `replot_figs.m` | `replot_norms()` |
| `save_data.m` | `save_results()` / `load_results()` |

---

## Python Virtual Environment

```
.venv/bin/python   →   /opt/anaconda3/bin/python3  (3.12.2)
.venv/lib/python3.12/site-packages/   →   project packages
```

Jupyter kernel registered as **"Python (Master Thesis)"**.

---

## License

Code: MIT.  
Original MATLAB implementation © Gautam Iyer — used with permission for research purposes.
