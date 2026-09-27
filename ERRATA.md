# Errata

Corrections made to `optimal_mixing_thesis_report.tex` after the original draft
(`optimal_mixing_report.tex`, kept unchanged for reference). Figures and the
underlying simulation outputs (Tables 1–2, stopping times, mix-norm values)
were not altered.

- **Sign convention.** The original draft mixed a positive decay rate with a
  negative fitted slope (`λ = -0.44` in the table vs. `e^{-λt}` in the text).
  The current report adopts a single convention throughout:
  `‖θ‖ ≈ ‖θ₀‖e^{-rt}` with `r > 0` and timescale `τ = 1/r`.
- **Citation attribution.** The `a^{-1}` mixing-rate scaling is Iyer, Kiselev
  & Xu (2014)'s result (Theorem 1.1 and its §4 numerics), not Lin, Thiffeault
  & Doering (2011)'s: LTD2011 does not use a support-size parameter `a` or
  state an `a^{-1}` law. The report now attributes this scaling correctly.
- **Terminology.** The "Saddle-Point Case" for the velocity-field degeneracy
  is retitled "First-Order-Critical Case," matching LTD2011's own treatment
  (they resolve the degeneracy via a second-derivative eigenvalue problem,
  not a saddle point).
- **Citation removed.** All four uses of Drivas (2022) were removed; that
  paper addresses anomalous dissipation under molecular diffusion, a
  different regime, and does not support any claim it was cited for here.
- **Overstated claims softened.** A resolution-dependence discussion that
  read the N=32→64 exponent shift as supporting convergence toward theory
  was corrected: the higher-resolution estimate moves further from, not
  closer to, the theoretical benchmark, so two resolutions cannot establish
  convergence either way. A reproducibility claim of being "free of any
  hidden dependence on library versions, random seeds, or execution
  environment" was softened to reproducibility "for the tested software
  environment," since cross-environment testing was not performed.
- **FFT count.** The reported per-step FFT breakdown in `mixing.py`'s RHS
  evaluation was corrected from 5 forward/5 inverse to the actual 3
  forward/7 inverse.
- **Reference file.** `references/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf`
  had been a mismatched, unrelated paper; it has been replaced with the
  correct Lin–Thiffeault–Doering (2011) PDF.

## Notes following the resolution study

These notes do not change any thesis number; they record how later work
qualifies statements in the thesis documents. Details are in
[`RESOLUTION_STUDY.md`](RESOLUTION_STUDY.md).

- **Source of the resolution dependence.** The report's conclusion that two
  resolutions cannot establish convergence of the fitted exponent stands.
  Runs at N = 128, 256 and 512 show that the change from about 1.61
  (N = 32) to 1.78 (N = 64) is not explained by spatial discretisation
  error over the common resolved time intervals: on a fixed set of time
  windows the exponent changes by less than 5e-3 between N = 32 and
  N = 512. It changes primarily because the fit window (last two thirds of
  each run, which ends at the L^p stopping time) moves to later physical
  times as the resolution increases, while the effective exponent itself
  varies in time. With the thesis protocol the exponent continues to
  change at higher resolution (1.66, 1.48, 1.40 at N = 128, 256, 512), and
  no asymptotic exponent is established.
- **`master_thesis.tex` (submitted thesis), Section on the Python
  simulation.** It describes the b^{-1} scaling as "theoretically
  predicted" and attributes the deviation to the gap between the greedy
  strategy and the global optimum. The Iyer-Kiselev-Xu result is a lower
  bound on the mix norm, whose rate scales as b^{-1}; it is not a
  prediction that the LTD strategy must attain. The resolution study
  attributes the variation of the fitted exponent primarily to the choice
  of fit window and does not test the global-optimum comparison. The
  submitted document is left unchanged as a record.
- **RK45 tolerances.** `optimal_mixing_report.ipynb` stated that
  `rtol = 1e-6`, `atol = 1e-8` match MATLAB's `ode45` defaults. MATLAB's
  defaults are `RelTol = 1e-3`, `AbsTol = 1e-6`, and the original
  `gen_figures.m` does not change them, so the Python runs are integrated
  more tightly than the MATLAB original. Tightening further to
  `rtol = 1e-8`, `atol = 1e-10` changes the fitted rates by less than 1e-6
  (relative). The notebook text has been corrected. The same notebook's
  estimate of 10 to 15 minutes for the N = 64 sweep has been replaced by
  the measured time of a few seconds.
- **Nyquist lines of the Leray projection.** The first-derivative symbols
  zero the Nyquist mode while the inverse Laplacian keeps it (as in the
  MATLAB code), so the discrete projection is not exact on the lines
  k_x = N/2 or k_y = N/2 and the velocity has a small divergence there. The
  affected content is of order 1e-6 of the velocity energy at N = 32 and
  1e-11 at N = 128 for the initial data, decreasing strongly with N. The
  default scheme keeps this behaviour so that the thesis results remain
  exactly reproducible. A dealiased variant, which removes these modes,
  gives the same rates on common windows to 6e-6 (relative) at N = 128 and
  256; no measurable impact on the reported quantities was found.
