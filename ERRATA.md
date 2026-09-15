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
