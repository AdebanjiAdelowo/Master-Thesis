# Scientific Correction Pass — Validation Report

**Target document:** `optimal_mixing_thesis_report.tex` / `.pdf`
**Original file:** `optimal_mixing_report.tex` / `.pdf` — untouched, preserved as-is
**Scope:** mathematical correctness, numerical honesty, citation accuracy, terminology consistency, formatting artifacts

---

## ⚠️ Data-integrity issue found

`references/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf` in this repo is **not actually the LTD2011 paper**. It's a mismatched file — a commutative-algebra paper ("On a new invariant of finitely generated modules over local rings," Cuong/Cuong/Truong) with zero connection to fluid mixing. The actual verification below used a correct copy sourced separately (`~/Downloads/optimal_stirring.pdf`, arXiv:1009.0834, confirmed as Lin/Thiffeault/Doering, *J. Fluid Mech.* 675:465–476, 2011). **The file in `references/` should be replaced** with a correct copy of the paper.

---

## 1. Scientific Validation Report

| # | Original claim | Corrected claim | Reason | Source |
|---|---|---|---|---|
| 1 | Abstract/text mixed positive rate λ with negative fitted slope (`λ=-0.44` in table vs. `e^{-λt}`) | Adopted `r>0` convention: `‖θ‖≈‖θ₀‖e^{-rt}`, slope `=-r`, `τ=1/r`. Applied globally (abstract, §5.2, Table 1, Table 2, figure captions, §6, Conclusions). | Internal inconsistency — same symbol used with opposite signs | — |
| 2 | Abstract conflated rate and timescale ("mixing timescale grows as λ∝a^-1.78") | Abstract now distinguishes decay-rate magnitude `r∝a^-1.78` from timescale `τ=1/r` | Category error in original phrasing | — |
| 3 | `a^{-1}` scaling attributed to LTD2011 (fig:rate caption, Remark, Conclusions) | Attributed to Iyer–Kiselev–Xu Theorem 1.1 (p.2, eq. 1.2, `p=2` case) + §4/Fig. 2(c) numerics (`m(supp θ₀)=O(a²)` ⟹ rate `∝1/√(a²)=1/a`) | LTD2011 (verified against the real paper) never discusses a support-size parameter `a` or an `a^{-1}` scaling law; its Fig. 1(b)/§5 sweep is over an unrelated ratio `ℓ₀/L`. This scaling is Iyer2014's result, not LTD2011's. | Iyer2014 p.2 Thm 1.1, p.10 §4, p.11 Fig.2(c) |
| 4 | "greedy LTD velocity does not in general achieve the theoretically optimal mixing rate; see Iyer2014, Drivas2022" | Rewritten: instantaneous optimality ≠ global optimality; LTD2011 themselves flag this as open, citing their own statement that the local-in-time strategy "leaves room for further improvement" and that closing the analysis/simulation gap "constitutes a major open question" | Iyer2014 explicitly reports "good qualitative agreement" between LTD numerics and their bound (p.3) — the opposite of proving suboptimality. Drivas2022 never discusses greedy/steepest-descent stirring at all. | LTD2011 p.8 §5 (discussion after Fig. 1(b)) |
| 5 | Degeneracy section titled "Saddle-Point Case"; implementation warning called "saddle point" | Retitled "First-Order-Critical Case"; added accurate note that LTD2011 respond to this degeneracy by maximizing the *second* time derivative via an eigenvalue problem | LTD2011 never uses "saddle point" terminology for this case (verified against real paper §4, eqs. 4.5–4.10) | LTD2011 pp.6–7 §4 |
| 6 | Every `\cite{Drivas2022}` (4 occurrences: greedy-suboptimality remark, fig:rate remark, §6.2, future-work) | All removed; citation dropped from bibliography | Drivas2022 is about anomalous dissipation under molecular diffusion (κ→0 limit) — a different equation, different regime (viscous vs. inviscid), and never discusses enstrophy-constrained instantaneous stirring, the LTD velocity, or the `a^{-1}` scaling. None of its 4 uses were supported by its actual content. | Drivas2022, full read pp.1–13 |
| 7 | §6.2 concluded the N=32→64 exponent change (1.61→1.78) "supports... the hypothesis" that the gap is partly a resolution artefact | Rewritten: since the higher-resolution estimate moved *farther* from −1, not closer, two resolutions cannot establish convergence toward the theory, nor rule it out | The exponent moved in the wrong direction for the "converging to theory" story; original text overstated what the experiment shows | — (internal consistency) |
| 8 | "originally published table" / "original MATLAB-derived report" / "original study" (6 occurrences) — ambiguous with Iyer2014 | Reworded to explicitly say "Gautam Iyer's earlier MATLAB implementation" | Confirmed via `matlab_code/gen_figures.m`: default `N=64`, `a_range=.5:1/16:15/16`, `tol=1e-3` — an exact match to this report's parameters. Iyer2014's own numerical experiment (§4) uses N=768 and `a∈{6/12,...,11/12}` with a different piecewise initial condition — a different experiment entirely. The "original" being reproduced is the local MATLAB code, not the published Iyer2014 paper. | `matlab_code/gen_figures.m`; Iyer2014 p.10 §4 |
| 9 | "conservation laws stand in for explicit dealiasing" (×2: §3.3, Conclusions) | Reworded: conservation monitor is an *a posteriori* diagnostic; does not prevent aliasing, not a substitute for it | Overstated the monitor's role | — |
| 10 | "free of any hidden dependence on library versions, random seeds, or execution environment" | Softened to "deterministic reproducibility... for the tested software environment," with explicit note that environment/version independence was not tested | No cross-environment testing was documented anywhere in the repo | — |
| 11 | LTD optimal-velocity formula and Leray-projection derivation | No change — verified correct | `v=-Δ⁻¹P(θ∇Δ⁻¹θ)`, `u=Fv/‖∇v‖` matches LTD2011 eq. (4.2)–(4.4) exactly under notational correspondence (`φ=Δ⁻¹θ`, `1/τ↔F`) | LTD2011 pp.6–7, eqs. 4.1–4.4 |

## 2. Numerical Validation Report

**Checked directly against `python_code/mixing.py` (static read, no rerun):**
- FFT count per RHS call: **miscounted in the original** — traced every `fft2`/`ifft2` call in `make_convection_hat`; actual breakdown is **3 forward, 7 inverse** (10 total), not "5 forward, 5 inverse." Fixed.
- Resolution-check event formula, `rtol=1e-6`/`atol=1e-8`, `tol=1e-3`, Nyquist-zeroing only on first derivatives, N² Parseval factor, `idata_sin` L²-normalization, last-two-thirds regression in `replot_norms` (default `t_trunc_fraction=1/3`) — all **confirmed correct**, no changes needed.
- Default `a_range`/`N` in `mixing.py`'s `main()` matches `matlab_code/gen_figures.m` exactly — confirms the "original implementation" identity used in item 8 above.

**LaTeX-only bug found and fixed:** the resolution-check equation had an orphaned `\Bigg|` instead of a matching `\left|...\right|` pair around the `L⁴` term — a typo, not a code/logic bug.

**Retained without independent rerun** (per instruction not to force the simulation toward any target number): all entries in Table 1 (`tab:rates`) and Table 2 (`tab:resolution`), all six figures, and the reported stopping times/mix-norm values. These are the author's own simulation outputs and were not touched.

## 3. Citation Audit (final state)

- **`LTD2011`** (7 uses): optimal-velocity formula derivation; Leray projection; degeneracy condition; LTD's own "leaves room for improvement" admission; monotonic-decrease property; greedy-vs-global future-work question. All verified supported against the actual paper (sourced from a correct copy, since the repo's own file is mislabeled).
- **`Iyer2014`** (4 uses): the `a^{-1}`/support-area benchmark scaling (Thm 1.1 + §4 numerics); the goals-list benchmark comparison; the resolution-section benchmark comparison; the degeneracy-remark forward reference. All verified supported.
- **`Drivas2022`**: removed entirely (0 uses); dropped from bibliography.

## 4. Change Log

Edited `optimal_mixing_thesis_report.tex` (`optimal_mixing_report.tex` untouched):

- Title/subtitle: unchanged
- Abstract: rewritten (rate/timescale notation, citation attribution, resolution caveat)
- Introduction: 1 sentence reworded
- Goals: 2 bullets reworded
- Structure-of-Report: 1 sentence reworded
- §2.6 Degeneracy: retitled + rewritten, including its Remark
- §3.2 Pseudo-Spectral RHS: FFT count corrected
- §3.3 Aliasing: paragraph reworded
- §3.5 Resolution Check: LaTeX delimiter fix
- §5 intro: 1 sentence reworded
- §5.2 Mix Norm Decay: paragraph rewritten
- Table 1 (`tab:rates`): header/sign convention changed, caption reworded
- Figure `fig:rate` caption: reattributed
- Remark after Fig. `fig:rate`: rewritten
- §6 intro: 1 sentence reworded
- §6.1 Reproducibility: paragraph softened
- §6.2 Resolution-Dependence: major rewrite + Table 2 caption reworded
- Conclusions: intro paragraph rewritten, bullets 3–4 rewritten
- Future-work: bullets 3–4 reworded
- Bibliography: `Drivas2022` entry removed

Figures themselves: untouched. Recompiled cleanly to a 16-page PDF with no errors and no undefined references.

## 5. Final Verdict

**PASS WITH MINOR CAVEATS**

Caveats:
1. Please swap in a correct copy of the LTD2011 PDF in `references/` — the current file is wrong.
2. Table 1/Table 2 numeric entries were preserved as-is per instruction and were not independently re-derived in this pass beyond the static code checks described above.
