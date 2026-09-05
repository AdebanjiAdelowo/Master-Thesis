# Final Closure Report — Optimal Mixing Report

**Target:** `optimal_mixing_thesis_report.tex` / `.pdf`
**Preserved untouched:** `optimal_mixing_report.tex`, `optimal_mixing_report.pdf`, all figures in `figures/`
**Prior status:** `PASS WITH MINOR CAVEATS`

---

## 1. LTD2011 Reference Integrity

**What was wrong:** `references/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf` was a mismatched file — a commutative-algebra paper ("On a new invariant of finitely generated modules over local rings," Cuong/Cuong/Truong), unrelated to fluid mixing.

**Replacement file:** `~/Downloads/optimal_stirring.pdf`.

**Identity verification:** opened and read the file directly. Title page shows "Optimal stirring strategies for passive scalar mixing," authors Zhi Lin, Jean-Luc Thiffeault, Charles R. Doering (Institute for Mathematics and Its Applications / UW-Madison / U. Michigan), arXiv identifier 1009.0834v1 (4 Sep 2010), abstract matching the mixing/`H^{-1}`-norm/steepest-descent subject matter cited throughout the report. Cross-checked the journal citation (*J. Fluid Mech.* 675:465–476, 2011) against Iyer2014's own bibliography entry [18], which cites the identical paper the same way — independent third-party confirmation.

**Action taken:**
- Original mismatched file moved to `references/_incorrect_backup/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf.WRONG_FILE_commutative_algebra_paper` (preserved, not deleted).
- Verified `optimal_stirring.pdf` copied into `references/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf`.
- Re-opened the resulting file in `references/` and confirmed it now shows the correct title/authors/abstract.

**Final status:** RESOLVED.

---

## 2. Iyer2014 Citation Recheck

Four body citations, all rechecked directly against `Iyer_Kiselev_Xu_2014_Mix_Bounds.pdf`:

| Location | Claim supported | Paper support | Verdict |
|---|---|---|---|
| §2.6 Remark (the "degeneracy-remark forward reference") | Forward-pointer to the *support-size scaling* result used later — **not** used to support the degeneracy itself (that's cited to LTD2011 in the same paragraph) | Thm 1.1 (p.2) + §4 numerics (p.10) | PASS |
| Fig. `fig:rate` caption | `a^{-1}` scaling is "characteristic of the Iyer–Kiselev–Xu enstrophy-constrained lower bound," offered as a comparison benchmark, not a prediction | Thm 1.1, p=2 case (p.2, eq. 1.2) | PASS |
| Remark after `fig:rate` | Same lower bound, explicitly stated to be a limit on decay speed, "not an exact prediction for the dynamically chosen LTD decay rate" | Thm 1.1 + p.3 discussion of `m(A_λ)` dependence | PASS |
| §6.2 Resolution-Dependence paragraph | "Iyer–Kiselev–Xu benchmark scaling" used only as the `-1` reference point two resolutions are compared against | §4, Fig. 2(c) (`m(supp θ₀)=O(a²)` ⟹ linear-in-`a` prediction) | PASS |

No citation turns the lower bound into an exact prediction for the measured LTD rate. No changes required to any of the four occurrences.

**Total: 4 citations, all PASS.**

---

## 3. MATLAB Provenance

**Evidence found:** `README.md` states explicitly: *"Ported from Gautam Iyer's MATLAB code"* with a link to `math.cmu.edu/~gautam/research/201208-mix-bounds/`, and in its License section: *"Original MATLAB implementation © Gautam Iyer — used with permission for research purposes."* This is a direct, first-party provenance statement in the project's own documentation — not an inference drawn from matching parameters.

**Authorship established:** Yes.

**Wording retained:** "Gautam Iyer's earlier MATLAB implementation" (all occurrences) — kept unchanged, since it is directly supported by explicit repository documentation rather than by parameter correspondence alone.

---

## 4. Numerical Validation Scope

**Statically verified against `python_code/mixing.py`** (source-code reading, no execution): FFT call count and forward/inverse breakdown (corrected to 3 forward / 7 inverse in the prior pass); RK45 tolerances (`rtol=1e-6`, `atol=1e-8`); resolution-check tolerance (`1e-3`); Nyquist-zeroing restricted to first derivatives; `N²` Parseval factor; L²-normalization of initial data; last-two-thirds regression window (`t_trunc_fraction=1/3`); default `a_range`/`N` matching `matlab_code/gen_figures.m`.

**Independently regenerated in this or the prior pass:** none. No rerun of `mixing.py` was performed in either correction pass.

**Numerical outputs retained without regeneration:** all entries in Table 1 (`tab:rates`) and Table 2 (`tab:resolution`), all reported stopping times (`t_stop`) and mix-norm values, and all six figures. These are the author's own simulation outputs.

**Precise caveat (superseding the prior report's wording):** Tables 1 and 2 and the reported stopping-time/mix-norm values were not independently regenerated during either correction pass. Their associated algorithms, parameter choices, regression procedure, and diagnostics were statically checked against the implementation; the numerical outputs themselves were retained unchanged. No repository evidence (logs, saved run artifacts, or documented reruns) was found that independently demonstrates regeneration of these specific tables, so no stronger claim is made.

---

## 5. Final Citation Audit

- **`LTD2011`: 7 occurrences.** Supports: instantaneous steepest-descent/optimal-velocity formula and Leray projection (§2.4–2.5); the first-order degeneracy condition and LTD's own second-derivative eigenvalue treatment of it (§2.6); the local-in-time-vs-global-optimality open question, in LTD's own words (§2.6 Remark); the monotonic-decrease/robust-exponential-decay property (Conclusions); the greedy-vs-global future-work question.
- **`Iyer2014`: 4 occurrences.** Supports only the enstrophy-constrained exponential lower bound and its `O(a²)`-support-size (`a^{-1}`-rate / `a`-timescale) dependence, used strictly as a comparison benchmark, never as an exact prediction (§2.6 Remark forward-reference, Fig. `fig:rate` caption, its Remark, §6.2).
- **`Drivas2022`: 0 occurrences.** Confirmed absent from both body text and bibliography. Not restored — no statement in the report is actually supported by that paper's anomalous-dissipation (diffusive, κ→0) content.

---

## 6. Compilation Validation

- **Status:** compiled successfully from a clean state (`latexmk -C` then full rebuild).
- **Page count:** 16 pages.
- **Undefined citations:** none.
- **Undefined references:** none (38 labels resolved).
- **Substantive warnings:** none. Remaining warnings are cosmetic only — two `Overfull \hbox` line-breaking notices and hyperref "Token not allowed in a PDF string (Unicode)" bookmark warnings (math symbols in section-title PDF bookmarks), both pre-existing and non-substantive.
- **Figures:** all six (`fig1`–`fig6`) load and render correctly; unmodified from the original.
- **Tables/equations/bibliography:** all render correctly.

---

## 7. Files Changed

- `optimal_mixing_thesis_report.tex` — citation/attribution fixes (Iyer2014 bibliography title correction; one leftover "LTD-predicted `a^{-1}`" misattribution in §3.3 corrected to the Iyer–Kiselev–Xu benchmark).
- `optimal_mixing_thesis_report.pdf` — recompiled.
- `references/Lin_Thiffeault_Doering_2011_Optimal_Stirring.pdf` — replaced with the verified correct paper.
- `references/_incorrect_backup/` — new directory holding the preserved, renamed, incorrect file.
- `validation_report.md` — pre-existing deliverable from the prior pass (unchanged in this pass).
- `FINAL_CLOSURE_REPORT.md` — this file (new).

**Confirmed untouched:** `optimal_mixing_report.tex`, `optimal_mixing_report.pdf`, and every file under `figures/`.

---

## 8. Final Verdict

**PASS — portfolio ready**
