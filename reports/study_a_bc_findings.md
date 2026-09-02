# Study A + B/C: Why warm-start SGLD gives wide, undercovering intervals

**Date:** 2026-07-03
**Jobs:** Study A = 52791237 (8.5 min), Study B/C = 52791241 (2.8 min). Both completed
cleanly (empty error logs).
**Cell (both):** teacher N=5000, student N=500, test N=500, teacher quality = full,
J=2 causes, K=12 intervals.
**Raw outputs:** `responses_study_a/` and `responses_study_bc/` (report `.md` + per-seed
`.csv`).
**Design source:** `responses/sgld_literature_and_extension_plan.md` (Codex plan, §4
Studies A/B/C).

---

## TL;DR

1. **Parameter count is NOT the cause** of SGLD's poor uncertainty. The *smallest*
   network (298 params) is the *worst*; coverage improves with more parameters; R-hat
   stays high at every size. The problem is posterior geometry + generalized-loss
   calibration, not raw dimension.
2. **SGLD is not the best UQ engine.** After a scalar width-calibration brings every
   method to nominal 95% coverage, a **deep ensemble gives the tightest intervals and
   the best point estimate**; SGLD is the widest and weakest.
3. **A single scalar calibration fixes coverage for all engines** — validating the plan's
   claim that raw generalized-posterior spread needs correction/calibration, but not
   requiring the main method to be non-Bayesian.
4. **Caveat:** the calibration scalar in Study B/C is an *oracle* (fitted against the
   closed-form true CIF), not a deployable method. The deployable version is
   split-conformal, which we have not yet run. Only 2 seeds.

---

## Study A — Is parameter count the cause?

Controlled dimension ablation: student hidden dim ∈ {8, 32, 128}, sampler target =
full-network SGLD vs last-layer-only SGLD. Same teacher, same split, closed-form true CIF.
SGLD: eps=2e-4 (fixed), gamma=0.55, noise 1.0, 3 chains × 300 draws, 300 epochs. Seeds 42, 43.

Averaged over seeds (from `sgld_parameter_diagnostic_report.md`):

| Hidden | Params | Target | Ctd1 | Ctd2 | Dev | Coverage | Width | R-hat | ESS |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 298 | full | 0.497 | 0.555 | 6.287 | 0.316 | 0.173 | 1.61 | 3 |
| 8 | 298 | last_layer | 0.506 | 0.536 | 6.373 | 0.348 | 0.219 | 2.48 | 2 |
| 32 | 2722 | full | 0.578 | 0.618 | 5.998 | 0.728 | 0.317 | 1.81 | 2 |
| 32 | 2722 | last_layer | 0.655 | 0.667 | 5.277 | 0.783 | 0.192 | 2.10 | 2 |
| 128 | 35458 | full | 0.604 | 0.556 | 6.590 | 0.814 | 0.513 | 1.34 | 6 |
| 128 | 35458 | last_layer | 0.652 | 0.591 | 5.609 | 0.823 | 0.290 | 1.89 | 2 |

**Reading (against the plan's decision rule):**
- *Small hidden dims stay weak → parameter count is not the main explanation.* The
  298-param net has the **worst** coverage (0.32) and chance-level discrimination
  (Ctd1 ≈ 0.50). Coverage **improves** with parameters (0.32 → 0.73 → 0.81), the opposite
  of the "too many parameters" hypothesis.
- **R-hat stays high (1.3–2.5) at every size**, including 298 params — non-convergence is
  not driven by dimension.
- **Last-layer-only SGLD improves the point estimate, width, and deviance** (e.g. h=32:
  Ctd1 0.66 vs 0.58, width 0.19 vs 0.32, dev 5.28 vs 6.00) **but does not fix R-hat.**

**Conclusion:** the weakness is **posterior geometry + generalized-loss calibration**, not
raw parameter count. Last-layer inference is a promising modular engine for the point
estimate, but convergence/spread still needs a calibration layer.

---

## Study B/C — Stronger UQ comparators + scalar calibration

Engines at the headline cell: deep ensemble, nonparametric patient bootstrap ensemble,
MC-dropout, warm-start SGLD. Calibration = a single scalar that widens each engine's
interval, chosen on a 20% calibration split. Seeds 42, 43.

From `uq_engine_comparator_report.md`:

| Engine | Calibrated | Scale | Ctd1 | Ctd2 | Dev | Coverage | Width | rho(width,err) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deep | 0 | 1.00 | 0.759 | 0.703 | 5.269 | 0.391 | 0.062 | +0.496 |
| deep | 1 | 4.88 | 0.759 | 0.703 | 5.269 | **0.961** | **0.286** | +0.502 |
| bootstrap | 0 | 1.00 | 0.748 | 0.674 | 5.287 | 0.422 | 0.080 | +0.422 |
| bootstrap | 1 | 5.84 | 0.748 | 0.674 | 5.287 | 0.967 | 0.414 | +0.418 |
| mc_dropout | 0 | 1.00 | 0.716 | 0.684 | 5.507 | 0.454 | 0.097 | +0.222 |
| mc_dropout | 1 | 3.85 | 0.716 | 0.684 | 5.507 | 0.952 | 0.331 | +0.347 |
| sgld | 0 | 1.00 | 0.606 | 0.645 | 5.958 | 0.749 | 0.430 | +0.417 |
| sgld | 1 | 1.32 | 0.606 | 0.645 | 5.958 | 0.960 | **0.493** | +0.428 |

**Reading:**
- **Raw:** deep / bootstrap / MC-dropout are very tight (width 0.06–0.10) but badly
  undercover (0.39–0.45). SGLD covers best raw (0.75) but is widest (0.43).
- **Calibrated to nominal (~0.95–0.97):** width-at-nominal is the discriminator —
  deep ensemble **0.286 (tightest)** < MC-dropout 0.33 < bootstrap 0.41 <
  **SGLD 0.49 (widest)**.
- The deep ensemble also wins on **point estimate** (Ctd1 0.759 vs SGLD 0.606),
  **deviance** (5.27 vs 5.96), and **width–error correlation** (+0.50 vs +0.42).
- SGLD's only edge: smallest calibration scale (1.32 vs ~5) — i.e. closest to
  calibrated in raw form — but cosmetic given its worst calibrated width.
- **A single scalar fixes coverage for every engine** → calibration is the fast,
  general fix for undercoverage.

**Caveat (important, do not overstate):** the calibration scalar is an **oracle**,
fitted against the closed-form true CIF — the report labels it "a diagnostic for
discussion, not a deployable conformal method." Calibrated-coverage numbers are therefore
optimistic. The deployable analog is **split-conformal** calibration, not yet run.
Only 2 seeds.

---

## Implications for Aim 1a (per plan §5–6)

1. **Reframe the aim.** From "SGLD Bayesian DiSKD" to **loss-general,
   covariance-corrected generalized Bayesian DiSKD**. The Bayesian engine can be
   SGLD / last-layer Laplace / SWAG / last-layer sandwich correction; deep ensemble,
   bootstrap ensemble, and MC-dropout should be treated as strong non-Bayesian baselines.
2. **Deep ensemble + calibration is the strong baseline to beat**, not necessarily the
   headline method. The main method should aim to match or beat it with a Bayesian
   covariance-corrected generalized posterior.
3. **The problem is calibration + posterior geometry, not model size** — Study A closes
   the "too many parameters" question directly.

## Recommended next steps

1. **Run last-layer sandwich/Godambe correction**. Highest value for the Bayesian main
   story: it directly addresses the fact that teacher KL is a surrogate external
   likelihood, so raw generalized posterior variance need not equal repeated-sampling
   variance.
2. **Replace the oracle scalar calibration with real split-conformal only as a safety
   layer / deployable comparator**, not as the primary Bayesian contribution.
3. **Use deep ensemble + calibration as the strongest empirical baseline to beat.**
4. **Update proposal language** to "covariance-corrected generalized-posterior predictive
   interval" and explicitly state that non-Bayesian UQ engines are baselines.
5. **Non-conjugate loss extension** (Study E, start with IPCW-Brier) for the structural
   expansion of Aim 1a beyond Bernoulli-KL.
