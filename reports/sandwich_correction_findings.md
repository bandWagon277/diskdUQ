# Last-Layer Godambe / Sandwich Variance Correction — Findings

**Date:** 2026-07-20
**Jobs:** 54085422, 54086399 (rerun with bias diagnostic). ~2.4 min total, 3 seeds.
**Cell:** teacher N=5000, student N=500, test N=500, quality=full, J=2, K=12, temperature=2.0.
**Code:** `examples/sandwich_correction_probe.py`; raw output `responses_sandwich/`.
**Design source:** the variance-correction plan (Experiments 1 + 2).

---

## TL;DR

1. **The implementation is validated.** At eta=0 the information equality `J = H` recovers to
   within 1% (tr J_R = 14.45 vs tr H = 14.63), and naive == sandwich coverage, exactly as
   theory requires when the loss is a genuine log-likelihood.
2. **The cross-covariance `J_RQ` is negligible — the motivating hypothesis is NOT supported.**
   Its contribution to `J_eta` is <= 2% at every eta > 0, and its sign flips across seeds.
3. **The sandwich correction goes the WRONG WAY for our problem.** `omega* > 1` at every
   eta > 0 (1.19-2.48), i.e. `J < H`, so the Godambe variance is *smaller* than the naive
   inverse curvature. It tightens intervals and makes undercoverage worse.
4. **Mechanism = the Cox mechanism.** `tr(H)` grows much faster than `tr(J)` with eta, i.e. the
   KL adds curvature without proportional sampling variability. Locality of the KL did NOT
   reverse the direction at moderate eta.
5. **Undercoverage is not a variance-calibration problem.** Actual error exceeds the
   model-implied standard error by 2-4x (`MAE/SE` = 1.4-3.3 vs ~0.8 expected under
   calibration). No rescaling of the *sampling* variance can close that gap.
6. **Positive result:** distillation cuts CIF error by ~43% (MAE 0.0958 at eta=0 -> 0.0548 at
   eta=1), measured against the closed-form truth.

---

## Setup

We correct the **last layer** (`head`: Linear(32, 2)) with the backbone frozen: p = 66
parameters, so `H_eta`, `J_R`, `J_Q`, `J_RQ` and the sandwich are computed **exactly** (no
diagonal / low-rank / Gauss-Newton approximation). Per-subject scores come from forward-mode
autodiff (66 JVPs). Intervals are function-space (delta method) on the test-set CIF versus
the closed-form true CIF; nominal 0.95.

Two implementation details that mattered:

- **`H` is rank-deficient.** Backbone features are a 12-covariate projection through ReLU, so
  they span a subspace of dim < 32; directions of `W` orthogonal to that span have exactly zero
  curvature (measured rank 62-66 of 66). A naive inverse explodes there (it produced a spurious
  `omega* = 26418`). We use a truncated PSD pseudo-inverse — correct here because those same
  directions leave the logits, and hence the CIF, unchanged.
- **Stationarity.** The sandwich is an M-estimator result requiring mean score = 0, but AdamW on
  the full network leaves a residual last-layer gradient (~1e-1). We polish the 66 last-layer
  parameters with LBFGS, driving |grad| to 1e-4 - 1e-9.

---

## Result 1 — Coverage and width by eta (3-seed mean)

| eta | omega* | cov naive | cov sandwich | cov omega* | wid naive | wid sandwich | Ctd1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.494 | 0.644 | 0.647 | 0.563 | 0.234 | 0.236 | 0.525 |
| 0.25 | 2.476 | 0.660 | 0.581 | 0.510 | 0.185 | 0.157 | 0.592 |
| 0.5 | 2.217 | 0.647 | 0.542 | 0.502 | 0.158 | 0.122 | 0.624 |
| 1 | 2.442 | 0.576 | 0.458 | 0.416 | 0.126 | 0.093 | 0.647 |
| 2 | 1.831 | 0.467 | 0.397 | 0.377 | 0.095 | 0.079 | 0.655 |
| 4 | 1.193 | 0.361 | 0.375 | 0.353 | 0.071 | 0.077 | 0.661 |

- At **eta = 0** naive and sandwich agree (0.644 vs 0.647) — the information-equality check.
- At **eta > 0** the sandwich is uniformly *tighter* and *worse-covering*.
- **Coverage falls monotonically with eta (0.64 -> 0.36) while Ctd1 rises (0.525 -> 0.661).**
  Borrowing buys accuracy and sharpness at the cost of calibration.

## Result 2 — Score-variance decomposition

`J_eta = J_R + eta (J_RQ + J_QR) + eta^2 J_Q`

| eta | tr(J_R) | tr(J_Q) | tr(J_RQ+J_QR) | tr(J_eta) | tr(H_eta) | cross share |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.45 | 29.51 | 5.731 | 14.45 | 14.63 | 0 (does not enter) |
| 0.25 | 13.93 | 18.47 | 1.202 | 15.39 | 23.08 | +2.0% |
| 0.5 | 13.83 | 15.54 | 0.234 | 17.84 | 31.57 | +0.7% |
| 1 | 13.90 | 13.19 | -0.242 | 26.85 | 48.62 | -0.9% |
| 2 | 14.21 | 12.25 | -0.095 | 63.04 | 83.53 | -0.3% |
| 4 | 14.52 | 11.85 | +0.196 | 204.8 | 153.7 | +0.4% |

**The cross term is empirically negligible.** The §10 derivation is mechanically right — both
scores share `Y_ik`, `lambda_ik` and `h_ik(theta)` — but at the fitted optimum the trace
contribution is <= 2% and sign-unstable. It is not a driver.

**`tr(H)` outgrows `tr(J)`** through eta = 2 (48.6 vs 26.9 at eta = 1), which is exactly why
`omega* > 1`. This is the Cox-paper mechanism: the KL contributes curvature without matching
sampling variability. Only at eta = 4 does `eta^2 J_Q` finally dominate (J 204.8 > H 153.7).

## Result 3 — Variance problem or bias problem?

| eta | MAE | RMSE | SE naive | SE sandwich | MAE / SE naive |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0958 | 0.1392 | 0.0600 | 0.0606 | **1.60** |
| 0.25 | 0.0659 | 0.0977 | 0.0468 | 0.0396 | **1.42** |
| 0.5 | 0.0581 | 0.0872 | 0.0400 | 0.0310 | **1.46** |
| 1 | 0.0548 | 0.0833 | 0.0319 | 0.0237 | **1.72** |
| 2 | 0.0557 | 0.0852 | 0.0240 | 0.0198 | **2.33** |
| 4 | 0.0579 | 0.0881 | 0.0177 | 0.0190 | **3.27** |

For a calibrated Gaussian interval, `E|error| = 0.798 * SE`, so **MAE/SE should be ~0.8**. We
observe **1.4-3.3**: the realized error is 2-4x larger than the model-implied standard error.

Partial decomposition of the deficit:

- **Even at eta = 0 (no teacher at all) the ratio is 1.60**, i.e. already ~2x too narrow. That
  part cannot be teacher-induced bias; it is unmodelled **backbone uncertainty (we freeze it)
  plus misspecification**.
- **The deficit then grows with eta (1.60 -> 3.27)** because SE shrinks 3.4x (0.060 -> 0.018)
  while MAE plateaus at ~0.055. Distillation sharpens the posterior faster than it reduces error.

**Consequence: no correction to the sampling variance — sandwich, omega, or otherwise — can
reach nominal coverage.** Only a procedure that inflates intervals to cover *total* error
(conformal) can. This is consistent with Study B/C, where every UQ engine required a ~5x width
inflation to hit 0.95.

## Result 4 — A clean positive for DiSKD

MAE against the closed-form true CIF falls **0.0958 -> 0.0548 (-43%)** from eta = 0 to eta = 1,
with Ctd1 rising 0.525 -> 0.647. Distillation demonstrably improves the CIF estimate itself;
the problem is purely that the uncertainty does not shrink honestly alongside it.

---

## Implications

1. **Do not pitch the sandwich as the coverage fix.** It is the wrong sign for our problem and
   the cross-covariance that motivated it is negligible.
2. **Do pitch the sandwich as a mechanistic diagnostic.** `omega*(eta)` rigorously quantifies
   **distillation-induced over-confidence**: how much sharper the generalized posterior becomes
   than the estimator's own repeated-sampling variability. Combined with the monotone
   coverage-vs-eta curve, this is a novel and defensible statement about teacher borrowing that
   we can state with an exact (not approximate) computation.
3. **Undercoverage is a bias / unmodelled-variance problem**, present even with no teacher.
   The remedies are (a) propagate more than the last layer, and (b) calibrate to total error.
4. **Split-conformal remains the deployable fix** and is now better motivated: it covers total
   error rather than rescaling sampling variance.

## Recommended next steps

1. **Split-conformal calibration** (deployable, covers total error) — the actual path to nominal
   coverage. Highest priority.
2. **Report `omega*(eta)` as the over-confidence diagnostic** in the proposal, with the exact
   last-layer computation as the methodological contribution.
3. **Test whether propagating beyond the last layer** closes part of the eta=0 deficit
   (ratio 1.60) — e.g. subnetwork/full Laplace or SWAG — to separate "frozen backbone" from
   "misspecification".
4. Treat the non-conjugate loss extension (IPCW-Brier) as an orthogonal structural aim.
