# Per-horizon diagnostic verdict — bias vs under-dispersion vs propagation

**Job 56811711 (K=20, clean log-linear DGP, welling_teh SGLD, R=20 repeated datasets).**
Seeds 42-43. Raw: `responses_horizon/`. Nominal 0.95.

| eta | cov | CIF bias | PostSD | EmpSD | PostSD/EmpSD | miss_below (over) | miss_above (under) | hazard MSE | hazard bias | CIF MSE | R-hat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.765 | -0.006 | 0.0338 | 0.0401 | **0.84** | 0.157 | 0.078 | 1.6e-2 | -0.044 | 2.9e-3 | 1.05 |
| 1 | 0.706 | -0.010 | 0.0203 | 0.0141 | **1.44** | 0.162 | 0.131 | 1.9e-2 | -0.053 | 1.9e-3 | 1.07 |

**Setup validation:** chains converge (R-hat 1.05-1.07, vs headline ~2); point estimate is good
(posterior mean tracks true CIF; CIF MSE ~2-3e-3). So convergence and center are NOT the problem.

---

## Result per assumption

### A — Systematic bias (over/under-estimation)
- **Mean CIF bias ≈ 0** (-0.006 / -0.010): the posterior *center* tracks the truth (overlay panel).
  NOT a global bias.
- **But directional misses are one-sided at the FRONT.** `miss_below` (P{F0<L}, over-estimation)
  starts ~0.25-0.30 at horizon 1 and falls; `miss_above` (under-estimation) starts low and rises,
  crossing near mid-horizon. So **early risk is over-estimated for the low-risk subpopulation**
  (their true early CIF ≈ 0 but the interval's lower bound sits above it) — a heterogeneous /
  boundary bias that averages to ~0.
- **Hazard is biased** (hazard bias ≈ -0.05, under-estimated) but this does **not** propagate into a
  large CIF bias — the survival weighting/normalization absorbs it.
- **Verdict:** mild, one-sided, early-horizon over-estimation for a subgroup; not the dominant cause.

### B — Under-dispersion (posterior spread too small)  ← dominant at eta=0
- **eta=0: PostSD/EmpSD = 0.84 (<1), uniformly across horizons → CONFIRMED under-dispersion.** The
  posterior SD is ~16% smaller than the estimator's true repeated-sampling SD. This is the main
  driver of the eta=0 undercoverage; a ~1.2x widening (or conformal) would largely fix it.
- **eta=1: PostSD/EmpSD = 1.44 (>1) → NOT under-dispersed** vs the fixed-teacher sampling SD, yet
  coverage is *lower* (0.71). The teacher shrinks the estimator's data-variability (small EmpSD) while
  injecting bias, so the residual miss is **teacher bias, not spread**. Caveat: EmpSD here holds the
  teacher FIXED, so it excludes teacher uncertainty — it is a *conditional* sampling SD.
- **Verdict:** under-dispersion is the eta=0 story; at eta=1 the teacher removes it and the problem
  becomes bias.

### C — Cumulative propagation (hazard error accumulating into CIF)
- **Interval WIDTH grows with horizon** (band widens; PostSD rises) — the propagation signature is
  present in the *spread*.
- **But the CIF ERROR does NOT accumulate.** CIF MSE stays flat and low (~0.002-0.003) across all
  horizons, far **below** `cumsum(hazard MSE)` (which rises to ~0.3). The survival weighting
  S(t_{u-1}) and boundedness damp accumulation. Hazard MSE rises only at the last few *wide* quantile
  bins (large Delta -> hazard near saturation), and even that does not blow up CIF MSE.
- **Verdict:** propagation shapes the interval **width** (and the late-horizon coverage dip at the
  wide last bin), but it is **not** an error-accumulation problem — CIF error is controlled.

---

## Bottom line

In a converged, unbiased-center, clean-DGP setting the undercoverage decomposes cleanly:

- **eta = 0 (no teacher): under-dispersion (B).** Posterior is ~84% as wide as it should be; fix =
  variance correction / conformal widening (~1.2x). Small secondary early over-estimation (A).
- **eta = 1 (homogeneous teacher): teacher bias (A/transport).** The teacher over-tightens the
  estimator (EmpSD small) and shifts it; the posterior is already wide vs conditional sampling
  (ratio 1.44) yet still misses truth. A variance-only correction will NOT fix this; it needs the
  teacher's own uncertainty in the loop (retrained-teacher bootstrap) or a bias-aware calibration.
- **Cumulative propagation (C)** is real in the *width* but not the *error* — not the culprit here.

**Implication for the method:** at eta=0, calibration (conformal / sandwich-style widening) is the
right lever and should reach nominal. At eta>0, the teacher-induced bias means the honest coverage
target must account for teacher uncertainty (fixed-teacher intervals cannot cover it), which points
to a teacher-aware bootstrap benchmark and conformal-to-total-error, consistent with the earlier
Exp 1 finding.

## Notes / caveats
- Only 2 seeds; `SNR_B=0.5` (moderate signal). The DGP rate is tunable (`SNR_B`/`BASE_RATE`/
  `CENSOR_RATE`) — a rate sweep is the natural next check to see if any regime gives nominal raw
  coverage.
- Plot bug: eta=1 top-right clips the PostSD/EmpSD line (1.44) above the y-axis (ylim 1.1); the
  value is in the table.
