# Plan: Solve Undercoverage + Variance Correction on NN Parameters

**Goal.** Address two coupled issues: (i) the generalized-posterior CIF intervals **undercover**
(0.36–0.66 vs 0.95), and (ii) we cannot do a stable **variance correction on the full network** —
the last-layer Godambe was exact but insufficient (it is the wrong sign and misses representation
uncertainty; `responses/sandwich_correction_findings.md`).

This plan operationalizes Phases 2–4 of `Bayes_survival.md`. It deliberately does **not** try
full-network Godambe (intractable/unstable); instead it (a) validates the sampler, (b) decomposes
undercoverage into *variance vs bias* using two coverage targets, and (c) scales variance
correction up **one block at a time** (last-layer → last-two-layers → adapter) to find the largest
tractable block that helps.

## Why these three, given what we already know

- Study A: undercoverage is not parameter count. Study B/C: every engine undercovers raw and needs
  ~5× width inflation. Sandwich probe: at η=0 (no teacher) MAE/SE = 1.60 already — so a big part of
  the deficit is **frozen-backbone + misspecification**, not the teacher.
- The open question those leave: **is the raw undercoverage a sampler artifact, a
  representation-uncertainty gap, or irreducible bias?** The three experiments below answer exactly
  that, in order.

---

## Experiment 0 — SGLD sampler validation in a low-dim model (Phase 2)

**Question.** Does our SGLD actually sample the intended generalized posterior, or is it
underdispersed? If it fails here, no full-net SGLD result is interpretable.

**Design.** A toy **single-risk discrete logistic-hazard** model, `lambda_ik = sigmoid(x_ik^T theta)`,
parameter dim `p ≤ 20`, with a synthetic teacher hazard for the KL term. Gold standards are exact
because `p` is small.

- Samplers compared: (1) **exact Laplace** `N(theta_MAP, H^-1)`; (2) **MALA** (Metropolis-adjusted
  Langevin, accept/reject — an exact-in-the-limit gold standard, no external HMC dependency);
  (3) **our SGLD** with the production config.
- Grid: `eta ∈ {0, 1}`, `omega ∈ {0.5, 1, 2}`.
- Diagnostics: posterior mean / sd / 2.5-97.5 quantile agreement (SGLD vs MALA vs Laplace);
  per-parameter coverage of the true theta; **duplicate-data test** (copy the data ×2 → interval
  widths should shrink ≈ 1/sqrt(2)); Langevin-noise unit test on a Gaussian target (`Cov ≈ H^-1`).

**Decision rule.**
- SGLD ≈ MALA ≈ Laplace ⇒ the sampler is fine; undercoverage lives in the *target / bias*, not SGLD.
- SGLD narrower than MALA ⇒ SGLD is underdispersed (a sampler bug/temperature issue) — fix before
  anything else.

**Deliverable.** `responses_exp0/` table + the two unit tests passing.

---

## Experiment 1 — Subnetwork variance correction, one block at a time (Phase 4.2)

**Question.** We can't correct the whole network; does correcting a *larger tractable block* than
the last layer recover the missing representation uncertainty and improve true-CIF coverage?

**Design.** Reuse the exact machinery of `examples/sandwich_correction_probe.py` (forward-mode
scores, LBFGS stationarity polish, truncated PSD pseudo-inverse, delta-method CIF), but generalize
the "corrected block" `theta_b` to three nested levels, backbone-above frozen:

1. **L1 = last layer** (`head`, 66 params) — the current probe.
2. **L2 = last two layers** (last block Linear + head, ~1.1k params for hidden=32).
3. **L3 = adapter block** (both residual blocks + head, the largest still-exact Hessian, a few k
   params).

For each block, at `eta ∈ {0, 1}`, compute and compare four interval constructions on the test CIF:
- Laplace `H_b^-1`;
- Godambe sandwich `H_b^-1 J_b H_b^-1`;
- block-SGLD (sample `theta_b`, freeze the rest) — ties Exp 0's validated sampler to the real model;
- omega-trace-matched.

**Metrics (per block × method):** width, **coverage vs pseudo-target** and **coverage vs true CIF**
(see Exp 2), MAE, MAE/SE, R-hat/ESS (for SGLD), numerical rank of `H_b`, stationarity `|grad|`.

**Decision rule.** If coverage-vs-truth and MAE/SE improve monotonically L1 → L2 → L3, representation
uncertainty is a real, recoverable component and the subnetwork correction is the method. If they
plateau at L1, the residual is irreducible bias → conformal is the only fix (Phase 7).

**Deliverable.** `responses_subnet/` block-vs-method table.

---

## Experiment 2 — Two coverage targets: variance vs bias (Phase 3)

**Question.** When an interval misses the true CIF, is it failing at its own job (covering the
estimator's sampling target `phi_eta`) or is the *target itself* biased (`phi_eta ≠ phi_0`)?

**Design (folded into Exp 1's evaluation, no extra job).**
- Estimate the **pseudo-target** `phi_eta = F_j(t|z; theta_eta)` by averaging the point-estimate CIF
  over **R independent training replicates** (fresh cohorts, same DGP, same eta) — this is the
  Monte-Carlo estimate of the estimator's expectation.
- For every method/block report both `P(phi_eta in CI)` (proper frequentist job) and
  `P(phi_0 in CI)` (honesty vs truth), plus the decomposition
  `hat phi - phi_0 = (hat phi - phi_eta) [estimation variability] + (phi_eta - phi_0) [bias]`.

**Interpretation.** If methods cover `phi_eta` well but not `phi_0`, undercoverage is **bias**
(teacher/transport/approximation) → variance correction cannot fix it, conformal must. If they miss
even `phi_eta`, it is genuine **variance** underestimation → the subnetwork correction / sampler fix
is the lever.

**Deliverable.** the `phi_eta` vs `phi_0` coverage columns in the Exp 1 table + a bias-vs-variance
bar per block.

---

## What gets submitted now (sbatch)

1. **Experiment 0** — sampler validation (CPU/GPU, minutes; MALA on p≤20 is cheap).
2. **Experiment 1 + 2** — subnetwork correction with two-target coverage (GPU; the R-replicate
   pseudo-target loop is the main cost, ~R× a student fit).

Phases 5 (teacher-aware bootstrap) and 7 (conformal safety layer) are the *coverage* deliverables
and come next, but only after Exp 0–2 establish whether the gap is variance or bias — so we don't
prematurely commit to conformal-as-method vs correction-as-method.

## Explicitly deferred (per plan, not now)

- Full-network Godambe (intractable — the whole point of the subnetwork ladder).
- Confidence-weighted teacher (Phase 6, Version B) — orthogonal structural extension.
- Prior-gradient / (1+eta) / batching audit (Phase 1) — a code-correctness checklist to run
  alongside, but not a GPU experiment; Exp 0's duplicate-data and Gaussian unit tests already
  exercise the same scaling.

---

## REVISION after Codex review (2026-07-20)

Codex review (`codex_consultations/2026-07-20_variance_correction_plan_review.md`) accepted the
diagnostic direction but pushed back that the plan is still too variance-correction-heavy given the
evidence that undercoverage is largely bias. **Revised priorities and fixes:**

1. **Code audit (Phase 1): confirmed no scaling bug.** Prior gradient is not tempered by
   loss_scale (`samplers.py:261-263`), `(1+eta)` is restored (`models.py:457`), batching is
   subject-level (`models.py:237`). So undercoverage is not a scaling artifact. *Caveat:* `literal`
   drift targets a **tempered per-sample** objective, not the full-data posterior — so Exp 0 must
   validate **both `literal` and `welling_teh`** and report which matches the exact posterior.

2. **Reorder.** The two sbatch jobs now are **Exp 0 (sampler validation)** and **Exp 1 rebuilt as
   the variance-vs-bias decomposition *with split-conformal*** — the coverage deliverable comes
   *before* any larger sandwich block. The subnetwork ladder is demoted to a **gated L2 pilot**
   (run L2 only; commit to L3 only if L2 materially improves coverage-vs-pseudo-target).

3. **Exp 0 fixes.** Use a **small competing-risk** toy (softmax + no-event + interval KL — the
   production path), not only single-risk; MALA is the gold standard (Laplace is a curvature check
   only); interpret the duplicate-data 1/sqrt(2) test **per drift mode** (welling_teh should shrink;
   `literal`, a mean objective, should not); the Gaussian unit test compares to the **discretized ULA
   stationary covariance** (≈ H^-1 only as eps→0), not exactly H^-1.

4. **Exp 1/2 fixes.** Replicate-averaging estimates **E[F(theta_hat)]** (the repeated-training mean),
   *not* `F(theta_eta)`; I also fit a **large-n model** as the pseudo-true `theta_eta` proxy, and
   report coverage vs both plus the true CIF. **Teacher is held fixed** across replicates (this
   targets conditional student-data variance — the clean pseudo-target); a retrained-teacher variant
   is a separate, later target. This loop (R student fits) is the main cost, not "free."

5. **Move coverage deliverables forward.** Add **split-conformal** (deployable, censoring-aware
   score) inside Exp 1 now; **teacher-aware bootstrap** (Phase 5) is the next benchmark after Exp 0–1.
   Report **pointwise vs simultaneous** coverage separately throughout.

### Revised experiment set (what actually gets built + submitted)

- **Exp 0 — `sampler_validation_probe.py`** (small toy, cheap): competing-risk + single-risk logistic
  hazard, p≤~20; MALA (gold) vs exact Laplace vs SGLD in **both** drift modes; eta∈{0,1},
  omega∈{0.5,1,2}; duplicate-data (per-mode) + Gaussian-ULA unit tests. Answers: is SGLD
  under-dispersed, and which drift mode hits the intended target.
- **Exp 1 — `coverage_decomposition_probe.py`** (headline cell): R fixed-teacher student replicates →
  repeated-training mean + large-n pseudo-true. For last-layer {Laplace, Godambe, SGLD}, deep
  ensemble, bootstrap: **coverage vs pseudo-target vs true CIF** (pointwise + simultaneous), width,
  MAE, MAE/SE, and **split-conformal** on top. Plus a **gated L2 pilot** (rank/condition/null-space/
  stationarity/linearization checks). Answers: is undercoverage variance or bias, and does conformal
  fix it deployably.
