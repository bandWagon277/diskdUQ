# Guideline simulation — implementation status & results

Implements `references/guideline_implementation.md`. Code: `examples/guideline_sim.py` (self-contained;
reuses `diskd.metrics` for deviance / concordance / IBS). Slurm: `cluster/run_guideline_smoke.slurm`
(smoke), `cluster/run_guideline.slurm` (full, per experiment via EXP env).

## What the framework provides (Section 0 of the spec)
- **Common nonlinear DGP**: 12 covariates X~N(μ,Σ_ρ), Σ_ρ=ρ^|j−l|; nonlinear risk f_T(X) (quadratic,
  sin, interactions), logit λ_k = α_k + f(X), K=20. α_k Monte-Carlo-calibrated to ~37% cumulative
  event rate; independent censoring calibrated to ~30%. Heterogeneity injected by config only:
  μ-shift, ρ-shift, concept-shift γ (f_S=(1−γ)f_T+γf_flip), baseline-shift δ₀, teacher calibration
  distortion (a,b) on logit(p_T), teacher feature-restriction.
- **Student**: MLP (2×H ReLU + K-logit head), AdamW MAP with objective NLL + η·KD (at-risk Bernoulli
  forward-KL to the frozen teacher hazards), then **last-layer Laplace** posterior on the head
  (Hessian → Gaussian → draws) — the spec's chosen posterior approximation.
- **Teacher**: MLP trained on the source cohort (n_S), predictions **frozen**; optional post-hoc
  logit distortion logit(p*)=a+b·logit(p) and predictor-subset restriction (teacher quality).
- **η-selectors**: CV-Deviance (5-fold held-out deviance, min), CV-C (same fits, held-out C^td, max),
  LPML/CPO (full-data Laplace, generalized-posterior CPO, max), WAIC (min), fixed η, and Oracle
  (min test deviance — benchmark only).
- **Metrics** (test set vs closed-form truth): predictive deviance, C^td (concordance on CIF),
  IBS, risk RMSE, 95% coverage + width, calibration slope/intercept; **negative-transfer rate**
  (I{Dev(η̂) > Dev(0)}) and **oracle regret** (Dev(η̂) − min_η Dev).

## Experiment dispatch (env `EXP=`)
| EXP | Spec experiment | knobs | selectors | status |
|---|---|---|---|---|
| 1 | Well-specified control | teacher conditions, fixed η | oracle | wired |
| 2 | **Calibration bias** (C0–C3) | distort (a,b) ∈ {(0,1),(.5,1),(1,1),(0,1.5)} | CV-Dev, CV-C, LPML, WAIC | wired |
| 3 | Predictor mean shift | μ_S ∈ {0,.5,1} | CV-Dev, CV-C, LPML | wired |
| 6 | Baseline-risk shift | δ₀ ∈ {0,.5,1} | CV-Dev, CV-C, LPML | wired |
| 7 | **Concept shift** | γ ∈ {0,.5,1} | CV-Dev, CV-C, LPML | wired |
| 8 | Teacher quality | good/fair/poor feature sets | CV-Dev, CV-C, LPML | wired |
| 4,5,9,10 | corr-shift / predictor-space / architecture / combined | — | — | knobs present; dispatch TODO |
| 11–14 | prediction outcomes (landmark, event-count, time-to-mth) | — | — | TODO (Cox-framework posterior-predictive) |

Central design principle followed: architecture is **fixed across η** (no joint tuning), teacher
predictions frozen before student fitting, test set never used for selection, paired comparison
within replicate, common seeds.

## How this connects to our earlier work
- The last-layer Laplace posterior is exactly the fix validated earlier (SGLD under-dispersion →
  Laplace gives p_WAIC≈df_eff); LPML/WAIC here reuse that machinery.
- The negative-transfer story is the concrete generalization of our biased-teacher result
  (single-risk: biased teacher at η=1 collapsed coverage 0.95→0.11) — now measured by NT-rate +
  regret across CV-Dev / CV-C / LPML selectors.

## Results
*(smoke job 58346508 running; full per-experiment tables land here after the R=100 runs.)*

### Expected headline (from the spec + our biased-teacher finding)
- Under calibration/concept shift, **CV-C keeps η large** (ranking preserved) while **CV-Dev and
  LPML shrink η toward 0** → CV-C shows higher negative-transfer rate on deviance/IBS/calibration.
- **η̂ decreases monotonically with heterogeneity severity** (γ↑, δ₀↑, teacher-quality↓).
- η-selection ≈ internal (η=0) when the teacher is harmful → "selection as a negative-transfer safeguard."
