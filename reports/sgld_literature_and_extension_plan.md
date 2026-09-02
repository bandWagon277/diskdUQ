# SGLD Literature And Aim 1a Extension Plan

## Refined Strategic Takeaway

After reviewing the current proposal and the follow-up comments, the most important change is narrative:

```text
Aim 1a should not be framed as "SGLD Bayesian DiSKD."
It should be framed as loss-general, covariance-corrected, calibrated generalized Bayesian DiSKD.
```

Non-Bayesian UQ engines should be used as baselines, not as the main method. The main method should remain Bayesian/generalized-Bayesian: SGLD, last-layer Laplace, SWAG, or related approximate posterior engines. The key correction is that raw generalized posterior variance is not automatically equal to repeated-sampling variance when the teacher KL is a surrogate external likelihood.

The safer and stronger phrase is:

```text
covariance-corrected generalized-posterior predictive uncertainty
```

or, for the current empirical SGLD object:

```text
warm-start stochastic trajectory ensemble around a DiSKD MAP
```

This is more honest given high R-hat, low ESS, and chain-length drift away from the MAP, and it creates room for SWAG, last-layer Laplace, conformal calibration, bootstrap ensembles, and deep ensembles.

The non-Bayesian methods remain useful as comparisons:

```text
deep ensembles / bootstrap ensembles / MC dropout / conformal are baselines or safety layers,
not the primary Bayesian contribution.
```

## 1. What the current proposal already says

The current proposal frames Bayesian DiSKD as a generalized posterior

```text
Pi_{eta,omega}(d theta | D, teacher) proportional to
pi_0(theta) exp{-omega [-ell(theta) + eta Q_KL(theta)]}.
```

The implemented preliminary result uses warm-start SGLD initialized at the AdamW MAP, fixed small step size, pooled chains, and honest reporting as finite-time local intervals rather than converged posterior draws. The key empirical facts are:

- The headline cell is teacher N=5000, student N=500, two risks, 12 time intervals, full teacher quality.
- CR -> CR distillation improves discrimination, deviance, coverage, and width over Internal CR in the target cell.
- R-hat remains high, ESS is low, and longer SGLD runs drift away from the MAP instead of converging.
- Coverage improves over AdamW restart spread but remains below nominal 95%.

Important scale check: the current student is small, not a modern million-parameter net.

| Model | Approx. parameters |
| --- | ---: |
| Student time-MLP, hidden=32, 2 layers | 2,722 |
| Teacher time-MLP, hidden=128, 2 layers | 35,458 |
| Student transformer, hidden=32, 2 layers | 25,890 |

So the main explanation should not be simply "too many parameters." The more defensible explanation is posterior geometry: non-identifiability, multimodality, flat/per-sample tempering, weak local curvature information, and stochastic-gradient noise.

## 2. What related literature suggests

SGLD is the canonical stochastic-gradient Bayesian sampler: Welling and Teh add calibrated Gaussian noise to stochastic-gradient updates so optimization transitions into posterior sampling as the step size is annealed.

But modern deep-learning Bayesian inference usually does not rely on plain full-network SGLD as the only tool.

- **SGHMC** adds momentum and friction to address noisy minibatch gradients and improve exploration over first-order Langevin dynamics.
- **pSGLD** adds adaptive preconditioning because deep-network curvature makes default SGLD inefficient.
- **Cyclical SG-MCMC** explicitly targets high-dimensional, multimodal neural-network posteriors: large steps explore new modes, small steps characterize local modes, and the paper reports ImageNet-scale experiments.
- **SGLD pitfalls** papers warn that practical constant-step SGLD can have an invariant distribution far from the target posterior and behave more like SGD when stochastic-gradient variance dominates.
- **SWAG** fits a low-rank plus diagonal Gaussian to SGD iterates and compares favorably to MC dropout, KFAC Laplace, SGLD, and temperature scaling.
- **Deep ensembles** are not fully Bayesian, but they are simple, parallel, and strong for calibrated predictive uncertainty, including ImageNet-scale experiments.
- **Laplace / last-layer Laplace** approximations and **Bayes by Backprop / VI** are common alternatives when full parameter-space MCMC is too expensive or unstable.
- Survival-specific recent work is also moving toward VI, MC dropout, SNGP, and local-linearized Bayesian survival models rather than plain SGLD.

Conclusion: it is normal that plain full-network SGLD is fragile. The literature supports treating SGLD as one approximate Bayesian/trajectory method, but the proposal should compare it to SWAG, last-layer Laplace, deep/bootstrap ensembles, MC dropout/SNGP, and conformal recalibration.

## 3. What SGLD might really be doing here

Given the current diagnostics, I would describe the method as:

```text
warm-start stochastic trajectory ensemble around a good DiSKD MAP,
calibrated through omega/noise/conformal scaling,
not a converged full-network posterior sampler.
```

This is still useful for Aim 1a if framed honestly. The statistical object is closer to local predictive uncertainty around a teacher-regularized student than exact posterior inference over all weights.

Likely failure modes:

1. **Per-sample tempering is too flat.** Literal mean-loss SGLD corresponds to a temperature about N relative to summed-loss posterior. Longer chains drift because the stationary target is too diffuse.
2. **Weight-space non-identifiability.** Many weight settings produce similar CIFs; R-hat in weight or cohort-mean functionals can remain bad even when predictions are usable.
3. **Curvature mismatch.** Isotropic noise is poorly aligned with sensitive directions; pSGLD can help in principle, but naive adaptive preconditioning can also suppress needed exploration.
4. **Multimodality.** Warm-start chains remain near one region; cold starts create between-mode artifacts; neither gives clean posterior coverage without stronger calibration.
5. **Loss is generalized, not a true likelihood.** The KL distillation term is a decision loss. Its posterior spread needs a learning-rate/omega calibration, not a default Bayes interpretation.

## 4. Recommended simulation studies

### Study A: Is parameter count the cause?

This should be written as a diagnostic experiment, not as a performance-seeking experiment. The mechanism question is:

```text
Is full-network SGLD weak because the parameter space is too large,
or because the generalized posterior / loss scale / posterior geometry
is poorly calibrated?
```

Run a controlled dimension ablation:

- Student hidden_dim: 8, 16, 32, 64, 128.
- Sampler target: full-network SGLD vs last-layer-only SGLD vs frozen-backbone Laplace/SWAG.
- Same teacher, same train/test split, same closed-form true CIF.
- Metrics: Ctd, deviance, pointwise coverage, simultaneous coverage, width, R-hat/ESS on CIF functionals, distance from MAP.

Decision rule:

- If last-layer-only SGLD or last-layer Laplace clearly improves coverage, R-hat, and ESS while full-network SGLD fails, the main problem is full parameter-space geometry.
- If last-layer-only also fails, the main problem is more likely generalized-loss calibration, teacher mismatch, loss temperature, censoring uncertainty, or rare-event uncertainty.
- If very small hidden dimensions remain unstable, "too many parameters" is not the main explanation.

This lets the proposal say: we diagnosed full-network posterior geometry and then selected a modular UQ engine accordingly.

### Study B: Stronger uncertainty comparators

At the current headline cell only, compare:

```text
teacher N = 5000, student N = 500, J = 2, K = 12, full teacher quality
```

- AdamW MAP.
- AdamW restart spread.
- Nonparametric patient bootstrap ensemble.
- Deep ensemble with different seeds.
- MC dropout at test time.
- SWAG from late AdamW/SGD trajectory.
- Last-layer Laplace.
- Current warm-start SGLD.
- Cyclical SGLD or SGHMC only as sampler-improvement arms.

This directly addresses the proposal weakness that AdamW restart spread is a weak baseline.

### Study C: Calibration layer

Use a validation/calibration split to tune one scalar:

- generalized posterior omega,
- SGLD noise scale,
- conformal scale on CIF interval residuals,
- or split-conformal lower/upper CIF band calibration.

Primary endpoint: achieve near-nominal marginal/pointwise coverage with minimal width inflation. Secondary endpoint: preserve the width-error monotonicity signal.

This should be written as part of the method, not as a post-hoc patch. In generalized Bayes, omega is not automatically fixed by a true likelihood. It is the spread-calibration parameter. Eta controls teacher borrowing; omega or conformal calibration controls uncertainty spread.

Immediate omega experiment:

```text
Run omega sweep for both full-network SGLD and last-layer-only SGLD.
Tune omega using calibration-split coverage or risk calibration; report independent test coverage,
width, deviance, Ctd, R-hat, and ESS.
```

Interpretation:

- If lower omega improves coverage with acceptable width and point quality, omega should be formalized as a Bayesian spread-calibration parameter.
- If no omega achieves acceptable coverage-width tradeoff, covariance correction or function-space calibration is needed.
- If last-layer has a more stable omega curve than full-network, use last-layer generalized posterior as the first theoretical method.

### Study C2: Sandwich/Godambe covariance correction

The strongest Bayesian route is to correct the generalized posterior covariance itself. The teacher-student KL term is a surrogate external likelihood: it adds curvature to the objective but does not necessarily carry the same sampling-noise structure as real external raw data. Therefore raw posterior covariance can differ from the repeated-sampling covariance of the DiSKD estimator.

For a low-dimensional or last-layer DiSKD parameter vector `theta`, define:

```text
L_eta(theta) = NLL_internal(theta) + eta * KL_teacher(theta)
H = second derivative of L_eta(theta) at theta_hat
V^{-1} = internal-only likelihood information at theta_hat
Sigma_2 = H^{-1}                      # raw generalized posterior covariance
Sigma_1 = Sigma_2 V^{-1} Sigma_2      # corrected repeated-sampling covariance
```

This matches the correction used in the external-KL surrogate paper: posterior covariance sees the combined curvature, while repeated-sampling variability is driven by the internal-data information structure. More generally, a Godambe/sandwich version replaces `V^{-1}` by empirical score variability:

```text
corrected covariance = H^{-1} J H^{-1}.
```

Recommended first implementation:

- Train a DiSKD MAP student.
- Freeze the backbone representation.
- Restrict `theta` to the final prediction head.
- Compute `H` from internal NLL + teacher KL.
- Compute `V^{-1}` from internal-only NLL curvature.
- Compare raw last-layer generalized Laplace intervals with corrected intervals after delta-mapping samples to CIF curves.

This keeps the main method Bayesian and uses non-Bayesian UQ only as baselines.

### Study D: Regime map

Extend the existing N x teacher-quality grid:

- Student N: 100, 200, 500, 1000.
- Teacher quality: low, reduced, full, biased/full, shifted/full.
- Censoring: low, moderate, high.
- Distillation weight selected by validation, including eta=0.

Purpose: show DiSKD-UQ helps in identifiable transfer regimes and backs off when teacher quality is poor.

### Study E: Non-conjugate generalized-loss extension

Replace Bernoulli/multinomial KL with losses that do not have a clean pseudo-prior interpretation:

- IPCW Brier / integrated Brier loss for survival or CIF.
- CRPS-like loss over the CIF curve.
- Calibration loss over risk groups or horizons.
- Jensen-Shannon / alpha divergence to teacher probabilities.
- Weighted loss for clinically important horizons.

Then form the same generalized posterior:

```text
Pi(d theta) proportional to pi_0(theta) exp{-omega [NLL + eta D_teacher + lambda R_calibration]}.
```

This demonstrates that Aim 1a is not limited to conjugate or Bernoulli-KL cases.

The first implementation should probably be IPCW Brier because it is familiar in survival analysis, explicitly handles censoring, and is easier to explain to reviewers than a novel CIF-curve score. A clean example loss is:

```text
L(theta) = NLL + eta D_teacher + lambda R_IPCW_Brier
```

with generalized posterior:

```text
Pi(d theta) proportional to pi_0(theta) exp{-omega L(theta)}.
```

That demonstrates the framework does not rely on beta/Dirichlet conjugacy or pseudo-count interpretation.

## 5. Recommended Aim 1a structure expansion

Do not frame the expansion as "we add Bayesian SGLD to DiSKD." Frame it as:

```text
Generalized Bayesian DiSKD for uncertainty-aware transfer learning:
define a clinically meaningful empirical loss, combine internal likelihood,
teacher discrepancy, and optional calibration/structure penalties, then obtain
predictive uncertainty through calibrated approximate posterior or ensemble inference.
```

Concretely, Aim 1a can expand along four axes:

1. **Loss-general DiSKD.** Allow teacher matching losses beyond Bernoulli KL: multinomial KL, overall-risk BCE, horizon risk, IPCW Brier, CRPS, JS/alpha divergence, calibration losses.
2. **Bayesian inference modularity.** Support SGLD, cyclical SG-MCMC/SGHMC, SWAG, last-layer Laplace, and covariance-corrected last-layer generalized posteriors as primary Bayesian UQ engines.
3. **Covariance/spread calibration layer.** Add sandwich/Godambe correction, omega tuning, and, if needed, conformal safety calibration because generalized posteriors need spread calibration under misspecification.
4. **Function-space reporting.** Evaluate uncertainty on CIF and survival curves, not weight posterior diagnostics alone. R-hat/ESS remain diagnostics, but coverage, width, deviance, Ctd, and width-error association are the clinical endpoints.

Deep ensembles, bootstrap ensembles, and MC dropout should be reported as strong empirical baselines, not as the central Bayesian method.

## 6. Immediate next steps

1. Run Study A first as a diagnostic experiment. This answers whether full-network parameter geometry is the main cause of weak SGLD behavior.
2. Run Study B only at the headline cell. This gives a credible comparator table against stronger UQ baselines, especially bootstrap ensemble and deep ensemble.
3. Run an omega sweep for full-network and last-layer SGLD. This directly answers whether omega should be treated as a generalized-posterior calibration parameter.
4. Run Study C2: last-layer sandwich/Godambe covariance correction. This is the most Bayesian way to address the generalized-posterior variance mismatch.
5. Add Study C's deployable calibration split experiment as a safety layer after the covariance-correction result.
6. Implement one non-conjugate loss example from Study E. Start with IPCW Brier unless there is a strong reason to use CRPS-like CIF loss first.
7. Update the proposal language. Replace "SGLD posterior credible interval" with "covariance-corrected generalized-posterior predictive interval" unless sampler diagnostics improve materially.

Suggested proposal language:

```text
We construct generalized Bayesian DiSKD posteriors from internal survival
likelihoods and teacher discrepancy losses. Because the teacher KL is a
surrogate external likelihood, raw generalized posterior covariance need not
match repeated-sampling variability. We therefore develop covariance-corrected
generalized Bayesian predictive intervals, beginning with a low-dimensional
last-layer Godambe/sandwich correction, and compare against non-Bayesian UQ
baselines such as deep ensembles and bootstrap ensembles.
```

One-sentence summary:

```text
Upgrade Aim 1a from "SGLD Bayesian DiSKD" to a loss-general,
covariance-corrected generalized Bayesian DiSKD framework.
```

## Sources

- Welling and Teh, "Bayesian Learning via Stochastic Gradient Langevin Dynamics": https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
- Chen, Fox, and Guestrin, "Stochastic Gradient Hamiltonian Monte Carlo": https://arxiv.org/abs/1402.4102
- Li et al., "Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks": https://arxiv.org/abs/1512.07666
- Zhang et al., "Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning": https://arxiv.org/abs/1902.03932
- Brosse, Durmus, and Moulines, "The promises and pitfalls of Stochastic Gradient Langevin Dynamics": https://arxiv.org/abs/1811.10072
- Maddox et al., "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (SWAG): https://arxiv.org/abs/1902.02476
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles": https://arxiv.org/abs/1612.01474
- Bissiri, Holmes, and Walker, "A General Framework for Updating Belief Distributions": https://arxiv.org/abs/1306.6430
- Syring and Martin, "Calibrating general posterior credible regions": https://arxiv.org/abs/1509.00922
- Lillelund, Magris, and Pedersen, "Efficient Training of Probabilistic Neural Networks for Survival Analysis": https://arxiv.org/abs/2404.06421
- Monod, Micheli, and Bhatt, "NeuralSurv: Deep Survival Analysis with Bayesian Uncertainty Quantification": https://arxiv.org/abs/2505.11054
- Candes, Lei, and Ren, "Conformalized Survival Analysis": https://arxiv.org/abs/2103.09763
