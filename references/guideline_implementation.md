Below is a **one-by-one implementation specification** that can be handed directly to whoever runs the simulations. I keep one common DGP and only change the heterogeneity component being studied, so results remain interpretable.

The design preserves the existing manuscript structure: fixed teacher predictions, target-only evaluation, subject-level 5-fold CV for frequentist η selection, LPML/CPO for Bayesian η selection, and last-layer Laplace for posterior-based criteria.  

---

# 0. Common implementation for all experiments

### Data

| Item                         | Setting                      |
| ---------------------------- | ---------------------------- |
| Source/teacher training size | (n_S=10,000)                 |
| Target/student training size | (n_T=500)                    |
| Independent test size        | (n_{\text{test}}=5,000)      |
| Number of predictors         | (p=12)                       |
| Time intervals               | (K=20)                       |
| Replicates                   | (R=100)                      |
| Main censoring rate          | approximately 30%            |
| Main event rate by (K)       | approximately 35–40%         |
| η grid                       | ({0,0.1,0.25,0.5,1,2,5})     |
| CV                           | subject-level 5-fold         |
| Temperature                  | fixed across all simulations |
| Test set                     | never used for tuning        |

The sample-size structure follows the original framework, which used (10,000/500/5,000) teacher/student/test samples. 

### Target predictor distribution

[
X=(X_1,\ldots,X_{12})^\top
\sim N(0,\Sigma_T),
]

with

[
(\Sigma_T)_{jl}=0.3^{|j-l|}.
]

### Target risk function

Use

[
\begin{aligned}
f_T(X)=&
0.50X_1-0.40X_2
+0.35(X_3^2-1)
+0.30\sin(X_4)\
&+0.30X_5X_6
-0.30X_7
+0.25X_8
+0.25X_9X_{10}
+0.20X_{11}.
\end{aligned}
]

(X_{12}) is noise.

Generate discrete hazards from

[
\operatorname{logit}{\lambda_k(X)}
==================================

\alpha_k+f_T(X),
\qquad k=1,\ldots,20.
]

Choose (\alpha_k) once through a large Monte Carlo calibration so that the cumulative event rate is about 35–40%.

Generate independent censoring and calibrate its parameter to approximately 30%.

### Models

Use the same student architecture, initialization strategy, optimizer, and training rule across candidate η values within a replicate.

Recommended fixed training setup:

| Component               | Setting                        |
| ----------------------- | ------------------------------ |
| Optimizer               | Adam                           |
| Learning rate           | (10^{-3})                      |
| Batch size              | 64                             |
| Maximum epochs          | 128                            |
| Early stopping          | patience 5                     |
| Teacher predictions     | computed once and frozen       |
| Student architecture    | fixed within experiments       |
| Posterior approximation | last-layer Laplace             |
| Random seeds            | common across compared methods |

Do **not** jointly tune architecture and η in these simulations.

---

# Experiment 1. Well-specified control

## Purpose

Establish that Bayesian inference and teacher borrowing work when there is no model misspecification or heterogeneity.

This keeps the current manuscript control experiment rather than introducing another DGP. The manuscript already shows that oracle teacher borrowing narrows intervals while maintaining coverage in this setting. 

## DGP

Use the existing correctly specified discrete-time logistic model.

Generate

[
\operatorname{logit}\lambda_k(X)
================================

\alpha_k+X^\top\beta,
]

using exactly the parameter settings currently used in the control simulation.

## Teacher conditions

1. No teacher: (\eta=0).
2. Oracle teacher: true hazards.
3. Estimated teacher, (n_S=1,000).
4. Estimated teacher, (n_S=5,000).
5. Estimated teacher, (n_S=10,000).

## η

Use

[
\eta\in{0,0.5,1}.
]

No η-selection experiment is required here.

## Metrics

Report:

* CIF/risk RMSE;
* predictive deviance;
* 95% coverage;
* mean interval width;
* empirical SD;
* posterior SD;
* posterior SD / empirical SD.

## Expected use

Sanity check only. Keep results concise.

---

# Experiment 2. η-selection under teacher calibration bias

This should be the **primary η-selection experiment**.

## Purpose

Compare:

[
\text{CV-C-index},\quad
\text{CV-Deviance},\quad
\text{LPML/CPO}
]

and determine which selector protects against negative transfer.

## Data

Generate source and target data from the same DGP.

Train a good teacher on (n_S=10,000).

Let the teacher predicted hazard probability be (p_{ik}^{T}).

Create distorted predictions:

[
\operatorname{logit}(p_{ik}^{T,*})
==================================

a+b,\operatorname{logit}(p_{ik}^{T}).
]

## Scenarios

| Scenario | (a) | (b) | Interpretation             |
| -------- | --: | --: | -------------------------- |
| C0       |   0 |   1 | calibrated                 |
| C1       | 0.5 |   1 | moderate calibration shift |
| C2       | 1.0 |   1 | severe calibration shift   |
| C3       |   0 | 1.5 | overconfident teacher      |

The (a)-shift is particularly important because ranking is largely preserved while absolute risk is wrong.

## η candidates

[
\eta\in{0,0.1,0.25,0.5,1,2,5}.
]

## Selector 1: CV-Deviance

For each η:

1. Split target training data into five folds.
2. Fit on four folds using local likelihood + teacher KL.
3. Do not use held-out likelihood or held-out KL during fitting.
4. Calculate predictive deviance on the held-out fold.
5. Sum/average across folds.
6. Select η with minimum deviance.

This matches the manuscript's current frequentist selection procedure. 

## Selector 2: CV-C

Use exactly the same five fits.

On each held-out fold calculate time-dependent C-index instead of deviance.

Select

[
\widehat\eta_C
==============

\arg\max_\eta C^{td}_{CV}(\eta).
]

## Selector 3: LPML/CPO

For every η, fit the Bayesian model using the entire target training sample.

Calculate CPO with both the subject likelihood and corresponding teacher KL contribution removed under leave-one-out evaluation, as defined in the manuscript. 

Select

[
\widehat\eta_{\rm LPML}
=======================

\arg\max_\eta LPML(\eta).
]

## Additional comparators

Include:

* Internal: (\eta=0);
* Fixed (\eta=0.5);
* Fixed (\eta=1);
* Oracle η.

Oracle η is

[
\eta_{\rm oracle}
=================

\arg\min_\eta
Dev_{\rm test}(\eta).
]

It is only a simulation benchmark.

## Main metrics

Report for each selector:

* mean/median selected η;
* (P(\widehat\eta=0));
* test predictive deviance;
* (C^{td});
* IBS;
* calibration intercept;
* calibration slope;
* risk RMSE;
* negative-transfer rate;
* oracle regret.

Define regret as

[
Regret=
Dev_{\rm test}(\widehat\eta)
----------------------------

\min_{\eta}Dev_{\rm test}(\eta).
]

This experiment should establish why C-index alone is insufficient for η selection.

---

# Experiment 3. Marginal predictor-distribution shift

## Purpose

Test covariate shift without changing (Y\mid X).

The previous framework already examined source-target covariate shift.  This experiment makes the severity parameter cleaner.

## Target

[
X_T\sim N(0,\Sigma_T).
]

## Source

[
X_S\sim N(\mu_S,\Sigma_T),
]

where

[
\mu_S
=====

(\delta,\ldots,\delta,0,\ldots,0)^\top,
]

with the first six coordinates shifted.

## Levels

[
\delta\in{0,0.5,1.0}.
]

Thus:

* H0: no shift;
* H1: moderate shift;
* H2: severe shift.

## Important constraint

Use the **same conditional outcome model**

[
P_S(Y\mid X)=P_T(Y\mid X).
]

Only (P(X)) changes.

## Methods

Compare:

* Internal;
* fixed η=1;
* CV-C;
* CV-Deviance;
* LPML.

## Main outputs

For each (\delta), report:

[
\widehat\eta
]

distribution and:

* predictive deviance;
* (C^{td});
* IBS;
* calibration;
* negative-transfer rate.

## Key figure

Plot

[
\delta
\quad\text{vs}\quad
\widehat\eta.
]

---

# Experiment 4. Covariance/correlation shift

## Purpose

Test heterogeneity in the joint predictor structure without mean shift.

## Target

[
X_T\sim N(0,\Sigma_{0.3}),
]

where

[
(\Sigma_\rho)_{jl}=\rho^{|j-l|}.
]

## Source

[
X_S\sim N(0,\Sigma_{\rho_S}).
]

## Levels

[
\rho_S\in{0.3,0.6,0.9}.
]

The conditional outcome mechanism remains identical in source and target.

## Methods and metrics

Exactly the same as Experiment 3.

Do not introduce any additional tuning parameters.

## Main question

Does η automatically decrease when the teacher is trained under increasingly different predictor dependence structures?

---

# Experiment 5. Predictor-space heterogeneity

This should be another major experiment.

## Purpose

Demonstrate the advantage of **prediction-level transfer when teacher and student do not share the same predictor space**.

## Underlying variables

Generate all (X_1,\ldots,X_{12}) for every subject.

Different models are simply given different subsets.

## Scenarios

| Scenario | Student predictors | Teacher predictors |
| -------- | ------------------ | ------------------ |
| P0       | (X_{1:12})         | (X_{1:12})         |
| P1       | (X_{1:12})         | (X_{1:6})          |
| P2       | (X_{1:6})          | (X_{1:12})         |
| P3       | (X_{1:8})          | (X_{1:4},X_{9:12}) |
| P4       | (X_{1:6})          | (X_{7:12})         |

Teacher predictions are generated for the same target subjects, but the student never receives the teacher's raw predictors or parameters.

## Important distinction

P1 asks whether a weak teacher can still help.

P2 asks whether a rich teacher can transfer information into a restricted student.

P3 asks whether prediction-level transfer works with partial overlap.

P4 is the strongest demonstration because teacher and student predictor sets do not overlap.

## Methods

Use:

* Internal;
* fixed η=1;
* CV-Deviance;
* LPML;
* optionally CV-C.

## Metrics

Primary:

* predictive deviance;
* (C^{td});
* IBS;
* risk RMSE;
* selected η;
* negative-transfer rate.

Secondary:

* calibration;
* coverage.

---

# Experiment 6. Baseline-risk shift

## Purpose

Create outcome heterogeneity where relative ranking may remain good but absolute risk changes.

## Target

[
\operatorname{logit}\lambda_{Tk}(X)
===================================

\alpha_k+f_T(X).
]

## Source

[
\operatorname{logit}\lambda_{Sk}(X)
===================================

\alpha_k+\delta_0+f_T(X).
]

## Levels

[
\delta_0\in{0,0.5,1.0}.
]

Everything else remains unchanged.

## Why important

The teacher may maintain strong discrimination while becoming poorly calibrated in the target population.

This provides a natural negative-transfer setting beyond the artificial post-hoc calibration experiment.

## Metrics

Emphasize:

* (C^{td});
* predictive deviance;
* IBS;
* calibration intercept/slope;
* selected η;
* negative-transfer rate.

The comparison between C-index and proper scoring rules is particularly important here.

---

# Experiment 7. Conditional outcome/concept shift

## Purpose

Create true (P(Y\mid X)) heterogeneity.

## Target risk

Use the common (f_T(X)).

## Severe source function

Define

[
\begin{aligned}
f_{\rm flip}(X)=&
-0.50X_1
+0.40X_2
+0.35(X_3^2-1)
+0.30\sin(X_4)\
&-0.30X_5X_6
-0.30X_7
+0.25X_8
-0.25X_9X_{10}
+0.20X_{11}.
\end{aligned}
]

Then let

[
f_S(X)
======

(1-\gamma)f_T(X)
+
\gamma f_{\rm flip}(X).
]

## Levels

[
\gamma\in{0,0.5,1}.
]

Interpretation:

* (\gamma=0): aligned;
* (\gamma=0.5): moderate concept shift;
* (\gamma=1): severe concept shift.

## Expected η behavior

The primary quantity is

[
\gamma\uparrow
\quad\Rightarrow\quad
\widehat\eta\downarrow.
]

## Methods

Compare:

* Internal;
* fixed η=.5;
* fixed η=1;
* CV-C;
* CV-Deviance;
* LPML;
* Oracle η.

## Main metrics

* selected η;
* (P(\widehat\eta=0));
* predictive deviance;
* IBS;
* risk RMSE;
* (C^{td});
* negative-transfer rate;
* oracle regret.

This is the strongest single-factor experiment for the “η prevents negative transfer” claim.

---

# Experiment 8. Teacher quality

## Purpose

Retain the teacher-quality experiment from the original DiSKD framework, but incorporate adaptive η selection.

The previous framework already defines Good/Fair/Poor teacher quality through predictor restriction. 

## Teacher configurations

### Good

[
X_{1:12}.
]

### Fair

[
{X_1,X_2,X_3,X_5,X_6,X_7,X_9,X_{10},X_{11}}.
]

### Poor

[
{X_1,X_2,X_5,X_6,X_9,X_{10}}.
]

Keep the student predictor space fixed at all 12 variables.

## Change from previous analysis

Do not only plot performance as a function of η.

Instead compare:

* fixed η=1;
* CV-C-selected η;
* CV-Deviance-selected η;
* LPML-selected η;
* internal.

## Outputs

Show:

1. selected η versus teacher quality;
2. test deviance;
3. C-index;
4. IBS;
5. negative-transfer rate.

Expected:

[
\text{teacher quality}\downarrow
\Rightarrow
\widehat\eta\downarrow.
]

---

# Experiment 9. Architecture mismatch

## Purpose

Show that prediction-space distillation does not require parameter-level architecture compatibility.

## DGP

Use the common nonlinear target DGP.

## Student

Use one fixed student architecture throughout, preferably the same Transformer/Time-MLP used in the main paper.

## Teacher architectures

Use three teachers:

1. discrete-time logistic regression;
2. 2-layer MLP;
3. Transformer.

All teachers use the same source training sample and predictor set.

## Critical implementation rule

Do **not** force their predictive accuracy to be identical.

Instead report teacher-only test performance first.

This separates:

[
\text{architecture mismatch}
]

from

[
\text{teacher quality}.
]

## Methods

For each teacher architecture:

* Internal;
* fixed η=1;
* CV-Deviance;
* LPML.

## Metrics

* teacher deviance;
* teacher (C^{td});
* student deviance;
* student (C^{td});
* IBS;
* selected η;
* negative-transfer rate.

Keep this experiment relatively small.

---

# Experiment 10. Combined heterogeneity stress test

## Purpose

Demonstrate whether η-selection remains protective when multiple mismatches occur simultaneously.

Use only three scenarios.

### A. Aligned

Common reference setting.

### B. Moderate heterogeneity

Use simultaneously:

[
\mu_{S,1:6}=0.5,
]

[
\rho_S=0.6,
]

[
\gamma=0.5,
]

plus 50% teacher/student predictor overlap.

### C. Severe heterogeneity

Use:

[
\mu_{S,1:6}=1,
]

[
\rho_S=0.9,
]

[
\gamma=1,
]

50% predictor overlap, plus teacher calibration distortion

[
\operatorname{logit}(p^*)
=========================

0.5+1.5\operatorname{logit}(p).
]

## Methods

Only compare:

* Internal;
* fixed η=1;
* CV-C;
* CV-Deviance;
* LPML;
* Oracle η.

## Primary metrics

Report:

| Metric                      | Purpose                   |
| --------------------------- | ------------------------- |
| selected η                  | borrowing adaptation      |
| predictive deviance         | main predictive criterion |
| oracle regret               | selector quality          |
| negative-transfer rate      | safety                    |
| (C^{td})                    | discrimination            |
| IBS                         | overall prediction        |
| calibration slope/intercept | absolute-risk accuracy    |

This should be the final heterogeneity experiment in the main manuscript.

---

# Experiment 11. Multiple prediction outcomes

Do **not** rerun every heterogeneity setting.

Use only:

1. aligned;
2. moderate combined heterogeneity;
3. severe combined heterogeneity.

This keeps computation manageable.

The Cox framework already provides posterior predictive event-count and event-time constructions that can be adapted directly. 

## 11A. Individual survival/event risk

For each test subject estimate

[
F(t_k\mid X_i)
]

at

[
k\in{5,10,15,20}.
]

### Metrics

* RMSE;
* MAE;
* Brier score;
* log score;
* calibration;
* 95% coverage;
* interval width.

---

# Experiment 12. Landmark event prediction

Set landmark

[
k_0=10.
]

Restrict evaluation to test subjects who remain at risk at (k_0).

Predict

[
P(T\le k_0+h\mid T>k_0,X)
]

for

[
h\in{3,5,10}.
]

## Metrics

* Brier score;
* log loss;
* AUC/time-dependent concordance;
* calibration intercept;
* calibration slope;
* probability RMSE;
* posterior interval coverage.

This corresponds conceptually to the conditional future-event prediction already defined in the Cox framework. 

---

# Experiment 13. Future cohort event-count prediction

At landmark (k_0=10), randomly select

[
M=500
]

test subjects still at risk.

For each posterior draw:

1. simulate each subject's future event/censoring path;
2. count events occurring between (k_0) and (K=20);
3. obtain one posterior predictive count (N^{(m)}).

Across posterior draws obtain

[
p(N_{\text{future}}\mid D).
]

The Cox framework uses exactly this posterior-predictive logic for future event counts. 

## Metrics across simulation replicates

* bias of posterior predictive mean;
* MAE;
* RMSE;
* 95% prediction interval coverage;
* prediction interval width.

Compare:

* Internal;
* CV-Deviance-selected DiSKD;
* LPML-selected DiSKD.

CV-C can be omitted here unless computational cost is small.

---

# Experiment 14. Time to the (m)-th future event

Use the same landmark cohort as Experiment 13.

Set

[
m=25.
]

For every posterior draw:

1. simulate future event/censoring trajectories;
2. sort observed future event times;
3. take the 25th event time;
4. if fewer than 25 events occur, record (W(25)=\infty).

This directly follows the posterior-predictive algorithm in the Cox framework. 

Use the posterior median of (W(25)) as the point prediction.

## Metrics

* MAE;
* RMSE;
* median bias;
* 95% prediction interval coverage;
* interval width;
* probability that (W(25)=\infty), if non-negligible.

---

# Negative-transfer definition used throughout

Use **predictive deviance as the primary definition**.

For replicate (r),

[
NT_r
====

I\left{
Dev_r(\widehat\eta)

>

Dev_r(\eta=0)
\right}.
]

Report

[
NT\ Rate
========

\frac1R\sum_{r=1}^{R}NT_r.
]

Also report the paired difference

[
\Delta Dev_r
============

Dev_r(\widehat\eta)-Dev_r(0).
]

Negative values are beneficial transfer.

As secondary definitions:

[
NT_{\rm IBS}
============

I{IBS(\widehat\eta)>IBS(0)},
]

and

[
NT_C
====

I{C^{td}(\widehat\eta)<C^{td}(0)}.
]

---

# Final experiment-to-metric matrix

| Exp. | Main varying factor         | η selection        | Primary metrics              |
| ---- | --------------------------- | ------------------ | ---------------------------- |
| 1    | none, well specified        | no                 | RMSE, coverage, width        |
| 2    | calibration bias            | CV-C, CV-Dev, LPML | η, deviance, NT, calibration |
| 3    | predictor mean shift        | CV-C, CV-Dev, LPML | η, deviance, NT              |
| 4    | predictor correlation shift | CV-Dev, LPML       | η, deviance, IBS             |
| 5    | predictor-space mismatch    | CV-Dev, LPML       | deviance, C-index, NT        |
| 6    | baseline-risk shift         | CV-C, CV-Dev, LPML | deviance, calibration, NT    |
| 7    | concept shift               | CV-C, CV-Dev, LPML | **η, regret, NT**            |
| 8    | teacher quality             | CV-Dev, LPML       | η, deviance, NT              |
| 9    | architecture mismatch       | CV-Dev, LPML       | deviance, η                  |
| 10   | combined heterogeneity      | all three          | **η, regret, NT, IBS**       |
| 11   | individual risk             | selected η         | RMSE, Brier, coverage        |
| 12   | landmark risk               | selected η         | Brier, log loss, calibration |
| 13   | event count                 | selected η         | RMSE, PI coverage            |
| 14   | time to 25th event          | selected η         | MAE, PI coverage             |

The **minimum main-text set** I would prioritize is Experiments **2, 3, 5, 7, 10, and 11–14**, while 1, 4, 8, and 9 can partly move to the appendix. This keeps the main narrative sharply centered on **heterogeneity → adaptive η → negative-transfer protection → broader predictive outcomes**.
