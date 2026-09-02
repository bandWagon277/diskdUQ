我把三个文档对齐以后，建议把现在的 simulation 重新收束成一条更清楚的主线：

> **核心问题不再是“Bayesian DiSKD 能不能做 competing-risk prediction”，而是：当 teacher 和 target 在不同层面 heterogeneous 时，η 能否自动控制 borrowing、避免 negative transfer，同时改善多种 prediction outcomes。**

这个方向正好对应 manuscript 里老师留下的四条批注：增加 C-index/internal prediction selection、和 frequentist 5-fold CV 的 η selection 比较、给“η 防 negative transfer”理论支撑、以及增加 predictor-space 等 heterogeneity。 而 revised manuscript 本身已经把 fixed-η 的 5-fold predictive-deviance CV 和 Bayesian LPML/CPO 建好了，所以不需要再另造一套 selection framework。

我建议**主 simulation 暂时不把 competing-risk-specific mismatch 当主轴**，保留现有 single-risk well-specified control，再加一个 nonlinear single-risk main DGP。这样 predictor shift、concept shift、feature-space mismatch、teacher miscalibration、architecture mismatch 都能被单独解释，不会和 cause-specific structure 混在一起。

---

# 一、最终 Simulation 总体结构

建议最终 manuscript 里分成下面 **6 个 simulation blocks**。

| Block | Experiment                           | 核心目的                                                            | 是否主文          |
| ----- | ------------------------------------ | --------------------------------------------------------------- | ------------- |
| S0    | Well-specified linear control        | 验证 Bayesian posterior / borrowing 本身没有问题                        | 保留，简化         |
| S1    | Aligned nonlinear reference          | teacher 与 target 完全一致时，建立 prediction/UQ reference               | 主文            |
| S2    | η-selection comparison               | CV-Deviance vs CV-C-index vs LPML/CPO                           | **主文重点**      |
| S3    | Predictor-distribution heterogeneity | mean / covariance / support shift                               | 主文            |
| S4    | Predictor-space heterogeneity        | teacher/student covariate availability 和 overlap 不同             | **主文重点**      |
| S5    | Outcome-mechanism heterogeneity      | baseline hazard shift、concept shift、teacher calibration bias    | **主文重点**      |
| S6    | Model heterogeneity                  | teacher quality / architecture mismatch                         | 主文或 appendix  |
| S7    | Combined severe heterogeneity        | 多种 shift 同时存在，专门测试 negative-transfer protection                 | **主文重点**      |
| S8    | Prediction-outcome extension         | individual risk、landmark event、event count、time-to-event-target | 主文 + appendix |

旧 DiSKD framework 已经固定了一个很合适的 computational skeleton：teacher $n_0=10,000$、student $n=500$、independent test $n=5,000$，20 个离散时间 interval。 Revised manuscript 则已经提出最终 repeated-sampling simulation 至少使用 100 个 independent target cohorts。

所以我建议直接沿用：

### Common sample sizes

| Quantity                   |          Primary setting |
| -------------------------- | -----------------------: |
| Teacher/source training    |             $n_S=10,000$ |
| Target/student training    |                $n_T=500$ |
| Independent test           |  $n_{\text{test}}=5,000$ |
| Time intervals             |                   $K=20$ |
| Simulation replicates      |                  **100** |
| η grid                     | ${0,0.1,0.25,0.5,1,2,5}$ |
| Main target censoring      |             $\approx30%$ |
| Main cumulative event rate |         $\approx35%-40%$ |
| Landmark                   |                 $k_0=10$ |
| Prediction horizons        |           $k=5,10,15,20$ |

η grid 我建议比现在 manuscript 的 ${0,0.5,1,2,5}$ 在 0 附近加密。因为 heterogeneity 越强，真正重要的是判断“**是不是应该只 borrow 一点点甚至完全不 borrow**”，0.1、0.25 比继续加 10 更有信息。

---

# 二、统一 Main DGP

现有 linear control 不需要动。它现在是 correctly specified logistic discrete hazard，$K=10$，$\beta=(0.8,-0.6,0.5)$，这个实验非常适合作为 sanity check。

真正新增 heterogeneity experiments 时，我建议统一使用一个 nonlinear DGP，而不是每个 heterogeneity 都换一套生成机制。

## Main target DGP

生成

[
X=(X_1,\ldots,X_{12})^\top
\sim N(0,\Sigma_T),
]

primary setting:

[
(\Sigma_T)_{jl}=0.3^{|j-l|}.
]

定义 nonlinear risk function

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

$X_{12}$ 为 pure noise。

离散 hazard：

[
\operatorname{logit}{\lambda_k^T(X)}
====================================

\alpha_k+f_T(X),
\qquad k=1,\ldots,20.
]

$\alpha_k$ 建议不要硬编码一个数，而是在 simulation 开始前通过大 Monte Carlo sample calibration，使：

[
P(T\le \tau_{20})\approx0.35.
]

这样不同 heterogeneity scenario 下可以区分：

* predictor distribution change；
* conditional outcome mechanism change；

而不会因为 intercept 随便设置导致 event rate 到处漂。

Censoring 独立生成，并 calibrate 到约 30%。

---

# 三、η selection 是整篇 simulation 最应该强化的部分

这部分我会做成一个单独的大 experiment，而不是只是 appendix sensitivity plot。

目前 manuscript 已经发现 point-estimate 5-fold CV 会倾向比较大的 η，而 LPML/WAIC/DIC 更倾向 moderate borrowing。 现在要把这个现象升级成一个 systematic simulation conclusion。

## Table A. η-selection methods

| Method         | Selection rule                               | Data used                    | Role                             |
| -------------- | -------------------------------------------- | ---------------------------- | -------------------------------- |
| Internal       | $\eta=0$                                     | target train                 | no-transfer reference            |
| Fixed-small    | $\eta=0.5$                                   | none                         | fixed borrowing reference        |
| Fixed-moderate | $\eta=1$                                     | none                         | fixed borrowing reference        |
| **CV-Dev**     | minimize 5-fold held-out predictive deviance | target CV folds              | **primary frequentist selector** |
| **CV-C**       | maximize held-out time-dependent C-index     | target CV folds              | comparator requested by advisor  |
| **LPML/CPO**   | maximize LPML                                | target posterior             | **primary Bayesian selector**    |
| WAIC           | minimize WAIC                                | target posterior             | secondary                        |
| Oracle-η       | minimize true/test predictive loss           | independent huge/test sample | simulation benchmark only        |

### CV implementation

对每一个 η 和每个 fold：

1. 用 4 folds 拟合 student；
2. training objective 包括 student likelihood + teacher KL；
3. **held-out subjects 的 likelihood 和 KL 都不能进入 training**；
4. CV-Dev 只计算 held-out target likelihood/deviance；
5. CV-C 只计算 held-out target $C^{td}$；
6. 5 folds aggregate 后选择 η。

这与 revised manuscript 当前 CV 定义完全一致。

---

# 四、为什么一定要比较 C-index selection

这个 experiment 可以非常漂亮。

### 专门设计一个“C-index 看不出来”的 heterogeneity

假设 teacher 原始 prediction 为 $p_T$，人为做：

[
\operatorname{logit}(p_T^\star)
===============================

a+b,\operatorname{logit}(p_T).
]

考虑：

| Level | $(a,b)$   | Interpretation                    |
| ----- | --------- | --------------------------------- |
| Cal-0 | $(0,1)$   | calibrated                        |
| Cal-1 | $(0.5,1)$ | moderate intercept miscalibration |
| Cal-2 | $(1,1)$   | severe intercept miscalibration   |
| Cal-3 | $(0,1.5)$ | over-confident teacher            |

尤其当 $b>0$ 且只是 intercept shift 时，**ranking 几乎不改变，因此 C-index 基本不变，但 absolute risk 已经错误**。

于是可以直接比较：

[
\widehat\eta_{\text{CV-C}}
\quad \text{vs}\quad
\widehat\eta_{\text{CV-Dev}}
\quad\text{vs}\quad
\widehat\eta_{\text{LPML}}.
]

预期：

* CV-C 仍可能选较大的 η；
* CV-Dev 会把 η 向 0 shrink；
* LPML/CPO 也会减少 borrowing；
* CV-C 的 C-index 看起来没问题，但 IBS / deviance / calibration / coverage 明显变差。

这会直接回答老师的：

> “add c-index/internal prediction selection good or not”

答案不是简单说“不好”，而是：

**C-index 可以用于 discrimination-oriented η selection，但不能作为 negative-transfer protection 的唯一 criterion。**

---

# 五、η 防止 negative transfer 的理论框架

这部分 manuscript 现有理论其实已经有了 80%。

你们已经定义 teacher-induced population shift

[
b_\eta
======

|\lambda_\eta-\lambda_0|_{\mathcal H},
]

并指出 heterogeneous transfer 下 fixed positive η 可能使 $b_\eta$ 不消失；若 $b_{\eta_n}=O(\epsilon_n)$，则 borrowing 不改变 first-order convergence rate。

我建议增加一个很简单但非常有针对性的 η-selection proposition。

设 target predictive risk 为

[
R(\eta)
=======

E_T\left[
-\log p_{\widehat\theta_\eta}(Y\mid X)
\right].
]

candidate set

[
\mathcal G_\eta=
{0,\eta_1,\ldots,\eta_G},
]

**关键是 $\eta=0\in\mathcal G_\eta$。**

定义

[
\widehat\eta_{\rm CV}
=====================

\arg\min_{\eta\in\mathcal G_\eta}
\widehat R_{\rm CV}(\eta).
]

如果

[
\sup_{\eta\in\mathcal G_\eta}
|\widehat R_{\rm CV}(\eta)-R(\eta)|
=o_p(1),
]

那么

[
R(\widehat\eta_{\rm CV})
\le
\min_{\eta\in\mathcal G_\eta}R(\eta)+o_p(1)
\le
R(0)+o_p(1).
]

这个结果非常适合你们。

它给出的 interpretation 是：

> **因为 no-transfer estimator $\eta=0$ 本身就在 candidate set 里，一个对 target proper predictive risk 一致的 η selector asymptotically cannot perform systematically worse than the internal-only option under that same risk.**

这就是“η selection acts as a negative-transfer safeguard”的理论版本。

注意这里最好用 **predictive deviance/log score**，而不是 C-index，因为 C-index 并不是 absolute predictive probability 的 proper loss。这样理论和前面的 C-index calibration counterexample 正好扣在一起。

LPML/CPO 则是 Bayesian parallel，因为它也是基于 leave-one-out target predictive density。你们 manuscript 已经把 CPO 定义为 leaving subject $i$ out 时同时移除 $L_i$ 和 $Q_i$。

---

# 六、Heterogeneity experiments 最终表

下面这张我建议就是你们真正的主 simulation design 表。

## Table B. Main heterogeneous transfer scenarios

| ID | Heterogeneity                  | Target                       | Teacher/source setting                  | Levels                                       | Main scientific question                                  |
| -- | ------------------------------ | ---------------------------- | --------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| H0 | Aligned                        | $X\sim N(0,\Sigma_T)$, $f_T$ | identical                               | reference                                    | borrowing 的 upper-bound benefit                           |
| H1 | Mean shift                     | $\mu_T=0$                    | $\mu_{S,1:6}=\delta$                    | $\delta=0,0.5,1$                             | marginal covariate shift 下是否还能安全 transfer                 |
| H2 | Correlation shift              | AR(1), $\rho_T=.3$           | AR(1), $\rho_S$                         | $.3,.6,.9$                                   | joint predictor structure 改变                              |
| H3 | Teacher covariate quality      | student all 12               | teacher subset                          | 12 / 9 / 6 variables                         | reproduce Good/Fair/Poor teacher                          |
| H4 | Predictor-space mismatch       | varying student set          | varying teacher set                     | same / nested / partial-overlap              | prediction-level transfer 是否不依赖 parameter space 对齐        |
| H5 | Baseline hazard shift          | $\alpha^T_k$                 | $\alpha^S_k=\alpha^T_k+\delta_0$        | $0,0.5,1$                                    | ranking 相近但 absolute risk 错误                              |
| H6 | Concept shift                  | $f_T$                        | $f_S=(1-\gamma)f_T+\gamma f_{\rm flip}$ | $\gamma=0,.5,1$                              | conditional $Y\mid X$ 改变时 negative transfer               |
| H7 | Teacher calibration distortion | correct target               | post-hoc $a+b\operatorname{logit}(p)$   | calibrated / intercept-shift / overconfident | CV-C 是否会漏掉 negative transfer                              |
| H8 | Architecture mismatch          | student Transformer          | teacher logistic / MLP / Transformer    | 3 architectures                              | prediction-space distillation 是否 robust to model mismatch |
| H9 | Combined shift                 | reference target             | H1+H4+H6 combinations                   | moderate / severe                            | η selector 的最终 stress test                                |

你们旧 framework 已经做过 mean-type covariate shift，并通过改变 source covariate mean 构造 moderate/severe shift。 也已经有 Good/Fair/Poor teacher covariate subsets。 所以 H1、H3 基本可以直接复用已有 code。

---

# 七、Predictor-space heterogeneity 要怎么具体生成

这是老师批注里最值得新增的一个。

设所有 subject 底层都有 $X_1,\ldots,X_{12}$，但 teacher 和 student 能访问的变量不同。

## Table C. Predictor-space experiment

| Scenario           | Student predictors | Teacher predictors | Overlap |
| ------------------ | ------------------ | ------------------ | ------: |
| P0 Same            | $X_{1:12}$         | $X_{1:12}$         |    100% |
| P1 Teacher-poor    | $X_{1:12}$         | $X_{1:6}$          |  nested |
| P2 Student-poor    | $X_{1:6}$          | $X_{1:12}$         |  nested |
| P3 Partial overlap | $X_{1:8}$          | $X_{1:4},X_{9:12}$ |     50% |
| P4 Complementary   | $X_{1:6}$          | $X_{7:12}$         |      0% |

P4 非常有意思。

student 完全看不到 teacher 使用的 predictors，但 external system 能针对同一个 target subject 输出 teacher prediction。

DiSKD 接收到的只有

[
\widetilde P_i,
]

而不是 teacher coefficients。

所以这个 experiment 可以非常直接地体现 paper 的核心 selling point：

> parameter-level transfer 在这种情况下基本无法进行，但 prediction-level distillation 仍然可以利用 complementary information。

---

# 八、Concept shift 的具体生成

建议不要每次随便改 coefficient，而是统一定义一个 heterogeneity parameter $\gamma$。

定义 severe-shift function：

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

然后

[
f_S(X)
======

(1-\gamma)f_T(X)
+
\gamma f_{\rm flip}(X),
]

其中：

[
\gamma\in{0,0.5,1}.
]

这样 heterogeneity 有一个明确 severity axis。

预期应该看到：

[
\gamma\uparrow
\Rightarrow
\eta_{\rm selected}\downarrow.
]

这张图本身就可以成为非常重要的结果：

**heterogeneity severity vs selected η**。

---

# 九、Negative transfer 一定要有正式定义

不能只说“performance dropped”。

建议每个 replicate 定义：

### Deviance negative transfer

[
NT_{\rm Dev}^{(r)}
==================

I{
Dev_{\rm selected}^{(r)}

>

Dev_{\eta=0}^{(r)}
}.
]

### IBS negative transfer

[
NT_{\rm IBS}^{(r)}
==================

I{
IBS_{\rm selected}^{(r)}

>

IBS_{\eta=0}^{(r)}
}.
]

### C-index negative transfer

[
NT_C^{(r)}
==========

I{
C_{\rm selected}^{(r)}
<
C_{\eta=0}^{(r)}
}.
]

最后报告：

[
\widehat P(NT)
==============

\frac1R\sum_{r=1}^R NT^{(r)}.
]

以及 paired performance difference：

[
\Delta Dev
==========

Dev_{\rm selected}-Dev_{\eta=0},
]

[
\Delta C
========

C_{\rm selected}-C_{\eta=0}.
]

**η selector 的成功标准不是“总选到 η>0”，恰恰是在 bad teacher 下经常选择 η≈0。**

这点 narrative 很重要。

---

# 十、Prediction outcomes 不应该只停留在 C-index

这也是我认为老师说“增加更多 prediction outcome”时最值得扩展的地方。

Revised manuscript 目前已有 $C^{td}$、predictive deviance、CIF MAE、IBS/IBLL 等。

另一个 Cox framework 已经定义了两种非常好的 posterior predictive outcome：

1. 给定 landmark 后未来窗口内发生 event 的概率；
2. future event count；
3. 达到第 $n_e$ 个 future event 的时间。

所以我建议最终 simulation 用下面 **四级 prediction targets**。

## Table D. Prediction outcomes and metrics

| Prediction target                  | Definition                     | Primary metrics                           | UQ metrics                         |
| ---------------------------------- | ------------------------------ | ----------------------------------------- | ---------------------------------- |
| **O1 Individual survival risk**    | $F(t_k\mid X)$, $k=5,10,15,20$ | RMSE, MAE, Brier, log score               | 95% coverage, width                |
| **O2 Landmark event prediction**   | $P(T\le k_0+h\mid T>k_0,X)$    | Brier, log loss, AUC/C-index, calibration | probability CI coverage            |
| **O3 Cohort future event count**   | $N(k_0,k_1)$                   | bias, MAE, RMSE                           | predictive interval coverage/width |
| **O4 Time to $m$-th future event** | $W(m)$                         | bias, MAE, RMSE                           | predictive interval coverage/width |

### O1: individual prediction

报告 horizons

[
k\in{5,10,15,20}.
]

特别强调：

* discrimination: $C^{td}$；
* proper prediction: predictive deviance / IBS；
* probability accuracy: RMSE/MAE；
* calibration slope/intercept；
* coverage/width。

---

### O2: landmark prediction

设

[
k_0=10.
]

预测：

[
P(T\le k_0+h\mid T>k_0,X),
]

其中

[
h\in{3,5,10}.
]

这个 outcome 很适合回答“teacher guidance 是否真的改善 clinically interpretable prediction”。

---

### O3: future event count

每 replicate 从 test cohort 选一个 landmark 时仍 at-risk 的固定 cohort，例如

[
M=500.
]

预测从 $k_0=10$ 到 $K=20$ 的 event 数：

[
N_{\rm future}.
]

每个 posterior draw 得一个 $N^{(m)}$。

报告：

* posterior predictive mean；
* median；
* 95% prediction interval；
* bias；
* RMSE；
* PI coverage；
* PI width。

---

### O4: time to the 25th future event

在同一个 $M=500$ landmark cohort 中：

[
W(25)
=====

\text{time until the 25th subsequent event}.
]

posterior predictive distribution 直接 forward simulate。

报告：

* median prediction error；
* MAE；
* RMSE；
* 95% PI coverage；
* interval width。

这个基本就是 `coxBayesianKL` 里 Q2 的 simulation version。

---

# 十一、每个 heterogeneity scenario 不需要报告所有 metric

否则文章会变成 Excel 森林。

建议分成 primary 和 secondary。

## Table E. Reporting hierarchy

| Category                 | Primary                        | Secondary / Appendix        |
| ------------------------ | ------------------------------ | --------------------------- |
| Discrimination           | $C^{td}$                       | horizon AUC                 |
| Overall prediction       | **Predictive deviance**        | IBS / IBLL                  |
| Calibration              | calibration slope + intercept  | calibration curve           |
| Oracle probability error | horizon RMSE                   | MAE                         |
| Bayesian UQ              | coverage + interval width      | PostSD / EmpSD              |
| η selection              | selected η distribution        | exact oracle-selection rate |
| Negative transfer        | **NT rate + paired ΔDeviance** | ΔIBS, ΔC                    |
| Event-count prediction   | RMSE + PI coverage             | bias, width                 |
| Event-time prediction    | MAE + PI coverage              | RMSE, width                 |

这样主文可以始终围绕三个东西：

[
\boxed{
\text{Prediction}
+
\text{Calibration}
+
\text{Negative-transfer protection}
}
]

而不是十几个 metric 平铺。

---

# 十二、η-selector comparison 最终输出表

这个我建议是 simulation 的核心结果表。

## Table F. η-selection evaluation

每个 heterogeneity level × selector 报：

| Selector  | Mean $\hat\eta$ | $P(\hat\eta=0)$ | Oracle regret | ΔDeviance vs internal | NT rate | ΔC-index | IBS | Coverage |
| --------- | --------------: | --------------: | ------------: | --------------------: | ------: | -------: | --: | -------: |
| CV-C      |                 |                 |               |                       |         |          |     |          |
| CV-Dev    |                 |                 |               |                       |         |          |     |          |
| LPML      |                 |                 |               |                       |         |          |     |          |
| WAIC      |                 |                 |               |                       |         |          |     |          |
| Fixed η=1 |                 |                 |               |                       |         |          |     |          |

Oracle regret 定义：

[
Regret
======

## L_{\rm test}(\widehat\eta)

\min_{\eta\in\mathcal G_\eta}
L_{\rm test}(\eta).
]

这是比“selected η 和 oracle η 一不一样”更稳定的评价。

因为 η=0.5 和 η=1 可能 performance 几乎完全一样，exact selection accuracy 会人为惩罚。

---

# 十三、最终 combined stress test

单因素 heterogeneity 解释 mechanism。

最后再做两个 combined scenarios。

## Moderate heterogeneous teacher

[
\mu_{S,1:6}=0.5,
\qquad
\rho_S=0.6,
\qquad
\gamma=0.5,
]

teacher/student predictor overlap 50%。

## Severe heterogeneous teacher

[
\mu_{S,1:6}=1,
\qquad
\rho_S=0.9,
\qquad
\gamma=1,
]

predictor overlap 50%，外加

[
\operatorname{logit}(\tilde p)
==============================

0.5+1.5\operatorname{logit}(p).
]

只比较：

* Internal；
* fixed η=1；
* CV-C selected；
* CV-Dev selected；
* LPML selected；
* Oracle selected。

这个实验最容易产生文章里一句非常漂亮的 conclusion：

> fixed borrowing exhibits substantial negative transfer as teacher-target heterogeneity increases, whereas target-predictive η selection progressively reduces borrowing and approaches the internal-only solution when external guidance becomes harmful.

---

# 十四、Implementation 固定下来，不要让 tuning 成为 confounder

这一点非常重要。

旧 framework 原本同时用 Optuna 调 network hyperparameters、$T$ 和 η。 但是**在现在这个 η-selection simulation 中，不建议这样做**。

否则你不知道：

> CV-Dev 比 CV-C 好，是因为 η selection 好，还是 network architecture 恰好调得好？

所以 simulation 应该：

## Table G. Implementation protocol

| Item                 | Final specification                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Software             | PyTorch                                                                                                          |
| Student architecture | fixed across all η within a scenario                                                                             |
| Teacher architecture | fixed except H8                                                                                                  |
| Max epochs           | 128                                                                                                              |
| Early stopping       | patience = 5                                                                                                     |
| Batch size           | 64                                                                                                               |
| Learning rate        | $10^{-3}$                                                                                                        |
| Optimizer            | Adam                                                                                                             |
| Same initialization  | same initialization across η within replicate if practical                                                       |
| CV                   | subject-level 5-fold                                                                                             |
| η grid               | $0,0.1,0.25,0.5,1,2,5$                                                                                           |
| Temperature          | fix $T$ at previously validated value, e.g. $T=2$                                                                |
| Architecture tuning  | performed once, not jointly with η                                                                               |
| Teacher predictions  | frozen before student fitting                                                                                    |
| Test set             | never used for η selection                                                                                       |
| Posterior method     | use one validated method consistently, preferably current last-layer Laplace for these heterogeneity experiments |
| Replications         | 100                                                                                                              |
| Random seeds         | common seeds across methods                                                                                      |
| Test cohort          | same test subjects across competing η selectors                                                                  |
| Metric comparison    | paired within replicate                                                                                          |

Revised manuscript 已经发现 short SGLD 的 posterior dispersion 会造成 selection criteria artifact，而 last-layer Laplace 修正以后 LPML/WAIC/DIC 的 behavior 才稳定。

所以既然这一轮的 scientific question 是 **heterogeneity/η selection**，我不会再让不同 computational posterior approximation 进来搅局。Computational-UQ comparison 可以继续留在另一节，不要和这里交叉。

---

# 十五、我建议最终 manuscript 的 simulation 顺序

最终写作顺序我会排成：

### Simulation 1: Well-specified control

保留已有 linear result。

目的只有一个：

> Bayesian formulation and posterior computation work under correct specification.

---

### Simulation 2: η selection under controlled teacher bias

做 logit intercept shift。

重点比较：

[
CV-C,\quad CV-Dev,\quad LPML.
]

这是回答老师关于 selection 的核心实验。

---

### Simulation 3: Predictor distribution shift

mean + covariance shift。

画：

[
\text{heterogeneity severity}
\longrightarrow
\widehat\eta.
]

同时报告 negative-transfer rate。

---

### Simulation 4: Predictor-space heterogeneity

Same / nested / partial / complementary predictors。

这个体现 DiSKD prediction-space transfer 的独特价值。

---

### Simulation 5: Conditional/outcome heterogeneity

baseline hazard shift + concept shift。

这是最容易产生 true negative transfer 的实验。

---

### Simulation 6: Architecture/teacher quality heterogeneity

Good/Fair/Poor + logistic/MLP/Transformer teacher。

已有 framework 大部分可以直接复用。

---

### Simulation 7: Combined stress test

moderate / severe combined shift。

证明 automatic η selection 是 safety mechanism。

---

### Simulation 8: Multiple prediction outcomes

在 H0、moderate、severe 三种 representative setting 下，不必重新跑全部 scenario。

只比较：

* individual risk；
* landmark event probability；
* future event count；
* time to 25th event。

这样新增 prediction outcomes 不会让 simulation 数量爆炸。

---

## 最终最重要的三张 Figure

如果让我现在替你们定 figure，我会优先做：

**Figure 1: Heterogeneity severity vs selected η**

横轴 heterogeneity，纵轴 $\hat\eta$，分别 CV-C / CV-Dev / LPML。

预期 CV-Dev、LPML 随 heterogeneity 增加向 0 收缩。

**Figure 2: Negative-transfer rate across selectors**

CV-C / CV-Dev / LPML / fixed η。

这个会非常直接。

**Figure 3: Prediction under calibration shift**

左：C-index
中：predictive deviance/IBS
右：calibration slope/coverage

CV-C 很可能左边“看起来很好”，但中间和右边暴露问题。这会是整篇关于“为什么 η selection 不能只靠 discrimination”的最有说服力的图。

---

### 一句话概括这套新 simulation 的 paper story

你们现在不要再把 simulation 写成“我们换很多 setting，然后 DiSKD 大多表现不错”。

更强的故事应该是：

[
\boxed{
\text{Heterogeneity}
\rightarrow
\text{teacher bias}
\rightarrow
\text{optimal borrowing decreases}
\rightarrow
\eta\text{-selection protects against negative transfer}
}
]

同时：

[
\boxed{
C\text{-index selection evaluates ranking,}
\quad
CV\text{-deviance/LPML evaluate target predictive distribution.}
}
]

因此 **CV-Deviance + LPML/CPO 应该成为主 η selectors，CV-C-index 是非常重要的 comparison，而不是最终推荐方法。**

这套设计和你们 revised manuscript 现在的 $b_\eta$ pseudo-target theory 是直接接得上的，也把旧 DiSKD framework 的 covariate shift / teacher quality 和 Cox framework 的 posterior predictive event-count / event-time outcomes 都真正整合进来了，而不是简单多加几个 sensitivity plots。  
