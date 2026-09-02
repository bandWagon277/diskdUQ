"""Synthetic data generators for DiSKD tutorials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CompetingRiskCohorts:
    """Independent synthetic cohorts for public DiSKD examples."""

    teacher: pd.DataFrame
    student: pd.DataFrame
    test: pd.DataFrame
    all_features: list[str]
    teacher_features: list[str]
    student_features: list[str]
    teacher_feature_quality: str


def simulate_competing_risks(
    n: int = 1000,
    seed: int | None = None,
    beta_risk1: float = 2.0,
    beta_risk2: float = 2.0,
    beta_shared: float = 8.0,
    censor_max: float = 500.0,
) -> pd.DataFrame:
    """Generate a two-risk nonlinear survival dataset with shared signal.

    The first four features primarily affect risk 1, the next four affect
    risk 2, and the last four affect both risks.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    z1 = x[:, 0:4].sum(axis=1)
    z2 = x[:, 4:8].sum(axis=1)
    z3 = x[:, 8:12].sum(axis=1)

    rate1 = (beta_risk1 * z1) ** 2 + (beta_shared * z3) ** 2
    rate2 = (beta_risk2 * z2) ** 2 + (beta_shared * z3) ** 2
    rate1 = np.clip(rate1, 1e-3, None)
    rate2 = np.clip(rate2, 1e-3, None)

    t1 = rng.exponential(scale=1.0 / rate1)
    t2 = rng.exponential(scale=1.0 / rate2)
    censor = rng.uniform(0.0, censor_max, size=n)

    event_time = np.minimum(t1, t2)
    event = np.where(t1 <= t2, 1, 2).astype(np.int64)
    censored = censor < event_time
    duration = np.where(censored, censor, event_time)
    event[censored] = 0

    df = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])])
    df["duration"] = duration.astype(float)
    df["event"] = event
    return df


def simulate_competing_risk_cohorts(
    n_teacher: int = 800,
    n_student: int = 160,
    n_test: int = 300,
    seed: int | None = None,
    teacher_feature_quality: str = "reduced",
    censor_max: float = 0.05,
    beta_risk1: float = 2.0,
    beta_risk2: float = 2.0,
    beta_shared: float = 8.0,
) -> CompetingRiskCohorts:
    """Generate separate teacher, student, and test cohorts.

    The data-generating process is shared across cohorts, but rows are sampled
    independently. The student cohort is intentionally small. Teacher quality is
    controlled by the feature subset available to the teacher, while the
    student keeps the full synthetic covariate set.

    `teacher_feature_quality` choices:

    - `"full"`: teacher sees all covariates.
    - `"reduced"`: teacher misses the shared signal block `x9..x12`.
    - `"low"`: teacher sees only the first risk-specific block `x1..x4`.
    """
    if n_teacher <= 0 or n_student <= 0 or n_test <= 0:
        raise ValueError("n_teacher, n_student, and n_test must be positive.")

    rng = np.random.default_rng(seed)
    teacher_seed, student_seed, test_seed = rng.integers(0, np.iinfo(np.int32).max, size=3)
    teacher = simulate_competing_risks(
        n=n_teacher,
        seed=int(teacher_seed),
        beta_risk1=beta_risk1,
        beta_risk2=beta_risk2,
        beta_shared=beta_shared,
        censor_max=censor_max,
    )
    student = simulate_competing_risks(
        n=n_student,
        seed=int(student_seed),
        beta_risk1=beta_risk1,
        beta_risk2=beta_risk2,
        beta_shared=beta_shared,
        censor_max=censor_max,
    )
    test = simulate_competing_risks(
        n=n_test,
        seed=int(test_seed),
        beta_risk1=beta_risk1,
        beta_risk2=beta_risk2,
        beta_shared=beta_shared,
        censor_max=censor_max,
    )

    all_features = [c for c in teacher.columns if c.startswith("x")]
    if teacher_feature_quality == "full":
        teacher_features = all_features
    elif teacher_feature_quality == "reduced":
        teacher_features = all_features[:8]
    elif teacher_feature_quality == "low":
        teacher_features = all_features[:4]
    else:
        raise ValueError("teacher_feature_quality must be 'full', 'reduced', or 'low'.")

    return CompetingRiskCohorts(
        teacher=teacher,
        student=student,
        test=test,
        all_features=all_features,
        teacher_features=teacher_features,
        student_features=all_features,
        teacher_feature_quality=teacher_feature_quality,
    )
