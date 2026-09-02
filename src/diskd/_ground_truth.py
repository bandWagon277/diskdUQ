"""Closed-form true cumulative incidence for `simulate_competing_risks`.

The simulator at `src/diskd/simulation.py` samples
  t_j ~ Exp(rate_j(Z_i)),  rate_j(Z) = (beta_risk_j * z_j)^2 + (beta_shared * z_shared)^2
with cause label = argmin_j t_j and uniform censoring. For two independent
exponential causes the true CIF admits a closed form:

  F_j^true(t | Z) = (rate_j(Z) / rate_total(Z)) * (1 - exp(-rate_total(Z) * t))

where rate_total = rate_1 + rate_2. Evaluating this at the discrete bin
endpoints used by the student gives the ground-truth discrete CIF that
the model's posterior should cover.

This module is used for `coverage` and `interval_width vs truth-distance`
diagnostics in the synthetic experiments only; it is not part of the
production `diskd` API and does not export from `diskd/__init__.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import TimeGrid


def true_cif_at_grid(
    data: pd.DataFrame,
    time_grid: TimeGrid,
    num_risks: int = 2,
    beta_risk1: float = 2.0,
    beta_risk2: float = 2.0,
    beta_shared: float = 8.0,
    rate_floor: float = 1e-3,
) -> np.ndarray:
    """Closed-form true CIF on the discrete bin right-endpoints.

    Assumes `data` was produced by `simulate_competing_risks(...)` with the
    same beta_* values. Reconstructs the latent rates from the 12 covariates
    and evaluates the closed-form CIF at each cut in `time_grid.cuts`.

    Returns
    -------
    Array of shape `[len(data), num_risks, K]` with the cause-specific
    cumulative incidence at each grid endpoint.
    """
    if num_risks != 2:
        raise NotImplementedError("Only the two-cause simulator is supported.")
    feat_cols = [f"x{i + 1}" for i in range(12)]
    if not all(c in data.columns for c in feat_cols):
        raise ValueError("data must contain columns x1..x12 from simulate_competing_risks.")
    x = data[feat_cols].to_numpy(dtype=float)
    z1 = x[:, 0:4].sum(axis=1)
    z2 = x[:, 4:8].sum(axis=1)
    z3 = x[:, 8:12].sum(axis=1)

    rate1 = np.clip((beta_risk1 * z1) ** 2 + (beta_shared * z3) ** 2, rate_floor, None)
    rate2 = np.clip((beta_risk2 * z2) ** 2 + (beta_shared * z3) ** 2, rate_floor, None)
    rate_total = rate1 + rate2

    cuts = np.asarray(time_grid.cuts, dtype=float)
    # [N, 1] * (1 - exp(- [N, 1] * [K]))
    rt = rate_total[:, None]
    surv = np.exp(-rt * cuts[None, :])             # [N, K]
    overall_cif = 1.0 - surv                       # [N, K]
    share1 = (rate1 / rate_total)[:, None]         # [N, 1]
    share2 = (rate2 / rate_total)[:, None]
    F = np.stack([share1 * overall_cif, share2 * overall_cif], axis=1)  # [N, J, K]
    return F


def coverage_from_quantiles(
    cif_quantiles: np.ndarray,
    true_cif: np.ndarray,
    low_idx: int = 0,
    high_idx: int = -1,
) -> dict:
    """Indicator-coverage statistics of a credible interval vs ground truth.

    Args
    ----
    cif_quantiles : array of shape [Q, N, J, K]
        Posterior quantiles produced by `credible_intervals(...)` on the
        SGLD or ensemble CIF predictions.
    true_cif : array of shape [N, J, K]
        Closed-form CIF from `true_cif_at_grid(...)`.
    low_idx, high_idx : int
        Quantile-axis indices defining the lower and upper interval
        endpoints (default: first and last quantile slice).

    Returns
    -------
    dict with keys:
      "overall": scalar coverage averaged over all (subject, cause, time)
      "per_cause": shape [J] coverage per cause (averaged over subject, time)
      "per_horizon": shape [K] coverage per time grid point
      "mean_interval_width": scalar mean of upper - lower
    """
    if cif_quantiles.ndim != 4:
        raise ValueError("cif_quantiles must have shape [Q, N, J, K].")
    if true_cif.shape != cif_quantiles.shape[1:]:
        raise ValueError(
            f"true_cif shape {true_cif.shape} does not match the [N, J, K] "
            f"slice {cif_quantiles.shape[1:]}."
        )
    lo = cif_quantiles[low_idx]   # [N, J, K]
    hi = cif_quantiles[high_idx]
    inside = (true_cif >= lo) & (true_cif <= hi)
    return {
        "overall": float(inside.mean()),
        "per_cause": inside.mean(axis=(0, 2)),       # [J]
        "per_horizon": inside.mean(axis=(0, 1)),     # [K]
        "mean_interval_width": float((hi - lo).mean()),
    }
