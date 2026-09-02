"""Posterior-sample driven credible intervals for DiSKD predictions.

Given a list of posterior state_dicts (e.g., from SGLD or from a multi-chain
sampler) plus a fitted model whose `net` matches the state_dict shape, these
utilities iterate over samples, run prediction (CIF / survival / hazard /
metric) at each, and return per-quantile arrays.

The interval-length, deviance-vs-coverage, and replicate-comparison plots
that this module enables are the building blocks for re-deriving the
paper's seed-replicate uncertainty estimates (Figure 4, Table 1, Table 3)
as Bayesian credible intervals.
"""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .models import DiscreteSurvivalModel


PREDICTOR_REGISTRY: dict[str, str] = {
    "cif": "predict_cif",
    "survival": "predict_survival",
    "hazard": "predict_hazard",
    "interval_probs": "predict_interval_probs",
}


def _iter_samples(model: DiscreteSurvivalModel, samples: Sequence[dict]):
    """Yield (sample_idx, model) with the model's state_dict swapped to each sample.

    On exit the model is restored to its original state.
    """
    if model.net is None:
        raise RuntimeError("Model is not fitted.")
    original_state = copy.deepcopy(model.net.state_dict())
    try:
        for i, state in enumerate(samples):
            model.net.load_state_dict(state)
            yield i, model
    finally:
        model.net.load_state_dict(original_state)


def posterior_predictions(
    model: DiscreteSurvivalModel,
    samples: Sequence[dict],
    data: pd.DataFrame,
    predictor: str = "cif",
) -> np.ndarray:
    """Run `model.predict_<predictor>(data)` for each posterior sample.

    Returns an array of shape `[S, *prediction_shape]`. For competing-risks
    CIF prediction this is `[S, N, J, K]`; for survival `[S, N, K]`.
    """
    if predictor not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"predictor must be one of {list(PREDICTOR_REGISTRY)}; got {predictor!r}"
        )
    method = PREDICTOR_REGISTRY[predictor]
    preds = []
    for _, m in _iter_samples(model, samples):
        out = getattr(m, method)(data)
        if hasattr(out, "numpy"):
            out = out.numpy()
        preds.append(np.asarray(out))
    return np.stack(preds, axis=0)


def credible_intervals(
    predictions: np.ndarray,
    q: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> np.ndarray:
    """Compute posterior quantiles along the sample axis (axis 0).

    Input shape `[S, ...]`; output shape `[len(q), ...]`.
    """
    if predictions.ndim < 2:
        raise ValueError("predictions must have shape [S, ...]; got 1-D.")
    return np.quantile(predictions, q, axis=0)


def credible_metric(
    model: DiscreteSurvivalModel,
    samples: Sequence[dict],
    metric_fn: Callable[[DiscreteSurvivalModel], float | np.ndarray],
    q: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply `metric_fn(model)` at every posterior sample.

    Returns `(quantiles, raw_values)`:
        - `quantiles` has shape `[len(q), *metric_shape]`
        - `raw_values` has shape `[S, *metric_shape]` for downstream
          diagnostics (e.g., R-hat, ESS).

    `metric_fn` should be a closure that uses the *current* model state.
    """
    vals = []
    for _, m in _iter_samples(model, samples):
        v = metric_fn(m)
        if hasattr(v, "numpy"):
            v = v.numpy()
        vals.append(np.asarray(v))
    arr = np.stack(vals, axis=0)
    return np.quantile(arr, q, axis=0), arr


def interval_width(quantiles: np.ndarray, low_idx: int = 0, high_idx: int = -1) -> np.ndarray:
    """Compute interval width along the quantile axis (axis 0).

    For default `q=(0.025, 0.5, 0.975)`, this returns the 95% credible
    interval width with shape `[*metric_shape]`.
    """
    return quantiles[high_idx] - quantiles[low_idx]


def gelman_rubin_rhat(samples_by_chain: np.ndarray) -> float:
    """Gelman-Rubin :math:`\\hat R` diagnostic for a scalar quantity.

    Args:
        samples_by_chain: array of shape `[M, N]` with `M` independent chains
            and `N` post-burn-in samples per chain.

    Returns:
        :math:`\\hat R = \\sqrt{(\\hat V / W)}` where
        :math:`\\hat V = (N-1)/N \\cdot W + B / N`,
        :math:`B` is the between-chain variance (scaled by `N`),
        and :math:`W` is the within-chain variance. Values close to 1.0
        indicate that the chains have converged to the same distribution;
        values above ~1.1 indicate non-convergence in the common practical
        threshold (Gelman & Rubin, 1992).
    """
    arr = np.asarray(samples_by_chain, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("samples_by_chain must have shape [M >= 2, N >= 2].")
    M, N = arr.shape
    chain_means = arr.mean(axis=1)               # [M]
    chain_vars = arr.var(axis=1, ddof=1)         # [M]
    overall_mean = chain_means.mean()
    B = (N / (M - 1)) * ((chain_means - overall_mean) ** 2).sum()
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    V_hat = ((N - 1) / N) * W + B / N
    return float(np.sqrt(V_hat / W))


def effective_sample_size(samples_by_chain: np.ndarray) -> float:
    """Effective sample size for a scalar quantity across MCMC chains.

    Implements the BDA3 (Gelman et al., 2013, Eq. 11.8) split-chain ESS
    using the autocovariance estimate truncated at the first negative
    odd lag pair (Geyer's initial monotone positive sequence). Returns a
    pooled scalar across all `M * N` samples; values close to `M * N`
    indicate near-independent draws, much smaller values warn of
    autocorrelation in the chain.
    """
    arr = np.asarray(samples_by_chain, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("samples_by_chain must have shape [M >= 2, N >= 2].")
    M, N = arr.shape

    # Within- and between-chain variance components.
    chain_means = arr.mean(axis=1)
    chain_vars = arr.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    B = (N / (M - 1)) * ((chain_means - overall_mean) ** 2).sum()
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    V_hat = ((N - 1) / N) * W + B / N

    # Lag-t autocorrelation pooled across chains.
    centered = arr - chain_means[:, None]
    rho = np.zeros(N)
    rho[0] = 1.0
    for t in range(1, N):
        cov_t = (centered[:, : N - t] * centered[:, t:]).mean()
        rho[t] = 1.0 - (W - cov_t) / V_hat

    # Geyer initial monotone positive sequence: truncate at first negative
    # sum of consecutive pairs (rho[2k] + rho[2k+1]).
    tau = 1.0
    for k in range(1, N // 2):
        pair = rho[2 * k - 1] + rho[2 * k]
        if pair < 0:
            break
        tau += 2 * pair
    if tau <= 0:
        return float("nan")
    return float(M * N / tau)


def coverage(
    quantiles: np.ndarray,
    truth: np.ndarray,
    low_idx: int = 0,
    high_idx: int = -1,
) -> float:
    """Empirical coverage: fraction of `truth` entries inside [q_low, q_high].

    Only meaningful when ground truth is known (simulation studies, or in
    held-out calibration cohorts after binarizing observed events).
    """
    lo = quantiles[low_idx]
    hi = quantiles[high_idx]
    if truth.shape != lo.shape:
        raise ValueError(
            f"truth shape {truth.shape} does not match quantile slice shape {lo.shape}."
        )
    return float(np.mean((truth >= lo) & (truth <= hi)))
