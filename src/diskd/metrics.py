"""Small evaluation helpers for discrete survival predictions."""

from __future__ import annotations

import numpy as np


def concordance_index(
    durations,
    events,
    risk_scores,
    event_of_interest: int | None = None,
    tied_tol: float = 1e-12,
) -> float:
    """Harrell-style concordance index for survival risk scores.

    Higher `risk_scores` should mean higher risk / earlier event. For
    competing risks, pass `event_of_interest` and event-specific scores such as
    the CIF for that cause. Comparable pairs are anchored by subjects who
    experienced the target event before another subject's observed time.
    """
    durations = np.asarray(durations, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    _validate_metric_inputs(durations, events, risk_scores)

    if event_of_interest is None:
        anchors = events > 0
    else:
        if event_of_interest <= 0:
            raise ValueError("event_of_interest must use one-indexed event labels.")
        anchors = events == int(event_of_interest)

    concordant = 0.0
    comparable = 0
    n = durations.shape[0]
    for i in range(n):
        if not anchors[i]:
            continue
        for j in range(n):
            if durations[i] >= durations[j]:
                continue
            comparable += 1
            diff = risk_scores[i] - risk_scores[j]
            if diff > tied_tol:
                concordant += 1.0
            elif abs(diff) <= tied_tol:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def competing_risk_c_index(
    cif,
    durations,
    events,
    event_of_interest: int | None = None,
    horizon_index: int = -1,
    tied_tol: float = 1e-12,
) -> float | np.ndarray:
    """Cause-specific C-index for competing-risks CIF predictions.

    `cif` must have shape `[N, J, K]`. The event label is one-indexed, so
    `event_of_interest=1` evaluates `cif[:, 0, horizon_index]`. When
    `event_of_interest` is omitted, returns one C-index per event cause.
    """
    cif = np.asarray(cif, dtype=float)
    if cif.ndim != 3:
        raise ValueError("cif must have shape [N, J, K].")
    if event_of_interest is None:
        return np.asarray(
            [
                competing_risk_c_index(
                    cif,
                    durations,
                    events,
                    event_of_interest=risk_index + 1,
                    horizon_index=horizon_index,
                    tied_tol=tied_tol,
                )
                for risk_index in range(cif.shape[1])
            ],
            dtype=float,
        )
    if not (1 <= event_of_interest <= cif.shape[1]):
        raise ValueError("event_of_interest must be in 1..J.")
    scores = cif[:, event_of_interest - 1, horizon_index]
    return concordance_index(
        durations,
        events,
        scores,
        event_of_interest=event_of_interest,
        tied_tol=tied_tol,
    )


def predictive_deviance(
    interval_probs,
    idx_durations,
    events,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> float | np.ndarray:
    """Predictive deviance for discrete-time survival probabilities.

    Parameters
    ----------
    interval_probs:
        Either event hazards with shape `[N, K]`, or full interval
        probabilities with shape `[N, J + 1, K]`; the final channel is the
        no-event category.
    idx_durations:
        Zero-indexed observed interval ids in `[0, K - 1]`.
    events:
        Event labels with `0=censored` and `1..J` for observed causes.
    reduction:
        `"mean"`, `"sum"`, or `"none"`.
    """
    log_lik = discrete_log_likelihood(interval_probs, idx_durations, events, eps=eps)
    deviance = -2.0 * log_lik
    if reduction == "mean":
        return float(np.mean(deviance))
    if reduction == "sum":
        return float(np.sum(deviance))
    if reduction == "none":
        return deviance
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


def brier_score(
    survival,
    idx_durations,
    events,
    eval_indices=None,
    censor_idx_durations=None,
    censor_events=None,
    eps: float = 1e-12,
) -> np.ndarray:
    """IPCW Brier score curve for single-risk survival predictions.

    `survival` must have shape `[N, K]` and `events` must use single-risk
    labels, with `0=censored` and `1=event`. Censoring weights are estimated
    from the evaluated sample unless `censor_idx_durations` and
    `censor_events` are supplied.
    """
    survival, idx, events, eval_indices = _validate_single_risk_survival_inputs(
        survival,
        idx_durations,
        events,
        eval_indices,
        eps=eps,
    )
    g_at_eval, g_before_subject = _censoring_weights(
        idx,
        events,
        eval_indices,
        max_index=survival.shape[1] - 1,
        censor_idx_durations=censor_idx_durations,
        censor_events=censor_events,
        eps=eps,
    )

    scores = np.zeros(eval_indices.shape[0], dtype=float)
    for pos, eval_idx in enumerate(eval_indices):
        pred_survival = survival[:, eval_idx]
        observed_event = (events == 1) & (idx <= eval_idx)
        observed_survival = idx > eval_idx

        subject_scores = np.zeros(idx.shape[0], dtype=float)
        subject_scores[observed_event] = pred_survival[observed_event] ** 2 / g_before_subject[observed_event]
        subject_scores[observed_survival] = (
            (1.0 - pred_survival[observed_survival]) ** 2 / g_at_eval[pos]
        )
        scores[pos] = subject_scores.mean()
    return scores


def integrated_brier_score(
    survival,
    idx_durations,
    events,
    eval_indices=None,
    time_points=None,
    censor_idx_durations=None,
    censor_events=None,
    eps: float = 1e-12,
) -> float:
    """Integrated Brier score for single-risk survival predictions."""
    scores = brier_score(
        survival,
        idx_durations,
        events,
        eval_indices=eval_indices,
        censor_idx_durations=censor_idx_durations,
        censor_events=censor_events,
        eps=eps,
    )
    eval_indices = _normalize_eval_indices(eval_indices, np.asarray(survival).shape[1])
    return _integrated_score(scores, eval_indices, time_points)


def negative_binomial_log_likelihood(
    survival,
    idx_durations,
    events,
    eval_indices=None,
    censor_idx_durations=None,
    censor_events=None,
    eps: float = 1e-12,
) -> np.ndarray:
    """IPCW negative binomial log-likelihood curve for single-risk survival."""
    survival, idx, events, eval_indices = _validate_single_risk_survival_inputs(
        survival,
        idx_durations,
        events,
        eval_indices,
        eps=eps,
    )
    g_at_eval, g_before_subject = _censoring_weights(
        idx,
        events,
        eval_indices,
        max_index=survival.shape[1] - 1,
        censor_idx_durations=censor_idx_durations,
        censor_events=censor_events,
        eps=eps,
    )

    scores = np.zeros(eval_indices.shape[0], dtype=float)
    for pos, eval_idx in enumerate(eval_indices):
        pred_survival = survival[:, eval_idx]
        observed_event = (events == 1) & (idx <= eval_idx)
        observed_survival = idx > eval_idx

        subject_scores = np.zeros(idx.shape[0], dtype=float)
        subject_scores[observed_event] = (
            -np.log1p(-pred_survival[observed_event]) / g_before_subject[observed_event]
        )
        subject_scores[observed_survival] = -np.log(pred_survival[observed_survival]) / g_at_eval[pos]
        scores[pos] = subject_scores.mean()
    return scores


def integrated_negative_binomial_log_likelihood(
    survival,
    idx_durations,
    events,
    eval_indices=None,
    time_points=None,
    censor_idx_durations=None,
    censor_events=None,
    eps: float = 1e-12,
) -> float:
    """Integrated NBLL for single-risk survival predictions."""
    scores = negative_binomial_log_likelihood(
        survival,
        idx_durations,
        events,
        eval_indices=eval_indices,
        censor_idx_durations=censor_idx_durations,
        censor_events=censor_events,
        eps=eps,
    )
    eval_indices = _normalize_eval_indices(eval_indices, np.asarray(survival).shape[1])
    return _integrated_score(scores, eval_indices, time_points)


def discrete_log_likelihood(
    interval_probs,
    idx_durations,
    events,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-subject discrete survival log-likelihood."""
    probs = _as_full_interval_probs(interval_probs, eps=eps)
    idx = np.asarray(idx_durations, dtype=int).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)

    if probs.shape[0] != idx.shape[0] or idx.shape[0] != events.shape[0]:
        raise ValueError("interval_probs, idx_durations, and events must have matching length.")
    if idx.size == 0:
        return np.asarray([], dtype=float)
    if idx.min() < 0 or idx.max() >= probs.shape[2]:
        raise ValueError("idx_durations must be in [0, K - 1].")

    num_risks = probs.shape[1] - 1
    if events.min() < 0 or events.max() > num_risks:
        raise ValueError("events must be coded as 0 for censoring and 1..J for causes.")

    probs = np.clip(probs, eps, 1.0)
    out = np.zeros(idx.shape[0], dtype=float)
    for i, stop in enumerate(idx):
        if stop > 0:
            out[i] += np.log(probs[i, -1, :stop]).sum()
        if events[i] > 0:
            out[i] += np.log(probs[i, events[i] - 1, stop])
        else:
            out[i] += np.log(probs[i, -1, stop])
    return out


def monotone_non_decreasing(values: np.ndarray, axis: int = -1, tol: float = 1e-8) -> bool:
    return bool(np.all(np.diff(values, axis=axis) >= -tol))


def monotone_non_increasing(values: np.ndarray, axis: int = -1, tol: float = 1e-8) -> bool:
    return bool(np.all(np.diff(values, axis=axis) <= tol))


def _as_full_interval_probs(interval_probs, eps: float) -> np.ndarray:
    probs = np.asarray(interval_probs, dtype=float)
    if probs.ndim == 2:
        if not np.all(np.isfinite(probs)):
            raise ValueError("interval_probs must be finite.")
        if np.any((probs < -eps) | (probs > 1.0 + eps)):
            raise ValueError("single-risk hazards must be in [0, 1].")
        hazard = np.clip(probs, eps, 1.0 - eps)
        return np.stack([hazard, 1.0 - hazard], axis=1)
    if probs.ndim != 3:
        raise ValueError("interval_probs must have shape [N, K] or [N, J + 1, K].")
    if not np.all(np.isfinite(probs)):
        raise ValueError("interval_probs must be finite.")
    if np.any(probs < -eps):
        raise ValueError("interval probabilities must be nonnegative.")
    channel_sum = probs.sum(axis=1, keepdims=True)
    if np.any(channel_sum <= 0):
        raise ValueError("interval probability channel sums must be positive.")
    return np.clip(probs / channel_sum, eps, 1.0)


def _validate_metric_inputs(durations: np.ndarray, events: np.ndarray, scores: np.ndarray) -> None:
    if durations.shape[0] != events.shape[0] or events.shape[0] != scores.shape[0]:
        raise ValueError("durations, events, and risk_scores must have matching length.")
    if durations.ndim != 1 or events.ndim != 1 or scores.ndim != 1:
        raise ValueError("durations, events, and risk_scores must be one-dimensional.")
    if not np.all(np.isfinite(durations)) or not np.all(np.isfinite(scores)):
        raise ValueError("durations and risk_scores must be finite.")
    if np.any(events < 0):
        raise ValueError("events must be nonnegative.")


def _validate_single_risk_survival_inputs(
    survival,
    idx_durations,
    events,
    eval_indices,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2:
        raise ValueError("survival must have shape [N, K] for single-risk predictions.")
    if not np.all(np.isfinite(survival)):
        raise ValueError("survival must be finite.")
    if np.any((survival < -eps) | (survival > 1.0 + eps)):
        raise ValueError("survival probabilities must be in [0, 1].")

    idx = np.asarray(idx_durations, dtype=int).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    if survival.shape[0] != idx.shape[0] or idx.shape[0] != events.shape[0]:
        raise ValueError("survival, idx_durations, and events must have matching length.")
    if idx.size == 0:
        raise ValueError("at least one subject is required.")
    if idx.min() < 0 or idx.max() >= survival.shape[1]:
        raise ValueError("idx_durations must be in [0, K - 1].")
    if np.any((events < 0) | (events > 1)):
        raise ValueError("single-risk IBS/INBLL require events coded as 0 or 1.")

    return (
        np.clip(survival, eps, 1.0 - eps),
        idx,
        events,
        _normalize_eval_indices(eval_indices, survival.shape[1]),
    )


def _normalize_eval_indices(eval_indices, num_durations: int) -> np.ndarray:
    if eval_indices is None:
        out = np.arange(num_durations, dtype=int)
    else:
        out = np.asarray(eval_indices, dtype=int).reshape(-1)
    if out.size == 0:
        raise ValueError("eval_indices must contain at least one index.")
    if out.min() < 0 or out.max() >= num_durations:
        raise ValueError("eval_indices must be in [0, K - 1].")
    return out


def _censoring_weights(
    idx: np.ndarray,
    events: np.ndarray,
    eval_indices: np.ndarray,
    max_index: int,
    censor_idx_durations=None,
    censor_events=None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    if censor_idx_durations is None:
        censor_idx = idx
        censor_events_array = events
    else:
        if censor_events is None:
            raise ValueError("censor_events is required when censor_idx_durations is supplied.")
        censor_idx = np.asarray(censor_idx_durations, dtype=int).reshape(-1)
        censor_events_array = np.asarray(censor_events, dtype=int).reshape(-1)
        if censor_idx.shape[0] != censor_events_array.shape[0]:
            raise ValueError("censor_idx_durations and censor_events must have matching length.")
        if censor_idx.size == 0:
            raise ValueError("censoring reference data must contain at least one subject.")
        if censor_idx.min() < 0 or censor_idx.max() > max_index:
            raise ValueError("censor_idx_durations must be in [0, K - 1].")
        if np.any((censor_events_array < 0) | (censor_events_array > 1)):
            raise ValueError("censor_events must be coded as 0 or 1.")

    censoring_observed = censor_events_array == 0
    censor_survival = _kaplan_meier_survival(censor_idx, censoring_observed, max_index=max_index)
    g_at_eval = np.clip(censor_survival[eval_indices], eps, None)

    g_before_subject = np.ones(idx.shape[0], dtype=float)
    after_first_interval = idx > 0
    g_before_subject[after_first_interval] = censor_survival[idx[after_first_interval] - 1]
    return g_at_eval, np.clip(g_before_subject, eps, None)


def _kaplan_meier_survival(idx: np.ndarray, event_observed: np.ndarray, max_index: int) -> np.ndarray:
    survival = np.ones(max_index + 1, dtype=float)
    current = 1.0
    for time_idx in range(max_index + 1):
        at_risk = idx >= time_idx
        n_at_risk = int(at_risk.sum())
        if n_at_risk > 0:
            n_events = int(((idx == time_idx) & event_observed).sum())
            current *= 1.0 - (n_events / n_at_risk)
        survival[time_idx] = current
    return survival


def _integrated_score(scores: np.ndarray, eval_indices: np.ndarray, time_points=None) -> float:
    if scores.shape[0] == 1:
        return float(scores[0])
    if time_points is None:
        return float(np.mean(scores))

    time_points = np.asarray(time_points, dtype=float).reshape(-1)
    if time_points.shape[0] == eval_indices.shape[0]:
        grid = time_points
    elif time_points.shape[0] > eval_indices.max():
        grid = time_points[eval_indices]
    else:
        raise ValueError("time_points must match eval_indices or contain one value per duration index.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time_points must be finite.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time_points must be strictly increasing over eval_indices.")
    return float(np.trapz(scores, grid) / (grid[-1] - grid[0]))
