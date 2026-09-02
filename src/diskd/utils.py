"""Tensor utilities for discrete-time survival models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_at_risk_mask(idx_durations: torch.Tensor, num_durations: int) -> torch.Tensor:
    """Return a boolean mask with True for intervals included in each likelihood."""
    idx = idx_durations.long().view(-1)
    if idx.numel() == 0:
        return torch.zeros((0, num_durations), dtype=torch.bool, device=idx.device)
    if idx.min() < 0 or idx.max() >= num_durations:
        raise ValueError("Discrete durations must be in [0, num_durations - 1].")
    grid = torch.arange(num_durations, device=idx.device).view(1, -1)
    return grid <= idx.view(-1, 1)


def competing_interval_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert event logits [N, J, K] to interval probabilities [N, J + 1, K].

    The last category is the no-event category. Its logit is fixed at zero,
    matching the paper's denominator 1 + sum_j exp(r_j).
    """
    if logits.ndim != 3:
        raise ValueError("Competing-risk logits must have shape [N, J, K].")
    baseline = torch.zeros(
        logits.shape[0], 1, logits.shape[2], dtype=logits.dtype, device=logits.device
    )
    return F.softmax(torch.cat([logits, baseline], dim=1), dim=1)


def full_probs_from_event_probs(event_probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Append a no-event category to event probabilities if needed.

    Accepts either [N, J, K] event probabilities or [N, J + 1, K] full
    probabilities. Values are clamped and renormalized for stable KL terms.
    """
    if event_probs.ndim != 3:
        raise ValueError("Teacher probabilities must have shape [N, J, K] or [N, J + 1, K].")

    probs = event_probs.float().clamp_min(eps)
    channel_sum = probs.sum(dim=1, keepdim=True)
    if torch.all(torch.abs(channel_sum - 1.0) <= 1e-4):
        return probs / channel_sum.clamp_min(eps)

    if torch.any(channel_sum > 1.0 + 1e-4):
        probs = probs / channel_sum.clamp_min(eps)
        channel_sum = probs.sum(dim=1, keepdim=True)

    no_event = (1.0 - channel_sum).clamp_min(eps)
    if probs.shape[1] >= 2 and torch.all(channel_sum <= 1.0 + 1e-4):
        full = torch.cat([probs.clamp_min(eps), no_event], dim=1)
    else:
        raise ValueError("Teacher probabilities have invalid channel sums.")

    return full / full.sum(dim=1, keepdim=True).clamp_min(eps)


def temperature_scale_probs(probs: torch.Tensor, temperature: float, eps: float = 1e-7) -> torch.Tensor:
    """Temperature-scale a probability distribution along the category dimension."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scaled = probs.clamp_min(eps).pow(1.0 / temperature)
    return scaled / scaled.sum(dim=1, keepdim=True).clamp_min(eps)


def competing_targets(
    idx_durations: torch.Tensor,
    events: torch.Tensor,
    num_risks: int,
    num_durations: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build full interval targets [N, J + 1, K] with no-event as the last channel."""
    idx = idx_durations.long().view(-1)
    event = events.long().view(-1)
    if idx.shape[0] != event.shape[0]:
        raise ValueError("idx_durations and events must have the same length.")
    if event.min() < 0 or event.max() > num_risks:
        raise ValueError("Events must be coded as 0 for censoring and 1..num_risks for causes.")

    out_dtype = dtype or torch.float32
    target = torch.zeros(
        idx.shape[0], num_risks + 1, num_durations, dtype=out_dtype, device=idx.device
    )
    target[:, -1, :] = 1.0
    observed = event > 0
    if observed.any():
        rows = torch.arange(idx.shape[0], device=idx.device)[observed]
        target[rows, -1, idx[observed]] = 0.0
        target[rows, event[observed] - 1, idx[observed]] = 1.0
    return target


def competing_survival(probs: torch.Tensor) -> torch.Tensor:
    """Compute survival [N, K] from full competing-risk interval probabilities."""
    if probs.ndim != 3:
        raise ValueError("probs must have shape [N, J + 1, K].")
    return torch.cumprod(probs[:, -1, :], dim=1)


def competing_cif(probs: torch.Tensor) -> torch.Tensor:
    """Compute cumulative incidence functions [N, J, K]."""
    if probs.ndim != 3:
        raise ValueError("probs must have shape [N, J + 1, K].")
    event_probs = probs[:, :-1, :]
    no_event = probs[:, -1, :]
    survival = torch.cumprod(no_event, dim=1)
    survival_prev = torch.cat(
        [torch.ones((probs.shape[0], 1), dtype=probs.dtype, device=probs.device), survival[:, :-1]],
        dim=1,
    )
    density = survival_prev.unsqueeze(1) * event_probs
    return torch.cumsum(density, dim=2)
