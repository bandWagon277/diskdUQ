"""Loss functions for DiSKD."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    competing_cif,
    competing_interval_probs,
    competing_targets,
    full_probs_from_event_probs,
    make_at_risk_mask,
    temperature_scale_probs,
)


def _reduce(loss_i: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss_i.mean()
    if reduction == "sum":
        return loss_i.sum()
    if reduction == "none":
        return loss_i
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


class SingleRiskNLLLoss(nn.Module):
    """Discrete-time single-risk negative log-likelihood."""

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Single-risk logits must have shape [N, K].")
        num_durations = logits.shape[1]
        mask = make_at_risk_mask(idx_durations, num_durations).to(logits.device)

        target = torch.zeros_like(logits)
        observed = events.view(-1).to(logits.device).float() > 0
        if observed.any():
            rows = torch.arange(logits.shape[0], device=logits.device)[observed]
            target[rows, idx_durations.to(logits.device).long().view(-1)[observed]] = 1.0

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return _reduce((bce * mask).sum(dim=1), reduction)


class SingleRiskKDLoss(nn.Module):
    """Single-risk NLL plus at-risk Bernoulli KD on teacher hazards."""

    def __init__(self, eta: float = 1.0):
        super().__init__()
        if eta < 0:
            raise ValueError("eta must be nonnegative.")
        self.eta = float(eta)
        self.nll = SingleRiskNLLLoss()

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        teacher_hazard: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        nll_i = self.nll(logits, idx_durations, events, reduction="none")
        if self.eta == 0:
            return _reduce(nll_i, reduction)
        if teacher_hazard is None:
            raise ValueError("teacher_hazard is required when eta > 0.")
        if teacher_hazard.shape != logits.shape:
            raise ValueError("teacher_hazard must have shape [N, K].")

        mask = make_at_risk_mask(idx_durations.to(logits.device), logits.shape[1])
        student_hazard = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        teacher = teacher_hazard.to(logits.device).float().clamp(1e-7, 1 - 1e-7)
        kd = F.binary_cross_entropy(student_hazard, teacher, reduction="none")
        kd_i = (kd * mask).sum(dim=1)
        return _reduce((nll_i + self.eta * kd_i) / (1.0 + self.eta), reduction)


class CompetingRiskNLLLoss(nn.Module):
    """Discrete-time competing-risk negative log-likelihood."""

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("Competing-risk logits must have shape [N, J, K].")
        _, num_risks, num_durations = logits.shape
        mask = make_at_risk_mask(idx_durations.to(logits.device), num_durations)
        probs = competing_interval_probs(logits).clamp_min(1e-7)
        target = competing_targets(
            idx_durations.to(logits.device),
            events.to(logits.device),
            num_risks,
            num_durations,
            dtype=logits.dtype,
        )
        ce = -(target * torch.log(probs)).sum(dim=1)
        return _reduce((ce * mask).sum(dim=1), reduction)


class CompetingRiskKDLoss(nn.Module):
    """Competing-risk NLL plus teacher-to-student interval KL."""

    def __init__(self, eta: float = 1.0, temperature: float = 1.0, scale_temperature: bool = True):
        super().__init__()
        if eta < 0:
            raise ValueError("eta must be nonnegative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.eta = float(eta)
        self.temperature = float(temperature)
        self.scale_temperature = bool(scale_temperature)
        self.nll = CompetingRiskNLLLoss()

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        nll_i = self.nll(logits, idx_durations, events, reduction="none")
        if self.eta == 0:
            return _reduce(nll_i, reduction)
        if teacher_probs is None:
            raise ValueError("teacher_probs is required when eta > 0.")

        _, _, num_durations = logits.shape
        mask = make_at_risk_mask(idx_durations.to(logits.device), num_durations)
        student_full = competing_interval_probs(logits)
        teacher_full = full_probs_from_event_probs(teacher_probs.to(logits.device))
        if teacher_full.shape != student_full.shape:
            raise ValueError("teacher_probs must match student shape [N, J, K] or [N, J + 1, K].")

        teacher_t = temperature_scale_probs(teacher_full, self.temperature)
        student_t = temperature_scale_probs(student_full, self.temperature)
        kl = teacher_t * (teacher_t.clamp_min(1e-7).log() - student_t.clamp_min(1e-7).log())
        kl_i = (kl.sum(dim=1) * mask).sum(dim=1)
        if self.scale_temperature:
            kl_i = (self.temperature * self.temperature) * kl_i
        return _reduce((nll_i + self.eta * kl_i) / (1.0 + self.eta), reduction)


class OverallToCompetingRiskKDLoss(nn.Module):
    """Distill an overall-event teacher hazard into a competing-risk student."""

    def __init__(self, eta: float = 1.0):
        super().__init__()
        if eta < 0:
            raise ValueError("eta must be nonnegative.")
        self.eta = float(eta)
        self.nll = CompetingRiskNLLLoss()

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        teacher_overall_hazard: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        nll_i = self.nll(logits, idx_durations, events, reduction="none")
        if self.eta == 0:
            return _reduce(nll_i, reduction)
        if teacher_overall_hazard is None:
            raise ValueError("teacher_overall_hazard is required when eta > 0.")
        if teacher_overall_hazard.shape != (logits.shape[0], logits.shape[2]):
            raise ValueError("teacher_overall_hazard must have shape [N, K].")

        probs = competing_interval_probs(logits)
        student_overall = probs[:, :-1, :].sum(dim=1).clamp(1e-7, 1 - 1e-7)
        teacher = teacher_overall_hazard.to(logits.device).float().clamp(1e-7, 1 - 1e-7)
        mask = make_at_risk_mask(idx_durations.to(logits.device), logits.shape[2])
        kd = F.binary_cross_entropy(student_overall, teacher, reduction="none")
        kd_i = (kd * mask).sum(dim=1)
        return _reduce((nll_i + self.eta * kd_i) / (1.0 + self.eta), reduction)


class BinaryHorizonToCompetingRiskKDLoss(nn.Module):
    """Distill fixed-horizon event probability into a competing-risk student's CIF."""

    def __init__(self, eta: float = 1.0, risk_index: int = 0, horizon_index: int | None = None):
        super().__init__()
        if eta < 0:
            raise ValueError("eta must be nonnegative.")
        self.eta = float(eta)
        self.risk_index = int(risk_index)
        self.horizon_index = horizon_index
        self.nll = CompetingRiskNLLLoss()

    def forward(
        self,
        logits: torch.Tensor,
        idx_durations: torch.Tensor,
        events: torch.Tensor,
        teacher_horizon_risk: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        nll_i = self.nll(logits, idx_durations, events, reduction="none")
        if self.eta == 0:
            return _reduce(nll_i, reduction)
        if teacher_horizon_risk is None:
            raise ValueError("teacher_horizon_risk is required when eta > 0.")
        if teacher_horizon_risk.shape[0] != logits.shape[0]:
            raise ValueError("teacher_horizon_risk must have shape [N].")
        if not (0 <= self.risk_index < logits.shape[1]):
            raise ValueError("risk_index must be in [0, num_risks - 1].")

        horizon = logits.shape[2] - 1 if self.horizon_index is None else int(self.horizon_index)
        if not (0 <= horizon < logits.shape[2]):
            raise ValueError("horizon_index must be in [0, num_durations - 1].")

        cif = competing_cif(competing_interval_probs(logits))
        student_risk = cif[:, self.risk_index, horizon].clamp(1e-7, 1 - 1e-7)
        teacher = teacher_horizon_risk.to(logits.device).float().view(-1).clamp(1e-7, 1 - 1e-7)
        kd_i = F.binary_cross_entropy(student_risk, teacher, reduction="none")
        return _reduce((nll_i + self.eta * kd_i) / (1.0 + self.eta), reduction)

