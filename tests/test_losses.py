import torch
import torch.nn.functional as F

from diskd.losses import (
    BinaryHorizonToCompetingRiskKDLoss,
    CompetingRiskKDLoss,
    CompetingRiskNLLLoss,
    OverallToCompetingRiskKDLoss,
    SingleRiskKDLoss,
    SingleRiskNLLLoss,
)
from diskd.utils import competing_cif, competing_interval_probs


def _logits_from_competing_probs(full_probs: torch.Tensor) -> torch.Tensor:
    return torch.log(full_probs[:, :-1, :] / full_probs[:, -1:, :])


def test_single_risk_loss_is_finite():
    logits = torch.randn(4, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 1, 0])
    loss = SingleRiskNLLLoss()(logits, idx, events)
    assert torch.isfinite(loss)


def test_single_risk_loss_matches_manual_likelihood():
    hazard = torch.tensor(
        [
            [0.20, 0.30, 0.40],
            [0.10, 0.25, 0.50],
        ]
    )
    logits = torch.logit(hazard)
    idx = torch.tensor([1, 2])
    events = torch.tensor([1, 0])

    expected = torch.stack(
        [
            -torch.log(1.0 - hazard[0, 0]) - torch.log(hazard[0, 1]),
            -torch.log(1.0 - hazard[1, 0])
            - torch.log(1.0 - hazard[1, 1])
            - torch.log(1.0 - hazard[1, 2]),
        ]
    )

    observed = SingleRiskNLLLoss()(logits, idx, events, reduction="none")
    assert torch.allclose(observed, expected)


def test_single_risk_kd_eta_zero_matches_nll():
    logits = torch.randn(4, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 1, 0])
    nll = SingleRiskNLLLoss()(logits, idx, events)
    kd = SingleRiskKDLoss(eta=0)(logits, idx, events)
    assert torch.allclose(nll, kd)


def test_competing_risk_loss_is_finite():
    logits = torch.randn(4, 2, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 2, 0])
    loss = CompetingRiskNLLLoss()(logits, idx, events)
    assert torch.isfinite(loss)


def test_competing_risk_loss_matches_manual_likelihood():
    full_probs = torch.tensor(
        [
            [[0.20, 0.30, 0.25], [0.10, 0.20, 0.15], [0.70, 0.50, 0.60]],
            [[0.25, 0.10, 0.20], [0.05, 0.30, 0.20], [0.70, 0.60, 0.60]],
        ]
    )
    logits = _logits_from_competing_probs(full_probs)
    idx = torch.tensor([1, 2])
    events = torch.tensor([2, 0])

    expected = torch.stack(
        [
            -torch.log(full_probs[0, -1, 0]) - torch.log(full_probs[0, 1, 1]),
            -torch.log(full_probs[1, -1, 0])
            - torch.log(full_probs[1, -1, 1])
            - torch.log(full_probs[1, -1, 2]),
        ]
    )

    observed = CompetingRiskNLLLoss()(logits, idx, events, reduction="none")
    assert torch.allclose(observed, expected)


def test_competing_kd_eta_zero_matches_nll():
    logits = torch.randn(4, 2, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 2, 0])
    nll = CompetingRiskNLLLoss()(logits, idx, events)
    kd = CompetingRiskKDLoss(eta=0, temperature=2.0)(logits, idx, events)
    assert torch.allclose(nll, kd)


def test_competing_kd_loss_is_finite():
    logits = torch.randn(4, 2, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 2, 0])
    teacher = competing_interval_probs(torch.randn(4, 2, 5))[:, :2, :]
    loss = CompetingRiskKDLoss(eta=1.5, temperature=2.0)(logits, idx, events, teacher)
    assert torch.isfinite(loss)


def test_competing_kd_loss_matches_manual_at_risk_kl():
    student_full = torch.tensor(
        [
            [[0.20, 0.30, 0.25], [0.10, 0.20, 0.15], [0.70, 0.50, 0.60]],
            [[0.25, 0.10, 0.20], [0.05, 0.30, 0.20], [0.70, 0.60, 0.60]],
        ]
    )
    teacher_full = torch.tensor(
        [
            [[0.25, 0.25, 0.20], [0.15, 0.10, 0.20], [0.60, 0.65, 0.60]],
            [[0.20, 0.15, 0.25], [0.10, 0.25, 0.15], [0.70, 0.60, 0.60]],
        ]
    )
    logits = _logits_from_competing_probs(student_full)
    idx = torch.tensor([1, 2])
    events = torch.tensor([2, 0])
    eta = 2.0

    nll_i = CompetingRiskNLLLoss()(logits, idx, events, reduction="none")
    kl_ik = (teacher_full * (teacher_full.log() - student_full.log())).sum(dim=1)
    kd_i = torch.stack([kl_ik[0, :2].sum(), kl_ik[1, :3].sum()])
    expected = (nll_i + eta * kd_i) / (1.0 + eta)

    observed = CompetingRiskKDLoss(eta=eta, temperature=1.0)(
        logits,
        idx,
        events,
        teacher_full[:, :-1, :],
        reduction="none",
    )
    assert torch.allclose(observed, expected)


def test_overall_to_competing_kd_loss_is_finite():
    logits = torch.randn(4, 2, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 2, 0])
    teacher = torch.sigmoid(torch.randn(4, 5))
    loss = OverallToCompetingRiskKDLoss(eta=1.0)(logits, idx, events, teacher)
    assert torch.isfinite(loss)


def test_overall_to_competing_kd_loss_matches_manual_hazard_bce():
    student_full = torch.tensor(
        [
            [[0.20, 0.30, 0.25], [0.10, 0.20, 0.15], [0.70, 0.50, 0.60]],
            [[0.25, 0.10, 0.20], [0.05, 0.30, 0.20], [0.70, 0.60, 0.60]],
        ]
    )
    logits = _logits_from_competing_probs(student_full)
    idx = torch.tensor([1, 2])
    events = torch.tensor([2, 0])
    teacher = torch.tensor([[0.35, 0.40, 0.30], [0.20, 0.35, 0.25]])
    eta = 0.5

    nll_i = CompetingRiskNLLLoss()(logits, idx, events, reduction="none")
    student_overall = student_full[:, :-1, :].sum(dim=1)
    bce = F.binary_cross_entropy(student_overall, teacher, reduction="none")
    kd_i = torch.stack([bce[0, :2].sum(), bce[1, :3].sum()])
    expected = (nll_i + eta * kd_i) / (1.0 + eta)

    observed = OverallToCompetingRiskKDLoss(eta=eta)(
        logits,
        idx,
        events,
        teacher,
        reduction="none",
    )
    assert torch.allclose(observed, expected)


def test_binary_horizon_to_competing_kd_loss_is_finite():
    logits = torch.randn(4, 2, 5)
    idx = torch.tensor([0, 2, 3, 4])
    events = torch.tensor([1, 0, 2, 0])
    teacher = torch.sigmoid(torch.randn(4))
    loss = BinaryHorizonToCompetingRiskKDLoss(eta=1.0, risk_index=0)(logits, idx, events, teacher)
    assert torch.isfinite(loss)


def test_binary_horizon_to_competing_kd_loss_matches_manual_cif_bce():
    student_full = torch.tensor(
        [
            [[0.20, 0.30, 0.25], [0.10, 0.20, 0.15], [0.70, 0.50, 0.60]],
            [[0.25, 0.10, 0.20], [0.05, 0.30, 0.20], [0.70, 0.60, 0.60]],
        ]
    )
    logits = _logits_from_competing_probs(student_full)
    idx = torch.tensor([1, 2])
    events = torch.tensor([2, 0])
    teacher = torch.tensor([0.30, 0.45])
    eta = 1.25

    nll_i = CompetingRiskNLLLoss()(logits, idx, events, reduction="none")
    risk1_horizon = competing_cif(student_full)[:, 0, 1]
    kd_i = F.binary_cross_entropy(risk1_horizon, teacher, reduction="none")
    expected = (nll_i + eta * kd_i) / (1.0 + eta)

    observed = BinaryHorizonToCompetingRiskKDLoss(
        eta=eta,
        risk_index=0,
        horizon_index=1,
    )(logits, idx, events, teacher, reduction="none")
    assert torch.allclose(observed, expected)
