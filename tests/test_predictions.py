import torch

from diskd.utils import competing_cif, competing_interval_probs, competing_survival


def test_competing_interval_probs_sum_to_one():
    logits = torch.randn(3, 2, 6)
    probs = competing_interval_probs(logits)
    assert probs.shape == (3, 3, 6)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3, 6), atol=1e-6)


def test_competing_survival_is_monotone_non_increasing():
    probs = competing_interval_probs(torch.randn(3, 2, 6))
    survival = competing_survival(probs)
    assert torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-8)


def test_competing_cif_is_monotone_non_decreasing():
    probs = competing_interval_probs(torch.randn(3, 2, 6))
    cif = competing_cif(probs)
    assert cif.shape == (3, 2, 6)
    assert torch.all(cif[:, :, 1:] >= cif[:, :, :-1] - 1e-8)


def test_competing_survival_and_cif_match_manual_formula():
    probs = torch.tensor(
        [
            [
                [0.20, 0.10, 0.30],
                [0.10, 0.20, 0.10],
                [0.70, 0.70, 0.60],
            ]
        ]
    )

    survival = competing_survival(probs)
    cif = competing_cif(probs)

    expected_survival = torch.tensor([[0.70, 0.49, 0.294]])
    expected_cif = torch.tensor(
        [
            [
                [0.20, 0.20 + 0.70 * 0.10, 0.20 + 0.70 * 0.10 + 0.49 * 0.30],
                [0.10, 0.10 + 0.70 * 0.20, 0.10 + 0.70 * 0.20 + 0.49 * 0.10],
            ]
        ]
    )

    assert torch.allclose(survival, expected_survival)
    assert torch.allclose(cif, expected_cif)
