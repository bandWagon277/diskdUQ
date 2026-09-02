import numpy as np
import pytest

from diskd import (
    brier_score,
    competing_risk_c_index,
    concordance_index,
    discrete_log_likelihood,
    integrated_brier_score,
    integrated_negative_binomial_log_likelihood,
    negative_binomial_log_likelihood,
    predictive_deviance,
)


def test_predictive_deviance_matches_manual_competing_likelihood():
    probs = np.asarray(
        [
            [[0.10, 0.20], [0.05, 0.10], [0.85, 0.70]],
            [[0.30, 0.20], [0.10, 0.20], [0.60, 0.60]],
        ]
    )
    idx = np.asarray([1, 0])
    events = np.asarray([1, 0])

    expected_log_lik = np.asarray([
        np.log(0.85) + np.log(0.20),
        np.log(0.60),
    ])

    assert np.allclose(discrete_log_likelihood(probs, idx, events), expected_log_lik)
    assert np.isclose(predictive_deviance(probs, idx, events, reduction="sum"), -2.0 * expected_log_lik.sum())


def test_concordance_index_uses_higher_score_as_higher_risk():
    durations = np.asarray([1.0, 2.0, 3.0])
    events = np.asarray([1, 0, 1])
    scores = np.asarray([0.9, 0.2, 0.7])

    assert concordance_index(durations, events, scores) == 1.0


def test_competing_risk_c_index_is_cause_specific():
    durations = np.asarray([1.0, 2.0, 3.0, 4.0])
    events = np.asarray([2, 1, 0, 2])
    cif = np.zeros((4, 2, 2))
    cif[:, 0, -1] = np.asarray([0.2, 0.9, 0.1, 0.3])
    cif[:, 1, -1] = np.asarray([0.8, 0.3, 0.2, 0.6])

    assert competing_risk_c_index(cif, durations, events, event_of_interest=2) == 1.0
    assert np.allclose(competing_risk_c_index(cif, durations, events), np.asarray([1.0, 1.0]))


def test_single_risk_brier_score_and_nbll_match_manual_values():
    survival = np.asarray(
        [
            [0.9, 0.7],
            [0.6, 0.2],
            [0.8, 0.5],
        ]
    )
    idx = np.asarray([0, 1, 1])
    events = np.asarray([1, 1, 0])

    expected_brier = np.asarray([
        (0.9**2 + (1.0 - 0.6) ** 2 + (1.0 - 0.8) ** 2) / 3.0,
        (0.7**2 + 0.2**2) / 3.0,
    ])
    expected_nbll = np.asarray([
        (-np.log1p(-0.9) - np.log(0.6) - np.log(0.8)) / 3.0,
        (-np.log1p(-0.7) - np.log1p(-0.2)) / 3.0,
    ])

    assert np.allclose(brier_score(survival, idx, events), expected_brier)
    assert np.isclose(integrated_brier_score(survival, idx, events), expected_brier.mean())
    assert np.allclose(negative_binomial_log_likelihood(survival, idx, events), expected_nbll)
    assert np.isclose(
        integrated_negative_binomial_log_likelihood(survival, idx, events),
        expected_nbll.mean(),
    )


def test_single_risk_ibs_rejects_competing_risk_survival_shape():
    survival = np.ones((2, 2, 3)) * 0.5
    idx = np.asarray([0, 1])
    events = np.asarray([1, 0])

    with pytest.raises(ValueError, match="single-risk"):
        integrated_brier_score(survival, idx, events)


def test_single_risk_ibs_rejects_competing_risk_event_labels():
    survival = np.ones((2, 3)) * 0.5
    idx = np.asarray([0, 1])
    events = np.asarray([1, 2])

    with pytest.raises(ValueError, match="0 or 1"):
        integrated_negative_binomial_log_likelihood(survival, idx, events)
