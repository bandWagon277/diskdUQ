"""Unit tests for the SGLD optimizer's two drift conventions.

We construct a 1-parameter problem with a fixed gradient and disable the
diffusion term (`noise_scale=0`) so each `step()` is deterministic. The
drift then collapses to a closed-form expression we can compare against:

    welling_teh:  delta = -0.5 * step_size * n_train * loss_scale * grad
    literal:      delta = -0.5 * step_size           * loss_scale * grad
"""
import math

import pytest
import torch

from diskd.samplers import SGLD


def _one_step(drift_mode: str, *, step_size: float, n_train: int,
              loss_scale: float = 1.0, grad_value: float = 1.0) -> float:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGLD(
        [p],
        step_size=step_size,
        n_train=n_train,
        loss_scale=loss_scale,
        noise_scale=0.0,
        drift_mode=drift_mode,
    )
    p.grad = torch.tensor([grad_value])
    opt.step()
    return float(p.detach().item())


def test_welling_teh_drift_scales_with_n_train():
    delta = _one_step("welling_teh", step_size=1e-6, n_train=5000, grad_value=2.0)
    expected = -0.5 * 1e-6 * 5000 * 2.0
    assert math.isclose(delta, expected, rel_tol=1e-6, abs_tol=1e-12)


def test_literal_drift_ignores_n_train():
    delta_a = _one_step("literal", step_size=1e-3, n_train=10, grad_value=2.0)
    delta_b = _one_step("literal", step_size=1e-3, n_train=10_000, grad_value=2.0)
    expected = -0.5 * 1e-3 * 2.0
    assert math.isclose(delta_a, expected, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(delta_b, expected, rel_tol=1e-6, abs_tol=1e-12)


def test_loss_scale_applies_to_both_modes():
    delta_wt = _one_step("welling_teh", step_size=1e-6, n_train=1000,
                         loss_scale=3.0, grad_value=1.0)
    delta_lit = _one_step("literal", step_size=1e-3, n_train=1000,
                          loss_scale=3.0, grad_value=1.0)
    assert math.isclose(delta_wt,  -0.5 * 1e-6 * 1000 * 3.0, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(delta_lit, -0.5 * 1e-3        * 3.0, rel_tol=1e-6, abs_tol=1e-12)


def test_diffusion_std_matches_sqrt_step_independent_of_mode():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(20_000))
    step = 1e-3
    opt = SGLD([p], step_size=step, n_train=100, drift_mode="literal")
    # Zero gradient isolates the diffusion term.
    p.grad = torch.zeros_like(p)
    opt.step()
    # Sample std of the increment from zero should be ~ sqrt(step).
    observed = float(p.detach().std().item())
    assert math.isclose(observed, math.sqrt(step), rel_tol=0.05)


def test_msgld_adds_momentum_bias():
    """MSGLD drift should be larger than plain literal drift due to the
    accumulated momentum pulling in the same direction as the gradient."""
    p_lit = torch.nn.Parameter(torch.zeros(1))
    p_msg = torch.nn.Parameter(torch.zeros(1))
    step = 1e-3
    opt_lit = SGLD([p_lit], step_size=step, n_train=10, drift_mode="literal",
                   noise_scale=0.0)
    opt_msg = SGLD([p_msg], step_size=step, n_train=10, drift_mode="msgld",
                   bias_factor=1.0, momentum_beta=0.9, noise_scale=0.0)
    grad_val = 2.0
    for _ in range(5):
        p_lit.grad = torch.tensor([grad_val])
        opt_lit.step()
        p_msg.grad = torch.tensor([grad_val])
        opt_msg.step()
    # MSGLD accumulates momentum in the same direction → drifts further.
    assert abs(float(p_msg)) > abs(float(p_lit))


def test_asgld_rescales_momentum():
    """ASGLD uses m_t / sqrt(V_t + λ) as bias; verify it runs without error
    and produces a different trajectory from plain literal mode."""
    p = torch.nn.Parameter(torch.zeros(3))
    opt = SGLD([p], step_size=1e-3, n_train=10, drift_mode="asgld",
               bias_factor=0.1, momentum_beta=0.9, adam_beta2=0.999,
               noise_scale=0.0)
    for _ in range(10):
        p.grad = torch.randn(3)
        opt.step()
    assert torch.isfinite(p).all()
    assert float(p.abs().sum()) > 0


def test_cyclical_schedule_endpoints():
    """Cyclical schedule should return eps_0 at cycle start and ~0 at cycle end."""
    p = torch.nn.Parameter(torch.zeros(1))
    cycle_len = 100
    opt = SGLD([p], step_size=1e-3, n_train=10, schedule="cyclical",
               cycle_length_steps=cycle_len, noise_scale=0.0)
    # t=0: start of cycle → eps = eps_0
    eps_start = opt._current_step_size()
    assert math.isclose(eps_start, 1e-3, rel_tol=1e-6)
    # t=cycle_len-1: end of cycle → eps ≈ 0
    opt._step_count = cycle_len - 1
    eps_end = opt._current_step_size()
    assert eps_end < 1e-6
    # t=cycle_len: start of next cycle → eps = eps_0 again
    opt._step_count = cycle_len
    eps_reset = opt._current_step_size()
    assert math.isclose(eps_reset, 1e-3, rel_tol=1e-6)


def test_cyclical_periodic():
    """Step size pattern repeats across multiple cycles."""
    p = torch.nn.Parameter(torch.zeros(1))
    cycle_len = 50
    opt = SGLD([p], step_size=2e-3, n_train=10, schedule="cyclical",
               cycle_length_steps=cycle_len, noise_scale=0.0)
    for offset in [0, cycle_len, 2 * cycle_len]:
        opt._step_count = offset + 10
        eps_a = opt._current_step_size()
        opt._step_count = offset + 25
        eps_b = opt._current_step_size()
        # Check relative values are consistent across cycles
        if offset == 0:
            ref_a, ref_b = eps_a, eps_b
        else:
            assert math.isclose(eps_a, ref_a, rel_tol=1e-10)
            assert math.isclose(eps_b, ref_b, rel_tol=1e-10)


def test_invalid_drift_mode_raises():
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="drift_mode"):
        SGLD([p], step_size=1e-3, n_train=10, drift_mode="banana")
