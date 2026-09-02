"""Stochastic Gradient Langevin Dynamics (SGLD) for Bayesian DiSKD.

Two drift conventions are supported. Both share the same diffusion term
``sqrt(step) * eps`` and the same polynomial step-size schedule.

Welling-Teh convention (``drift_mode='welling_teh'``, default), as in
Welling & Teh (2011):

    theta_{t+1} = theta_t - 0.5 * step * N * grad(L_batch) + sqrt(step) * eps

Here ``L_batch`` is the *mean*-batch loss returned by the DiSKD loss modules
(reduction='mean'), so the unbiased full-data gradient estimator is
``N * grad(L_batch)``. The ``N`` factor (``n_train``) is multiplied inside
the optimizer. The SGLD target is the full posterior
``pi_0(theta) * prod_i p(x_i | theta)``. With this convention, the
SGD-equivalent effective learning rate is ``0.5 * step * N`` and ``step``
must be ~ ``1e-7`` for an Adam-like effective lr at ``N ~ 10^3 - 10^4``.

Literal convention (``drift_mode='literal'``), Option B in the cluster
runbook:

    theta_{t+1} = theta_t - 0.5 * step * grad(L_batch) + sqrt(step) * eps

The mean-batch gradient is used directly without the ``N`` multiplier, so
``step`` is interpreted at the same scale as an SGD/AdamW learning rate
(e.g. ``1e-3``). The diffusion term is the same as in Welling-Teh, which
means the noise-to-drift ratio per step is ``sqrt(2/step) / 1`` instead of
``sqrt(2/step) / N`` -- much larger for the same step size, which helps
push chains out of local modes when Welling-Teh chains have frozen.
Formally, this samples the tempered posterior
``pi_0(theta) * (prod_i p(x_i | theta))^(1/N)``; treat the resulting draws
as a posterior on the per-sample log-likelihood rather than the full data.

Momentum SGLD (``drift_mode='msgld'``), from Kim, Song & Liang (2020)
"SGLD Algorithms with Adaptive Drifts", Algorithm 1:

    m_t = beta1 * m_{t-1} + (1 - beta1) * grad_{t-1}
    theta_{t+1} = theta_t - 0.5 * step * (grad_t + a * m_t) + sqrt(step) * eps

The momentum term ``m_t`` is an exponentially decaying average of past
gradients. The bias factor ``a`` controls how strongly the momentum pulls
the chain along persistent gradient directions, helping escape narrow
ravines and mode boundaries. Theorem 3.1 of the paper proves ergodicity.

Adam SGLD (``drift_mode='asgld'``), from Kim, Song & Liang (2020),
Algorithm 2:

    m_t = beta1 * m_{t-1} + (1 - beta1) * grad_{t-1}
    V_t = beta2 * V_{t-1} + (1 - beta2) * grad_{t-1}^2
    theta_{t+1} = theta_t - 0.5 * step * (grad_t + a * m_t / sqrt(V_t + lam)) + sqrt(step) * eps

The second-moment ``V_t`` rescales the momentum to be approximately
isotropic near stationary points, combining the preconditioning benefit of
Adam with the posterior-sampling property of SGLD. Theorem 3.2 proves
ergodicity.

All modes share the same polynomial step-size schedule and diffusion
``sqrt(step) * eps``. An optional isotropic Gaussian prior is available via
``prior_sigma``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


DRIFT_MODES = ("welling_teh", "literal", "msgld", "asgld", "psgld")


class SGLD(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    Args:
        params: Parameters to optimize.
        step_size: Initial SGLD step size (a.k.a. `epsilon_0`). The natural
            scale depends on ``drift_mode``: in ``'welling_teh'`` mode the
            SGD-equivalent lr is ``0.5 * step_size * n_train`` (so
            ``step_size ~ 1e-7`` is typical for ``N ~ 10^3 - 10^4``); in
            ``'literal'`` mode ``step_size`` is the SGD-equivalent lr
            directly (so ``step_size ~ 1e-3`` matches an Adam-like rate).
        n_train: Training set size N. Used in ``'welling_teh'`` mode to
            rescale the mean-batch gradient up to a full-data gradient
            estimator; recorded but not applied to the drift in
            ``'literal'`` mode.
        final_step_size: If not None, step size linearly (in log-space)
            decays from `step_size` to `final_step_size` over
            `total_steps` updates, following `eps_t = a (b + t)^(-gamma)`.
            With `final_step_size=None`, step size is constant.
        total_steps: Total number of `step()` calls used for the schedule.
        gamma: Exponent in the polynomial decay. Welling & Teh require
            gamma in (0.5, 1] for convergence. Default 0.55.
        noise_scale: Multiplier on the Gaussian noise std. Defaults to 1.0
            (true SGLD). Set to 0.0 to recover plain SGD with the same
            step-size schedule (useful for unit-testing the trajectory).
        prior_sigma: If not None, applies an isotropic Gaussian prior
            :math:`\\pi_0(\\theta) = \\mathcal{N}(0, \\sigma^2 I)` to the
            full parameter vector. The negative log prior contributes a
            ridge-style gradient term :math:`\\theta / \\sigma^2` per step.
            With `prior_sigma=None`, the prior is flat (default, matches
            R01 Eq.~(4) Section C.2.1.4 with :math:`\\pi_0(\\theta)`
            constant).
        drift_mode: ``'welling_teh'`` (default) multiplies the mean-batch
            gradient by ``n_train`` to target the full posterior;
            ``'literal'`` (Option B) uses the mean-batch gradient directly
            and targets the tempered posterior at temperature ``N``;
            ``'msgld'`` adds momentum-based adaptive bias (Kim et al. 2020,
            Algorithm 1); ``'asgld'`` adds Adam-style adaptive bias
            (Algorithm 2). Both ``msgld`` and ``asgld`` use the literal
            gradient convention (no ``n_train`` multiplier). See module
            docstring for the update rules.
        bias_factor: Multiplier ``a`` on the adaptive bias term for
            ``msgld``/``asgld``. Default 1.0 for msgld, 0.1 for asgld
            (paper recommendations). Ignored by welling_teh/literal.
        momentum_beta: Exponential smoothing factor ``beta_1`` for the
            first-moment estimate in ``msgld``/``asgld``. Default 0.9.
        adam_beta2: Smoothing factor ``beta_2`` for the second-moment
            estimate in ``asgld``. Default 0.999. Ignored by other modes.
        ada_eps: Small constant ``lambda`` added to sqrt(V_t) in ``asgld``
            to avoid division by zero. Default 1e-8.
        schedule: Step-size schedule. ``'polynomial'`` (default) uses the
            monotone decay ``eps_t = a(b+t)^{-gamma}``; ``'cyclical'``
            (Zhang et al., ICLR 2020) uses cosine-annealed cycles
            ``eps_t = (eps_0/2)(cos(pi * t_cycle / M) + 1)`` where ``M``
            is the cycle length in steps and ``t_cycle`` is the step
            index within the current cycle. Each cycle resets eps to
            ``eps_0``, enabling periodic exploration of new modes.
        cycle_length_steps: Steps per cycle in cyclical mode. Ignored
            by polynomial mode. Typical: 50 epochs × steps_per_epoch.
    """

    def __init__(
        self,
        params,
        step_size: float,
        n_train: int,
        final_step_size: Optional[float] = None,
        total_steps: Optional[int] = None,
        gamma: float = 0.55,
        noise_scale: float = 1.0,
        loss_scale: float = 1.0,
        prior_sigma: Optional[float] = None,
        drift_mode: str = "welling_teh",
        bias_factor: Optional[float] = None,
        momentum_beta: float = 0.9,
        adam_beta2: float = 0.999,
        ada_eps: float = 1e-8,
        schedule: str = "polynomial",
        cycle_length_steps: int = 750,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if n_train <= 0:
            raise ValueError("n_train must be positive.")
        if final_step_size is not None:
            if total_steps is None or total_steps <= 0:
                raise ValueError("total_steps required with final_step_size.")
            if final_step_size <= 0 or final_step_size > step_size:
                raise ValueError("final_step_size must be in (0, step_size].")
            if not (0.5 < gamma <= 1.0):
                raise ValueError("gamma must be in (0.5, 1] for Welling-Teh convergence.")
        if noise_scale < 0:
            raise ValueError("noise_scale must be nonnegative.")
        if loss_scale <= 0:
            raise ValueError("loss_scale must be positive.")
        if prior_sigma is not None and prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive when provided.")
        if drift_mode not in DRIFT_MODES:
            raise ValueError(f"drift_mode must be one of {DRIFT_MODES}, got {drift_mode!r}.")
        if schedule not in ("polynomial", "cyclical"):
            raise ValueError(f"schedule must be 'polynomial' or 'cyclical', got {schedule!r}.")
        if schedule == "cyclical" and cycle_length_steps <= 0:
            raise ValueError("cycle_length_steps must be positive for cyclical schedule.")

        # Resolve per-mode defaults for the adaptive bias factor.
        _is_adaptive = drift_mode in ("msgld", "asgld")
        if bias_factor is None:
            bias_factor = {"msgld": 1.0, "asgld": 0.1}.get(drift_mode, 0.0)
        if _is_adaptive and momentum_beta <= 0:
            raise ValueError("momentum_beta must be positive for adaptive modes.")

        defaults = dict(
            step_size=step_size,
            final_step_size=final_step_size,
            total_steps=total_steps,
            gamma=gamma,
            noise_scale=noise_scale,
            loss_scale=loss_scale,
            prior_sigma=prior_sigma,
            drift_mode=drift_mode,
            bias_factor=bias_factor,
            momentum_beta=momentum_beta,
            adam_beta2=adam_beta2,
            ada_eps=ada_eps,
        )
        super().__init__(params, defaults)
        self.n_train = int(n_train)
        self.loss_scale = float(loss_scale)
        self.prior_sigma = float(prior_sigma) if prior_sigma is not None else None
        self.drift_mode = drift_mode
        self.bias_factor = float(bias_factor)
        self.momentum_beta = float(momentum_beta)
        self.adam_beta2 = float(adam_beta2)
        self.ada_eps = float(ada_eps)
        # welling_teh multiplies by n_train; all other modes use mean-batch
        # gradient directly (literal, msgld, asgld, psgld).
        self._grad_n_factor = float(n_train) if drift_mode == "welling_teh" else 1.0
        self._step_count = 0
        self._schedule = schedule
        self._eps_0 = float(step_size)
        self._cycle_length_steps = int(cycle_length_steps)

        # Solve polynomial-decay coefficients eps_t = a*(b+t)^(-gamma) so that
        # eps_0 = step_size and eps_{T-1} = final_step_size.
        if final_step_size is not None and total_steps is not None:
            T = total_steps - 1
            ratio = (step_size / final_step_size) ** (1.0 / gamma)
            # eps_0 / eps_T = ((b+T)/b)^gamma  =>  (b+T)/b = ratio
            #  =>  b = T / (ratio - 1)
            if ratio <= 1.0 + 1e-12:
                # final and initial nearly identical; fall back to constant
                self._a = step_size
                self._b = 1.0
            else:
                self._b = T / (ratio - 1.0)
                self._a = step_size * (self._b ** gamma)
        else:
            self._a = None
            self._b = None

    def _current_step_size(self) -> float:
        if self._schedule == "cyclical":
            t_in_cycle = self._step_count % self._cycle_length_steps
            return self._eps_0 * 0.5 * (math.cos(math.pi * t_in_cycle / self._cycle_length_steps) + 1)
        group = self.param_groups[0]
        if self._a is None:
            return float(group['step_size'])
        gamma = group['gamma']
        return float(self._a * ((self._b + self._step_count) ** (-gamma)))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        eps = self._current_step_size()
        is_adaptive = self.drift_mode in ("msgld", "asgld")
        is_precond = self.drift_mode == "psgld"
        for group in self.param_groups:
            noise_scale = group['noise_scale']
            beta1 = group['momentum_beta']
            beta2 = group['adam_beta2']
            a = group['bias_factor']
            ada_lam = group['ada_eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad * self._grad_n_factor * self.loss_scale
                if self.prior_sigma is not None:
                    grad = grad + p.data / (self.prior_sigma * self.prior_sigma)

                # --- Adaptive bias (MSGLD / ASGLD) ---
                if is_adaptive:
                    state = self.state[p]
                    if len(state) == 0:
                        state['m'] = torch.zeros_like(p)
                        if self.drift_mode == "asgld":
                            state['v'] = torch.zeros_like(p)
                        state['prev_grad'] = torch.zeros_like(p)

                    state['m'].mul_(beta1).add_(state['prev_grad'], alpha=1 - beta1)
                    if self.drift_mode == "asgld":
                        state['v'].mul_(beta2).addcmul_(
                            state['prev_grad'], state['prev_grad'], value=1 - beta2)
                        bias = state['m'] / (state['v'].sqrt() + ada_lam)
                    else:
                        bias = state['m']

                    drift = grad + a * bias
                    state['prev_grad'] = grad.clone()
                    p.data.add_(drift, alpha=-0.5 * eps)
                    if noise_scale > 0:
                        p.data.add_(torch.randn_like(p), alpha=(eps ** 0.5) * noise_scale)

                # --- Preconditioned SGLD (pSGLD, Li et al. 2016) ---
                # G = diag(sqrt(v_t) + lambda),  v_t = EMA of grad^2
                # drift: -0.5 * eps * G^{-1} * grad
                # noise: sqrt(eps) * G^{-1/2} * N(0,I)
                elif is_precond:
                    state = self.state[p]
                    if len(state) == 0:
                        state['v'] = torch.zeros_like(p)
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    g_inv = 1.0 / (state['v'].sqrt() + ada_lam)
                    p.data.add_(grad * g_inv, alpha=-0.5 * eps)
                    if noise_scale > 0:
                        p.data.add_(torch.randn_like(p) * g_inv.sqrt(),
                                    alpha=(eps ** 0.5) * noise_scale)

                # --- Standard SGLD (literal / welling_teh) ---
                else:
                    p.data.add_(grad, alpha=-0.5 * eps)
                    if noise_scale > 0:
                        p.data.add_(torch.randn_like(p), alpha=(eps ** 0.5) * noise_scale)

        self._step_count += 1
        return loss

    @property
    def step_count(self) -> int:
        return self._step_count
