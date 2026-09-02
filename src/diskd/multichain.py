"""Multi-chain SGLD wrapper for DiSKD students.

Runs M independent SGLD chains with different random seeds and concatenates
their posterior samples. The class is thin by design: each chain is just a
fresh `DiSKDStudent` (or `DiscreteSurvivalModel`) instance fit with
`optimizer='sgld'`, and the wrapper aggregates their state-dict trajectories.

Multi-chain serves two purposes here:
  1. It mirrors the paper's "20 random seeds" replicate design (each chain
     starts from a fresh initialization), so credible intervals computed from
     the combined samples can be directly compared against the paper's seed
     replicates.
  2. It reduces dependence on any one chain's mixing behavior, which SGLD
     is known to be fragile about in high-dimensional neural-net parameter
     spaces.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .models import DiSKDStudent, DiscreteSurvivalModel


class MultiChainSampler:
    """Aggregate posterior samples across multiple independent SGLD chains.

    Args:
        base_model: An *unfitted* `DiscreteSurvivalModel` or `DiSKDStudent`
            configured with `optimizer='sgld'`. The wrapper deep-copies it
            for each chain to keep configuration consistent.
        n_chains: Number of independent SGLD chains to run.
        seeds: Optional list of `torch.manual_seed` values for each chain.
            Defaults to `range(n_chains)` so runs are reproducible.

    Attributes:
        chains: List of fitted per-chain model instances. `chains[0]` is the
            "lead" chain whose `predict_*` methods are reused by the
            uncertainty helpers.
        posterior_samples: Combined list of state_dicts across all chains
            (length `n_chains * samples_per_chain`).
    """

    def __init__(
        self,
        base_model: DiscreteSurvivalModel,
        n_chains: int = 5,
        seeds: Optional[list[int]] = None,
    ):
        if n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        if base_model.optimizer != "sgld":
            raise ValueError("base_model must be configured with optimizer='sgld'.")
        self.base_model = base_model
        self.n_chains = int(n_chains)
        self.seeds = list(seeds) if seeds is not None else list(range(self.n_chains))
        if len(self.seeds) != self.n_chains:
            raise ValueError("len(seeds) must equal n_chains.")
        self.chains: list[DiscreteSurvivalModel] = []
        self.posterior_samples: list[dict] = []

    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        duration_col: str = "duration",
        event_col: str = "event",
        valid_data: pd.DataFrame | None = None,
    ) -> "MultiChainSampler":
        self.chains = []
        self.posterior_samples = []
        for chain_id, seed in enumerate(self.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            chain = copy.deepcopy(self.base_model)
            chain.fit(
                data,
                feature_cols=feature_cols,
                duration_col=duration_col,
                event_col=event_col,
                valid_data=valid_data,
            )
            self.chains.append(chain)
            self.posterior_samples.extend(chain.posterior_samples)
        return self

    @property
    def lead_chain(self) -> DiscreteSurvivalModel:
        """Return the first fitted chain (used as a prediction stub)."""
        if not self.chains:
            raise RuntimeError("MultiChainSampler is not fitted.")
        return self.chains[0]

    def __len__(self) -> int:
        return len(self.posterior_samples)


class WarmStartMultiChainSampler(MultiChainSampler):
    """Multi-chain sampler where every chain starts from a shared pretrained state_dict."""

    def __init__(
        self,
        base_model: DiscreteSurvivalModel,
        pretrained_state: dict,
        n_chains: int = 5,
        seeds: Optional[list[int]] = None,
    ):
        super().__init__(base_model, n_chains=n_chains, seeds=seeds)
        self.pretrained_state = pretrained_state

    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        duration_col: str = "duration",
        event_col: str = "event",
        valid_data: pd.DataFrame | None = None,
    ) -> "WarmStartMultiChainSampler":
        self.chains = []
        self.posterior_samples = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            chain = copy.deepcopy(self.base_model)
            chain._warm_start_state = copy.deepcopy(self.pretrained_state)
            chain.fit(
                data,
                feature_cols=feature_cols,
                duration_col=duration_col,
                event_col=event_col,
                valid_data=valid_data,
            )
            self.chains.append(chain)
            self.posterior_samples.extend(chain.posterior_samples)
        return self


def fit_multichain_diskd(
    data: pd.DataFrame,
    teacher_model,
    n_chains: int = 5,
    samples_per_chain: int = 10,
    seeds: Optional[list[int]] = None,
    feature_cols: Optional[list[str]] = None,
    duration_col: str = "duration",
    event_col: str = "event",
    **student_kwargs,
) -> MultiChainSampler:
    """Convenience constructor: configure DiSKDStudent for SGLD and fit M chains.

    Any unrecognized keyword argument is forwarded to `DiSKDStudent.__init__`.
    The optimizer is forced to `'sgld'` and `sgld_samples_per_chain` is set
    to `samples_per_chain`.
    """
    base = DiSKDStudent(
        teacher_model=teacher_model,
        optimizer="sgld",
        sgld_samples_per_chain=samples_per_chain,
        **student_kwargs,
    )
    sampler = MultiChainSampler(base, n_chains=n_chains, seeds=seeds)
    sampler.fit(
        data,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
    )
    return sampler
