"""Lightweight model wrappers for DiSKD."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import (
    BinaryHorizonToCompetingRiskKDLoss,
    CompetingRiskKDLoss,
    CompetingRiskNLLLoss,
    OverallToCompetingRiskKDLoss,
    SingleRiskKDLoss,
    SingleRiskNLLLoss,
)
from .networks import build_backbone
from .preprocessing import FeaturePreprocessor, TimeGrid, fit_time_grid, transform_durations
from .utils import competing_cif, competing_interval_probs, competing_survival


@dataclass
class FitHistory:
    losses: list[float]
    val_losses: list[float] | None = None


class DiscreteSurvivalModel:
    """A small discrete-time neural survival model.

    This wrapper is intentionally minimal and aimed at examples. Advanced
    experiments can use the loss functions directly.
    """

    def __init__(
        self,
        num_risks: int = 1,
        num_durations: int = 20,
        backbone: str = "time_mlp",
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 20,
        optimizer: str = "adamw",
        device: str | torch.device | None = None,
        time_grid: TimeGrid | None = None,
        sgld_step_size: float = 3e-7,
        sgld_final_step_size: float | None = 3e-9,
        sgld_gamma: float = 1.0,
        sgld_burnin_epochs: int | None = None,
        sgld_samples_per_chain: int = 10,
        sgld_noise_scale: float = 1.0,
        sgld_prior_sigma: float | None = None,
        sgld_drift_mode: str = "welling_teh",
        sgld_bias_factor: float | None = None,
        sgld_momentum_beta: float = 0.9,
        sgld_adam_beta2: float = 0.999,
        sgld_schedule: str = "polynomial",
        sgld_cycle_length: int = 50,
        sgld_burn_in_cycles: int | None = None,
        sgld_draws_per_cycle: int = 1,
    ):
        if num_risks <= 0:
            raise ValueError("num_risks must be positive.")
        self.num_risks = int(num_risks)
        self.num_durations = int(num_durations)
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.nhead = nhead
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer.lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.time_grid = time_grid
        self.preprocessor: FeaturePreprocessor | None = None
        self.net: torch.nn.Module | None = None
        self.history: FitHistory | None = None
        # SGLD-specific configuration (used only when optimizer == 'sgld')
        self.sgld_step_size = float(sgld_step_size)
        self.sgld_final_step_size = sgld_final_step_size
        self.sgld_gamma = float(sgld_gamma)
        self.sgld_burnin_epochs = sgld_burnin_epochs
        self.sgld_samples_per_chain = int(sgld_samples_per_chain)
        self.sgld_noise_scale = float(sgld_noise_scale)
        self.sgld_prior_sigma = float(sgld_prior_sigma) if sgld_prior_sigma is not None else None
        self.sgld_drift_mode = str(sgld_drift_mode)
        self.sgld_bias_factor = sgld_bias_factor
        self.sgld_momentum_beta = float(sgld_momentum_beta)
        self.sgld_adam_beta2 = float(sgld_adam_beta2)
        self.sgld_schedule = str(sgld_schedule)
        self.sgld_cycle_length = int(sgld_cycle_length)
        self.sgld_burn_in_cycles = sgld_burn_in_cycles
        self.sgld_draws_per_cycle = int(sgld_draws_per_cycle)
        # Populated during fit() when optimizer == 'sgld'.
        self.posterior_samples: list[dict] = []

    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        duration_col: str = "duration",
        event_col: str = "event",
        valid_data: pd.DataFrame | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
    ) -> "DiscreteSurvivalModel":
        x, idx, events = self._prepare_fit_data(data, feature_cols, duration_col, event_col)
        valid_tensors = self._prepare_validation_data(valid_data, duration_col, event_col)
        self._fit_tensors(
            x,
            idx,
            events,
            valid_tensors=valid_tensors,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        return self

    def _prepare_fit_data(self, data: pd.DataFrame, feature_cols, duration_col: str, event_col: str):
        if feature_cols is None:
            feature_cols = [c for c in data.columns if c not in {duration_col, event_col}]
        self.feature_cols = list(feature_cols)
        self.time_grid = self.time_grid or fit_time_grid(data[duration_col].values, self.num_durations)
        idx = transform_durations(data[duration_col].values, self.time_grid)
        events = data[event_col].to_numpy(dtype=np.int64)
        self.preprocessor = FeaturePreprocessor(numeric_cols=self.feature_cols)
        x = self.preprocessor.fit_transform(data)
        return x, idx, events

    def _prepare_validation_data(
        self,
        data: pd.DataFrame | None,
        duration_col: str,
        event_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if data is None:
            return None
        if self.time_grid is None or self.preprocessor is None:
            raise RuntimeError("Fit preprocessing before preparing validation data.")
        x = self.preprocessor.transform(data)
        idx = transform_durations(data[duration_col].values, self.time_grid)
        events = data[event_col].to_numpy(dtype=np.int64)
        return x, idx, events

    def _build_net(self, in_features: int):
        self.net = build_backbone(
            self.backbone,
            in_features,
            self.num_durations,
            self.num_risks,
            self.hidden_dim,
            self.hidden_layers,
            self.dropout,
            self.nhead,
        ).to(self.device)

    def _sgld_loss_scale(self) -> float:
        """Per-loss-function gradient scale to recover the R01 Eq 4 posterior.

        DiSKD students apply `(NLL + eta*KD)/(1+eta)` per-sample normalization
        in their KD losses, so `loss_scale = 1 + eta` undoes the (1+eta) factor
        and makes the SGLD target distribution match the generalized posterior
        with omega = 1. Plain `DiscreteSurvivalModel` uses unnormalized NLL,
        so the default is 1.0.
        """
        return 1.0

    def _build_optimizer(self, n_train: int | None = None, total_steps: int | None = None):
        if self.net is None:
            raise RuntimeError("Network is not built.")
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        if self.optimizer == "adam":
            return torch.optim.Adam(self.net.parameters(), lr=self.lr)
        if self.optimizer == "sgld":
            from .samplers import SGLD

            if n_train is None:
                raise RuntimeError("SGLD requires n_train; called from _fit_tensors.")
            steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size) if n_train else 1
            return SGLD(
                self.net.parameters(),
                step_size=self.sgld_step_size,
                n_train=n_train,
                final_step_size=self.sgld_final_step_size,
                total_steps=total_steps,
                gamma=self.sgld_gamma,
                noise_scale=self.sgld_noise_scale,
                loss_scale=self._sgld_loss_scale(),
                prior_sigma=self.sgld_prior_sigma,
                drift_mode=self.sgld_drift_mode,
                bias_factor=self.sgld_bias_factor,
                momentum_beta=self.sgld_momentum_beta,
                adam_beta2=self.sgld_adam_beta2,
                schedule=self.sgld_schedule,
                cycle_length_steps=self.sgld_cycle_length * steps_per_epoch,
            )
        raise ValueError("optimizer must be 'adamw', 'adam', or 'sgld'.")

    def _fit_tensors(
        self,
        x: np.ndarray,
        idx: np.ndarray,
        events: np.ndarray,
        loss_fn=None,
        teacher_tensor=None,
        valid_tensors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
    ):
        self._build_net(x.shape[1])
        assert self.net is not None
        if getattr(self, '_warm_start_state', None) is not None:
            self.net.load_state_dict(self._warm_start_state)
            self._warm_start_state = None
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None.")
        loss_fn = loss_fn or (SingleRiskNLLLoss() if self.num_risks == 1 else CompetingRiskNLLLoss())

        tensors = [
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
            torch.tensor(events, dtype=torch.long),
        ]
        if teacher_tensor is not None:
            tensors.append(torch.tensor(teacher_tensor, dtype=torch.float32))
        loader = DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=True)
        n_train = int(tensors[0].shape[0])
        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
        total_steps = steps_per_epoch * self.epochs
        optimizer = self._build_optimizer(n_train=n_train, total_steps=total_steps)

        # SGLD trajectory configuration.
        is_sgld = self.optimizer == "sgld"
        is_cyclical = is_sgld and self.sgld_schedule == "cyclical"
        if is_sgld:
            self.posterior_samples = []
            if is_cyclical:
                cycle_steps = self.sgld_cycle_length * steps_per_epoch
                n_cycles = max(1, total_steps // cycle_steps)
                burn_in_cycles = (
                    self.sgld_burn_in_cycles
                    if self.sgld_burn_in_cycles is not None
                    else max(1, n_cycles // 4)
                )
                # Collection window: last 10% of each post-burn-in cycle.
                collection_start = int(0.9 * cycle_steps)
                draws_per_cycle = self.sgld_draws_per_cycle
                if draws_per_cycle > 1:
                    _collect_thin = max(1, (cycle_steps - collection_start) // draws_per_cycle)
                else:
                    _collect_thin = 1
                max_draws = (n_cycles - burn_in_cycles) * draws_per_cycle
                # These are unused in cyclical mode but set for the else-branch.
                burnin_steps = 0
                thin = 1
            else:
                burnin_epochs = (
                    self.sgld_burnin_epochs
                    if self.sgld_burnin_epochs is not None
                    else max(1, self.epochs // 2)
                )
                burnin_steps = burnin_epochs * steps_per_epoch
                sample_steps = max(0, total_steps - burnin_steps)
                if self.sgld_samples_per_chain <= 0:
                    raise ValueError("sgld_samples_per_chain must be positive.")
                thin = max(1, sample_steps // self.sgld_samples_per_chain) if sample_steps > 0 else 1
                max_draws = self.sgld_samples_per_chain
            if early_stopping_patience is not None:
                raise ValueError("early_stopping_patience must be None for SGLD.")
        else:
            burnin_steps = 0
            thin = 1
            is_cyclical = False

        losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.net.train()
        global_step = 0
        for _ in range(self.epochs):
            epoch_losses = []
            for batch in loader:
                xb, idxb, eb = [t.to(self.device) for t in batch[:3]]
                tb = batch[3].to(self.device) if len(batch) == 4 else None
                optimizer.zero_grad()
                logits = self._reshape_logits(self.net(xb))
                if tb is None:
                    loss = loss_fn(logits, idxb, eb)
                else:
                    loss = loss_fn(logits, idxb, eb, tb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                global_step += 1
                _should_collect = False
                if is_sgld and len(self.posterior_samples) < max_draws:
                    if is_cyclical:
                        _cyc_idx = global_step // cycle_steps
                        _t_in_cyc = global_step % cycle_steps
                        _in_sampling = _cyc_idx >= burn_in_cycles
                        _in_tail = _t_in_cyc >= collection_start
                        if draws_per_cycle == 1:
                            _at_cycle_end = _t_in_cyc == cycle_steps - 1
                            _should_collect = _in_sampling and _at_cycle_end
                        else:
                            _tail_offset = _t_in_cyc - collection_start
                            _should_collect = (
                                _in_sampling and _in_tail
                                and _tail_offset % _collect_thin == 0
                            )
                    else:
                        _should_collect = (
                            global_step > burnin_steps
                            and (global_step - burnin_steps - 1) % thin == 0
                        )
                if _should_collect:
                    self.posterior_samples.append(
                        {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    )
            losses.append(float(np.mean(epoch_losses)))

            if valid_tensors is None:
                continue

            val_loss = self._validation_nll_from_tensors(valid_tensors)
            val_losses.append(val_loss)
            if early_stopping_patience is None:
                continue

            if val_loss < best_val - early_stopping_min_delta:
                best_val = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = FitHistory(losses=losses, val_losses=val_losses or None)

    @torch.no_grad()
    def _validation_nll_from_tensors(self, valid_tensors: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        if self.net is None:
            raise RuntimeError("Network is not built.")
        x, idx, events = valid_tensors
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
        event_tensor = torch.tensor(events, dtype=torch.long, device=self.device)
        self.net.eval()
        logits = self._reshape_logits(self.net(x_tensor))
        loss_fn = SingleRiskNLLLoss() if self.num_risks == 1 else CompetingRiskNLLLoss()
        loss = loss_fn(logits, idx_tensor, event_tensor)
        self.net.train()
        return float(loss.detach().cpu())

    def _reshape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_risks == 1:
            return logits
        if logits.ndim == 2:
            return logits.view(logits.shape[0], self.num_risks, self.num_durations)
        return logits

    def _transform_features(self, data: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model is not fitted.")
        return self.preprocessor.transform(data)

    @torch.no_grad()
    def predict_logits(self, data: pd.DataFrame) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("Model is not fitted.")
        x = torch.tensor(self._transform_features(data), dtype=torch.float32, device=self.device)
        self.net.eval()
        return self._reshape_logits(self.net(x)).detach().cpu()

    @torch.no_grad()
    def predict_interval_probs(self, data: pd.DataFrame) -> torch.Tensor:
        logits = self.predict_logits(data)
        if self.num_risks == 1:
            hazard = torch.sigmoid(logits)
            return torch.stack([hazard, 1.0 - hazard], dim=1)
        return competing_interval_probs(logits)

    @torch.no_grad()
    def predict_hazard(self, data: pd.DataFrame) -> np.ndarray:
        """Predict discrete interval event hazards.

        Returns `[N, K]` for single-risk models and `[N, J, K]` for
        competing-risk models. For competing risks, entries are the
        cause-specific discrete hazards; the no-event category is omitted.
        """
        probs = self.predict_interval_probs(data)
        if self.num_risks == 1:
            return probs[:, 0, :].numpy()
        return probs[:, :-1, :].numpy()

    @torch.no_grad()
    def predict_survival(self, data: pd.DataFrame) -> np.ndarray:
        probs = self.predict_interval_probs(data)
        if self.num_risks == 1:
            survival = torch.cumprod(probs[:, 1, :], dim=1)
        else:
            survival = competing_survival(probs)
        return survival.numpy()

    @torch.no_grad()
    def predict_cif(self, data: pd.DataFrame, risk: int | None = None) -> np.ndarray:
        probs = self.predict_interval_probs(data)
        if self.num_risks == 1:
            out = 1.0 - torch.cumprod(probs[:, 1, :], dim=1)
            return out.numpy()
        cif = competing_cif(probs)
        if risk is not None:
            if not (1 <= risk <= self.num_risks):
                raise ValueError("risk must be one-indexed in 1..num_risks.")
            return cif[:, risk - 1, :].numpy()
        return cif.numpy()


class DiSKDStudent(DiscreteSurvivalModel):
    """Discrete survival student trained with teacher prediction guidance."""

    def __init__(
        self,
        *args,
        teacher_model=None,
        teacher_type: str = "competing",
        eta: float = 1.0,
        temperature: float = 1.0,
        binary_risk_index: int = 0,
        binary_horizon_index: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_type = teacher_type
        self.eta = eta
        self.temperature = temperature
        self.binary_risk_index = binary_risk_index
        self.binary_horizon_index = binary_horizon_index

    def _sgld_loss_scale(self) -> float:
        # DiSKD KD losses are normalized as `(NLL + eta*KD)/(1+eta)`; rescale
        # gradients by (1 + eta) to recover the omega = 1 generalized posterior
        # from R01 Eq 4 (C.2.1.4). With eta = 0 this reduces to 1.0 (plain NLL).
        return 1.0 + float(self.eta)

    def fit(
        self,
        data: pd.DataFrame,
        feature_cols: list[str] | None = None,
        duration_col: str = "duration",
        event_col: str = "event",
        valid_data: pd.DataFrame | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
    ) -> "DiSKDStudent":
        x, idx, events = self._prepare_fit_data(data, feature_cols, duration_col, event_col)
        valid_tensors = self._prepare_validation_data(valid_data, duration_col, event_col)
        loss_fn, teacher_tensor = self._teacher_loss_and_tensor(data)
        self._fit_tensors(
            x,
            idx,
            events,
            loss_fn=loss_fn,
            teacher_tensor=teacher_tensor,
            valid_tensors=valid_tensors,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        return self

    def _teacher_loss_and_tensor(self, data: pd.DataFrame):
        if self.eta == 0:
            base = SingleRiskNLLLoss() if self.num_risks == 1 else CompetingRiskNLLLoss()
            return base, None
        if self.teacher_model is None:
            raise ValueError("teacher_model is required when eta > 0.")

        if self.num_risks == 1:
            teacher_probs = self.teacher_model.predict_interval_probs(data)
            return SingleRiskKDLoss(self.eta), teacher_probs[:, 0, :].numpy()

        if self.teacher_type == "competing":
            teacher_probs = self.teacher_model.predict_interval_probs(data)
            return CompetingRiskKDLoss(self.eta, self.temperature), teacher_probs[:, : self.num_risks, :].numpy()
        if self.teacher_type == "overall":
            teacher_probs = self.teacher_model.predict_interval_probs(data)
            overall = teacher_probs[:, 0, :] if teacher_probs.shape[1] == 2 else teacher_probs[:, :-1, :].sum(dim=1)
            return OverallToCompetingRiskKDLoss(self.eta), overall.numpy()
        if self.teacher_type == "binary_horizon":
            if hasattr(self.teacher_model, "predict_proba"):
                if self.preprocessor is None:
                    raise RuntimeError("Student preprocessor must be fitted before binary teacher prediction.")
                x_teacher = self.preprocessor.transform(data)
                pred = self.teacher_model.predict_proba(x_teacher)[:, 1]
            elif callable(self.teacher_model):
                pred = self.teacher_model(data)
            else:
                raise ValueError("binary_horizon teacher must be callable or expose predict_proba.")
            return (
                BinaryHorizonToCompetingRiskKDLoss(
                    self.eta,
                    risk_index=self.binary_risk_index,
                    horizon_index=self.binary_horizon_index,
                ),
                np.asarray(pred),
            )
        raise ValueError("teacher_type must be 'competing', 'overall', or 'binary_horizon'.")
