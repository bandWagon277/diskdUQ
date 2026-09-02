"""Preprocessing utilities for simulation tutorials and model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TimeGrid:
    """Discrete time grid represented by right endpoints."""

    cuts: np.ndarray

    @property
    def num_durations(self) -> int:
        return int(len(self.cuts))


def fit_time_grid(durations, num_durations: int, scheme: str = "quantiles") -> TimeGrid:
    """Fit discrete-time cut points from observed durations."""
    if num_durations <= 0:
        raise ValueError("num_durations must be positive.")
    values = np.asarray(durations, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("durations must be a nonempty one-dimensional array.")
    if np.any(~np.isfinite(values)):
        raise ValueError("durations must be finite.")

    if scheme == "quantiles":
        probs = np.linspace(0.0, 1.0, num_durations + 1)[1:]
        cuts = np.quantile(values, probs)
    elif scheme == "equidistant":
        cuts = np.linspace(values.min(), values.max(), num_durations + 1)[1:]
    else:
        raise ValueError("scheme must be 'quantiles' or 'equidistant'.")

    cuts = np.maximum.accumulate(cuts)
    cuts[-1] = max(cuts[-1], values.max())
    return TimeGrid(cuts=cuts.astype(float))


def transform_durations(durations, time_grid: TimeGrid) -> np.ndarray:
    """Map continuous durations to integer interval indices in [0, K - 1]."""
    values = np.asarray(durations, dtype=float)
    idx = np.searchsorted(time_grid.cuts, values, side="left")
    return np.clip(idx, 0, time_grid.num_durations - 1).astype(np.int64)


class FeaturePreprocessor:
    """Small DataFrame feature preprocessor.

    Numeric columns are standardized by default. Categorical columns are one-hot
    encoded when provided. Columns listed in `passthrough_cols` are kept as-is.
    """

    def __init__(
        self,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        passthrough_cols: list[str] | None = None,
    ):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.passthrough_cols = passthrough_cols or []
        self.transformer: ColumnTransformer | None = None

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        transformers = []
        if self.numeric_cols:
            transformers.append(("numeric", StandardScaler(), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols))
        if self.passthrough_cols:
            transformers.append(("passthrough", "passthrough", self.passthrough_cols))
        if not transformers:
            raise ValueError("At least one feature column is required.")
        self.transformer = ColumnTransformer(transformers)
        self.transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Call fit before transform.")
        out = self.transformer.transform(df)
        if hasattr(out, "toarray"):
            out = out.toarray()
        return np.asarray(out, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

