"""Neural network backbones for discrete-time survival models."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_time_embedding(num_durations: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    positions = torch.arange(num_durations, device=device, dtype=dtype or torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype or torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    emb = torch.zeros(num_durations, dim, device=device, dtype=dtype or torch.float32)
    emb[:, 0::2] = torch.sin(positions * div)
    if dim > 1:
        emb[:, 1::2] = torch.cos(positions * div[: emb[:, 1::2].shape[1]])
    return emb


class MLPBackbone(nn.Module):
    """Plain MLP that outputs `[N, out_features]`."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 64, hidden_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_features
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(last, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeEmbeddingMLP(nn.Module):
    """MLP with sinusoidal time embeddings for interval-specific logits."""

    def __init__(self, in_features: int, num_durations: int, num_risks: int = 1, hidden_dim: int = 64, hidden_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_durations = num_durations
        self.num_risks = num_risks
        self.feature_projection = nn.Linear(in_features, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(hidden_layers):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            if dropout:
                blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, num_risks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(self.num_durations, h.shape[-1], device=x.device, dtype=x.dtype)
        h = h + emb.unsqueeze(0)
        h = self.blocks(h)
        out = self.head(h)
        if self.num_risks == 1:
            return out.squeeze(-1)
        return out.permute(0, 2, 1)


class TransformerBackbone(nn.Module):
    """Transformer encoder over discrete time embeddings."""

    def __init__(
        self,
        in_features: int,
        num_durations: int,
        num_risks: int = 1,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_durations = num_durations
        self.num_risks = num_risks
        self.feature_projection = nn.Linear(in_features, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=hidden_layers)
        self.head = nn.Linear(hidden_dim, num_risks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_projection(x).unsqueeze(1)
        emb = sinusoidal_time_embedding(self.num_durations, h.shape[-1], device=x.device, dtype=x.dtype)
        h = h + emb.unsqueeze(0)
        h = self.encoder(h)
        out = self.head(h)
        if self.num_risks == 1:
            return out.squeeze(-1)
        return out.permute(0, 2, 1)


def build_backbone(
    backbone: str,
    in_features: int,
    num_durations: int,
    num_risks: int,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    dropout: float = 0.1,
    nhead: int = 4,
) -> nn.Module:
    if backbone == "mlp":
        return MLPBackbone(in_features, num_risks * num_durations, hidden_dim, hidden_layers, dropout)
    if backbone == "time_mlp":
        return TimeEmbeddingMLP(in_features, num_durations, num_risks, hidden_dim, hidden_layers, dropout)
    if backbone == "transformer":
        return TransformerBackbone(
            in_features,
            num_durations,
            num_risks,
            hidden_dim,
            hidden_layers,
            nhead=nhead,
            dropout=dropout,
        )
    raise ValueError("backbone must be 'mlp', 'time_mlp', or 'transformer'.")
