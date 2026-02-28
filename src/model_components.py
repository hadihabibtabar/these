from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for tabular field tokens."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, D]
        return x + self.pe[:, : x.size(1), :]


class TransformerExpert(nn.Module):
    """Transformer expert that consumes embedded feature tokens."""

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int,
        expert_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_positional_encoding: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by num_heads={num_heads}."
            )
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be one of {'mean', 'cls'}")

        self.pooling = pooling
        self.positional_encoding = (
            PositionalEncoding(embedding_dim, num_fields + (1 if pooling == "cls" else 0))
            if use_positional_encoding
            else None
        )
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embedding_dim)) if pooling == "cls" else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, expert_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, D]
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        hidden = self.encoder(x)
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            pooled = hidden.mean(dim=1)

        pooled = self.dropout(self.out_norm(pooled))
        return self.out_proj(pooled)


class MLPExpert(nn.Module):
    """MLP expert baseline for ablation against transformer experts."""

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int,
        expert_dim: int,
        hidden_units: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = embedding_dim * num_fields
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_units:
            layers.extend([
                nn.Linear(prev, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.out_proj = nn.Linear(prev, expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        return self.out_proj(self.backbone(x))


class TaskGate(nn.Module):
    """Task-specific gate: softmax over experts."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits


class TaskTower(nn.Module):
    """Task-specific prediction tower."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_units:
            hidden_layers.extend([
                nn.Linear(prev, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = width
        self.backbone = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()
        self.output = nn.Linear(prev, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        logits = self.output(hidden)
        return logits, hidden


@dataclass
class GatingDiagnostics:
    gate_probabilities: torch.Tensor  # [T, B, E]
    task_entropy: torch.Tensor        # [T]
    expert_utilization: torch.Tensor  # [T, E]
    load_balance_loss: torch.Tensor   # scalar


def summarize_gates(gate_probabilities: torch.Tensor, eps: float = 1e-8) -> GatingDiagnostics:
    """Summarize gate entropy and expert utilization for regularization/monitoring."""
    entropy = -(gate_probabilities * (gate_probabilities + eps).log()).sum(dim=-1).mean(dim=-1)
    utilization = gate_probabilities.mean(dim=1)
    uniform = torch.full_like(utilization, 1.0 / utilization.size(-1))
    balance = torch.mean((utilization - uniform) ** 2)
    return GatingDiagnostics(
        gate_probabilities=gate_probabilities,
        task_entropy=entropy,
        expert_utilization=utilization,
        load_balance_loss=balance,
    )
