from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head for contrastive SSL objectives."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=-1)


class NTXentLoss(nn.Module):
    """Symmetric NT-Xent loss used by SimCLR-style training."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        reps = torch.cat([z1, z2], dim=0)  # [2N, D]
        similarity = torch.matmul(reps, reps.T) / self.temperature

        # Remove diagonal self-similarity.
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(mask, float("-inf"))

        # Positive index for each anchor.
        positives = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=similarity.device),
                torch.arange(0, batch_size, device=similarity.device),
            ]
        )
        return F.cross_entropy(similarity, positives)


class MaskedFeatureReconstructionHead(nn.Module):
    """Masked feature modeling head for tabular SSL."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)


class JointSSLObjective(nn.Module):
    """Joint contrastive + masked reconstruction SSL objective."""

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.reconstruction_loss = nn.MSELoss()

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        reconstruction: torch.Tensor | None = None,
        reconstruction_target: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        losses = {}
        total = torch.tensor(0.0, device=z1.device)

        if self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(z1, z2)
            total = total + self.contrastive_weight * contrastive
            losses["contrastive"] = float(contrastive.item())

        if (
            self.reconstruction_weight > 0
            and reconstruction is not None
            and reconstruction_target is not None
        ):
            reconstruction_loss = self.reconstruction_loss(reconstruction, reconstruction_target)
            total = total + self.reconstruction_weight * reconstruction_loss
            losses["reconstruction"] = float(reconstruction_loss.item())

        losses["total"] = float(total.item())
        return total, losses
