"""Sheaf smoothness regularizer.

Penalizes total sheaf disagreement to prevent transport maps from
diverging unnecessarily:

    L_sheaf = (1 / |E|T) sum_{(u,v), t} D_uv(t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SheafSmoothnessLoss(nn.Module):
    """Sheaf Laplacian smoothness regularizer.

    Encourages the learned transport maps to maintain local consistency
    while still allowing disagreement to grow where genuinely needed
    (approaching dissociation).
    """

    def forward(self, disagreements: list[Tensor]) -> Tensor:
        """Compute sheaf smoothness loss.

        Args:
            disagreements: List of per-frame disagreement tensors [E_t].

        Returns:
            Scalar loss — mean disagreement over all edges and time steps.
        """
        if not disagreements:
            return torch.tensor(0.0, requires_grad=True)

        total = torch.tensor(0.0, device=disagreements[0].device)
        count = 0

        for D_t in disagreements:
            if D_t.numel() > 0:
                total = total + D_t.sum()
                count += D_t.numel()

        if count == 0:
            return torch.tensor(0.0, device=disagreements[0].device, requires_grad=True)

        return total / count
