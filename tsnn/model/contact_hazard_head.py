"""Contact Hazard Head (Component 4a).

Computes per-contact local instability scores from sheaf disagreement
and edge features. This is Eq. 12 from the paper:

    r_uv(t) = MLP_risk(-D_uv(t) || e_uv(t) || delta_e_uv(t))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.model.layers.mlp import MLP


class ContactHazardHead(nn.Module):
    """Predicts per-contact instability risk scores.

    Args:
        edge_dim: Edge feature dimension.
        hidden_dim: Hidden dimension for risk MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Input: -D_uv(t) [1] + e_uv(t) [edge_dim] + delta_e_uv(t) [edge_dim]
        risk_input_dim = 1 + edge_dim + edge_dim
        self.risk_mlp = MLP(
            risk_input_dim, hidden_dim, 1,
            num_layers=3, dropout=dropout,
        )

    def forward(
        self,
        disagreements: Tensor,
        edge_attr: Tensor,
        delta_edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Compute contact risk scores.

        Args:
            disagreements: Sheaf disagreement D_uv(t) per edge [E].
            edge_attr: Current edge features [E, F_edge].
            delta_edge_attr: Temporal change in edge features [E, F_edge].
                If None, uses zeros.

        Returns:
            Risk scores r_uv(t) per contact [E, 1].
        """
        neg_D = -disagreements.unsqueeze(-1)  # [E, 1] — negated per Eq. 12

        if delta_edge_attr is None:
            delta_edge_attr = torch.zeros_like(edge_attr)

        risk_input = torch.cat([neg_D, edge_attr, delta_edge_attr], dim=-1)
        risk = self.risk_mlp(risk_input)  # [E, 1]
        return risk
