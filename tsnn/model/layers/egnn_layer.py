"""E(3)-Equivariant Graph Neural Network Layer.

Implements the EGNN layer from Satorras et al. (2021), which performs
equivariant message passing that respects E(3) symmetry: rotations,
reflections, and translations.

Scalar features h are invariant; coordinate features x are equivariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.model.layers.mlp import MLP


class EGNNLayer(nn.Module):
    """Single E(3)-equivariant message passing layer.

    Updates both invariant node features h and equivariant positions x.

    Args:
        hidden_dim: Dimension of node features.
        edge_dim: Dimension of edge features (0 if none).
        activation: Activation class.
        update_coords: Whether to update coordinates (set False for final layer).
        coord_weight_clamp: Clamp coordinate updates for stability.
        norm: Use LayerNorm in MLPs.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        activation: type[nn.Module] = nn.SiLU,
        update_coords: bool = True,
        coord_weight_clamp: float = 100.0,
        norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.coord_weight_clamp = coord_weight_clamp

        # Message MLP: takes h_i, h_j, ||x_i - x_j||^2, edge_attr
        msg_in = 2 * hidden_dim + 1 + edge_dim
        self.msg_mlp = MLP(msg_in, hidden_dim, hidden_dim, num_layers=2,
                           activation=activation, norm=norm)

        # Node update MLP
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=2,
                            activation=activation, norm=norm)

        # Coordinate update (scalar weight per message)
        if update_coords:
            self.coord_mlp = MLP(hidden_dim, hidden_dim, 1, num_layers=2,
                                 activation=activation, norm=False)

        # Attention weight for messages
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            h: Node features [N, D].
            x: Node positions [N, 3].
            edge_index: [2, E].
            edge_attr: Optional edge features [E, edge_dim].

        Returns:
            Updated (h, x).
        """
        row, col = edge_index  # row=source, col=target
        diff = x[row] - x[col]  # [E, 3]
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)  # [E, 1]

        # Build message input
        msg_input = [h[row], h[col], dist_sq]
        if edge_attr is not None:
            msg_input.append(edge_attr)
        msg_input = torch.cat(msg_input, dim=-1)

        # Compute messages
        msg = self.msg_mlp(msg_input)  # [E, D]

        # Attention-weighted messages
        att = self.att_mlp(msg)  # [E, 1]
        msg = msg * att

        # Aggregate messages per node
        agg = torch.zeros_like(h)
        agg.index_add_(0, col, msg)

        # Update node features
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = h + h_new  # Residual connection

        # Update coordinates (equivariant)
        x_new = x
        if self.update_coords:
            coord_weights = self.coord_mlp(msg)  # [E, 1]
            coord_weights = coord_weights.clamp(-self.coord_weight_clamp,
                                                 self.coord_weight_clamp)
            # Weighted sum of displacement vectors -> equivariant
            coord_delta = torch.zeros_like(x)
            coord_delta.index_add_(0, col, diff * coord_weights)
            x_new = x + coord_delta

        return h_new, x_new
