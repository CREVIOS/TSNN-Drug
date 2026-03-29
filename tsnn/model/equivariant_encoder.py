"""E(3)-equivariant local encoder (Component 2).

Encodes each MD frame's local geometry into invariant node states h_v^(0)(t)
using an EGNN-style message passing network. This is Eq. 7 in the paper:

    h_v^(0)(t) = phi_enc({x_u(t), x_v(t), d_uv(t), e_uv(t)} for u in N(v))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.model.layers.egnn_layer import EGNNLayer
from tsnn.model.layers.mlp import MLP


class EquivariantEncoder(nn.Module):
    """E(3)-equivariant graph encoder using stacked EGNN layers.

    Processes a single frame graph and produces invariant node embeddings
    suitable for the temporal sheaf transport block.

    Args:
        node_input_dim: Raw node feature dimension.
        edge_input_dim: Raw edge feature dimension.
        hidden_dim: Internal and output hidden dimension.
        num_layers: Number of EGNN layers.
        dropout: Dropout probability.
        update_coords: Whether to update coordinates through layers.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        update_coords: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project raw features to hidden dim
        self.node_embed = MLP(node_input_dim, hidden_dim, hidden_dim,
                              num_layers=1)
        self.edge_embed = MLP(edge_input_dim, hidden_dim, hidden_dim,
                              num_layers=1) if edge_input_dim > 0 else None

        # Stack of EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim if edge_input_dim > 0 else 0,
                    update_coords=update_coords and (i < num_layers - 1),
                    norm=True,
                )
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Encode a single frame.

        Args:
            node_features: Raw node features [N, F_node].
            positions: Atomic/residue coordinates [N, 3].
            edge_index: Graph connectivity [2, E].
            edge_attr: Raw edge features [E, F_edge].

        Returns:
            Node embeddings h [N, hidden_dim] — invariant under E(3).
        """
        # Clamp inputs for numerical stability under mixed precision
        node_features = node_features.clamp(-100, 100)
        h = self.node_embed(node_features)
        x = positions.clamp(-500, 500).clone()

        if self.edge_embed is not None and edge_attr is not None:
            ea = self.edge_embed(edge_attr)
        else:
            ea = None

        for layer in self.layers:
            h, x = layer(h, x, edge_index, ea)
            h = self.dropout(h)

        return h
