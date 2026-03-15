"""Temporal Sheaf Transport Block (Component 3).

Implements the core novel contribution: temporal sheaf transport on MD
trajectories. This block processes a sequence of per-frame node embeddings
using learned orthogonal transport maps (Householder reflections) and
GRU-based temporal updates.

Key equations from the paper:
- Eq. 8:  U_v(t) via Householder composition, Q_uv(t) = U_v^T U_u
- Eq. 10: Transported messages m_{u<-v}(t) = MLP(h_u || Q_uv h_v || e_uv)
- Eq. 11: h_v(t) = GRU(h_v(t^-), sum_u m_{v<-u}(t))
- Eq. 1:  D_uv(t) = ||h_u(t) - Q_uv(t) h_v(t)||^2  (sheaf disagreement)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from tsnn.utils.scatter import scatter_add

from tsnn.model.householder import HouseholderFrameBuilder
from tsnn.model.layers.mlp import MLP
from tsnn.model.layers.sheaf_gru import SheafGRU


@dataclass
class SheafTransportOutput:
    """Output of the temporal sheaf transport block."""
    h_final: Tensor              # Final node states [N, D]
    h_sequence: list[Tensor]     # Per-frame node states
    disagreements: list[Tensor]  # Per-frame edge disagreements [E_t]
    transport_maps: list[Tensor] # Per-frame Q_uv [E_t, D, D]


class TemporalSheafTransport(nn.Module):
    """Temporal sheaf transport block with Householder orthogonal maps.

    Processes a sequence of per-frame graphs, maintaining temporal state
    via GRU updates with sheaf-transported messages.

    Args:
        hidden_dim: Node embedding dimension.
        edge_dim: Edge feature dimension.
        householder_depth: Number of Householder reflections (k).
        num_sheaf_layers: Number of sheaf message-passing iterations per frame.
        dropout: Dropout probability.
        static_frames: If True, use time-invariant frames (ablation 2).
        identity_transport: If True, set Q_uv=I (ablation 1).
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        householder_depth: int = 4,
        num_sheaf_layers: int = 1,
        dropout: float = 0.0,
        static_frames: bool = False,
        identity_transport: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_frames = static_frames
        self.identity_transport = identity_transport

        # Householder frame builder (learns orthogonal transport maps)
        if not identity_transport:
            self.frame_builder = HouseholderFrameBuilder(
                hidden_dim, householder_depth
            )

        # Message MLP: h_u(t^-) || Q_uv h_v(t^-) || e_uv(t)
        msg_input_dim = 2 * hidden_dim + edge_dim
        self.message_mlp = MLP(
            msg_input_dim, hidden_dim, hidden_dim,
            num_layers=2, dropout=dropout,
        )

        # GRU for temporal update
        self.gru = SheafGRU(hidden_dim, hidden_dim, norm=True, dropout=dropout)

        self.num_sheaf_layers = num_sheaf_layers

    def _compute_transport(
        self, h: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Compute transport maps Q_uv for current frame.

        Returns:
            Q: Transport maps [E, D, D].
        """
        if self.identity_transport:
            E = edge_index.shape[1]
            return torch.eye(
                self.hidden_dim, device=h.device, dtype=h.dtype
            ).unsqueeze(0).expand(E, -1, -1)

        _, Q = self.frame_builder(h, edge_index)
        return Q

    def _transported_message(
        self,
        h: Tensor,
        Q: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
    ) -> Tensor:
        """Compute transported messages (Eq. 10).

        Args:
            h: Current node states [N, D].
            Q: Transport maps [E, D, D].
            edge_index: [2, E].
            edge_attr: Optional edge features [E, F].

        Returns:
            Aggregated messages per node [N, D].
        """
        row, col = edge_index  # row=target u, col=source v
        h_source = h[col]  # [E, D]

        # Transport: Q_uv @ h_v
        h_transported = torch.bmm(Q, h_source.unsqueeze(-1)).squeeze(-1)  # [E, D]

        # Concatenate for message MLP
        msg_parts = [h[row], h_transported]
        if edge_attr is not None:
            msg_parts.append(edge_attr)
        msg_input = torch.cat(msg_parts, dim=-1)

        messages = self.message_mlp(msg_input)  # [E, D]

        # Scatter-add to target nodes
        N = h.shape[0]
        agg = scatter_add(messages, row, dim=0, dim_size=N)  # [N, D]
        return agg

    def _compute_disagreement(
        self, h: Tensor, Q: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Compute sheaf disagreement D_uv(t) (Eq. 1).

        D_uv(t) = ||h_u(t) - Q_uv(t) h_v(t)||^2

        Args:
            h: Node states [N, D].
            Q: Transport maps [E, D, D].
            edge_index: [2, E].

        Returns:
            Disagreement per edge [E].
        """
        row, col = edge_index
        h_u = h[row]  # [E, D]
        h_v = h[col]  # [E, D]

        # Q_uv @ h_v
        Qh_v = torch.bmm(Q, h_v.unsqueeze(-1)).squeeze(-1)  # [E, D]

        # ||h_u - Q_uv h_v||^2
        diff = h_u - Qh_v
        disagreement = (diff * diff).sum(dim=-1)  # [E]
        return disagreement

    def forward(
        self,
        h_sequence: list[Tensor],
        edge_index_sequence: list[Tensor],
        edge_attr_sequence: list[Tensor] | None = None,
        num_nodes_per_frame: list[int] | None = None,
    ) -> SheafTransportOutput:
        """Process a temporal sequence of frame embeddings.

        Args:
            h_sequence: List of per-frame node embeddings [N_t, D].
            edge_index_sequence: List of per-frame edge indices [2, E_t].
            edge_attr_sequence: Optional list of per-frame edge features.
            num_nodes_per_frame: Number of nodes per frame (for variable sizes).

        Returns:
            SheafTransportOutput with final states, sequences, and disagreements.
        """
        T = len(h_sequence)
        h_prev = h_sequence[0]

        all_h = []
        all_disagreements = []
        all_Q = []

        # For static frames ablation: compute U matrices once from first frame
        # but recompute Q per-frame since edge topology may change
        if self.static_frames and not self.identity_transport:
            U_static, _ = self.frame_builder(h_prev, edge_index_sequence[0])

        for t in range(T):
            h_t = h_sequence[t]
            ei_t = edge_index_sequence[t]
            ea_t = edge_attr_sequence[t] if edge_attr_sequence else None

            # Fuse per-frame encoder output with temporal state via residual
            # This ensures gradients flow to every h_sequence[t]
            h_fused = h_prev + h_t

            # Compute transport maps
            if self.static_frames and not self.identity_transport:
                from tsnn.model.householder import compute_transport_maps
                Q_t = compute_transport_maps(U_static, ei_t)
            else:
                Q_t = self._compute_transport(h_fused, ei_t)

            # Multiple sheaf message-passing iterations per frame
            h_current = h_fused
            for _ in range(self.num_sheaf_layers):
                agg = self._transported_message(h_current, Q_t, ei_t, ea_t)
                h_current = self.gru(agg, h_current)

            # Compute sheaf disagreement on updated states
            D_t = self._compute_disagreement(h_current, Q_t, ei_t)

            all_h.append(h_current)
            all_disagreements.append(D_t)
            all_Q.append(Q_t)

            h_prev = h_current

        return SheafTransportOutput(
            h_final=all_h[-1],
            h_sequence=all_h,
            disagreements=all_disagreements,
            transport_maps=all_Q,
        )
