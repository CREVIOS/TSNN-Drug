"""Householder reflection utilities for orthogonal transport map construction.

Implements Eq. 8 from the paper:
    U_v(t) = prod_{i=1}^{k} H(f_i_v(t))
    Q_uv(t) = U_v(t)^T @ U_u(t)  in O(d)

where H(f) = I - 2*f*f^T / ||f||^2 is a Householder reflection.

The composition of k Householder reflections is guaranteed to be in O(d),
providing the orthogonal transport maps between local frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def householder_reflection(v: Tensor) -> Tensor:
    """Compute a single Householder reflection matrix H(v) = I - 2*v*v^T / ||v||^2.

    Args:
        v: Reflection vectors [*, d].

    Returns:
        Reflection matrices [*, d, d].
    """
    # Normalize to avoid numerical issues
    v_norm_sq = (v.float() * v.float()).sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [*, 1]
    v = v.unsqueeze(-1)  # [*, d, 1]
    v_t = v.transpose(-2, -1)  # [*, 1, d]
    d = v.shape[-2]
    I = torch.eye(d, device=v.device, dtype=v.dtype).expand_as(
        torch.zeros(*v.shape[:-2], d, d, device=v.device)
    )
    H = I - 2.0 * torch.matmul(v, v_t) / v_norm_sq.unsqueeze(-1)
    return H


def compose_householder(frame_vectors: Tensor) -> Tensor:
    """Compose k Householder reflections into an orthogonal matrix.

    Args:
        frame_vectors: [*, k, d] — k reflection vectors per node.

    Returns:
        Orthogonal matrices U [*, d, d] in O(d).
    """
    k = frame_vectors.shape[-2]
    d = frame_vectors.shape[-1]
    batch_shape = frame_vectors.shape[:-2]

    U = torch.eye(d, device=frame_vectors.device, dtype=frame_vectors.dtype)
    U = U.expand(*batch_shape, d, d).contiguous()

    for i in range(k):
        v_i = frame_vectors[..., i, :]  # [*, d]
        H_i = householder_reflection(v_i)  # [*, d, d]
        U = torch.matmul(U, H_i)

    return U


def compute_transport_maps(
    U: Tensor, edge_index: Tensor
) -> Tensor:
    """Compute pairwise transport maps Q_uv = U_v^T @ U_u for each edge.

    Args:
        U: Per-node orthogonal matrices [N, d, d].
        edge_index: Edge indices [2, E] where row=u (target), col=v (source).

    Returns:
        Transport maps Q_uv [E, d, d] in O(d).
    """
    row, col = edge_index
    U_u = U[row]  # [E, d, d]
    U_v = U[col]  # [E, d, d]
    # Q_uv = U_v^T @ U_u
    Q_uv = torch.bmm(U_v.transpose(-2, -1), U_u)
    return Q_uv


class HouseholderFrameBuilder(nn.Module):
    """Learns frame vectors from node embeddings and constructs orthogonal maps.

    Args:
        hidden_dim: Dimension of node embeddings (= d for the orthogonal maps).
        householder_depth: Number of Householder reflections k.
    """

    def __init__(self, hidden_dim: int, householder_depth: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = householder_depth

        # MLP that produces k frame vectors from node embedding
        self.frame_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * householder_depth),
        )

    def forward(self, h: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Compute orthogonal matrices and transport maps.

        Args:
            h: Node embeddings [N, D].
            edge_index: [2, E].

        Returns:
            U: Per-node orthogonal matrices [N, D, D].
            Q: Per-edge transport maps [E, D, D].
        """
        N, D = h.shape
        # Produce frame vectors
        frame_flat = self.frame_mlp(h)  # [N, k*D]
        frame_vectors = frame_flat.view(N, self.k, D)  # [N, k, D]

        # Compose Householder reflections
        U = compose_householder(frame_vectors)  # [N, D, D]

        # Compute transport maps
        Q = compute_transport_maps(U, edge_index)  # [E, D, D]

        return U, Q
