"""Multi-Scale Sheaf Spectral Decomposition (Innovation 2).

Decomposes molecular dynamics into frequency bands using Chebyshev
polynomial filtering on the sheaf Laplacian. Each band captures
different timescale dynamics:

- Low frequency:  global conformational changes, complex-level dissociation
- Mid frequency:  residue-level rearrangements, contact breaking
- High frequency: local atomic vibrations, thermal noise

Uses Chebyshev recurrence to avoid expensive O(N³) eigendecomposition:
    T_0(x) = x
    T_1(x) = L̃·x
    T_{k+1}(x) = 2·L̃·T_k(x) - T_{k-1}(x)

Cost: O(K · nnz(L_F) · d) per layer — linear in edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.utils.scatter import scatter_add


def _build_sheaf_laplacian_sparse(
    Q: Tensor, edge_index: Tensor, N: int, D: int
) -> tuple[Tensor, Tensor]:
    """Build sparse sheaf Laplacian action as a function.

    Rather than materializing the full (N·D × N·D) matrix, returns
    components for efficient sparse matrix-vector multiplication.

    Returns:
        diag_blocks: [N, D, D] diagonal blocks of L_F
        off_diag_data: (Q, edge_index) for off-diagonal multiplication
    """
    row, col = edge_index  # row=u, col=v
    E = edge_index.shape[1]

    # Diagonal blocks: [L_F]_vv = Σ Q_e^T Q_e for all edges touching v
    # Since Q_uv ∈ O(d), Q^T Q = I, so diagonal = degree * I
    # But we compute it properly for non-orthogonal generalization
    QTQ = torch.bmm(Q.transpose(-2, -1), Q)  # [E, D, D]

    diag = torch.zeros(N, D, D, device=Q.device, dtype=Q.dtype)
    for i in range(D):
        for j in range(D):
            diag[:, i, j].index_add_(0, row, QTQ[:, i, j])
            diag[:, i, j].index_add_(0, col, QTQ[:, i, j])

    return diag, Q


def sheaf_laplacian_matvec(
    h: Tensor, Q: Tensor, edge_index: Tensor
) -> Tensor:
    """Compute L_F @ h efficiently without materializing L_F.

    L_F h_v = (Σ_{e:v◁e} F_{v◁e}^T F_{v◁e}) h_v
              - Σ_{u:u~v} F_{v◁e}^T F_{u◁e} h_u

    For orthogonal Q_uv = U_v^T U_u:
    L_F h_v = deg(v) · h_v - Σ_{u~v} Q_vu h_u

    Args:
        h: Node states [N, D].
        Q: Transport maps [E, D, D].
        edge_index: [2, E].

    Returns:
        L_F @ h [N, D].
    """
    row, col = edge_index
    N, D = h.shape

    # Transported neighbor features: Q_uv @ h_v for each edge
    Qh = torch.bmm(Q, h[col].unsqueeze(-1)).squeeze(-1)  # [E, D]

    # Off-diagonal: -Σ Q_uv h_v  (aggregated per target node u)
    off_diag = scatter_add(Qh, row, dim=0, dim_size=N)  # [N, D]

    # Diagonal: deg(v) * h_v (for orthogonal maps, Q^T Q = I)
    degree = torch.zeros(N, device=h.device, dtype=h.dtype)
    degree.index_add_(0, row, torch.ones(row.shape[0], device=h.device, dtype=h.dtype))
    degree.index_add_(0, col, torch.ones(col.shape[0], device=h.device, dtype=h.dtype))
    diag = degree.unsqueeze(-1) * h  # [N, D]

    return diag - off_diag


def estimate_lambda_max(
    Q: Tensor, edge_index: Tensor, N: int, D: int, num_iters: int = 5
) -> Tensor:
    """Estimate largest eigenvalue of L_F via power iteration."""
    v = torch.randn(N, D, device=Q.device, dtype=Q.dtype)
    v = v / v.norm()

    for _ in range(num_iters):
        Lv = sheaf_laplacian_matvec(v, Q, edge_index)
        lambda_est = (v * Lv).sum() / ((v * v).sum() + 1e-8)
        v = Lv / (Lv.norm() + 1e-8)

    return lambda_est.clamp(min=1.0)


class ChebyshevSheafFilter(nn.Module):
    """Chebyshev polynomial filter on the sheaf Laplacian.

    Computes p_θ(L̃_F) x = Σ_{k=0}^{K} θ_k T_k(L̃_F x)
    where L̃_F = (2/λ_max) L_F - I is rescaled to [-1, 1].

    Args:
        hidden_dim: Feature dimension.
        order: Chebyshev polynomial order K.
    """

    def __init__(self, hidden_dim: int, order: int = 4):
        super().__init__()
        self.order = order
        self.hidden_dim = hidden_dim

        # Learnable Chebyshev coefficients per output dimension
        self.coeffs = nn.Parameter(torch.ones(order + 1) / (order + 1))

    def forward(
        self, h: Tensor, Q: Tensor, edge_index: Tensor, lambda_max: Tensor
    ) -> Tensor:
        """Apply Chebyshev filter.

        Args:
            h: Node features [N, D].
            Q: Transport maps [E, D, D].
            edge_index: [2, E].
            lambda_max: Estimated largest eigenvalue (scalar).

        Returns:
            Filtered features [N, D].
        """
        N, D = h.shape

        # Rescale: L̃ = (2/λ_max) L - I
        def L_tilde_matvec(x):
            Lx = sheaf_laplacian_matvec(x, Q, edge_index)
            return (2.0 / (lambda_max + 1e-8)) * Lx - x

        # Chebyshev recurrence
        T_prev = h  # T_0 = x
        T_curr = L_tilde_matvec(h)  # T_1 = L̃·x

        # Weighted sum: softmax coefficients for stability
        weights = torch.softmax(self.coeffs, dim=0)
        result = weights[0] * T_prev + weights[1] * T_curr

        for k in range(2, self.order + 1):
            T_next = 2.0 * L_tilde_matvec(T_curr) - T_prev
            result = result + weights[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        return result


class MultiScaleSheafDecomposition(nn.Module):
    """Multi-scale decomposition via band-pass Chebyshev filters.

    Decomposes node features into frequency bands using different
    Chebyshev filters, each capturing a different spatial scale.

    Args:
        hidden_dim: Feature dimension.
        num_bands: Number of frequency bands (default 3: low/mid/high).
        chebyshev_order: Polynomial order per band.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_bands: int = 3,
        chebyshev_order: int = 4,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim

        # One Chebyshev filter per band
        self.band_filters = nn.ModuleList([
            ChebyshevSheafFilter(hidden_dim, chebyshev_order)
            for _ in range(num_bands)
        ])

        # Per-band projection (mix filtered features)
        self.band_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_bands)
        ])

        # Fusion: combine bands back into single representation
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_bands, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h: Tensor, Q: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Decompose and recombine multi-scale features.

        Args:
            h: Node features [N, D].
            Q: Transport maps [E, D, D].
            edge_index: [2, E].

        Returns:
            Multi-scale enhanced features [N, D].
        """
        N = h.shape[0]

        # Estimate lambda_max for rescaling
        lambda_max = estimate_lambda_max(Q, edge_index, N, self.hidden_dim)

        # Apply each band filter
        band_outputs = []
        for filt, proj in zip(self.band_filters, self.band_projections):
            filtered = filt(h, Q, edge_index, lambda_max)
            projected = proj(filtered)
            band_outputs.append(projected)

        # Fuse bands
        fused = self.fusion(torch.cat(band_outputs, dim=-1))

        # Residual connection
        return h + fused
