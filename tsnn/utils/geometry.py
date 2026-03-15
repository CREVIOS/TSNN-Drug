"""Geometric utility functions for molecular dynamics processing."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_distances(pos: Tensor, edge_index: Tensor) -> Tensor:
    """Compute pairwise distances for edges.

    Args:
        pos: Node positions [N, 3].
        edge_index: Edge indices [2, E].

    Returns:
        Distances [E].
    """
    diff = pos[edge_index[0]] - pos[edge_index[1]]
    return torch.norm(diff, dim=-1)


def compute_displacement(pos_t: Tensor, pos_prev: Tensor) -> Tensor:
    """Compute per-node displacement vectors between frames.

    Args:
        pos_t: Current positions [N, 3].
        pos_prev: Previous positions [N, 3].

    Returns:
        Displacement vectors [N, 3].
    """
    return pos_t - pos_prev


def compute_rmsf(
    positions: list[Tensor], window_size: int | None = None
) -> Tensor:
    """Compute root-mean-square fluctuation per node over a trajectory window.

    Args:
        positions: List of position tensors [N, 3], one per frame.
        window_size: If given, use only the last `window_size` frames.

    Returns:
        RMSF per node [N].
    """
    if window_size is not None:
        positions = positions[-window_size:]
    stacked = torch.stack(positions, dim=0)  # [T, N, 3]
    mean_pos = stacked.mean(dim=0, keepdim=True)  # [1, N, 3]
    deviations = (stacked - mean_pos).pow(2).sum(dim=-1)  # [T, N]
    return deviations.mean(dim=0).sqrt()  # [N]


def compute_unit_vectors(pos: Tensor, edge_index: Tensor) -> Tensor:
    """Compute unit displacement vectors along edges.

    Args:
        pos: Node positions [N, 3].
        edge_index: Edge indices [2, E].

    Returns:
        Unit vectors [E, 3].
    """
    diff = pos[edge_index[0]] - pos[edge_index[1]]
    dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    return diff / dist


def radius_graph(pos: Tensor, cutoff: float, batch: Tensor | None = None) -> Tensor:
    """Build radius graph: edges between nodes within cutoff distance.

    Args:
        pos: Node positions [N, 3].
        cutoff: Distance cutoff.
        batch: Batch assignment vector [N] to prevent cross-graph edges.

    Returns:
        Edge index [2, E].
    """
    dist_matrix = torch.cdist(pos, pos)  # [N, N]
    mask = (dist_matrix < cutoff) & (dist_matrix > 1e-6)

    if batch is not None:
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        mask = mask & batch_mask

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index
