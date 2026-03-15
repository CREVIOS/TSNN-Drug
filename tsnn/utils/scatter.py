"""Scatter operation compatibility layer.

Uses torch_geometric's scatter if available, falls back to pure PyTorch.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from torch_geometric.utils import scatter
    _HAS_PYG_SCATTER = True
except ImportError:
    _HAS_PYG_SCATTER = False


def scatter_add(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None
) -> Tensor:
    if _HAS_PYG_SCATTER:
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    return out.index_add_(dim, index, src)


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None
) -> Tensor:
    if _HAS_PYG_SCATTER:
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out_sum = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out_sum.index_add_(dim, index, src)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1)
    for _ in range(src.dim() - 1):
        count = count.unsqueeze(-1)
    return out_sum / count
