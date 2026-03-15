"""Sheaf-aware GRU cell for temporal updates in the sheaf transport block."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SheafGRU(nn.Module):
    """GRU cell with LayerNorm for temporal state updates.

    Wraps nn.GRUCell with optional LayerNorm on the output for
    improved training stability in deep temporal unrolling.

    Args:
        input_dim: Dimension of aggregated messages.
        hidden_dim: Dimension of hidden state.
        norm: Whether to apply LayerNorm to the output.
        dropout: Dropout on the output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """Update hidden state.

        Args:
            x: Input (aggregated messages) [N, input_dim].
            h_prev: Previous hidden state [N, hidden_dim].

        Returns:
            New hidden state [N, hidden_dim].
        """
        h_new = self.gru_cell(x, h_prev)
        h_new = self.norm(h_new)
        h_new = self.dropout(h_new)
        return h_new
