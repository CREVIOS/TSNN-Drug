"""Pairwise ranking loss for congeneric series ordering.

Ensures that within congeneric series (matched molecular pairs),
the model correctly orders compounds by k_off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss within congeneric series.

    For pairs (i,j) where koff_i > koff_j:
        L = max(0, margin - (pred_i - pred_j))

    Args:
        margin: Minimum required margin between predictions.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        series_ids: Tensor | None = None,
    ) -> Tensor:
        """Compute pairwise ranking loss.

        Args:
            predictions: Predicted log k_off [B].
            targets: Target log k_off [B].
            series_ids: Congeneric series assignment [B].
                If None, all pairs are compared.

        Returns:
            Scalar loss.
        """
        valid = ~torch.isnan(targets)
        if valid.sum() < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        pred = predictions[valid]
        tgt = targets[valid]
        B = pred.shape[0]

        if series_ids is not None:
            sid = series_ids[valid]
        else:
            sid = None

        # Build all pairs (i, j) where tgt_i > tgt_j
        idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=pred.device)

        # Filter to same series if series_ids provided
        if sid is not None:
            same_series = sid[idx_i] == sid[idx_j]
            idx_i = idx_i[same_series]
            idx_j = idx_j[same_series]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Determine ordering
        tgt_diff = tgt[idx_i] - tgt[idx_j]  # positive if i has higher koff
        pred_diff = pred[idx_i] - pred[idx_j]

        # For pairs where tgt_i > tgt_j, pred_i should be > pred_j
        # For pairs where tgt_i < tgt_j, flip the sign
        signs = torch.sign(tgt_diff)
        # Filter out tied pairs
        non_tied = signs != 0
        if non_tied.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        signs = signs[non_tied]
        pred_diff = pred_diff[non_tied]

        # Loss: max(0, margin - sign * pred_diff)
        loss = torch.clamp(self.margin - signs * pred_diff, min=0)
        return loss.mean()
