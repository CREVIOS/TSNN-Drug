"""Regression loss for log k_off prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KoffRegressionLoss(nn.Module):
    """L2 regression loss on log k_off.

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred_log_koff: Tensor, target_log_koff: Tensor) -> Tensor:
        """Compute L2 loss.

        Args:
            pred_log_koff: Predicted log k_off [B].
            target_log_koff: Target log k_off [B].

        Returns:
            Scalar loss.
        """
        # Filter out samples without koff labels (NaN)
        valid = ~torch.isnan(target_log_koff)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_log_koff.device, requires_grad=True)

        return self.mse(pred_log_koff[valid], target_log_koff[valid])
