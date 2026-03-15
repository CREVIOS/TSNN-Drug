"""Discrete-time survival negative log-likelihood loss.

Handles right-censored observations as described in Section 7.3:

For observed dissociation at time bin t_event:
    L = -[log(lambda(t_event)) + sum_{t < t_event} log(1 - lambda(t))]

For censored observation (no dissociation observed):
    L = -sum_{t <= T_obs} log(1 - lambda(t))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SurvivalNLLLoss(nn.Module):
    """Discrete-time survival negative log-likelihood with censoring support.

    Args:
        eps: Small constant for numerical stability in log.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        hazard: Tensor,
        event_times: Tensor,
        censored: Tensor,
    ) -> Tensor:
        """Compute survival NLL.

        Args:
            hazard: Predicted hazard rates [T, B] in (0, 1).
            event_times: Time bin of event (or last observed time) [B].
                Integer indices into the T dimension.
            censored: Boolean mask — True if right-censored [B].

        Returns:
            Scalar loss.
        """
        T, B = hazard.shape
        device = hazard.device

        # Clamp hazard for numerical stability
        hazard = hazard.clamp(self.eps, 1.0 - self.eps)

        log_hazard = torch.log(hazard)           # [T, B]
        log_survival = torch.log(1.0 - hazard)   # [T, B]

        total_loss = torch.zeros(B, device=device)

        for b in range(B):
            t_event = int(event_times[b].item())
            t_event = min(t_event, T - 1)  # Clamp to valid range

            if censored[b]:
                # Censored: only survival terms up to observation time
                total_loss[b] = -log_survival[:t_event + 1, b].sum()
            else:
                # Observed: survival up to t_event-1, then event at t_event
                if t_event > 0:
                    total_loss[b] = -log_survival[:t_event, b].sum()
                total_loss[b] -= log_hazard[t_event, b]

        return total_loss.mean()


class SurvivalNLLLossVectorized(nn.Module):
    """Vectorized version of SurvivalNLLLoss for efficiency.

    Uses masking instead of loops over the batch dimension.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        hazard: Tensor,
        event_times: Tensor,
        censored: Tensor,
    ) -> Tensor:
        """Compute survival NLL (vectorized).

        Args:
            hazard: [T, B].
            event_times: [B] integer indices.
            censored: [B] boolean.

        Returns:
            Scalar loss.
        """
        T, B = hazard.shape
        device = hazard.device

        hazard = hazard.clamp(self.eps, 1.0 - self.eps)
        event_times = event_times.clamp(0, T - 1).long()

        log_hazard = torch.log(hazard)
        log_survival = torch.log(1.0 - hazard)

        # Create time mask: 1 for t <= event_time
        time_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        event_mask = time_indices <= event_times.unsqueeze(0)       # [T, B]

        # Survival contribution: sum log(1-lambda) for t < event_time
        pre_event_mask = time_indices < event_times.unsqueeze(0)    # [T, B]
        survival_contrib = (log_survival * pre_event_mask).sum(dim=0)  # [B]

        # Event contribution: log(lambda) at event_time (uncensored only)
        event_loglik = log_hazard.gather(
            0, event_times.unsqueeze(0)
        ).squeeze(0)  # [B]
        event_contrib = event_loglik * (~censored).float()  # [B]

        # For censored: include survival at event_time too
        censored_extra = log_survival.gather(
            0, event_times.unsqueeze(0)
        ).squeeze(0) * censored.float()  # [B]

        nll = -(survival_contrib + event_contrib + censored_extra)
        return nll.mean()
