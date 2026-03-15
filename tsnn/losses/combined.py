"""Combined loss for Stage C kinetics fine-tuning (Eq. 16).

    L = L_surv + alpha * L_reg + beta * L_rank + gamma * L_sheaf
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.losses.survival_nll import SurvivalNLLLossVectorized
from tsnn.losses.regression import KoffRegressionLoss
from tsnn.losses.ranking import PairwiseRankingLoss
from tsnn.losses.sheaf_smoothness import SheafSmoothnessLoss


class CombinedLoss(nn.Module):
    """Combined kinetics loss from Eq. 16 of the paper.

    L = L_surv + alpha * L_reg + beta * L_rank + gamma * L_sheaf

    Args:
        alpha: Weight for regression loss.
        beta: Weight for ranking loss.
        gamma: Weight for sheaf smoothness regularizer.
        ranking_margin: Margin for pairwise ranking.
        use_survival: If False, skip survival loss (ablation 7).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.05,
        gamma: float = 0.01,
        ranking_margin: float = 0.5,
        use_survival: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_survival = use_survival

        if use_survival:
            self.survival_loss = SurvivalNLLLossVectorized()
        self.regression_loss = KoffRegressionLoss()
        self.ranking_loss = PairwiseRankingLoss(margin=ranking_margin)
        self.sheaf_loss = SheafSmoothnessLoss()

    def forward(
        self,
        model_output,
        targets: dict,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            model_output: TSNNOutput from the model.
            targets: Dict with keys:
                'log_koff': Target log k_off [B].
                'event_times': Discretized event times [B].
                'censored': Boolean censoring mask [B].
                'series_ids': Optional congeneric series IDs [B].

        Returns:
            Dict with individual losses and 'total'.
        """
        losses = {}
        device = model_output.log_koff.device
        total = torch.tensor(0.0, device=device)

        # Survival NLL
        if self.use_survival and model_output.hazard is not None:
            l_surv = self.survival_loss(
                model_output.hazard,
                targets["event_times"],
                targets["censored"],
            )
            losses["survival"] = l_surv
            total = total + l_surv

        # Regression on log k_off
        l_reg = self.regression_loss(
            model_output.log_koff,
            targets["log_koff"],
        )
        losses["regression"] = l_reg
        total = total + self.alpha * l_reg

        # Pairwise ranking
        l_rank = self.ranking_loss(
            model_output.log_koff,
            targets["log_koff"],
            targets.get("series_ids"),
        )
        losses["ranking"] = l_rank
        total = total + self.beta * l_rank

        # Sheaf smoothness regularizer
        l_sheaf = self.sheaf_loss(model_output.disagreements)
        losses["sheaf"] = l_sheaf
        total = total + self.gamma * l_sheaf

        losses["total"] = total
        return losses
