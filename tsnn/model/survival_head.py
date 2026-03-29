"""Complex-level Survival Head (Component 4b).

Aggregates contact-level risk scores into complex-level hazard and
survival estimates. Implements Eqs. 13-14 from the paper:

    lambda(t) = sigma(MLP_surv(mean(r_uv(t))))          — Eq. 13
    S(t) = prod_{tau <= t} (1 - lambda(tau))              — Eq. 14

Also provides a direct log k_off regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from tsnn.utils.scatter import scatter_mean

from tsnn.model.layers.mlp import MLP


class SurvivalHead(nn.Module):
    """Complex-level hazard and survival decoder.

    Args:
        hidden_dim: Dimension of node/risk features.
        num_nodes_feature_dim: Dimension of pooled node features for koff.
        dropout: Dropout probability.
        use_survival: If False, skip survival computation (ablation 7).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        risk_dim: int = 1,
        dropout: float = 0.0,
        use_survival: bool = True,
    ):
        super().__init__()
        self.use_survival = use_survival

        # Hazard MLP: mean risk -> hazard rate
        if use_survival:
            self.hazard_mlp = MLP(
                risk_dim, hidden_dim, 1,
                num_layers=2, dropout=dropout, norm=False,
            )

        # Direct log k_off regression from pooled node features
        self.koff_mlp = MLP(
            hidden_dim, hidden_dim, 1,
            num_layers=3, dropout=dropout,
        )

    def forward(
        self,
        risk_scores_sequence: list[Tensor],
        edge_to_complex: list[Tensor],
        h_final: Tensor,
        node_to_complex: Tensor,
        num_complexes: int,
    ) -> dict[str, Tensor]:
        """Compute hazard, survival, and k_off predictions.

        Args:
            risk_scores_sequence: List of risk scores per frame [E_t, 1].
            edge_to_complex: List of edge-to-complex assignment per frame [E_t].
            h_final: Final node embeddings [N, D].
            node_to_complex: Node-to-complex assignment [N].
            num_complexes: Number of complexes in batch.

        Returns:
            Dict with keys: 'log_koff', 'hazard', 'survival'.
        """
        output = {}

        # Pool node features per complex for k_off regression
        complex_features = scatter_mean(
            h_final, node_to_complex, dim=0, dim_size=num_complexes
        )  # [B, D]
        output["log_koff"] = self.koff_mlp(complex_features).squeeze(-1)  # [B]

        if self.use_survival and len(risk_scores_sequence) > 0:
            hazard_rates = []
            for t, (risk_t, e2c_t) in enumerate(
                zip(risk_scores_sequence, edge_to_complex)
            ):
                # Mean-pool risk scores per complex (Eq. 13)
                mean_risk = scatter_mean(
                    risk_t, e2c_t, dim=0, dim_size=num_complexes
                )  # [B, 1]
                # Hazard rate through sigmoid
                lambda_t = torch.sigmoid(self.hazard_mlp(mean_risk))  # [B, 1]
                hazard_rates.append(lambda_t.squeeze(-1))  # [B]

            # Stack: [T, B]
            hazard = torch.stack(hazard_rates, dim=0)  # [T, B]
            output["hazard"] = hazard

            # Survival: S(t) = prod_{tau<=t} (1 - lambda(tau))  (Eq. 14)
            # Clamp hazard to (0, 1) to prevent cumprod collapse
            survival = torch.cumprod((1.0 - hazard).clamp(min=1e-7, max=1.0), dim=0)  # [T, B]
            output["survival"] = survival
        else:
            output["hazard"] = None
            output["survival"] = None

        return output
