"""Pretraining losses for Stage A (MD) and Stage B (dissociation).

Stage A (self-supervised MD pretraining):
- Next-frame contact prediction
- Contact persistence prediction
- Sheaf disagreement forecasting
- Masked interaction-type prediction
- Temporal contrastive learning

Stage B (dissociation-aware pretraining):
- Contact rupture prediction
- Local hazard prediction
- Time-to-escape-bin classification
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tsnn.utils.scatter import scatter_mean


class NextFrameContactLoss(nn.Module):
    """Predict whether each contact exists in the next frame (BCE).

    Stage A auxiliary task.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        edge_scores: Tensor,
        future_contact_exists: Tensor,
    ) -> Tensor:
        """
        Args:
            edge_scores: Predicted logits for contact at t+delta [E].
            future_contact_exists: Binary target [E].
        """
        return self.bce(edge_scores, future_contact_exists.float())


class ContactPersistenceLoss(nn.Module):
    """Predict how many frames a contact persists (regression).

    Stage A auxiliary task.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_persistence: Tensor,
        true_persistence: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_persistence: Predicted frames of persistence [E].
            true_persistence: True number of frames [E].
        """
        return self.mse(pred_persistence, true_persistence.float())


class DisagreementForecastLoss(nn.Module):
    """Predict future sheaf disagreement from current state.

    Stage A auxiliary task: helps the model learn that growing
    disagreement is predictable from current geometry.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_disagreement: Tensor,
        future_disagreement: Tensor,
    ) -> Tensor:
        return self.mse(pred_disagreement, future_disagreement)


class MaskedInteractionTypeLoss(nn.Module):
    """Predict masked edge interaction types (cross-entropy).

    Stage A auxiliary task: like masked language modeling but for
    molecular interaction types.
    """

    def __init__(self, num_types: int = 9):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_types = num_types

    def forward(
        self,
        pred_types: Tensor,
        true_types: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_types: Predicted type logits [E, num_types].
            true_types: True type indices [E].
            mask: Boolean mask for which edges were masked [E].
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_types.device, requires_grad=True)
        return self.ce(pred_types[mask], true_types[mask])


class TemporalContrastiveLoss(nn.Module):
    """Contrastive loss between temporal windows.

    Windows from the same trajectory should be closer in embedding
    space than windows from different trajectories.

    Stage A auxiliary task.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: Tensor,
        trajectory_ids: Tensor,
    ) -> Tensor:
        """InfoNCE-style contrastive loss.

        Args:
            embeddings: Window-level embeddings [B, D].
            trajectory_ids: Which trajectory each window came from [B].
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature  # [B, B]

        # Positive mask: same trajectory
        pos_mask = trajectory_ids.unsqueeze(0) == trajectory_ids.unsqueeze(1)
        pos_mask.fill_diagonal_(False)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # InfoNCE: for each anchor, log-softmax over all others
        # excluding self-similarity
        mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        sim = sim.masked_fill(~mask, float("-inf"))

        log_softmax = F.log_softmax(sim, dim=1)
        loss = -(log_softmax * pos_mask.float()).sum() / pos_mask.float().sum()
        return loss


class ContactRuptureLoss(nn.Module):
    """Predict whether each contact breaks within next K frames.

    Stage B auxiliary task using DD-13M ground truth.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        rupture_logits: Tensor,
        rupture_labels: Tensor,
    ) -> Tensor:
        """
        Args:
            rupture_logits: Predicted rupture probability logits [E].
            rupture_labels: Binary: does contact break in next K frames [E].
        """
        return self.bce(rupture_logits, rupture_labels.float())


class TimeToEscapeLoss(nn.Module):
    """Classify time-to-escape into discrete bins.

    Stage B auxiliary task.
    """

    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_bins = num_bins

    def forward(
        self,
        pred_logits: Tensor,
        time_bins: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_logits: Predicted bin logits [B, num_bins].
            time_bins: True bin indices [B].
        """
        return self.ce(pred_logits, time_bins)


class StageALoss(nn.Module):
    """Combined loss for Stage A MD pretraining.

    Args:
        w_contact: Weight for next-frame contact prediction.
        w_persistence: Weight for contact persistence.
        w_disagreement: Weight for disagreement forecasting.
        w_interaction: Weight for masked interaction type.
        w_contrastive: Weight for temporal contrastive.
    """

    def __init__(
        self,
        w_contact: float = 1.0,
        w_persistence: float = 0.5,
        w_disagreement: float = 0.5,
        w_interaction: float = 0.5,
        w_contrastive: float = 0.3,
    ):
        super().__init__()
        self.contact_loss = NextFrameContactLoss()
        self.persistence_loss = ContactPersistenceLoss()
        self.disagreement_loss = DisagreementForecastLoss()
        self.interaction_loss = MaskedInteractionTypeLoss()
        self.contrastive_loss = TemporalContrastiveLoss()

        self.weights = {
            "contact": w_contact,
            "persistence": w_persistence,
            "disagreement": w_disagreement,
            "interaction": w_interaction,
            "contrastive": w_contrastive,
        }

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute all Stage A losses.

        Returns dict with individual losses and total.
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if "contact_scores" in predictions and "future_contacts" in targets:
            l = self.contact_loss(predictions["contact_scores"],
                                  targets["future_contacts"])
            losses["contact"] = l
            total = total + self.weights["contact"] * l

        if "persistence_pred" in predictions and "persistence_true" in targets:
            l = self.persistence_loss(predictions["persistence_pred"],
                                      targets["persistence_true"])
            losses["persistence"] = l
            total = total + self.weights["persistence"] * l

        if "disagreement_pred" in predictions and "disagreement_true" in targets:
            l = self.disagreement_loss(predictions["disagreement_pred"],
                                       targets["disagreement_true"])
            losses["disagreement"] = l
            total = total + self.weights["disagreement"] * l

        if "type_pred" in predictions and "type_true" in targets:
            l = self.interaction_loss(
                predictions["type_pred"],
                targets["type_true"],
                targets.get("type_mask", torch.ones_like(targets["type_true"], dtype=torch.bool)),
            )
            losses["interaction"] = l
            total = total + self.weights["interaction"] * l

        if "embeddings" in predictions and "trajectory_ids" in targets:
            l = self.contrastive_loss(predictions["embeddings"],
                                      targets["trajectory_ids"])
            losses["contrastive"] = l
            total = total + self.weights["contrastive"] * l

        losses["total"] = total
        return losses


class StageBLoss(nn.Module):
    """Combined loss for Stage B dissociation pretraining.

    Args:
        w_rupture: Weight for contact rupture prediction.
        w_hazard: Weight for local hazard prediction.
        w_escape: Weight for time-to-escape bin classification.
    """

    def __init__(
        self,
        w_rupture: float = 1.0,
        w_hazard: float = 1.0,
        w_escape: float = 0.5,
    ):
        super().__init__()
        self.rupture_loss = ContactRuptureLoss()
        self.hazard_mse = nn.MSELoss()
        self.escape_loss = TimeToEscapeLoss()

        self.weights = {
            "rupture": w_rupture,
            "hazard": w_hazard,
            "escape": w_escape,
        }

    def forward(self, predictions: dict, targets: dict) -> dict:
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        if "rupture_logits" in predictions and "rupture_labels" in targets:
            l = self.rupture_loss(predictions["rupture_logits"],
                                  targets["rupture_labels"])
            losses["rupture"] = l
            total = total + self.weights["rupture"] * l

        if "hazard_pred" in predictions and "hazard_true" in targets:
            l = self.hazard_mse(predictions["hazard_pred"],
                                targets["hazard_true"])
            losses["hazard"] = l
            total = total + self.weights["hazard"] * l

        if "escape_logits" in predictions and "escape_bins" in targets:
            l = self.escape_loss(predictions["escape_logits"],
                                 targets["escape_bins"])
            losses["escape"] = l
            total = total + self.weights["escape"] * l

        losses["total"] = total
        return losses
