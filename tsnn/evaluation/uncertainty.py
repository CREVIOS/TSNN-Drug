"""Sheaf-Calibrated Uncertainty via Conformal Prediction (Innovation 3).

Uses the distribution of sheaf disagreements D_uv(t) across protein-ligand
contacts as a geometrically principled uncertainty score, wrapped with
split conformal prediction for calibrated confidence intervals.

Coverage guarantee (distribution-free, finite-sample):
    P(y_{n+1} ∈ C(x_{n+1})) ≥ 1 - α

Requires only exchangeability of calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class ConformalPrediction:
    """Conformal prediction output."""
    point_estimate: np.ndarray       # [B] predicted log k_off
    lower_bound: np.ndarray          # [B] lower CI
    upper_bound: np.ndarray          # [B] upper CI
    uncertainty_score: np.ndarray    # [B] sheaf-based uncertainty
    interval_width: np.ndarray       # [B] CI width
    alpha: float                     # miscoverage rate
    q_hat: float                     # conformal quantile


def compute_sheaf_uncertainty(
    disagreements: list[Tensor],
    cross_masks: list[Tensor],
    beta: float = 0.5,
) -> Tensor:
    """Compute uncertainty score from sheaf disagreements.

    u(complex) = Var_{contacts}[D_uv(T)] + β · max_{contacts} D_uv(T)

    High variance across contacts = heterogeneous binding = uncertain.
    High max disagreement = at least one strained contact = uncertain.

    Args:
        disagreements: Per-frame sheaf disagreements [E_t].
        cross_masks: Boolean masks for protein-ligand edges [E_t].
        beta: Weight for max disagreement term.

    Returns:
        Uncertainty score (scalar tensor).
    """
    # Use last frame (most evolved state)
    D_final = disagreements[-1]
    cross_mask = cross_masks[-1]

    # Filter to cross-contacts only
    D_cross = D_final[cross_mask]

    if D_cross.numel() == 0:
        return torch.tensor(1.0, device=D_final.device)

    variance = D_cross.var().clamp(min=1e-8)
    maximum = D_cross.max()

    uncertainty = variance + beta * maximum
    return uncertainty


def compute_temporal_uncertainty(
    disagreements: list[Tensor],
    cross_masks: list[Tensor],
) -> Tensor:
    """Compute uncertainty from temporal evolution of disagreements.

    Rising disagreement over time = system becoming unstable = uncertain.

    Args:
        disagreements: Per-frame sheaf disagreements.
        cross_masks: Per-frame cross-edge masks.

    Returns:
        Temporal uncertainty score (scalar tensor).
    """
    if len(disagreements) < 2:
        return torch.tensor(0.0, device=disagreements[0].device)

    mean_Ds = []
    for D_t, cm_t in zip(disagreements, cross_masks):
        D_cross = D_t[cm_t]
        if D_cross.numel() > 0:
            mean_Ds.append(D_cross.mean())
        else:
            mean_Ds.append(torch.tensor(0.0, device=D_t.device))

    mean_Ds = torch.stack(mean_Ds)

    # Slope of mean disagreement over time (positive = growing instability)
    T = len(mean_Ds)
    t = torch.arange(T, device=mean_Ds.device, dtype=mean_Ds.dtype)
    slope = ((t * mean_Ds).sum() - t.sum() * mean_Ds.sum() / T) / \
            ((t * t).sum() - t.sum() ** 2 / T + 1e-8)

    return slope.clamp(min=0.0)


class SheafConformalPredictor:
    """Conformal prediction wrapper using sheaf-derived uncertainty.

    Split conformal prediction with sheaf disagreement as the
    nonconformity normalization, providing calibrated prediction
    intervals with finite-sample coverage guarantees.

    Usage:
        1. predictor = SheafConformalPredictor(alpha=0.1)
        2. predictor.calibrate(cal_predictions, cal_targets, cal_uncertainties)
        3. result = predictor.predict(test_predictions, test_uncertainties)
    """

    def __init__(self, alpha: float = 0.1, min_uncertainty: float = 0.01):
        """
        Args:
            alpha: Desired miscoverage rate (e.g., 0.1 for 90% coverage).
            min_uncertainty: Floor on uncertainty to prevent zero-width intervals.
        """
        self.alpha = alpha
        self.min_uncertainty = min_uncertainty
        self.q_hat: float | None = None
        self._calibrated = False

    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> float:
        """Calibrate on held-out data.

        Computes the conformal quantile q̂ such that:
            P(|y - ŷ| / u ≤ q̂) ≥ 1 - α

        Args:
            predictions: Predicted log k_off [n].
            targets: True log k_off [n].
            uncertainties: Sheaf-based uncertainty scores [n].

        Returns:
            Calibrated quantile q̂.
        """
        n = len(predictions)
        uncertainties = np.maximum(uncertainties, self.min_uncertainty)

        # Nonconformity scores: normalized residuals
        scores = np.abs(targets - predictions) / uncertainties

        # Conformal quantile (finite-sample correction)
        quantile_level = min((1 - self.alpha) * (1 + 1 / n), 1.0)
        self.q_hat = float(np.quantile(scores, quantile_level))
        self._calibrated = True
        return self.q_hat

    def predict(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> ConformalPrediction:
        """Generate prediction intervals.

        Args:
            predictions: Point estimates [B].
            uncertainties: Sheaf-based uncertainty scores [B].

        Returns:
            ConformalPrediction with intervals and metadata.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first.")

        uncertainties = np.maximum(uncertainties, self.min_uncertainty)
        half_width = self.q_hat * uncertainties

        return ConformalPrediction(
            point_estimate=predictions,
            lower_bound=predictions - half_width,
            upper_bound=predictions + half_width,
            uncertainty_score=uncertainties,
            interval_width=2 * half_width,
            alpha=self.alpha,
            q_hat=self.q_hat,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


def compute_coverage_metrics(
    predictions: ConformalPrediction,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute coverage and interval width metrics.

    Args:
        predictions: ConformalPrediction output.
        targets: True log k_off values.

    Returns:
        Dict with PICP, MPIW, and other metrics.
    """
    covered = (targets >= predictions.lower_bound) & (targets <= predictions.upper_bound)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(predictions.interval_width))

    # Conditional coverage by uncertainty quantile
    n = len(targets)
    q25 = int(n * 0.25)
    q75 = int(n * 0.75)
    sort_idx = np.argsort(predictions.uncertainty_score)
    low_u_coverage = float(np.mean(covered[sort_idx[:q25]])) if q25 > 0 else 0.0
    high_u_coverage = float(np.mean(covered[sort_idx[q75:]])) if q75 < n else 0.0

    return {
        "picp": picp,                           # Prediction Interval Coverage Probability
        "mpiw": mpiw,                           # Mean Prediction Interval Width
        "target_coverage": 1 - predictions.alpha,
        "coverage_gap": picp - (1 - predictions.alpha),
        "low_uncertainty_coverage": low_u_coverage,
        "high_uncertainty_coverage": high_u_coverage,
        "mean_uncertainty": float(np.mean(predictions.uncertainty_score)),
        "q_hat": predictions.q_hat,
    }
