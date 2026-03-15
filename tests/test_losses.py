"""Tests for all loss functions."""

import torch
import pytest

from tsnn.losses.regression import KoffRegressionLoss
from tsnn.losses.ranking import PairwiseRankingLoss
from tsnn.losses.sheaf_smoothness import SheafSmoothnessLoss
from tsnn.losses.combined import CombinedLoss


def test_regression_loss_basic():
    loss_fn = KoffRegressionLoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.1, 2.2, 2.8])
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss > 0


def test_regression_loss_nan_filtering():
    loss_fn = KoffRegressionLoss()
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.1, float("nan"), 2.8])
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)


def test_regression_loss_all_nan():
    loss_fn = KoffRegressionLoss()
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([float("nan"), float("nan")])
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss == 0.0


def test_ranking_loss_correct_order():
    """Loss should be small when predictions match target ordering."""
    loss_fn = PairwiseRankingLoss(margin=0.5)
    pred = torch.tensor([3.0, 2.0, 1.0])  # Correct order
    target = torch.tensor([3.0, 2.0, 1.0])
    loss = loss_fn(pred, target)
    assert loss == 0.0  # Margin is satisfied


def test_ranking_loss_wrong_order():
    """Loss should be positive when predictions are misranked."""
    loss_fn = PairwiseRankingLoss(margin=0.5)
    pred = torch.tensor([1.0, 2.0, 3.0])  # Reversed
    target = torch.tensor([3.0, 2.0, 1.0])
    loss = loss_fn(pred, target)
    assert loss > 0


def test_ranking_loss_with_series():
    """Should only compare within same series."""
    loss_fn = PairwiseRankingLoss(margin=0.5)
    pred = torch.tensor([1.0, 3.0, 2.0, 4.0])
    target = torch.tensor([1.0, 3.0, 2.0, 4.0])
    series = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(pred, target, series)
    assert torch.isfinite(loss)


def test_sheaf_smoothness():
    loss_fn = SheafSmoothnessLoss()
    disagreements = [torch.rand(20), torch.rand(20), torch.rand(20)]
    loss = loss_fn(disagreements)
    assert torch.isfinite(loss)
    assert loss > 0


def test_sheaf_smoothness_empty():
    loss_fn = SheafSmoothnessLoss()
    loss = loss_fn([])
    assert torch.isfinite(loss)


def test_combined_loss_all_components():
    """Combined loss should compute all four terms."""
    from dataclasses import dataclass

    @dataclass
    class MockOutput:
        log_koff: torch.Tensor
        hazard: torch.Tensor
        survival: torch.Tensor
        disagreements: list

    loss_fn = CombinedLoss(alpha=0.1, beta=0.05, gamma=0.01)

    B, T = 4, 5
    output = MockOutput(
        log_koff=torch.randn(B),
        hazard=torch.sigmoid(torch.randn(T, B)),
        survival=None,
        disagreements=[torch.rand(20) for _ in range(T)],
    )

    targets = {
        "log_koff": torch.randn(B),
        "event_times": torch.randint(1, T, (B,)),
        "censored": torch.zeros(B, dtype=torch.bool),
    }

    losses = loss_fn(output, targets)

    assert "total" in losses
    assert "survival" in losses
    assert "regression" in losses
    assert "ranking" in losses
    assert "sheaf" in losses
    assert torch.isfinite(losses["total"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
