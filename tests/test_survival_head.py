"""Tests for the survival head and survival loss."""

import torch
import pytest

from tsnn.model.survival_head import SurvivalHead
from tsnn.losses.survival_nll import SurvivalNLLLoss, SurvivalNLLLossVectorized


def test_survival_monotonically_decreasing():
    """Survival function S(t) must be monotonically non-increasing."""
    head = SurvivalHead(hidden_dim=32, use_survival=True)

    T, B = 10, 4
    risk_seq = [torch.randn(20, 1) for _ in range(T)]
    e2c = [torch.randint(0, B, (20,)) for _ in range(T)]
    h_final = torch.randn(50, 32)
    n2c = torch.randint(0, B, (50,))

    output = head(risk_seq, e2c, h_final, n2c, B)

    survival = output["survival"]  # [T, B]
    assert survival is not None

    for b in range(B):
        s = survival[:, b]
        for t in range(1, T):
            assert s[t] <= s[t-1] + 1e-6, \
                f"S(t) must be non-increasing. S({t})={s[t]:.4f} > S({t-1})={s[t-1]:.4f}"


def test_survival_bounded_zero_one():
    """Survival function should be in [0, 1]."""
    head = SurvivalHead(hidden_dim=32, use_survival=True)

    T, B = 10, 3
    risk_seq = [torch.randn(15, 1) for _ in range(T)]
    e2c = [torch.randint(0, B, (15,)) for _ in range(T)]
    h_final = torch.randn(30, 32)
    n2c = torch.randint(0, B, (30,))

    output = head(risk_seq, e2c, h_final, n2c, B)

    survival = output["survival"]
    assert (survival >= -1e-6).all(), "Survival should be >= 0"
    assert (survival <= 1 + 1e-6).all(), "Survival should be <= 1"


def test_hazard_bounded_zero_one():
    """Hazard rates should be in (0, 1) after sigmoid."""
    head = SurvivalHead(hidden_dim=32, use_survival=True)

    T, B = 8, 2
    risk_seq = [torch.randn(10, 1) * 10 for _ in range(T)]  # Large values
    e2c = [torch.randint(0, B, (10,)) for _ in range(T)]
    h_final = torch.randn(20, 32)
    n2c = torch.randint(0, B, (20,))

    output = head(risk_seq, e2c, h_final, n2c, B)

    hazard = output["hazard"]
    assert (hazard > 0).all(), "Hazard should be > 0"
    assert (hazard < 1).all(), "Hazard should be < 1"


def test_survival_nll_loss_uncensored():
    """Loss should be finite for uncensored observations."""
    loss_fn = SurvivalNLLLoss()

    T, B = 10, 5
    hazard = torch.sigmoid(torch.randn(T, B))
    event_times = torch.randint(1, T, (B,))
    censored = torch.zeros(B, dtype=torch.bool)

    loss = loss_fn(hazard, event_times, censored)
    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_survival_nll_loss_censored():
    """Loss should work with censored observations."""
    loss_fn = SurvivalNLLLoss()

    T, B = 10, 5
    hazard = torch.sigmoid(torch.randn(T, B))
    event_times = torch.randint(1, T, (B,))
    censored = torch.ones(B, dtype=torch.bool)  # All censored

    loss = loss_fn(hazard, event_times, censored)
    assert torch.isfinite(loss), f"Loss should be finite for censored data"


def test_survival_nll_vectorized_matches_loop():
    """Vectorized and loop versions should give same result."""
    loss_loop = SurvivalNLLLoss()
    loss_vec = SurvivalNLLLossVectorized()

    T, B = 8, 4
    hazard = torch.sigmoid(torch.randn(T, B))
    event_times = torch.randint(1, T, (B,))
    censored = torch.tensor([True, False, False, True])

    l1 = loss_loop(hazard, event_times, censored)
    l2 = loss_vec(hazard, event_times, censored)

    assert torch.allclose(l1, l2, atol=1e-4), \
        f"Loop ({l1:.4f}) and vectorized ({l2:.4f}) should match"


def test_no_survival_ablation():
    """With use_survival=False, should still produce log_koff."""
    head = SurvivalHead(hidden_dim=32, use_survival=False)

    h_final = torch.randn(30, 32)
    n2c = torch.randint(0, 3, (30,))

    output = head([], [], h_final, n2c, 3)

    assert output["log_koff"].shape == (3,)
    assert output["hazard"] is None
    assert output["survival"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
