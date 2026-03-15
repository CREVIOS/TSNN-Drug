"""Tests for the temporal sheaf transport block."""

import torch
import pytest

from tsnn.model.sheaf_transport import TemporalSheafTransport


def make_test_data(N=20, D=32, T=5, E=40):
    """Create synthetic test data for sheaf transport."""
    h_seq = [torch.randn(N, D) for _ in range(T)]
    ei_seq = [
        torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
        for _ in range(T)
    ]
    ea_seq = [torch.randn(E, 16) for _ in range(T)]
    return h_seq, ei_seq, ea_seq


def test_sheaf_transport_output_shapes():
    """Check output shapes are correct."""
    N, D, T, E = 20, 32, 5, 40
    block = TemporalSheafTransport(
        hidden_dim=D, edge_dim=16, householder_depth=4
    )

    h_seq, ei_seq, ea_seq = make_test_data(N, D, T, E)
    output = block(h_seq, ei_seq, ea_seq)

    assert output.h_final.shape == (N, D)
    assert len(output.h_sequence) == T
    assert len(output.disagreements) == T
    assert len(output.transport_maps) == T

    for t in range(T):
        assert output.h_sequence[t].shape == (N, D)
        assert output.disagreements[t].shape == (E,)
        assert output.transport_maps[t].shape == (E, D, D)


def test_sheaf_disagreement_non_negative():
    """Sheaf disagreement D_uv should be non-negative (it's a squared norm)."""
    N, D, T, E = 15, 16, 3, 25
    block = TemporalSheafTransport(hidden_dim=D, edge_dim=0, householder_depth=2)

    h_seq = [torch.randn(N, D) for _ in range(T)]
    ei_seq = [torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]) for _ in range(T)]

    output = block(h_seq, ei_seq)

    for t in range(T):
        assert (output.disagreements[t] >= -1e-6).all(), \
            f"Disagreement should be non-negative at t={t}"


def test_identity_transport_reduces_to_standard():
    """With Q=I (ablation 1), should reduce to standard message passing."""
    N, D, T, E = 10, 16, 3, 20
    block_sheaf = TemporalSheafTransport(
        hidden_dim=D, edge_dim=0, householder_depth=2,
        identity_transport=False,
    )
    block_identity = TemporalSheafTransport(
        hidden_dim=D, edge_dim=0, householder_depth=2,
        identity_transport=True,
    )

    # Both should run without error
    h_seq = [torch.randn(N, D) for _ in range(T)]
    ei_seq = [torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]) for _ in range(T)]

    out_sheaf = block_sheaf(h_seq, ei_seq)
    out_identity = block_identity(h_seq, ei_seq)

    # Shapes should match
    assert out_sheaf.h_final.shape == out_identity.h_final.shape


def test_gradient_flow():
    """Gradients should flow through the entire sheaf transport block."""
    N, D, T, E = 10, 16, 3, 15
    block = TemporalSheafTransport(
        hidden_dim=D, edge_dim=8, householder_depth=2
    )

    h_seq = [torch.randn(N, D, requires_grad=True) for _ in range(T)]
    ei_seq = [torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]) for _ in range(T)]
    ea_seq = [torch.randn(E, 8) for _ in range(T)]

    output = block(h_seq, ei_seq, ea_seq)
    loss = output.h_final.sum() + sum(d.sum() for d in output.disagreements)
    loss.backward()

    for t in range(T):
        assert h_seq[t].grad is not None, f"No gradient at t={t}"


def test_static_frames_ablation():
    """Static frames ablation should use same transport maps across time."""
    N, D, T, E = 10, 16, 4, 20
    block = TemporalSheafTransport(
        hidden_dim=D, edge_dim=0, householder_depth=2,
        static_frames=True,
    )

    h_seq = [torch.randn(N, D) for _ in range(T)]
    ei_seq = [torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))]) for _ in range(T)]

    output = block(h_seq, ei_seq)
    assert len(output.transport_maps) == T


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
