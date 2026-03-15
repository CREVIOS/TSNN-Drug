"""Tests for the E(3)-equivariant encoder."""

import torch
import pytest

from tsnn.model.equivariant_encoder import EquivariantEncoder


def test_encoder_output_shape():
    encoder = EquivariantEncoder(
        node_input_dim=16, edge_input_dim=10,
        hidden_dim=32, num_layers=2,
    )

    N = 20
    h = torch.randn(N, 16)
    x = torch.randn(N, 3)
    ei = torch.stack([torch.randint(0, N, (40,)), torch.randint(0, N, (40,))])
    ea = torch.randn(40, 10)

    out = encoder(h, x, ei, ea)
    assert out.shape == (N, 32)


def test_encoder_gradient_flow():
    encoder = EquivariantEncoder(
        node_input_dim=16, edge_input_dim=0,
        hidden_dim=32, num_layers=2,
    )

    N = 10
    h = torch.randn(N, 16, requires_grad=True)
    x = torch.randn(N, 3, requires_grad=True)
    ei = torch.stack([torch.randint(0, N, (20,)), torch.randint(0, N, (20,))])

    out = encoder(h, x, ei)
    out.sum().backward()

    assert h.grad is not None
    assert x.grad is not None


def test_encoder_handles_no_edges():
    """Should handle graphs with no edges gracefully."""
    encoder = EquivariantEncoder(
        node_input_dim=8, edge_input_dim=0,
        hidden_dim=16, num_layers=1,
    )

    N = 5
    h = torch.randn(N, 8)
    x = torch.randn(N, 3)
    ei = torch.zeros(2, 0, dtype=torch.long)

    out = encoder(h, x, ei)
    assert out.shape == (N, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
