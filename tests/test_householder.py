"""Tests for Householder reflection and orthogonal transport maps.

Critical correctness tests:
1. H(v) is orthogonal: H^T H = I
2. H(v) is a reflection: H = H^T, H^2 = I
3. Composed product is orthogonal: U^T U = I
4. Transport maps are orthogonal: Q^T Q = I
"""

import torch
import pytest

from tsnn.model.householder import (
    householder_reflection,
    compose_householder,
    compute_transport_maps,
    HouseholderFrameBuilder,
)


def test_householder_reflection_orthogonal():
    """H(v) should satisfy H^T H = I."""
    v = torch.randn(5, 8)
    H = householder_reflection(v)  # [5, 8, 8]

    I = torch.eye(8).unsqueeze(0).expand(5, -1, -1)
    HtH = torch.bmm(H.transpose(-2, -1), H)

    assert torch.allclose(HtH, I, atol=1e-5), \
        f"H^T H should be I. Max error: {(HtH - I).abs().max()}"


def test_householder_reflection_symmetric():
    """H(v) should be symmetric: H = H^T."""
    v = torch.randn(3, 6)
    H = householder_reflection(v)

    assert torch.allclose(H, H.transpose(-2, -1), atol=1e-6), \
        "Householder reflection should be symmetric"


def test_householder_reflection_involutory():
    """H(v) should satisfy H^2 = I."""
    v = torch.randn(4, 10)
    H = householder_reflection(v)

    H2 = torch.bmm(H, H)
    I = torch.eye(10).unsqueeze(0).expand(4, -1, -1)

    assert torch.allclose(H2, I, atol=1e-5), \
        "Householder reflection should be involutory (H^2 = I)"


def test_compose_householder_orthogonal():
    """Product of k Householder reflections should be orthogonal."""
    for k in [1, 2, 4, 8]:
        frame_vectors = torch.randn(10, k, 16)
        U = compose_householder(frame_vectors)  # [10, 16, 16]

        I = torch.eye(16).unsqueeze(0).expand(10, -1, -1)
        UtU = torch.bmm(U.transpose(-2, -1), U)

        assert torch.allclose(UtU, I, atol=1e-4), \
            f"U^T U should be I for k={k}. Max error: {(UtU - I).abs().max()}"


def test_transport_maps_orthogonal():
    """Q_uv = U_v^T U_u should be orthogonal."""
    N, d = 20, 32
    U = compose_householder(torch.randn(N, 4, d))  # [N, d, d]

    # Random edges
    edge_index = torch.stack([
        torch.randint(0, N, (50,)),
        torch.randint(0, N, (50,)),
    ])

    Q = compute_transport_maps(U, edge_index)  # [50, d, d]

    I = torch.eye(d).unsqueeze(0).expand(50, -1, -1)
    QtQ = torch.bmm(Q.transpose(-2, -1), Q)

    assert torch.allclose(QtQ, I, atol=1e-4), \
        f"Q^T Q should be I. Max error: {(QtQ - I).abs().max()}"


def test_frame_builder_shapes():
    """HouseholderFrameBuilder should produce correct shapes."""
    D = 64
    builder = HouseholderFrameBuilder(hidden_dim=D, householder_depth=4)

    h = torch.randn(15, D)
    edge_index = torch.stack([
        torch.randint(0, 15, (30,)),
        torch.randint(0, 15, (30,)),
    ])

    U, Q = builder(h, edge_index)

    assert U.shape == (15, D, D), f"U shape should be [15, {D}, {D}]"
    assert Q.shape == (30, D, D), f"Q shape should be [30, {D}, {D}]"

    # Check orthogonality
    I = torch.eye(D).unsqueeze(0).expand(15, -1, -1)
    assert torch.allclose(torch.bmm(U.transpose(-2, -1), U), I, atol=1e-4)


def test_frame_builder_gradient_flow():
    """Gradients should flow through the Householder composition."""
    D = 32
    builder = HouseholderFrameBuilder(hidden_dim=D, householder_depth=4)

    h = torch.randn(10, D, requires_grad=True)
    edge_index = torch.stack([
        torch.randint(0, 10, (20,)),
        torch.randint(0, 10, (20,)),
    ])

    U, Q = builder(h, edge_index)
    loss = Q.sum()
    loss.backward()

    assert h.grad is not None, "Gradients should flow to input"
    assert not torch.all(h.grad == 0), "Gradients should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
