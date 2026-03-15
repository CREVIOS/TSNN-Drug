"""E(3) equivariance tests.

The EGNN encoder should be equivariant: applying a rotation/translation
to input coordinates should produce the same scalar outputs (invariant)
and appropriately transformed coordinate outputs (equivariant).

The full TSNN pipeline should produce SE(3)-invariant scalar outputs
(hazard, survival, k_off).
"""

import torch
import pytest

from tsnn.model.layers.egnn_layer import EGNNLayer
from tsnn.model.equivariant_encoder import EquivariantEncoder


def random_rotation():
    """Generate a random 3x3 rotation matrix via QR decomposition."""
    M = torch.randn(3, 3)
    Q, R = torch.linalg.qr(M)
    # Ensure it's a proper rotation (det = +1)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_egnn_layer_invariant_features():
    """Node features h should be invariant under E(3) transforms."""
    N, D, E_dim = 15, 64, 0
    layer = EGNNLayer(hidden_dim=D, edge_dim=E_dim, update_coords=False)
    layer.eval()

    h = torch.randn(N, D)
    x = torch.randn(N, 3)
    edge_index = torch.stack([
        torch.randint(0, N, (40,)),
        torch.randint(0, N, (40,)),
    ])

    # Original
    h_out1, _ = layer(h, x, edge_index)

    # Apply rotation + translation
    R = random_rotation()
    t = torch.randn(1, 3)
    x_transformed = x @ R.T + t

    h_out2, _ = layer(h, x_transformed, edge_index)

    assert torch.allclose(h_out1, h_out2, atol=1e-4), \
        f"Node features should be invariant. Max diff: {(h_out1 - h_out2).abs().max()}"


def test_egnn_layer_equivariant_coords():
    """Coordinate outputs should transform equivariantly."""
    N, D = 15, 64
    layer = EGNNLayer(hidden_dim=D, edge_dim=0, update_coords=True)
    layer.eval()

    h = torch.randn(N, D)
    x = torch.randn(N, 3)
    edge_index = torch.stack([
        torch.randint(0, N, (40,)),
        torch.randint(0, N, (40,)),
    ])

    # Original
    _, x_out1 = layer(h, x, edge_index)

    # Apply rotation + translation
    R = random_rotation()
    t = torch.randn(1, 3)
    x_transformed = x @ R.T + t

    _, x_out2 = layer(h, x_transformed, edge_index)

    # x_out2 should equal x_out1 @ R^T + t
    x_out1_transformed = x_out1 @ R.T + t

    assert torch.allclose(x_out1_transformed, x_out2, atol=1e-3), \
        f"Coords should be equivariant. Max diff: {(x_out1_transformed - x_out2).abs().max()}"


def test_encoder_invariance():
    """Full encoder should produce invariant node embeddings."""
    encoder = EquivariantEncoder(
        node_input_dim=16,
        edge_input_dim=0,
        hidden_dim=32,
        num_layers=2,
        update_coords=True,
    )
    encoder.eval()

    N = 12
    h = torch.randn(N, 16)
    x = torch.randn(N, 3)
    edge_index = torch.stack([
        torch.randint(0, N, (30,)),
        torch.randint(0, N, (30,)),
    ])

    # Original
    out1 = encoder(h, x, edge_index)

    # Transformed
    R = random_rotation()
    t = torch.randn(1, 3)
    out2 = encoder(h, x @ R.T + t, edge_index)

    assert torch.allclose(out1, out2, atol=1e-3), \
        f"Encoder output should be SE(3)-invariant. Max diff: {(out1 - out2).abs().max()}"


def test_encoder_translation_invariance():
    """Encoder should be translation invariant."""
    encoder = EquivariantEncoder(
        node_input_dim=16,
        edge_input_dim=0,
        hidden_dim=32,
        num_layers=2,
    )
    encoder.eval()

    N = 10
    h = torch.randn(N, 16)
    x = torch.randn(N, 3)
    edge_index = torch.stack([
        torch.randint(0, N, (25,)),
        torch.randint(0, N, (25,)),
    ])

    out1 = encoder(h, x, edge_index)

    t = torch.randn(1, 3) * 100  # Large translation
    out2 = encoder(h, x + t, edge_index)

    assert torch.allclose(out1, out2, atol=1e-3), \
        "Encoder should be translation invariant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
