"""End-to-end integration test for the full TSNN pipeline.

Tests the complete forward pass through all 4 components:
1. Graph building
2. E(3)-equivariant encoding
3. Temporal sheaf transport
4. Contact hazard + survival heads

Also tests backward pass (gradient flow) and loss computation.
"""

import torch
import pytest
from torch_geometric.data import Data

from tsnn.model.tsnn import TSNN, TSNNConfig, TSNNOutput
from tsnn.losses.combined import CombinedLoss


def create_synthetic_frames(N=30, T=5, N_lig=10, D_node=32, D_edge=28):
    """Create synthetic temporal frame data for testing."""
    frames = []
    cross_masks = []
    e2c_list = []

    for t in range(T):
        pos = torch.randn(N, 3) * 5.0
        x = torch.randn(N, D_node)

        # Build radius graph
        dist = torch.cdist(pos, pos)
        mask = (dist < 5.0) & (dist > 0.01)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        E = edge_index.shape[1]
        edge_attr = torch.randn(E, D_edge)

        is_ligand = torch.zeros(N, dtype=torch.bool)
        is_ligand[:N_lig] = True
        cross_mask = is_ligand[edge_index[0]] != is_ligand[edge_index[1]]

        frame = {
            "node_features": x,
            "positions": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }
        frames.append(frame)
        cross_masks.append(cross_mask)
        e2c_list.append(torch.zeros(E, dtype=torch.long))

    n2c = torch.zeros(N, dtype=torch.long)
    return frames, cross_masks, n2c, e2c_list


def test_full_forward_pass():
    """Test complete forward pass produces all expected outputs."""
    config = TSNNConfig(
        node_input_dim=32,
        edge_input_dim=28,
        hidden_dim=64,
        encoder_layers=2,
        householder_depth=2,
    )
    model = TSNN(config)
    model.eval()

    frames, cross_masks, n2c, e2c_list = create_synthetic_frames(
        N=30, T=5, D_node=32, D_edge=28
    )

    with torch.no_grad():
        output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)

    assert isinstance(output, TSNNOutput)
    assert output.log_koff.shape == (1,)
    assert output.hazard is not None
    assert output.hazard.shape[1] == 1  # 1 complex
    assert output.survival is not None
    assert len(output.disagreements) == 5  # T frames
    assert len(output.risk_scores) == 5
    assert output.h_final.shape[0] == 30  # N nodes


def test_full_backward_pass():
    """Test gradients flow through the entire pipeline."""
    config = TSNNConfig(
        node_input_dim=32,
        edge_input_dim=28,
        hidden_dim=64,
        encoder_layers=2,
        householder_depth=2,
    )
    model = TSNN(config)
    model.train()

    frames, cross_masks, n2c, e2c_list = create_synthetic_frames()

    output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)

    # Simple loss
    loss = output.log_koff.sum()
    if output.hazard is not None:
        loss = loss + output.hazard.sum()
    loss.backward()

    # Check all model components received gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_combined_loss_with_model():
    """Test the combined loss (Eq. 16) works with model output."""
    config = TSNNConfig(
        node_input_dim=32,
        edge_input_dim=28,
        hidden_dim=64,
        encoder_layers=2,
        householder_depth=2,
    )
    model = TSNN(config)
    model.train()

    loss_fn = CombinedLoss(alpha=0.1, beta=0.05, gamma=0.01)

    frames, cross_masks, n2c, e2c_list = create_synthetic_frames(T=5)
    output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)

    targets = {
        "log_koff": torch.tensor([2.5]),
        "event_times": torch.tensor([3]),
        "censored": torch.tensor([False]),
    }

    losses = loss_fn(output, targets)

    assert "total" in losses
    assert "regression" in losses
    assert "sheaf" in losses
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    # Verify gradients propagated
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert grad_count == total_params, \
        f"Only {grad_count}/{total_params} params got gradients"


def test_ablation_no_sheaf():
    """Ablation 1: Q_uv = I (no sheaf transport)."""
    config = TSNNConfig(
        node_input_dim=32, edge_input_dim=28, hidden_dim=64,
        encoder_layers=2, identity_transport=True,
    )
    model = TSNN(config)
    frames, cross_masks, n2c, e2c_list = create_synthetic_frames()
    output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)
    assert output.log_koff.shape == (1,)


def test_ablation_static_frames():
    """Ablation 2: time-invariant U_v."""
    config = TSNNConfig(
        node_input_dim=32, edge_input_dim=28, hidden_dim=64,
        encoder_layers=2, static_frames=True,
    )
    model = TSNN(config)
    frames, cross_masks, n2c, e2c_list = create_synthetic_frames()
    output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)
    assert output.log_koff.shape == (1,)


def test_ablation_no_survival():
    """Ablation 7: no survival head, direct regression only."""
    config = TSNNConfig(
        node_input_dim=32, edge_input_dim=28, hidden_dim=64,
        encoder_layers=2, use_survival=False,
    )
    model = TSNN(config)
    frames, cross_masks, n2c, e2c_list = create_synthetic_frames()
    output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)
    assert output.log_koff.shape == (1,)
    assert output.hazard is None
    assert output.survival is None


def test_householder_depth_ablation():
    """Ablation 10: varying Householder depth k."""
    for k in [1, 2, 4, 8]:
        config = TSNNConfig(
            node_input_dim=32, edge_input_dim=28, hidden_dim=64,
            encoder_layers=2, householder_depth=k,
        )
        model = TSNN(config)
        frames, cross_masks, n2c, e2c_list = create_synthetic_frames()
        output = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)
        assert output.log_koff.shape == (1,), f"Failed for k={k}"


def test_model_deterministic():
    """Same input should produce same output (eval mode)."""
    config = TSNNConfig(
        node_input_dim=32, edge_input_dim=28, hidden_dim=64,
        encoder_layers=2, householder_depth=2,
    )
    model = TSNN(config)
    model.eval()

    torch.manual_seed(42)
    frames, cross_masks, n2c, e2c_list = create_synthetic_frames()

    with torch.no_grad():
        out1 = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)
        out2 = model(frames, cross_masks, n2c, e2c_list, num_complexes=1)

    assert torch.allclose(out1.log_koff, out2.log_koff), \
        "Model should be deterministic in eval mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
