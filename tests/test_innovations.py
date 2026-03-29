"""Tests for all three innovations: Sheaf ODE, Multi-Scale Spectral, Conformal UQ."""

import numpy as np
import torch
import pytest

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.model.sheaf_ode import SheafODEFunc, ContinuousSheafTransport
from tsnn.model.spectral_sheaf import (
    sheaf_laplacian_matvec,
    estimate_lambda_max,
    ChebyshevSheafFilter,
    MultiScaleSheafDecomposition,
)
from tsnn.evaluation.uncertainty import (
    compute_sheaf_uncertainty,
    compute_temporal_uncertainty,
    SheafConformalPredictor,
    compute_coverage_metrics,
)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def make_graph(N=30, n_lig=10, T=5, D=128, edge_dim=28, device="cpu"):
    """Create synthetic MD trajectory data."""
    frames, cross_masks, e2c_list = [], [], []
    for t in range(T):
        pos = torch.randn(N, 3, device=device) * 5
        x = torch.randn(N, 29, device=device)
        dist = torch.cdist(pos, pos)
        mask = (dist < 5.0) & (dist > 0.01)
        ei = mask.nonzero(as_tuple=False).t().contiguous()
        E = ei.shape[1]
        ea = torch.randn(E, edge_dim, device=device)
        is_lig = torch.zeros(N, dtype=torch.bool, device=device)
        is_lig[:n_lig] = True
        cm = is_lig[ei[0]] != is_lig[ei[1]]
        frames.append({"node_features": x, "positions": pos, "edge_index": ei, "edge_attr": ea})
        cross_masks.append(cm)
        e2c_list.append(torch.zeros(E, dtype=torch.long, device=device))
    n2c = torch.zeros(N, dtype=torch.long, device=device)
    return frames, cross_masks, n2c, e2c_list


def make_transport(N=30, D=128, edge_cutoff=5.0, device="cpu"):
    """Create node features, edge index, and transport maps."""
    h = torch.randn(N, D, device=device)
    pos = torch.randn(N, 3, device=device) * 5
    dist = torch.cdist(pos, pos)
    mask = (dist < edge_cutoff) & (dist > 0.01)
    ei = mask.nonzero(as_tuple=False).t().contiguous()
    E = ei.shape[1]
    ea = torch.randn(E, 28, device=device)
    # Random orthogonal transport maps
    Q = torch.linalg.qr(torch.randn(E, D, D, device=device))[0]
    return h, ei, ea, Q


# ═══════════════════════════════════════════════════════════
# Innovation 1: Sheaf Neural ODE
# ═══════════════════════════════════════════════════════════

class TestSheafODE:
    def test_ode_func_output_shape(self):
        """ODE function produces correct dh/dt shape."""
        D = 64
        h, ei, ea, Q = make_transport(N=20, D=D)
        func = SheafODEFunc(D, edge_dim=28)
        func.set_graph(ei, ea, Q)
        t = torch.tensor(0.0)
        dh_dt = func(t, h)
        assert dh_dt.shape == h.shape

    def test_ode_func_bounded(self):
        """ODE dynamics are bounded by tanh (|dh/dt| ≤ 1)."""
        D = 64
        h, ei, ea, Q = make_transport(N=20, D=D)
        func = SheafODEFunc(D, edge_dim=28)
        func.set_graph(ei, ea, Q)
        dh_dt = func(torch.tensor(0.0), h * 100)  # Large input
        assert dh_dt.abs().max() <= 1.0 + 1e-5

    def test_ode_func_no_edges(self):
        """ODE function handles empty graphs gracefully."""
        D = 64
        h = torch.randn(10, D)
        ei = torch.zeros(2, 0, dtype=torch.long)
        Q = torch.zeros(0, D, D)
        func = SheafODEFunc(D)
        func.set_graph(ei, None, Q)
        dh_dt = func(torch.tensor(0.0), h)
        assert (dh_dt == 0).all()

    def test_continuous_transport_output(self):
        """ContinuousSheafTransport produces correct output structure."""
        D = 64
        transport = ContinuousSheafTransport(
            hidden_dim=D, edge_dim=28, householder_depth=2,
            ode_solver="euler", ode_step_size=0.5,
        )
        h_seq = [torch.randn(20, D) for _ in range(3)]
        ei_seq = []
        ea_seq = []
        for _ in range(3):
            pos = torch.randn(20, 3) * 5
            dist = torch.cdist(pos, pos)
            mask = (dist < 5.0) & (dist > 0.01)
            ei = mask.nonzero(as_tuple=False).t().contiguous()
            ei_seq.append(ei)
            ea_seq.append(torch.randn(ei.shape[1], 28))

        out = transport(h_seq, ei_seq, ea_seq)
        assert out.h_final.shape == (20, D)
        assert len(out.h_sequence) == 3
        assert len(out.disagreements) == 3
        assert len(out.transport_maps) == 3

    def test_continuous_transport_gradient_flow(self):
        """Gradients flow through ODE integration."""
        D = 64
        transport = ContinuousSheafTransport(
            hidden_dim=D, edge_dim=28, householder_depth=2,
            ode_solver="euler", ode_step_size=0.5,
        )
        h_seq = [torch.randn(15, D, requires_grad=True) for _ in range(3)]
        ei_seq, ea_seq = [], []
        for _ in range(3):
            pos = torch.randn(15, 3) * 5
            dist = torch.cdist(pos, pos)
            ei = (dist < 5.0).nonzero(as_tuple=False).t().contiguous()
            ei_seq.append(ei)
            ea_seq.append(torch.randn(ei.shape[1], 28))

        out = transport(h_seq, ei_seq, ea_seq)
        loss = out.h_final.sum()
        loss.backward()

        # All input embeddings should receive gradients
        for t, h_t in enumerate(h_seq):
            assert h_t.grad is not None, f"No gradient at frame {t}"

        # All parameters should have gradients
        for name, p in transport.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_ode_vs_gru_same_interface(self):
        """ODE and GRU transports have compatible interfaces."""
        D = 64
        from tsnn.model.sheaf_transport import TemporalSheafTransport

        gru_transport = TemporalSheafTransport(
            hidden_dim=D, edge_dim=28, householder_depth=2
        )
        ode_transport = ContinuousSheafTransport(
            hidden_dim=D, edge_dim=28, householder_depth=2,
            ode_solver="euler", ode_step_size=0.5,
        )

        h_seq = [torch.randn(10, D) for _ in range(3)]
        ei_seq, ea_seq = [], []
        for _ in range(3):
            pos = torch.randn(10, 3) * 5
            dist = torch.cdist(pos, pos)
            ei = (dist < 5.0).nonzero(as_tuple=False).t().contiguous()
            ei_seq.append(ei)
            ea_seq.append(torch.randn(ei.shape[1], 28))

        out_gru = gru_transport(h_seq, ei_seq, ea_seq)
        out_ode = ode_transport(h_seq, ei_seq, ea_seq)

        # Same output structure
        assert out_gru.h_final.shape == out_ode.h_final.shape
        assert len(out_gru.h_sequence) == len(out_ode.h_sequence)
        assert len(out_gru.disagreements) == len(out_ode.disagreements)

    def test_full_model_with_ode(self):
        """Full TSNN model works with ODE transport."""
        config = TSNNConfig(hidden_dim=64, use_ode=True, ode_solver="euler")
        model = TSNN(config)
        frames, cm, n2c, e2c = make_graph(N=20, T=3, D=64)
        out = model(frames, cm, n2c, e2c, 1)
        assert out.log_koff.shape == (1,)
        loss = out.log_koff.sum()
        loss.backward()

    def test_ode_numerical_stability(self):
        """ODE transport handles large inputs without NaN."""
        D = 64
        transport = ContinuousSheafTransport(
            hidden_dim=D, edge_dim=28, householder_depth=2,
            ode_solver="euler", ode_step_size=0.5,
        )
        h_seq = [torch.randn(15, D) * 100 for _ in range(3)]  # Large values
        ei_seq, ea_seq = [], []
        for _ in range(3):
            pos = torch.randn(15, 3) * 50
            dist = torch.cdist(pos, pos)
            ei = (dist < 50.0).nonzero(as_tuple=False).t().contiguous()
            ei_seq.append(ei)
            ea_seq.append(torch.randn(ei.shape[1], 28) * 10)

        out = transport(h_seq, ei_seq, ea_seq)
        assert not torch.isnan(out.h_final).any()
        assert not torch.isinf(out.h_final).any()


# ═══════════════════════════════════════════════════════════
# Innovation 2: Multi-Scale Spectral Decomposition
# ═══════════════════════════════════════════════════════════

class TestSpectralSheaf:
    def test_sheaf_laplacian_matvec_shape(self):
        """Sheaf Laplacian matvec produces correct shape."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        Lh = sheaf_laplacian_matvec(h, Q, ei)
        assert Lh.shape == h.shape

    def test_sheaf_laplacian_positive_semidefinite(self):
        """x^T L_F x ≥ 0 for the sheaf Laplacian."""
        D = 16
        h, ei, _, Q = make_transport(N=15, D=D)
        Lh = sheaf_laplacian_matvec(h, Q, ei)
        energy = (h * Lh).sum()
        assert energy >= -1e-4, f"Negative energy: {energy}"

    def test_lambda_max_positive(self):
        """Estimated lambda_max is positive."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        lam = estimate_lambda_max(Q, ei, 20, D)
        assert lam > 0

    def test_chebyshev_filter_shape(self):
        """Chebyshev filter preserves shape."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        filt = ChebyshevSheafFilter(D, order=4)
        lam = estimate_lambda_max(Q, ei, 20, D)
        out = filt(h, Q, ei, lam)
        assert out.shape == h.shape

    def test_chebyshev_filter_gradient(self):
        """Gradients flow through Chebyshev filter."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        h.requires_grad_(True)
        filt = ChebyshevSheafFilter(D, order=4)
        lam = estimate_lambda_max(Q, ei, 20, D)
        out = filt(h, Q, ei, lam)
        out.sum().backward()
        assert h.grad is not None
        assert filt.coeffs.grad is not None

    def test_multiscale_decomposition_shape(self):
        """Multi-scale decomposition preserves shape."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        ms = MultiScaleSheafDecomposition(D, num_bands=3, chebyshev_order=4)
        out = ms(h, Q, ei)
        assert out.shape == h.shape

    def test_multiscale_residual(self):
        """Multi-scale output differs from input (not identity)."""
        D = 32
        h, ei, _, Q = make_transport(N=20, D=D)
        ms = MultiScaleSheafDecomposition(D, num_bands=3, chebyshev_order=4)
        out = ms(h, Q, ei)
        assert not torch.allclose(out, h, atol=1e-3)

    def test_full_model_with_multiscale(self):
        """Full TSNN model works with multi-scale decomposition."""
        config = TSNNConfig(hidden_dim=64, use_multiscale=True, num_bands=3)
        model = TSNN(config)
        frames, cm, n2c, e2c = make_graph(N=20, T=3, D=64)
        out = model(frames, cm, n2c, e2c, 1)
        assert out.log_koff.shape == (1,)
        out.log_koff.sum().backward()

    def test_full_model_ode_plus_multiscale(self):
        """Full TSNN model with BOTH ODE and multi-scale."""
        config = TSNNConfig(
            hidden_dim=64, use_ode=True, use_multiscale=True,
            ode_solver="euler", num_bands=2, chebyshev_order=2,
        )
        model = TSNN(config)
        frames, cm, n2c, e2c = make_graph(N=20, T=3, D=64)
        out = model(frames, cm, n2c, e2c, 1)
        assert out.log_koff.shape == (1,)
        # Use combined loss so hazard head also gets gradients
        loss = out.log_koff.sum()
        if out.hazard is not None:
            loss = loss + out.hazard.sum()
        if out.survival is not None:
            loss = loss + out.survival.sum()
        loss.backward()
        # Check key components have gradients (encoder, sheaf, survival)
        for name, p in model.named_parameters():
            if p.requires_grad and ("encoder" in name or "sheaf" in name
                                     or "survival" in name or "multiscale" in name):
                assert p.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════
# Innovation 3: Conformal Prediction
# ═══════════════════════════════════════════════════════════

class TestConformalPrediction:
    def test_sheaf_uncertainty_basic(self):
        """Sheaf uncertainty score is non-negative."""
        D = [torch.rand(100) * 5]
        cm = [torch.ones(100, dtype=torch.bool)]
        u = compute_sheaf_uncertainty(D, cm)
        assert u >= 0

    def test_sheaf_uncertainty_high_variance(self):
        """High-variance disagreements → high uncertainty."""
        D_uniform = [torch.ones(100) * 2.0]
        D_varied = [torch.tensor([0.0] * 50 + [10.0] * 50)]
        cm = [torch.ones(100, dtype=torch.bool)]

        u_uniform = compute_sheaf_uncertainty(D_uniform, cm)
        u_varied = compute_sheaf_uncertainty(D_varied, cm)
        assert u_varied > u_uniform

    def test_sheaf_uncertainty_empty_contacts(self):
        """Handles zero cross-edges gracefully."""
        D = [torch.rand(50)]
        cm = [torch.zeros(50, dtype=torch.bool)]
        u = compute_sheaf_uncertainty(D, cm)
        assert u == 1.0  # Default uncertainty for no contacts

    def test_temporal_uncertainty(self):
        """Rising disagreement → positive temporal uncertainty."""
        T = 5
        # Rising disagreement
        Ds_rising = [torch.ones(20) * (t + 1) for t in range(T)]
        cms = [torch.ones(20, dtype=torch.bool)] * T
        u_rising = compute_temporal_uncertainty(Ds_rising, cms)
        assert u_rising > 0

        # Flat disagreement
        Ds_flat = [torch.ones(20) * 2.0 for _ in range(T)]
        u_flat = compute_temporal_uncertainty(Ds_flat, cms)
        assert u_flat < u_rising

    def test_conformal_calibration(self):
        """Conformal predictor calibrates correctly."""
        np.random.seed(42)
        n = 200
        preds = np.random.randn(n) * 2
        targets = preds + np.random.randn(n) * 0.5  # Noise
        uncertainties = np.abs(np.random.randn(n)) + 0.1

        cp = SheafConformalPredictor(alpha=0.1)
        q_hat = cp.calibrate(preds, targets, uncertainties)
        assert q_hat > 0
        assert cp.is_calibrated

    def test_conformal_coverage_guarantee(self):
        """Empirical coverage ≥ 1-α on held-out data."""
        np.random.seed(42)
        n_cal = 500
        n_test = 1000

        # Generate calibration data
        cal_preds = np.random.randn(n_cal) * 2
        cal_targets = cal_preds + np.random.randn(n_cal) * 0.5
        cal_unc = np.abs(np.random.randn(n_cal)) + 0.1

        # Calibrate
        cp = SheafConformalPredictor(alpha=0.1)
        cp.calibrate(cal_preds, cal_targets, cal_unc)

        # Test
        test_preds = np.random.randn(n_test) * 2
        test_targets = test_preds + np.random.randn(n_test) * 0.5
        test_unc = np.abs(np.random.randn(n_test)) + 0.1

        result = cp.predict(test_preds, test_unc)
        metrics = compute_coverage_metrics(result, test_targets)

        # Coverage should be ≥ 1 - α (with some statistical slack)
        assert metrics["picp"] >= 0.85, f"Coverage {metrics['picp']} < 0.85"
        assert metrics["mpiw"] > 0
        assert metrics["q_hat"] > 0

    def test_conformal_interval_width(self):
        """Higher uncertainty → wider intervals."""
        np.random.seed(42)
        n = 200
        preds = np.random.randn(n)
        targets = preds + np.random.randn(n) * 0.3
        unc = np.abs(np.random.randn(n)) + 0.1

        cp = SheafConformalPredictor(alpha=0.1)
        cp.calibrate(preds, targets, unc)

        # Low uncertainty → narrow interval
        result_low = cp.predict(np.array([0.0]), np.array([0.1]))
        result_high = cp.predict(np.array([0.0]), np.array([10.0]))
        assert result_high.interval_width[0] > result_low.interval_width[0]

    def test_conformal_not_calibrated_error(self):
        """Raises error if predict() called before calibrate()."""
        cp = SheafConformalPredictor()
        with pytest.raises(RuntimeError):
            cp.predict(np.array([1.0]), np.array([0.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
