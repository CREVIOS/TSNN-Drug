"""Top-level TSNN model (Equivariant Temporal Sheaf Network).

Wires together all four components from the paper:
1. Hybrid residue-atom graph (handled by data pipeline)
2. E(3)-equivariant local encoder
3. Temporal sheaf transport block
4. Contact hazard + survival heads

The model processes temporal sequences of molecular dynamics frames
and outputs dissociation hazard, survival curves, and log k_off predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from tsnn.model.equivariant_encoder import EquivariantEncoder
from tsnn.model.sheaf_transport import TemporalSheafTransport, SheafTransportOutput
from tsnn.model.contact_hazard_head import ContactHazardHead
from tsnn.model.survival_head import SurvivalHead


@dataclass
class TSNNConfig:
    """Configuration for the TSNN model."""
    # Encoder — defaults match build_node_features (29) and build_edge_features (28)
    node_input_dim: int = 29
    edge_input_dim: int = 28
    hidden_dim: int = 128
    encoder_layers: int = 4
    encoder_dropout: float = 0.1
    update_coords: bool = True

    # Sheaf transport
    householder_depth: int = 4
    num_sheaf_layers: int = 1
    sheaf_dropout: float = 0.1
    static_frames: bool = False
    identity_transport: bool = False

    # Hazard head
    edge_feature_dim: int = 28

    # Survival head
    use_survival: bool = True
    survival_dropout: float = 0.1

    # Ablation: no equivariant encoder (use RBF only)
    no_equivariant: bool = False
    rbf_dim: int = 16

    # Innovation 1: Continuous-time sheaf Neural ODE
    use_ode: bool = False
    ode_solver: str = "rk4"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-3

    # Innovation 2: Multi-scale sheaf spectral decomposition
    use_multiscale: bool = False
    chebyshev_order: int = 4
    num_bands: int = 3


@dataclass
class TSNNOutput:
    """Output of the TSNN model."""
    log_koff: Tensor                        # [B] log k_off predictions
    hazard: Tensor | None                   # [T, B] per-timestep hazard rates
    survival: Tensor | None                 # [T, B] survival curves
    disagreements: list[Tensor]             # Per-frame sheaf disagreements
    risk_scores: list[Tensor]               # Per-frame contact risk scores
    h_final: Tensor                         # [N, D] final node embeddings
    h_sequence: list[Tensor]                # Per-frame node embeddings


class TSNN(nn.Module):
    """Equivariant Temporal Sheaf Network for protein-ligand dissociation.

    Args:
        config: TSNNConfig with all hyperparameters.
    """

    def __init__(self, config: TSNNConfig | None = None):
        super().__init__()
        if config is None:
            config = TSNNConfig()
        self.config = config
        D = config.hidden_dim

        # Component 2: E(3)-equivariant encoder
        if not config.no_equivariant:
            self.encoder = EquivariantEncoder(
                node_input_dim=config.node_input_dim,
                edge_input_dim=config.edge_input_dim,
                hidden_dim=D,
                num_layers=config.encoder_layers,
                dropout=config.encoder_dropout,
                update_coords=config.update_coords,
            )
        else:
            # Ablation 3: RBF-only encoder (no equivariance)
            from tsnn.model.layers.mlp import MLP
            self.encoder = MLP(
                config.node_input_dim + config.rbf_dim,
                D, D, num_layers=3,
            )

        # Component 3: temporal sheaf transport
        if config.use_ode:
            from tsnn.model.sheaf_ode import ContinuousSheafTransport
            self.sheaf_transport = ContinuousSheafTransport(
                hidden_dim=D,
                edge_dim=config.edge_input_dim,
                householder_depth=config.householder_depth,
                dropout=config.sheaf_dropout,
                static_frames=config.static_frames,
                identity_transport=config.identity_transport,
                ode_solver=config.ode_solver,
                ode_rtol=config.ode_rtol,
                ode_atol=config.ode_atol,
            )
        else:
            self.sheaf_transport = TemporalSheafTransport(
                hidden_dim=D,
                edge_dim=config.edge_input_dim,
                householder_depth=config.householder_depth,
                num_sheaf_layers=config.num_sheaf_layers,
                dropout=config.sheaf_dropout,
                static_frames=config.static_frames,
                identity_transport=config.identity_transport,
            )

        # Innovation 2: multi-scale spectral decomposition (optional)
        self.multiscale = None
        if config.use_multiscale:
            from tsnn.model.spectral_sheaf import MultiScaleSheafDecomposition
            self.multiscale = MultiScaleSheafDecomposition(
                hidden_dim=D,
                num_bands=config.num_bands,
                chebyshev_order=config.chebyshev_order,
            )

        # Component 4a: contact hazard head
        self.hazard_head = ContactHazardHead(
            edge_dim=config.edge_input_dim,
            hidden_dim=D,
            dropout=config.sheaf_dropout,
        )

        # Component 4b: survival head
        self.survival_head = SurvivalHead(
            hidden_dim=D,
            risk_dim=1,
            dropout=config.survival_dropout,
            use_survival=config.use_survival,
        )

    def encode_frame(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Encode a single MD frame into node embeddings.

        Args:
            node_features: [N, F_node].
            positions: [N, 3].
            edge_index: [2, E].
            edge_attr: [E, F_edge].

        Returns:
            Node embeddings [N, D].
        """
        if not self.config.no_equivariant:
            return self.encoder(node_features, positions, edge_index, edge_attr)
        else:
            # RBF-only ablation: use distance-expanded features
            from tsnn.utils.chemistry import _rbf_expansion
            from tsnn.utils.geometry import compute_distances
            distances = compute_distances(positions, edge_index)
            # Mean-pool RBF over neighbors
            rbf = _rbf_expansion(distances, self.config.rbf_dim)
            from tsnn.utils.scatter import scatter_mean
            rbf_pooled = scatter_mean(rbf, edge_index[1], dim=0,
                                       dim_size=node_features.shape[0])
            combined = torch.cat([node_features, rbf_pooled], dim=-1)
            return self.encoder(combined)

    def forward(
        self,
        frames: list[dict[str, Tensor]],
        cross_edge_masks: list[Tensor],
        node_to_complex: Tensor,
        edge_to_complex_list: list[Tensor],
        num_complexes: int,
    ) -> TSNNOutput:
        """Full forward pass over a temporal sequence of MD frames.

        Args:
            frames: List of dicts with keys:
                'node_features': [N_t, F_node]
                'positions': [N_t, 3]
                'edge_index': [2, E_t]
                'edge_attr': [E_t, F_edge]
            cross_edge_masks: Boolean masks indicating protein-ligand edges [E_t].
            node_to_complex: Node-to-complex batch assignment [N].
            edge_to_complex_list: Edge-to-complex assignment per frame.
            num_complexes: Number of complexes in batch.

        Returns:
            TSNNOutput with all predictions.
        """
        T = len(frames)

        # Step 1: Encode each frame independently (Component 2)
        h_encoded = []
        for frame in frames:
            h_t = self.encode_frame(
                frame["node_features"],
                frame["positions"],
                frame["edge_index"],
                frame.get("edge_attr"),
            )
            h_encoded.append(h_t)

        # Step 2: Temporal sheaf transport (Component 3)
        edge_indices = [f["edge_index"] for f in frames]
        edge_attrs = [f.get("edge_attr") for f in frames]

        sheaf_out: SheafTransportOutput = self.sheaf_transport(
            h_encoded, edge_indices, edge_attrs
        )

        # Step 2b: Multi-scale spectral enhancement (Innovation 2)
        if self.multiscale is not None:
            enhanced_h = []
            for t in range(T):
                Q_t = sheaf_out.transport_maps[t]
                ei_t = edge_indices[t]
                h_t = sheaf_out.h_sequence[t]
                h_ms = self.multiscale(h_t, Q_t, ei_t)
                enhanced_h.append(h_ms)
            sheaf_out.h_sequence = enhanced_h
            sheaf_out.h_final = enhanced_h[-1]

        # Step 3: Contact hazard for cross-contact edges (Component 4a)
        risk_scores_seq = []
        cross_edge_to_complex = []

        for t in range(T):
            cross_mask = cross_edge_masks[t]  # Boolean [E_t]
            D_t = sheaf_out.disagreements[t]  # [E_t]
            ea_t = frames[t].get("edge_attr")

            # Filter to cross-contacts only
            D_cross = D_t[cross_mask]
            ea_cross = ea_t[cross_mask] if ea_t is not None else None

            # Compute delta edge features (temporal change)
            if t > 0 and ea_t is not None:
                ea_prev = frames[t - 1].get("edge_attr")
                if ea_prev is not None and ea_prev.shape == ea_t.shape:
                    delta_ea = ea_t[cross_mask] - ea_prev[cross_mask]
                else:
                    delta_ea = None
            else:
                delta_ea = None

            if ea_cross is not None:
                risk_t = self.hazard_head(D_cross, ea_cross, delta_ea)
            else:
                risk_t = -D_cross.unsqueeze(-1)  # Fallback: use raw disagreement

            risk_scores_seq.append(risk_t)
            cross_edge_to_complex.append(edge_to_complex_list[t][cross_mask])

        # Step 4: Survival head (Component 4b)
        surv_out = self.survival_head(
            risk_scores_seq,
            cross_edge_to_complex,
            sheaf_out.h_final,
            node_to_complex,
            num_complexes,
        )

        return TSNNOutput(
            log_koff=surv_out["log_koff"],
            hazard=surv_out["hazard"],
            survival=surv_out["survival"],
            disagreements=sheaf_out.disagreements,
            risk_scores=risk_scores_seq,
            h_final=sheaf_out.h_final,
            h_sequence=sheaf_out.h_sequence,
        )
