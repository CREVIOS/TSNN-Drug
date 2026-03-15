"""Tests for graph construction."""

import torch
import pytest

from tsnn.data.graph_builder import GraphBuilder, GraphBuilderConfig


def test_graph_builder_basic():
    config = GraphBuilderConfig(
        pocket_cutoff=10.0, edge_cutoff=5.0, include_water=False
    )
    builder = GraphBuilder(config)

    N_lig, N_prot = 10, 20
    graph = builder.build_frame_graph(
        ligand_positions=torch.randn(N_lig, 3),
        ligand_atom_types=["C"] * N_lig,
        protein_positions=torch.randn(N_prot, 3),
        protein_node_types=["ALA"] * N_prot,
        protein_is_residue=[True] * N_prot,
    )

    assert graph.x is not None
    assert graph.pos.shape == (N_lig + N_prot, 3)
    assert graph.edge_index.shape[0] == 2
    assert graph.is_ligand.sum() == N_lig


def test_graph_builder_with_water():
    config = GraphBuilderConfig(include_water=True, edge_cutoff=8.0)
    builder = GraphBuilder(config)

    graph = builder.build_frame_graph(
        ligand_positions=torch.randn(5, 3),
        ligand_atom_types=["C"] * 5,
        protein_positions=torch.randn(15, 3),
        protein_node_types=["ALA"] * 15,
        protein_is_residue=[True] * 15,
        water_positions=torch.randn(3, 3),
    )

    assert graph.num_nodes == 5 + 15 + 3
    assert graph.is_water.sum() == 3


def test_cross_edge_mask():
    config = GraphBuilderConfig(edge_cutoff=20.0)  # Large cutoff for all edges
    builder = GraphBuilder(config)

    # Place ligand and protein close together
    lig_pos = torch.zeros(3, 3)
    prot_pos = torch.ones(5, 3) * 2

    graph = builder.build_frame_graph(
        ligand_positions=lig_pos,
        ligand_atom_types=["C"] * 3,
        protein_positions=prot_pos,
        protein_node_types=["ALA"] * 5,
        protein_is_residue=[True] * 5,
    )

    # Should have cross edges
    assert graph.cross_edge_mask.any(), "Should detect protein-ligand edges"


def test_pocket_selection():
    config = GraphBuilderConfig(pocket_cutoff=5.0)
    builder = GraphBuilder(config)

    lig_pos = torch.zeros(3, 3)
    prot_pos = torch.tensor([
        [1.0, 0.0, 0.0],   # Within 5A
        [3.0, 0.0, 0.0],   # Within 5A
        [10.0, 0.0, 0.0],  # Outside
        [20.0, 0.0, 0.0],  # Outside
    ])

    mask = builder.pocket_selection(lig_pos, prot_pos)
    assert mask[0] and mask[1], "Close residues should be selected"
    assert not mask[2] and not mask[3], "Far residues should not be selected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
