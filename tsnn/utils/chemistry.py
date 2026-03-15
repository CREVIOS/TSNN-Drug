"""Chemical feature extraction utilities."""

from __future__ import annotations

import torch
from torch import Tensor

# Standard atom types in protein-ligand systems
ATOM_TYPES = [
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H",
    "Se", "B", "Si", "Fe", "Zn", "Mg", "Ca", "Mn", "Cu", "Co",
    "OTHER",
]
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

# Standard amino acid residue types
RESIDUE_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "OTHER",
]
RESIDUE_TYPE_TO_IDX = {r: i for i, r in enumerate(RESIDUE_TYPES)}

# Edge interaction types
EDGE_TYPES = [
    "covalent",
    "backbone",
    "hbond",
    "hydrophobic",
    "salt_bridge",
    "pi_stacking",
    "protein_ligand",
    "water_mediated",
    "context",
]
EDGE_TYPE_TO_IDX = {e: i for i, e in enumerate(EDGE_TYPES)}

NUM_ATOM_TYPES = len(ATOM_TYPES)
NUM_RESIDUE_TYPES = len(RESIDUE_TYPES)
NUM_EDGE_TYPES = len(EDGE_TYPES)


def get_atom_features(
    atom_types: list[str],
    partial_charges: Tensor | None = None,
    solvent_exposure: Tensor | None = None,
) -> Tensor:
    """Build atom-level node feature vectors.

    Args:
        atom_types: List of atom element symbols.
        partial_charges: Optional partial charges [N].
        solvent_exposure: Optional SASA values [N].

    Returns:
        Feature tensor [N, F] where F = NUM_ATOM_TYPES + extras.
    """
    n = len(atom_types)
    one_hot = torch.zeros(n, NUM_ATOM_TYPES)
    for i, atype in enumerate(atom_types):
        idx = ATOM_TYPE_TO_IDX.get(atype, ATOM_TYPE_TO_IDX["OTHER"])
        one_hot[i, idx] = 1.0

    features = [one_hot]

    if partial_charges is not None:
        features.append(partial_charges.view(n, 1))
    if solvent_exposure is not None:
        features.append(solvent_exposure.view(n, 1))

    return torch.cat(features, dim=-1)


def get_residue_features(
    residue_types: list[str],
    partial_charges: Tensor | None = None,
    solvent_exposure: Tensor | None = None,
    torsion_angles: Tensor | None = None,
) -> Tensor:
    """Build residue-level node feature vectors.

    Args:
        residue_types: List of 3-letter residue codes.
        partial_charges: Optional mean partial charges per residue [N].
        solvent_exposure: Optional SASA per residue [N].
        torsion_angles: Optional phi/psi/chi angles [N, num_torsions].

    Returns:
        Feature tensor [N, F].
    """
    n = len(residue_types)
    one_hot = torch.zeros(n, NUM_RESIDUE_TYPES)
    for i, rtype in enumerate(residue_types):
        idx = RESIDUE_TYPE_TO_IDX.get(rtype, RESIDUE_TYPE_TO_IDX["OTHER"])
        one_hot[i, idx] = 1.0

    features = [one_hot]

    if partial_charges is not None:
        features.append(partial_charges.view(n, 1))
    if solvent_exposure is not None:
        features.append(solvent_exposure.view(n, 1))
    if torsion_angles is not None:
        if torsion_angles.dim() == 1:
            torsion_angles = torsion_angles.unsqueeze(-1)
        features.append(torsion_angles)

    return torch.cat(features, dim=-1)


def get_edge_features(
    distances: Tensor,
    edge_types: list[str],
    unit_vectors: Tensor | None = None,
    num_rbf: int = 16,
    rbf_cutoff: float = 15.0,
) -> Tensor:
    """Build edge feature vectors with RBF distance expansion.

    Args:
        distances: Edge distances [E].
        edge_types: List of edge type strings.
        unit_vectors: Optional unit direction vectors [E, 3].
        num_rbf: Number of radial basis functions.
        rbf_cutoff: Maximum distance for RBF.

    Returns:
        Edge features [E, F].
    """
    rbf = _rbf_expansion(distances, num_rbf, rbf_cutoff)

    e = len(edge_types)
    type_one_hot = torch.zeros(e, NUM_EDGE_TYPES, device=distances.device)
    for i, etype in enumerate(edge_types):
        idx = EDGE_TYPE_TO_IDX.get(etype, 0)
        type_one_hot[i, idx] = 1.0

    features = [rbf, type_one_hot]

    if unit_vectors is not None:
        features.append(unit_vectors)

    return torch.cat(features, dim=-1)


def _rbf_expansion(
    distances: Tensor, num_rbf: int = 16, cutoff: float = 15.0
) -> Tensor:
    """Radial basis function expansion of distances.

    Args:
        distances: [E].
        num_rbf: Number of basis functions.
        cutoff: Maximum distance.

    Returns:
        RBF features [E, num_rbf].
    """
    centers = torch.linspace(0.0, cutoff, num_rbf, device=distances.device)
    gamma = 1.0 / (cutoff / num_rbf)
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-gamma * diff.pow(2))
