#!/usr/bin/env python3
"""Preprocess raw datasets into unified HDF5 format for TSNN training.

Converts raw data (GROMACS, HDF5, H5MD, TSV, JSON) into per-complex HDF5 files
containing trajectory windows with node features, positions, and labels.

Usage:
    python scripts/preprocess_data.py --mdd           # Process MDD → Stage A
    python scripts/preprocess_data.py --misato         # Process MISATO → Stage A
    python scripts/preprocess_data.py --dd13m          # Process DD-13M → Stage B
    python scripts/preprocess_data.py --kinetics       # Merge koff labels → Stage C
    python scripts/preprocess_data.py --all            # Process everything
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────

ATOM_TYPES = {
    "C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5,
    "Cl": 6, "Br": 7, "I": 8, "H": 9, "Ca": 10, "Zn": 11,
    "Fe": 12, "Mg": 13, "Na": 14, "K": 15, "Se": 16,
}
N_ATOM_TYPES = 17

RESIDUE_TYPES = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
N_RESIDUE_TYPES = 20

# Node feature dim: 17 atom + 20 residue + 3 flags (lig/prot/water)
# + 1 charge + 1 SASA + 1 displacement + 1 RMSF + 1 torsion = 45 raw
# We project to 29 in the model's input dim. Features beyond 29 are
# handled by the feature builder; here we store the essential ones.
NODE_DIM = 29


def _build_node_features(
    elements: list[str],
    residues: list[str] | None = None,
    is_ligand: np.ndarray | None = None,
    is_water: np.ndarray | None = None,
    charges: np.ndarray | None = None,
) -> np.ndarray:
    """Build per-node feature vectors [N, NODE_DIM]."""
    N = len(elements)
    feat = np.zeros((N, NODE_DIM), dtype=np.float32)

    # Atom type one-hot (0..16)
    for i, elem in enumerate(elements):
        idx = ATOM_TYPES.get(elem, ATOM_TYPES.get(elem[:1], 0))
        if idx < N_ATOM_TYPES:
            feat[i, idx] = 1.0

    # Residue type one-hot (17..36) — only for protein nodes
    if residues is not None:
        for i, res in enumerate(residues):
            if res:
                idx = RESIDUE_TYPES.get(res[:3].upper(), 0)
                feat[i, N_ATOM_TYPES + idx] = 1.0

    # Flags: is_ligand (idx 37), is_protein (idx 38), is_water (idx 39)
    # Mapped to positions within NODE_DIM=29
    flag_start = min(N_ATOM_TYPES + N_RESIDUE_TYPES, NODE_DIM - 3)
    if is_ligand is not None:
        feat[:, flag_start] = is_ligand.astype(np.float32)
        feat[:, flag_start + 1] = (~is_ligand.astype(bool)).astype(np.float32)
        if is_water is not None:
            feat[:, flag_start + 1] -= is_water.astype(np.float32)
            feat[:, flag_start + 2] = is_water.astype(np.float32)

    # Partial charge
    if charges is not None and NODE_DIM > flag_start + 3:
        feat[:, flag_start + 3] = np.clip(charges, -3, 3)

    return feat


# ─────────────────────────────────────────────────────────────
# MDD Preprocessing (GROMACS → HDF5)
# ─────────────────────────────────────────────────────────────

def preprocess_mdd(raw_dir: Path, out_dir: Path, window_size: int = 20, stride: int = 10):
    """Convert MDD GROMACS trajectories to HDF5 windows.

    Each complex → one HDF5 with positions [T, N, 3] and node features.
    """
    import h5py

    mdd_raw = raw_dir / "mdd" / "extracted"
    if not mdd_raw.exists():
        # Try the direct tarball extraction path
        mdd_raw = raw_dir / "mdd"

    out_stage_a = out_dir / "stage_a" / "mdd"
    out_stage_a.mkdir(parents=True, exist_ok=True)

    # Find complex directories
    complex_dirs = sorted([
        d for d in mdd_raw.rglob("*")
        if d.is_dir() and list(d.glob("*.gro"))
    ])

    if not complex_dirs:
        logger.warning(f"No GROMACS files found in {mdd_raw}")
        logger.info("Expected structure: mdd/extracted/<complex_id>/*.gro + *.xtc")
        return

    logger.info(f"Found {len(complex_dirs)} MDD complexes")

    try:
        import MDAnalysis as mda
    except ImportError:
        logger.error("MDAnalysis required: pip install MDAnalysis")
        return

    n_processed = 0
    for cdir in complex_dirs:
        complex_id = cdir.name

        gro_files = list(cdir.glob("*.gro"))
        xtc_files = list(cdir.glob("*.xtc"))

        if not gro_files or not xtc_files:
            continue

        gro = str(gro_files[0])
        xtc = str(xtc_files[0])

        try:
            u = mda.Universe(gro, xtc)
        except Exception as e:
            logger.warning(f"  Skipping {complex_id}: {e}")
            continue

        # Identify ligand vs protein vs water
        try:
            protein = u.select_atoms("protein")
            # Try common ligand selection strategies
            ligand = u.select_atoms("not protein and not resname SOL WAT HOH TIP3 NA CL")
            if len(ligand) == 0:
                ligand = u.select_atoms("resname LIG UNK UNL")
            water = u.select_atoms("resname SOL WAT HOH TIP3")
        except Exception:
            logger.warning(f"  Skipping {complex_id}: atom selection failed")
            continue

        if len(protein) == 0 or len(ligand) == 0:
            logger.warning(f"  Skipping {complex_id}: no protein or ligand found")
            continue

        # Combine atoms: ligand first, then protein (skip water for now)
        all_atoms = ligand + protein
        N = len(all_atoms)
        n_lig = len(ligand)
        T = len(u.trajectory)

        # Extract trajectory positions
        positions = np.zeros((T, N, 3), dtype=np.float32)
        for t, ts in enumerate(u.trajectory):
            positions[t] = all_atoms.positions / 10.0  # Angstrom → nm → keep Angstrom

        # Build node features (static — same for all frames)
        elements = [a.element if hasattr(a, "element") and a.element else a.name[:1]
                    for a in all_atoms]
        residues = [a.resname if hasattr(a, "resname") else "" for a in all_atoms]
        is_ligand = np.zeros(N, dtype=bool)
        is_ligand[:n_lig] = True

        node_features = _build_node_features(
            elements=elements,
            residues=residues,
            is_ligand=is_ligand,
        )

        # Save to HDF5
        h5_path = out_stage_a / f"{complex_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("positions", data=positions, compression="gzip")
            f.create_dataset("node_features", data=node_features, compression="gzip")
            f.attrs["complex_id"] = complex_id
            f.attrs["n_ligand"] = n_lig
            f.attrs["n_protein"] = len(protein)
            f.attrs["n_frames"] = T
            f.attrs["source"] = "MDD"

        n_processed += 1
        if n_processed % 50 == 0:
            logger.info(f"  Processed {n_processed}/{len(complex_dirs)}")

    logger.info(f"MDD preprocessing complete: {n_processed} complexes → {out_stage_a}")


# ─────────────────────────────────────────────────────────────
# MISATO Preprocessing (HDF5 → HDF5 windows)
# ─────────────────────────────────────────────────────────────

def preprocess_misato(raw_dir: Path, out_dir: Path):
    """Convert MISATO HDF5 to per-complex HDF5 files."""
    import h5py

    md_path = raw_dir / "misato" / "MD.hdf5"
    if not md_path.exists():
        logger.error(f"MISATO MD.hdf5 not found at {md_path}")
        return

    out_stage_a = out_dir / "stage_a" / "misato"
    out_stage_a.mkdir(parents=True, exist_ok=True)

    # Load split files
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = raw_dir / "misato" / f"{split_name}_MD.txt"
        if split_file.exists():
            with open(split_file) as f:
                splits[split_name] = [line.strip() for line in f if line.strip()]

    logger.info(f"Opening MISATO MD.hdf5 ({md_path.stat().st_size / 1e9:.1f} GB)")

    with h5py.File(md_path, "r") as f:
        pdb_ids = list(f.keys())
        logger.info(f"  Found {len(pdb_ids)} complexes")

        n_processed = 0
        for pdb_id in pdb_ids:
            grp = f[pdb_id]

            # Get positions
            if "atoms_coordinates_ref" not in grp:
                continue

            # MISATO stores trajectory as multiple coordinate arrays
            # or as a single reference + perturbations
            coords_ref = grp["atoms_coordinates_ref"][:]  # [N, 3]

            # Check for trajectory frames
            frame_keys = sorted([
                k for k in grp.keys()
                if k.startswith("atoms_coordinates_") and k != "atoms_coordinates_ref"
            ])

            if frame_keys:
                T = len(frame_keys) + 1
                N = coords_ref.shape[0]
                positions = np.zeros((T, N, 3), dtype=np.float32)
                positions[0] = coords_ref
                for i, fk in enumerate(frame_keys):
                    positions[i + 1] = grp[fk][:]
            else:
                # Single frame — use as reference
                positions = coords_ref[np.newaxis, :, :]  # [1, N, 3]

            N = positions.shape[1]

            # Element information
            if "atoms_element" in grp:
                elements_raw = grp["atoms_element"][:]
                elements = [e.decode() if isinstance(e, bytes) else str(e)
                           for e in elements_raw]
            else:
                elements = ["C"] * N

            # Molecule boundaries (ligand = last molecule typically)
            if "molecules_begin_atom_index" in grp:
                mol_starts = grp["molecules_begin_atom_index"][:]
                # Convention: ligand is first molecule in MISATO
                if len(mol_starts) >= 2:
                    n_lig = int(mol_starts[1])
                else:
                    n_lig = N // 4  # fallback
            else:
                n_lig = N // 4

            is_ligand = np.zeros(N, dtype=bool)
            is_ligand[:n_lig] = True

            node_features = _build_node_features(
                elements=elements,
                is_ligand=is_ligand,
            )

            # Determine split
            split = "train"
            for s, ids in splits.items():
                if pdb_id in ids:
                    split = s
                    break

            # Save per-complex
            split_dir = out_stage_a / split
            split_dir.mkdir(parents=True, exist_ok=True)
            h5_out = split_dir / f"{pdb_id}.h5"

            with h5py.File(h5_out, "w") as fout:
                fout.create_dataset("positions", data=positions.astype(np.float32),
                                    compression="gzip")
                fout.create_dataset("node_features", data=node_features,
                                    compression="gzip")
                fout.attrs["complex_id"] = pdb_id
                fout.attrs["n_ligand"] = n_lig
                fout.attrs["n_frames"] = positions.shape[0]
                fout.attrs["source"] = "MISATO"

            n_processed += 1
            if n_processed % 500 == 0:
                logger.info(f"  Processed {n_processed}/{len(pdb_ids)}")

    logger.info(f"MISATO preprocessing complete: {n_processed} complexes → {out_stage_a}")


# ─────────────────────────────────────────────────────────────
# DD-13M Preprocessing (H5MD → HDF5 windows)
# ─────────────────────────────────────────────────────────────

def preprocess_dd13m(raw_dir: Path, out_dir: Path):
    """Convert DD-13M H5MD dissociation trajectories to HDF5.

    Each trajectory is a complete unbinding event — critical for Stage B.
    """
    import h5py

    dd13m_raw = raw_dir / "dd13m"
    if not dd13m_raw.exists():
        logger.error(f"DD-13M not found at {dd13m_raw}")
        return

    out_stage_b = out_dir / "stage_b"
    out_stage_b.mkdir(parents=True, exist_ok=True)

    # Find H5MD files
    h5md_files = sorted(dd13m_raw.rglob("*.h5md")) + sorted(dd13m_raw.rglob("*.h5"))
    pdb_files = sorted(dd13m_raw.rglob("*.pdb"))

    logger.info(f"DD-13M: found {len(h5md_files)} H5MD files, {len(pdb_files)} PDB files")

    if not h5md_files:
        # Try directory-based structure
        traj_dirs = sorted([
            d for d in dd13m_raw.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        logger.info(f"  Found {len(traj_dirs)} trajectory directories")

        for traj_dir in traj_dirs:
            _process_dd13m_directory(traj_dir, out_stage_b)
        return

    n_processed = 0
    for h5_path in h5md_files:
        try:
            with h5py.File(h5_path, "r") as f:
                # H5MD format: particles/all/position/value [T, N, 3]
                if "particles" in f:
                    pos_data = f["particles"]["all"]["position"]["value"]
                elif "positions" in f:
                    pos_data = f["positions"]
                else:
                    # Try to find position data
                    for key in f.keys():
                        if "position" in key.lower() or "coord" in key.lower():
                            pos_data = f[key]
                            break
                    else:
                        continue

                positions = pos_data[:].astype(np.float32)
                T, N = positions.shape[0], positions.shape[1]

                # Extract complex ID from path
                complex_id = h5_path.stem
                parent_name = h5_path.parent.name
                if parent_name != dd13m_raw.name:
                    complex_id = f"{parent_name}_{complex_id}"

                # Try to get atom types
                elements = ["C"] * N  # Default
                if "particles" in f and "species" in f["particles"]["all"]:
                    species = f["particles"]["all"]["species"]["value"][0]
                    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
                    elements = [elem_map.get(int(s), "C") for s in species]

                # For DD-13M, ligand identification varies
                # Try metadata first
                n_lig = N // 5  # Conservative default
                if "particles" in f and "all" in f["particles"]:
                    grp = f["particles"]["all"]
                    if "molecule_id" in grp:
                        mol_ids = grp["molecule_id"]["value"][0]
                        # Ligand is typically molecule 0 or the smallest molecule
                        unique_mols = np.unique(mol_ids)
                        mol_sizes = {m: np.sum(mol_ids == m) for m in unique_mols}
                        lig_mol = min(mol_sizes, key=mol_sizes.get)
                        n_lig = mol_sizes[lig_mol]
                        # Reorder so ligand atoms come first
                        lig_mask = mol_ids == lig_mol
                        lig_idx = np.where(lig_mask)[0]
                        prot_idx = np.where(~lig_mask)[0]
                        reorder = np.concatenate([lig_idx, prot_idx])
                        positions = positions[:, reorder, :]
                        elements = [elements[i] for i in reorder]

                is_ligand = np.zeros(N, dtype=bool)
                is_ligand[:n_lig] = True

                node_features = _build_node_features(
                    elements=elements,
                    is_ligand=is_ligand,
                )

                # Compute dissociation time (frame where ligand RMSD > threshold)
                lig_pos = positions[:, :n_lig, :]
                prot_com = positions[:, n_lig:, :].mean(axis=1, keepdims=True)  # [T, 1, 3]
                lig_com = lig_pos.mean(axis=1, keepdims=True)  # [T, 1, 3]
                distances = np.linalg.norm(lig_com - prot_com, axis=-1).squeeze()  # [T]
                initial_dist = distances[0]
                # Dissociation = distance increases by > 10 Angstroms
                dissoc_frames = np.where(distances > initial_dist + 10.0)[0]
                dissoc_time = int(dissoc_frames[0]) if len(dissoc_frames) > 0 else T - 1

                h5_out = out_stage_b / f"{complex_id}.h5"
                with h5py.File(h5_out, "w") as fout:
                    fout.create_dataset("positions", data=positions, compression="gzip")
                    fout.create_dataset("node_features", data=node_features,
                                        compression="gzip")
                    fout.attrs["complex_id"] = complex_id
                    fout.attrs["n_ligand"] = n_lig
                    fout.attrs["n_frames"] = T
                    fout.attrs["dissociation_frame"] = dissoc_time
                    fout.attrs["source"] = "DD-13M"

                n_processed += 1
                if n_processed % 100 == 0:
                    logger.info(f"  Processed {n_processed}/{len(h5md_files)}")

        except Exception as e:
            logger.warning(f"  Error processing {h5_path.name}: {e}")
            continue

    logger.info(f"DD-13M preprocessing complete: {n_processed} trajectories → {out_stage_b}")


def _process_dd13m_directory(traj_dir: Path, out_dir: Path):
    """Process a single DD-13M trajectory directory (PDB-based)."""
    import h5py

    complex_id = traj_dir.name
    pdb_files = sorted(traj_dir.glob("*.pdb"))

    if len(pdb_files) < 2:
        return

    try:
        import MDAnalysis as mda
    except ImportError:
        logger.error("MDAnalysis required for PDB-based DD-13M processing")
        return

    try:
        # Load first PDB to get topology
        u = mda.Universe(str(pdb_files[0]))
        N = len(u.atoms)

        positions = np.zeros((len(pdb_files), N, 3), dtype=np.float32)
        for i, pdb in enumerate(pdb_files):
            u_frame = mda.Universe(str(pdb))
            positions[i] = u_frame.atoms.positions

        protein = u.select_atoms("protein")
        ligand = u.select_atoms("not protein and not resname SOL WAT HOH TIP3 NA CL")
        n_lig = len(ligand) if len(ligand) > 0 else N // 5

        elements = [a.element if hasattr(a, "element") and a.element else a.name[:1]
                    for a in u.atoms]
        is_ligand = np.zeros(N, dtype=bool)
        is_ligand[:n_lig] = True

        node_features = _build_node_features(elements=elements, is_ligand=is_ligand)

        h5_out = out_dir / f"{complex_id}.h5"
        with h5py.File(h5_out, "w") as f:
            f.create_dataset("positions", data=positions, compression="gzip")
            f.create_dataset("node_features", data=node_features, compression="gzip")
            f.attrs["complex_id"] = complex_id
            f.attrs["n_ligand"] = n_lig
            f.attrs["n_frames"] = len(pdb_files)
            f.attrs["source"] = "DD-13M"

    except Exception as e:
        logger.warning(f"  Error processing {complex_id}: {e}")


# ─────────────────────────────────────────────────────────────
# Kinetics labels (BindingDB + KOFFI → Stage C)
# ─────────────────────────────────────────────────────────────

def preprocess_kinetics(raw_dir: Path, out_dir: Path):
    """Merge koff labels from BindingDB + KOFFI into a unified CSV.

    Output: data/processed/stage_c/kinetics_labels.csv
    Columns: complex_id, pdb_id, smiles, target_name, koff, log_koff, source
    """
    out_stage_c = out_dir / "stage_c"
    out_stage_c.mkdir(parents=True, exist_ok=True)

    all_records = []

    # ── BindingDB ──
    bdb_dir = raw_dir / "bindingdb"
    tsv_files = list(bdb_dir.glob("*.tsv")) + list(bdb_dir.glob("**/*.tsv"))

    if tsv_files:
        logger.info(f"Processing BindingDB TSV: {tsv_files[0].name}")
        n_bdb = 0

        with open(tsv_files[0], "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                koff_str = row.get("koff (s-1)", "").strip()
                if not koff_str or koff_str in ("", "N/A", "nan"):
                    continue

                try:
                    # Handle ranges like ">1e-3" or "<0.5"
                    koff_clean = koff_str.lstrip("<>~≈ ")
                    koff = float(koff_clean)
                    if koff <= 0:
                        continue
                except (ValueError, OverflowError):
                    continue

                smiles = row.get("Ligand SMILES", "")
                target = row.get("Target Name", "")
                pdb_id = row.get("PDB ID(s) of Target Chain", "")

                all_records.append({
                    "pdb_id": pdb_id.split(",")[0].strip() if pdb_id else "",
                    "smiles": smiles,
                    "target_name": target,
                    "koff": koff,
                    "log_koff": float(np.log10(koff)) if koff > 0 else float("nan"),
                    "source": "BindingDB",
                })
                n_bdb += 1

        logger.info(f"  BindingDB: {n_bdb} koff records extracted")
    else:
        logger.warning("  BindingDB TSV not found — skipping")

    # ── KOFFI ──
    koffi_file = raw_dir / "koffi" / "koffi_all.json"
    if koffi_file.exists():
        logger.info("Processing KOFFI JSON")

        with open(koffi_file) as f:
            koffi_data = json.load(f)

        n_koffi = 0
        for record in koffi_data:
            koff = record.get("koff") or record.get("dissociation_rate_constant")
            if koff is None:
                continue

            try:
                koff = float(koff)
                if koff <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            all_records.append({
                "pdb_id": record.get("pdb_id", ""),
                "smiles": record.get("ligand_smiles", record.get("analyte", "")),
                "target_name": record.get("ligand_name", record.get("target", "")),
                "koff": koff,
                "log_koff": float(np.log10(koff)),
                "source": "KOFFI",
            })
            n_koffi += 1

        logger.info(f"  KOFFI: {n_koffi} koff records extracted")
    else:
        logger.warning("  KOFFI JSON not found — skipping")

    if not all_records:
        logger.warning("No kinetics records found. Download data first:")
        logger.warning("  python scripts/download_data.py --stage-c")
        return

    # ── Deduplicate ──
    # Group by (target, smiles) — keep record with most metadata
    dedup = {}
    for r in all_records:
        key = (r["target_name"].lower().strip(), r["smiles"])
        if key not in dedup:
            dedup[key] = r
        else:
            # Prefer record with PDB ID
            if r["pdb_id"] and not dedup[key]["pdb_id"]:
                dedup[key] = r

    records = list(dedup.values())

    # Assign complex IDs
    for i, r in enumerate(records):
        r["complex_id"] = r["pdb_id"] if r["pdb_id"] else f"kinetics_{i:05d}"

    # ── Save ──
    csv_path = out_stage_c / "kinetics_labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "complex_id", "pdb_id", "smiles", "target_name",
            "koff", "log_koff", "source",
        ])
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Kinetics labels saved: {csv_path}")
    logger.info(f"  Total: {len(records)} unique koff records")
    logger.info(f"  With PDB IDs: {sum(1 for r in records if r['pdb_id'])}")

    # ── Statistics ──
    koffs = [r["koff"] for r in records]
    log_koffs = [r["log_koff"] for r in records]
    logger.info(f"  koff range: {min(koffs):.2e} — {max(koffs):.2e} s⁻¹")
    logger.info(f"  log₁₀(koff) range: {min(log_koffs):.2f} — {max(log_koffs):.2f}")

    by_source = defaultdict(int)
    for r in records:
        by_source[r["source"]] += 1
    for src, count in by_source.items():
        logger.info(f"  {src}: {count} records")


# ─────────────────────────────────────────────────────────────
# Split generation
# ─────────────────────────────────────────────────────────────

def generate_splits(out_dir: Path):
    """Generate train/val/test splits for Stage C kinetics data."""
    stage_c = out_dir / "stage_c"
    csv_path = stage_c / "kinetics_labels.csv"

    if not csv_path.exists():
        logger.warning("kinetics_labels.csv not found — run --kinetics first")
        return

    import pandas as pd

    df = pd.read_csv(csv_path)
    logger.info(f"Generating splits for {len(df)} kinetics records")

    sys.path.insert(0, str(PROJECT_ROOT))
    from tsnn.data.splits.random_split import random_split

    ids = df["complex_id"].tolist()

    train, val, test = random_split(ids, train_ratio=0.7, val_ratio=0.15, seed=42)

    splits_dir = stage_c / "splits"
    splits_dir.mkdir(exist_ok=True)

    for name, split_ids in [("train", train), ("val", val), ("test", test)]:
        with open(splits_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(split_ids))
        logger.info(f"  {name}: {len(split_ids)} complexes")

    logger.info(f"Splits saved to {splits_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw datasets for TSNN training"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mdd", action="store_true")
    parser.add_argument("--misato", action="store_true")
    parser.add_argument("--dd13m", action="store_true")
    parser.add_argument("--kinetics", action="store_true",
                        help="Merge koff labels from BindingDB + KOFFI")
    parser.add_argument("--splits", action="store_true",
                        help="Generate train/val/test splits for Stage C")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)

    args = parser.parse_args()

    any_selected = (args.all or args.mdd or args.misato or args.dd13m
                    or args.kinetics or args.splits)
    if not any_selected:
        parser.print_help()
        print("\n\nRecommended order:")
        print("  1. python scripts/preprocess_data.py --kinetics   # Stage C labels (fast)")
        print("  2. python scripts/preprocess_data.py --splits     # Generate splits")
        print("  3. python scripts/preprocess_data.py --mdd        # Stage A (needs MDD)")
        print("  4. python scripts/preprocess_data.py --dd13m      # Stage B (needs DD-13M)")
        print("  5. python scripts/preprocess_data.py --misato     # Stage A scale (needs MISATO)")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.mdd:
        preprocess_mdd(args.raw_dir, args.out_dir, args.window_size, args.stride)

    if args.all or args.misato:
        preprocess_misato(args.raw_dir, args.out_dir)

    if args.all or args.dd13m:
        preprocess_dd13m(args.raw_dir, args.out_dir)

    if args.all or args.kinetics:
        preprocess_kinetics(args.raw_dir, args.out_dir)

    if args.all or args.splits:
        generate_splits(args.out_dir)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
