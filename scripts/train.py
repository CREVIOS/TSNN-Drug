#!/usr/bin/env python3
"""Main training entry point for TSNN.

Usage:
    python scripts/train.py                    # Full pipeline (Stage C)
    python scripts/train.py training.stage=a   # Stage A only
    python scripts/train.py training.stage=b   # Stage B only
    python scripts/train.py training.stage=c   # Stage C only
    python scripts/train.py model.householder_depth=8  # Ablation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.training.trainer import TSNNTrainer
from tsnn.losses.combined import CombinedLoss
from tsnn.data.collate import temporal_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config() -> dict:
    """Load default.yaml and apply CLI overrides."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {CONFIG_PATH}")
    else:
        config = {"model": {}, "data": {}, "training": {"stage": "c"}, "losses": {}}

    # Apply CLI overrides: key.subkey=value
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parts = key.split(".")
            # Type conversion
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            # Set nested key
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value

    # Ensure numeric strings (like scientific notation) are converted
    _coerce_numerics(config)
    return config


def _coerce_numerics(d: dict):
    """Recursively convert string values that look numeric to float/int."""
    for k, v in d.items():
        if isinstance(v, dict):
            _coerce_numerics(v)
        elif isinstance(v, str):
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass


def build_model_config(cfg: dict) -> TSNNConfig:
    """Build TSNNConfig from merged config dict."""
    model_cfg = cfg.get("model", {})
    config = TSNNConfig()
    for key, value in model_cfg.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """Create a DataLoader with proper collation."""
    from tsnn.data.graph_builder import GraphBuilderConfig

    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    graph_config = GraphBuilderConfig(
        pocket_cutoff=data_cfg.get("pocket_cutoff", 10.0),
        context_cutoff=data_cfg.get("context_cutoff", 15.0),
        edge_cutoff=data_cfg.get("edge_cutoff", 5.0),
        include_water=data_cfg.get("include_water", True),
        num_rbf=data_cfg.get("num_rbf", 16),
        rbf_cutoff=data_cfg.get("rbf_cutoff", 15.0),
    )

    stage = training_cfg.get("stage", "c")
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"

    # --- Stage C: try kinetics dataset with real koff labels ---
    if stage == "c":
        labels_csv = processed_dir / "stage_c" / "kinetics_labels.csv"
        split_file = processed_dir / "stage_c" / "splits" / f"{split}.txt"
        traj_dir = processed_dir / "stage_a"  # Reuse Stage A trajectories

        if labels_csv.exists():
            try:
                from tsnn.data.datasets.kinetics import KineticsDataset
                dataset = KineticsDataset(
                    labels_csv=str(labels_csv),
                    trajectory_dir=str(traj_dir) if traj_dir.exists() else None,
                    split_file=str(split_file) if split_file.exists() else None,
                    window_size=data_cfg.get("window_size", 20),
                    edge_cutoff=data_cfg.get("edge_cutoff", 5.0),
                )
                if len(dataset) > 0:
                    logger.info(f"Loaded kinetics dataset: {len(dataset)} samples ({split})")
                    return DataLoader(
                        dataset, batch_size=1, shuffle=(split == "train"),
                        num_workers=training_cfg.get("num_workers", 0),
                        collate_fn=temporal_collate_fn,
                    )
            except Exception as e:
                logger.warning(f"Failed to load kinetics dataset: {e}")

    # --- Stage B: try DD-13M dissociation trajectories ---
    if stage == "b":
        dd13m_dir = processed_dir / "stage_b"
        if dd13m_dir.exists() and list(dd13m_dir.glob("*.h5")):
            try:
                from tsnn.data.datasets.dd13m import DD13MDataset
                dataset = DD13MDataset(
                    root=str(dd13m_dir),
                    window_size=data_cfg.get("window_size", 20),
                    stride=data_cfg.get("stride", 10),
                    edge_cutoff=data_cfg.get("edge_cutoff", 5.0),
                )
                if len(dataset) > 0:
                    logger.info(f"Loaded DD-13M dataset: {len(dataset)} samples")
                    return DataLoader(
                        dataset, batch_size=1, shuffle=(split == "train"),
                        num_workers=training_cfg.get("num_workers", 0),
                        collate_fn=temporal_collate_fn,
                    )
            except Exception as e:
                logger.warning(f"Failed to load DD-13M dataset: {e}")

    # --- Stage A: try MISATO or MDD ---
    if stage in ("a", "all"):
        for source in ["misato", "mdd"]:
            source_dir = processed_dir / "stage_a" / source
            if not source_dir.exists():
                continue
            split_dir = source_dir / split if (source_dir / split).exists() else source_dir
            h5_files = list(split_dir.glob("*.h5"))
            if not h5_files:
                continue

            try:
                from tsnn.data.datasets.misato import MISATODataset
                dataset = MISATODataset(
                    root=str(source_dir),
                    split=split,
                    window_size=data_cfg.get("window_size", 20),
                    stride=data_cfg.get("stride", 10),
                    edge_cutoff=data_cfg.get("edge_cutoff", 5.0),
                )
                if len(dataset) > 0:
                    logger.info(f"Loaded {source.upper()} dataset: {len(dataset)} samples ({split})")
                    return DataLoader(
                        dataset, batch_size=1, shuffle=(split == "train"),
                        num_workers=training_cfg.get("num_workers", 0),
                        collate_fn=temporal_collate_fn,
                    )
            except Exception as e:
                logger.warning(f"Failed to load {source} dataset: {e}")

    # --- Fallback: try legacy MDD loader ---
    try:
        from tsnn.data.datasets.mdd import MDDDataset
        data_root = data_cfg.get("root", "/tmp/tsnn_data")
        dataset = MDDDataset(
            root=data_root, split=split, graph_config=graph_config,
            window_size=data_cfg.get("window_size", 20),
            stride=data_cfg.get("stride", 10),
        )
        if len(dataset) > 0:
            return DataLoader(
                dataset, batch_size=1, shuffle=(split == "train"),
                num_workers=training_cfg.get("num_workers", 0),
                collate_fn=temporal_collate_fn,
            )
    except Exception:
        pass

    logger.warning("No real data found. Using synthetic data for pipeline testing.")
    return _create_synthetic_loader(cfg)


def _create_synthetic_loader(cfg: dict):
    """Create a synthetic DataLoader for end-to-end testing."""
    from torch_geometric.data import Data

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    node_dim = model_cfg.get("node_input_dim", 29)
    edge_dim = model_cfg.get("edge_input_dim", 28)

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            import numpy as np
            N, T = 30, 5
            N_lig = 10
            frames = []
            for t in range(T):
                pos = torch.randn(N, 3) * 5.0
                x = torch.randn(N, node_dim)
                dist = torch.cdist(pos, pos)
                mask = (dist < 5.0) & (dist > 0.01)
                edge_index = mask.nonzero(as_tuple=False).t()
                E = edge_index.shape[1]
                edge_attr = torch.randn(E, edge_dim)

                is_ligand = torch.zeros(N, dtype=torch.bool)
                is_ligand[:N_lig] = True
                cross_mask = is_ligand[edge_index[0]] != is_ligand[edge_index[1]]

                frame = Data(
                    x=x, pos=pos, edge_index=edge_index,
                    edge_attr=edge_attr,
                    cross_edge_mask=cross_mask,
                    is_ligand=is_ligand,
                    num_nodes=N,
                )
                frames.append(frame)

            return {
                "frames": frames,
                "labels": {
                    "koff": float(np.random.normal(0, 2)),
                    "censored": False,
                    "contact_break_times": None,
                    "dissociation_time": float(np.random.randint(1, T)),
                    "series_id": idx % 5,
                },
                "complex_id": f"synthetic_{idx}",
            }

    return DataLoader(
        SyntheticDataset(),
        batch_size=1,
        shuffle=True,
        collate_fn=temporal_collate_fn,
    )


def main():
    cfg = load_config()
    model_config = build_model_config(cfg)
    training_cfg = cfg.get("training", {})
    stage = training_cfg.get("stage", "c")

    logger.info(f"TSNN Training — Stage {stage.upper()}")
    logger.info(f"Model config: {model_config}")

    device = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    trainer = TSNNTrainer(
        config=model_config,
        device=device,
        output_dir=training_cfg.get("output_dir", "checkpoints"),
        lr=training_cfg.get("lr", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
        grad_clip=training_cfg.get("grad_clip", 1.0),
        mixed_precision=training_cfg.get("mixed_precision", True),
    )

    train_loader = create_dataloader(cfg, "train")
    val_loader = None

    loss_cfg = cfg.get("losses", {})
    loss_fn = CombinedLoss(
        alpha=loss_cfg.get("alpha", 0.1),
        beta=loss_cfg.get("beta", 0.05),
        gamma=loss_cfg.get("gamma", 0.01),
        use_survival=model_config.use_survival,
    )

    stage_a_cfg = training_cfg.get("stage_a", {})
    stage_b_cfg = training_cfg.get("stage_b", {})
    stage_c_cfg = training_cfg.get("stage_c", {})

    if stage == "a":
        trainer.run_stage_a(train_loader, val_loader,
                            num_epochs=stage_a_cfg.get("num_epochs", 50),
                            lr=stage_a_cfg.get("lr", 3e-4))
    elif stage == "b":
        trainer.run_stage_b(train_loader, val_loader,
                            num_epochs=stage_b_cfg.get("num_epochs", 30),
                            lr=stage_b_cfg.get("lr", 1e-4))
    elif stage == "c":
        trainer.run_stage_c(train_loader, val_loader, loss_fn=loss_fn,
                            num_epochs=stage_c_cfg.get("num_epochs", 100),
                            lr=stage_c_cfg.get("lr", 5e-5))
    elif stage == "all":
        trainer.run_stage_a(train_loader, val_loader,
                            num_epochs=stage_a_cfg.get("num_epochs", 50))
        trainer.run_stage_b(train_loader, val_loader,
                            num_epochs=stage_b_cfg.get("num_epochs", 30))
        trainer.run_stage_c(train_loader, val_loader, loss_fn=loss_fn,
                            num_epochs=stage_c_cfg.get("num_epochs", 100))
    else:
        logger.error(f"Unknown stage: {stage}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
