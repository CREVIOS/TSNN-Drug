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

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.training.trainer import TSNNTrainer
from tsnn.losses.combined import CombinedLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_config_from_args() -> dict:
    """Parse command-line args as key=value overrides."""
    config = {
        "model": {},
        "data": {},
        "training": {"stage": "c"},
        "losses": {},
    }

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

            if len(parts) == 2:
                section, param = parts
                if section in config:
                    config[section][param] = value

    return config


def build_model_config(overrides: dict) -> TSNNConfig:
    """Build TSNNConfig from overrides."""
    config = TSNNConfig()
    for key, value in overrides.get("model", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_dummy_dataloader(batch_size: int = 2) -> DataLoader:
    """Create a dummy DataLoader for testing the pipeline."""
    from tsnn.data.datasets.mdd import MDDDataset
    from tsnn.data.graph_builder import GraphBuilderConfig

    # Use MDD with placeholder data
    dataset = MDDDataset(
        root="/tmp/tsnn_dummy_data",
        split="train",
        graph_config=GraphBuilderConfig(),
        window_size=5,
    )

    # If no real data, create minimal synthetic dataset
    if len(dataset) == 0:
        logger.warning("No data found. Using synthetic data for pipeline testing.")
        return _create_synthetic_loader(batch_size)

    return DataLoader(dataset, batch_size=1, shuffle=True)


def _create_synthetic_loader(batch_size: int):
    """Create a synthetic DataLoader for end-to-end testing."""
    from torch_geometric.data import Data

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            N, T = 30, 5
            N_lig = 10
            frames = []
            for t in range(T):
                pos = torch.randn(N, 3) * 5.0
                # Simple node features
                x = torch.randn(N, 32)
                # Radius graph
                dist = torch.cdist(pos, pos)
                mask = (dist < 5.0) & (dist > 0.01)
                edge_index = mask.nonzero(as_tuple=False).t()
                E = edge_index.shape[1]
                edge_attr = torch.randn(E, 28)

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

            import numpy as np
            return {
                "frames": frames,
                "labels": {
                    "koff": float(np.random.normal(0, 2)),
                    "censored": False,
                    "contact_break_times": None,
                    "dissociation_time": None,
                },
                "complex_id": f"synthetic_{idx}",
            }

    return DataLoader(SyntheticDataset(), batch_size=1, shuffle=True)


def main():
    args = build_config_from_args()
    model_config = build_model_config(args)
    stage = args.get("training", {}).get("stage", "c")

    logger.info(f"TSNN Training — Stage {stage.upper()}")
    logger.info(f"Model config: {model_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    trainer = TSNNTrainer(
        config=model_config,
        device=device,
        output_dir=args.get("training", {}).get("output_dir", "checkpoints"),
    )

    # Create data loaders (uses synthetic data if real data unavailable)
    train_loader = create_dummy_dataloader()
    val_loader = None  # TODO: create validation loader

    loss_fn = CombinedLoss(
        alpha=args.get("losses", {}).get("alpha", 0.1),
        beta=args.get("losses", {}).get("beta", 0.05),
        gamma=args.get("losses", {}).get("gamma", 0.01),
        use_survival=model_config.use_survival,
    )

    if stage == "a":
        trainer.run_stage_a(train_loader, val_loader, num_epochs=50)
    elif stage == "b":
        trainer.run_stage_b(train_loader, val_loader, num_epochs=30)
    elif stage == "c":
        trainer.run_stage_c(train_loader, val_loader, loss_fn=loss_fn, num_epochs=100)
    elif stage == "all":
        trainer.run_stage_a(train_loader, val_loader, num_epochs=50)
        trainer.run_stage_b(train_loader, val_loader, num_epochs=30)
        trainer.run_stage_c(train_loader, val_loader, loss_fn=loss_fn, num_epochs=100)
    else:
        logger.error(f"Unknown stage: {stage}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
