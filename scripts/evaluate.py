#!/usr/bin/env python3
"""Evaluation entry point: run benchmark across all 6 splits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.evaluation.benchmark import run_benchmark, evaluate_split
from tsnn.utils.io import load_checkpoint
from tsnn.data.collate import temporal_collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stage_c_best.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"

    # Load config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model from config
    model_cfg = cfg.get("model", {})
    config = TSNNConfig()
    for k, v in model_cfg.items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = TSNN(config).to(device)

    # Load checkpoint
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint at {checkpoint_path}, using random weights")

    # Build dataset factory
    def dataset_factory(split_name: str, split: str):
        """Create test dataset for a given split."""
        from tsnn.data.datasets.mdd import MDDDataset
        from tsnn.data.graph_builder import GraphBuilderConfig

        data_cfg = cfg.get("data", {})
        graph_config = GraphBuilderConfig(
            pocket_cutoff=data_cfg.get("pocket_cutoff", 10.0),
            context_cutoff=data_cfg.get("context_cutoff", 15.0),
            edge_cutoff=data_cfg.get("edge_cutoff", 5.0),
            include_water=data_cfg.get("include_water", True),
        )

        data_root = data_cfg.get("root", "/tmp/tsnn_data")
        split_root = Path(data_root) / "splits" / split_name
        if not split_root.exists():
            # Fall back to default split
            split_root = Path(data_root)

        dataset = MDDDataset(
            root=data_root,
            split=split,
            graph_config=graph_config,
            window_size=data_cfg.get("window_size", 20),
            random_window=False,
        )
        if len(dataset) == 0:
            raise FileNotFoundError(
                f"No data for split={split_name}/{split} in {data_root}"
            )
        return dataset

    logger.info("Running benchmark evaluation across all 6 splits...")
    results = run_benchmark(
        model=model,
        dataset_factory=dataset_factory,
        split_configs={},
        device=device,
        output_dir=output_dir,
    )

    logger.info("Benchmark results:")
    for split_name, metrics in results.items():
        logger.info(f"  {split_name}: {metrics}")


if __name__ == "__main__":
    main()
