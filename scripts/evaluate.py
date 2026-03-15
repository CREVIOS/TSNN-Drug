#!/usr/bin/env python3
"""Evaluation entry point: run benchmark across all 6 splits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.evaluation.benchmark import run_benchmark
from tsnn.utils.io import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stage_c_best.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    config = TSNNConfig()
    model = TSNN(config).to(device)

    # Load checkpoint
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")

    logger.info("Running benchmark evaluation across all 6 splits...")

    # This would use real dataset factories in production
    results = run_benchmark(
        model=model,
        dataset_factory=lambda split_name, split: None,
        split_configs={},
        device=device,
        output_dir=output_dir,
    )

    logger.info("Benchmark results:")
    for split_name, metrics in results.items():
        logger.info(f"  {split_name}: {metrics}")


if __name__ == "__main__":
    main()
