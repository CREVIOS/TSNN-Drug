"""I/O utilities for checkpoints, logging, and configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
    scheduler: Any = None,
) -> None:
    """Save a training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str = "cpu",
) -> dict:
    """Load a training checkpoint.

    Returns:
        Dict with 'epoch' and 'metrics' keys.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    logger.info(f"Loaded checkpoint from {path} (epoch {state.get('epoch', '?')})")
    return {"epoch": state.get("epoch", 0), "metrics": state.get("metrics", {})}


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
