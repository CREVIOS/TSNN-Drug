"""Benchmark runner for all six split types.

Runs evaluation across all splits and collects results into a table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from tsnn.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

SPLIT_NAMES = [
    "random",
    "cold_protein",
    "cold_scaffold",
    "pocket_cluster",
    "congeneric_series",
    "interaction_deleaked",
]


def run_benchmark(
    model,
    dataset_factory,
    split_configs: dict,
    device: str = "cuda",
    output_dir: str = "results",
) -> dict:
    """Run evaluation across all six benchmark splits.

    Args:
        model: Trained TSNN model.
        dataset_factory: Callable(split_name, split) -> Dataset.
        split_configs: Dict of split name -> config.
        device: Device for inference.
        output_dir: Directory to save results.

    Returns:
        Dict of split_name -> metrics dict.
    """
    model.eval()
    all_results = {}

    for split_name in SPLIT_NAMES:
        logger.info(f"Evaluating on {split_name} split...")

        try:
            test_dataset = dataset_factory(split_name, "test")
            metrics = evaluate_split(model, test_dataset, device)
            all_results[split_name] = metrics
            logger.info(f"  {split_name}: {metrics}")
        except Exception as e:
            logger.error(f"  Failed on {split_name}: {e}")
            all_results[split_name] = {"error": str(e)}

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX table
    latex = generate_latex_table(all_results)
    with open(output_path / "benchmark_table.tex", "w") as f:
        f.write(latex)

    return all_results


def evaluate_split(
    model,
    dataset,
    device: str = "cuda",
) -> dict:
    """Evaluate model on a single split.

    Args:
        model: TSNN model.
        dataset: Test dataset.
        device: Device.

    Returns:
        Metrics dict.
    """
    all_pred_koff = []
    all_true_koff = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            labels = sample["labels"]

            if labels.get("koff") is None:
                continue

            # Run forward pass
            # (This would need proper batching in production)
            output = _run_single_sample(model, sample, device)

            all_pred_koff.append(output["log_koff"])
            all_true_koff.append(labels["koff"])

    if not all_pred_koff:
        return {"error": "no valid samples"}

    pred = np.array(all_pred_koff)
    true = np.array(all_true_koff)

    return compute_all_metrics(pred, true)


def _run_single_sample(model, sample: dict, device: str) -> dict:
    """Run model on a single sample (simplified — production needs batching)."""
    frames = sample["frames"]

    frame_dicts = []
    for frame in frames:
        frame_dicts.append({
            "node_features": frame.x.to(device),
            "positions": frame.pos.to(device),
            "edge_index": frame.edge_index.to(device),
            "edge_attr": frame.edge_attr.to(device) if frame.edge_attr is not None else None,
        })

    cross_masks = [f.cross_edge_mask.to(device) for f in frames]
    n2c = torch.zeros(frames[0].num_nodes, dtype=torch.long, device=device)
    e2c = [torch.zeros(f.edge_index.shape[1], dtype=torch.long, device=device) for f in frames]

    output = model(frame_dicts, cross_masks, n2c, e2c, num_complexes=1)

    return {"log_koff": output.log_koff.cpu().item()}


def generate_latex_table(results: dict) -> str:
    """Generate a LaTeX table from benchmark results."""
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Split} & \textbf{RMSE} & \textbf{Spearman $\rho$} & \textbf{Pearson $r$} & \textbf{C-index} \\",
        r"\midrule",
    ]

    for split in SPLIT_NAMES:
        m = results.get(split, {})
        if "error" in m:
            lines.append(f"{split.replace('_', ' ').title()} & \\multicolumn{{4}}{{c}}{{Error}} \\\\")
        else:
            rmse = f"{m.get('rmse', float('nan')):.3f}"
            spearman = f"{m.get('spearman_rho', float('nan')):.3f}"
            pearson = f"{m.get('pearson_r', float('nan')):.3f}"
            cindex = f"{m.get('c_index', float('nan')):.3f}"
            name = split.replace("_", " ").title()
            lines.append(f"{name} & {rmse} & {spearman} & {pearson} & {cindex} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines)
