"""Ablation experiment runner.

Runs all 10 ablations from Section 8 of the paper:
1. No sheaf transport: Q_uv(t) = I
2. Static frames: time-invariant U_v
3. No E(3) encoder (RBF distance embeddings only)
4. No Stage B dissociation pretraining
5. No contact-level auxiliary task
6. No water-mediated edges
7. No survival head (direct log k_off regression only)
8. Atom-level vs hybrid residue-atom graph
9. Short (<=5ns) vs long (>=50ns) observation window
10. Householder depth k in {1, 2, 4, 8}
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tsnn.model.tsnn import TSNNConfig

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "no_sheaf_transport": {
        "description": "Q_uv(t) = I (identity transport)",
        "config_overrides": {"identity_transport": True},
    },
    "static_frames": {
        "description": "Time-invariant U_v (static frames)",
        "config_overrides": {"static_frames": True},
    },
    "no_equivariant_encoder": {
        "description": "No E(3) encoder (RBF distance embeddings only)",
        "config_overrides": {"no_equivariant": True},
    },
    "no_stage_b": {
        "description": "No Stage B dissociation pretraining",
        "config_overrides": {},  # Handled in training pipeline
        "training_override": {"skip_stage_b": True},
    },
    "no_contact_auxiliary": {
        "description": "No contact-level auxiliary task",
        "config_overrides": {},
        "loss_override": {"contact_aux_weight": 0.0},
    },
    "no_water_edges": {
        "description": "No water-mediated edges",
        "config_overrides": {},
        "data_override": {"include_water": False},
    },
    "no_survival_head": {
        "description": "Direct log k_off regression only",
        "config_overrides": {"use_survival": False},
    },
    "atom_only_graph": {
        "description": "Atom-level graph (no residue-level nodes)",
        "config_overrides": {},
        "data_override": {"graph_resolution": "atom"},
    },
    "short_window": {
        "description": "Short observation window (<=5ns equivalent)",
        "config_overrides": {},
        "data_override": {"window_size": 5},
    },
    "long_window": {
        "description": "Long observation window (>=50ns equivalent)",
        "config_overrides": {},
        "data_override": {"window_size": 50},
    },
    "householder_k1": {
        "description": "Householder depth k=1",
        "config_overrides": {"householder_depth": 1},
    },
    "householder_k2": {
        "description": "Householder depth k=2",
        "config_overrides": {"householder_depth": 2},
    },
    "householder_k4": {
        "description": "Householder depth k=4 (default)",
        "config_overrides": {"householder_depth": 4},
    },
    "householder_k8": {
        "description": "Householder depth k=8",
        "config_overrides": {"householder_depth": 8},
    },
}


@dataclass
class AblationOverrides:
    """Complete set of overrides for an ablation experiment."""
    config: TSNNConfig
    training_override: dict
    loss_override: dict
    data_override: dict


def get_ablation_config(
    base_config: TSNNConfig, ablation_name: str
) -> AblationOverrides:
    """Create a full ablation override bundle.

    Returns an AblationOverrides with model config AND training/loss/data
    overrides so callers can apply them all (fixes issue #12).
    """
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    ablation = ABLATION_CONFIGS[ablation_name]
    config = copy.deepcopy(base_config)

    for key, value in ablation.get("config_overrides", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Config key {key} not found in TSNNConfig")

    return AblationOverrides(
        config=config,
        training_override=ablation.get("training_override", {}),
        loss_override=ablation.get("loss_override", {}),
        data_override=ablation.get("data_override", {}),
    )


def run_all_ablations(
    base_config: TSNNConfig,
    train_fn,
    evaluate_fn,
    output_dir: str = "ablation_results",
) -> dict:
    """Run all ablation experiments.

    Args:
        base_config: Base model config.
        train_fn: Function(overrides: AblationOverrides, name: str) -> model.
            Receives the full AblationOverrides bundle so it can apply
            training_override, loss_override, and data_override.
        evaluate_fn: Function(model) -> metrics dict.
        output_dir: Output directory.

    Returns:
        Dict of ablation_name -> metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run baseline first
    logger.info("Running baseline...")
    baseline_overrides = AblationOverrides(
        config=base_config,
        training_override={},
        loss_override={},
        data_override={},
    )
    baseline_model = train_fn(baseline_overrides, "baseline")
    results["baseline"] = evaluate_fn(baseline_model)

    # Run each ablation
    for abl_name in ABLATION_CONFIGS:
        logger.info(f"Running ablation: {abl_name}")
        try:
            overrides = get_ablation_config(base_config, abl_name)
            model = train_fn(overrides, abl_name)
            metrics = evaluate_fn(model)
            results[abl_name] = metrics
            logger.info(f"  {abl_name}: {metrics}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[abl_name] = {"error": str(e)}

    # Save results
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate comparison table
    latex = generate_ablation_table(results)
    with open(output_path / "ablation_table.tex", "w") as f:
        f.write(latex)

    return results


def generate_ablation_table(results: dict) -> str:
    """Generate LaTeX ablation table."""
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Ablation} & \textbf{RMSE} & \textbf{Spearman $\rho$} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]

    baseline = results.get("baseline", {})
    baseline_rmse = baseline.get("rmse", float("nan"))
    baseline_rho = baseline.get("spearman_rho", float("nan"))

    lines.append(
        f"-- & Full model (baseline) & {baseline_rmse:.3f} & {baseline_rho:.3f} & -- \\\\"
    )
    lines.append(r"\midrule")

    ablation_order = [
        ("1", "no_sheaf_transport"),
        ("2", "static_frames"),
        ("3", "no_equivariant_encoder"),
        ("4", "no_stage_b"),
        ("5", "no_contact_auxiliary"),
        ("6", "no_water_edges"),
        ("7", "no_survival_head"),
        ("8", "atom_only_graph"),
        ("9a", "short_window"),
        ("9b", "long_window"),
        ("10a", "householder_k1"),
        ("10b", "householder_k2"),
        ("10c", "householder_k8"),
    ]

    for num, abl_name in ablation_order:
        m = results.get(abl_name, {})
        desc = ABLATION_CONFIGS.get(abl_name, {}).get("description", abl_name)
        if "error" in m:
            lines.append(f"{num} & {desc} & -- & -- & -- \\\\")
        else:
            rmse = m.get("rmse", float("nan"))
            rho = m.get("spearman_rho", float("nan"))
            delta = rho - baseline_rho if not (
                np.isnan(rho) or np.isnan(baseline_rho)
            ) else float("nan")
            lines.append(
                f"{num} & {desc} & {rmse:.3f} & {rho:.3f} & {delta:+.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines)


# Need numpy for nan checks in table generation
import numpy as np
