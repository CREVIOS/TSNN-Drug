"""Visualization utilities for sheaf disagreement and contact analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def plot_disagreement_heatmap(
    disagreements: np.ndarray,
    contact_labels: list[str] | None = None,
    time_labels: list[str] | None = None,
    title: str = "Sheaf Disagreement Heatmap",
    output_path: str | Path | None = None,
) -> None:
    """Plot a heatmap of sheaf disagreement over time and contacts.

    Args:
        disagreements: [T, E] array of disagreement values.
        contact_labels: Names for contacts (y-axis).
        time_labels: Names for time steps (x-axis).
        title: Plot title.
        output_path: If given, save to file.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not available; skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(12, max(4, disagreements.shape[1] * 0.3)))

    im = ax.imshow(
        disagreements.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Contact")
    ax.set_title(title)

    if contact_labels is not None:
        ax.set_yticks(range(len(contact_labels)))
        ax.set_yticklabels(contact_labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="D_uv(t)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved heatmap to {output_path}")
    plt.close()


def plot_survival_curves(
    survival: np.ndarray,
    complex_ids: list[str] | None = None,
    true_event_times: np.ndarray | None = None,
    title: str = "Predicted Survival Curves",
    output_path: str | Path | None = None,
) -> None:
    """Plot predicted survival curves.

    Args:
        survival: [T, B] survival function values.
        complex_ids: Complex identifiers for legend.
        true_event_times: If known, mark with vertical lines.
        title: Plot title.
        output_path: If given, save to file.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping survival plot")
        return

    T, B = survival.shape
    fig, ax = plt.subplots(figsize=(10, 6))

    for b in range(min(B, 20)):  # Limit to 20 curves
        label = complex_ids[b] if complex_ids else f"Complex {b}"
        ax.plot(range(T), survival[:, b], label=label, alpha=0.7)

        if true_event_times is not None:
            t_event = int(true_event_times[b])
            if t_event < T:
                ax.axvline(t_event, color="red", alpha=0.2, linestyle="--")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("S(t)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    if B <= 10:
        ax.legend(fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "spearman_rho",
    title: str = "Ablation Study Results",
    output_path: str | Path | None = None,
) -> None:
    """Bar chart comparing ablation results.

    Args:
        results: Dict of ablation_name -> metrics dict.
        metric: Which metric to plot.
        title: Plot title.
        output_path: If given, save to file.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping ablation plot")
        return

    names = []
    values = []
    for name, metrics in results.items():
        if metric in metrics:
            names.append(name.replace("_", "\n"))
            values.append(metrics[metric])

    if not values:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 6))
    bars = ax.bar(range(len(names)), values, color="steelblue", alpha=0.8)

    # Highlight baseline
    if "baseline" in results and metric in results["baseline"]:
        baseline_val = results["baseline"][metric]
        ax.axhline(baseline_val, color="red", linestyle="--", label="Baseline")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
