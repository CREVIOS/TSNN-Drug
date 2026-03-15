"""Mechanistic analysis: sheaf disagreement as contact-rupture early warning.

Implements the mechanistic analysis from the paper:
- Track sheaf disagreement D_uv(t) per contact over time
- Identify contacts with rising disagreement before rupture
- Compute lead time: frames before macroscopic separation where D exceeds threshold
- Generate per-complex visualizations
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def analyze_disagreement_trajectories(
    disagreement_sequences: list[list[np.ndarray]],
    contact_break_times: list[dict],
    edge_indices: list[list[np.ndarray]],
    threshold_quantile: float = 0.9,
) -> dict:
    """Analyze whether rising sheaf disagreement predicts contact rupture.

    Args:
        disagreement_sequences: Per-complex list of per-frame disagreement arrays.
        contact_break_times: Per-complex dict of {(u,v): break_frame}.
        edge_indices: Per-complex list of per-frame edge index arrays.
        threshold_quantile: Quantile for disagreement threshold.

    Returns:
        Analysis results dict.
    """
    all_lead_times = []
    all_aurocs = []
    rising_before_break = 0
    total_breaks = 0

    for cx_idx, (d_seq, breaks, ei_seq) in enumerate(
        zip(disagreement_sequences, contact_break_times, edge_indices)
    ):
        if breaks is None:
            continue

        T = len(d_seq)
        for (u, v), break_frame in breaks.items():
            # Find this edge across frames
            d_values = []
            for t in range(min(T, break_frame + 1)):
                ei = ei_seq[t]
                d_t = d_seq[t]
                # Find edge (u,v) in this frame
                mask = (ei[0] == u) & (ei[1] == v)
                if mask.any():
                    d_values.append(float(d_t[mask.nonzero(as_tuple=True)[0][0]]))

            if len(d_values) < 3:
                continue

            total_breaks += 1
            d_arr = np.array(d_values)

            # Check if disagreement is rising before break
            if len(d_arr) >= 5:
                last_half = d_arr[len(d_arr)//2:]
                first_half = d_arr[:len(d_arr)//2]
                if last_half.mean() > first_half.mean():
                    rising_before_break += 1

            # Compute lead time: first frame where D exceeds threshold
            threshold = np.quantile(d_arr, threshold_quantile)
            exceeds = np.where(d_arr > threshold)[0]
            if len(exceeds) > 0:
                lead_time = break_frame - exceeds[0]
                all_lead_times.append(max(0, lead_time))

    results = {
        "total_contacts_analyzed": total_breaks,
        "rising_before_break_fraction": (
            rising_before_break / total_breaks if total_breaks > 0 else 0
        ),
        "mean_lead_time_frames": (
            float(np.mean(all_lead_times)) if all_lead_times else 0
        ),
        "median_lead_time_frames": (
            float(np.median(all_lead_times)) if all_lead_times else 0
        ),
        "std_lead_time_frames": (
            float(np.std(all_lead_times)) if all_lead_times else 0
        ),
    }

    logger.info(f"Mechanistic analysis: {results}")
    return results


def compute_contact_break_auroc(
    disagreements: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute AUROC for contact break prediction from disagreement.

    Args:
        disagreements: Sheaf disagreement values [N_contacts].
        labels: Binary break labels [N_contacts].

    Returns:
        AUROC score.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return 0.5

    return float(roc_auc_score(labels, disagreements))


def generate_case_study(
    complex_id: str,
    disagreement_sequence: list[np.ndarray],
    edge_index_sequence: list[np.ndarray],
    contact_break_times: dict | None,
    output_dir: str | Path,
) -> None:
    """Generate a case study visualization for a single complex.

    Creates disagreement trajectory plots for key contacts.

    Args:
        complex_id: Complex identifier.
        disagreement_sequence: Per-frame disagreement arrays.
        edge_index_sequence: Per-frame edge indices.
        contact_break_times: Optional contact break times.
        output_dir: Output directory for plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping visualization")
        return

    T = len(disagreement_sequence)

    # Find top contacts by max disagreement
    contact_max_d = {}
    for t in range(T):
        ei = edge_index_sequence[t]
        d = disagreement_sequence[t]
        for e_idx in range(ei.shape[1]):
            u, v = int(ei[0, e_idx]), int(ei[1, e_idx])
            key = (min(u, v), max(u, v))
            contact_max_d[key] = max(contact_max_d.get(key, 0), float(d[e_idx]))

    # Get top-10 contacts by max disagreement
    top_contacts = sorted(contact_max_d.keys(),
                           key=lambda k: contact_max_d[k], reverse=True)[:10]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot disagreement trajectories
    ax = axes[0]
    for u, v in top_contacts:
        d_values = []
        frames = []
        for t in range(T):
            ei = edge_index_sequence[t]
            d = disagreement_sequence[t]
            mask = ((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u))
            if mask.any():
                d_values.append(float(d[mask.nonzero()[0][0]]))
                frames.append(t)

        if d_values:
            label = f"({u},{v})"
            if contact_break_times and (u, v) in contact_break_times:
                label += f" [break@{contact_break_times[(u,v)]}]"
            ax.plot(frames, d_values, label=label, alpha=0.7)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Sheaf Disagreement D_uv(t)")
    ax.set_title(f"Complex {complex_id}: Top Contact Disagreement Trajectories")
    ax.legend(fontsize=7, ncol=2)

    # Plot mean disagreement over all contacts
    ax2 = axes[1]
    mean_d = [float(d.mean()) if len(d) > 0 else 0 for d in disagreement_sequence]
    ax2.plot(range(T), mean_d, "k-", linewidth=2)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Mean Disagreement")
    ax2.set_title("Mean Sheaf Disagreement Over Time")

    # Mark contact break times
    if contact_break_times:
        for (u, v), bt in contact_break_times.items():
            if bt < T:
                ax2.axvline(bt, color="red", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / f"{complex_id}_disagreement.png", dpi=150)
    plt.close()

    logger.info(f"Case study saved: {output_dir / complex_id}_disagreement.png")
