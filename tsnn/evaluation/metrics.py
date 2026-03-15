"""Evaluation metrics for kinetics prediction.

Implements all metrics from Section 7.3 of the paper:
- Kinetics regression: RMSE, Spearman rho, Pearson r on log k_off
- Censored prediction: Concordance index (C-index), Integrated Brier Score
- Mechanistic early warning: AUROC/AUPRC for contact-break, lead time
- Efficiency: Training time, inference time, memory footprint
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)


def compute_all_metrics(
    pred_log_koff: np.ndarray,
    true_log_koff: np.ndarray,
    pred_hazard: np.ndarray | None = None,
    event_times: np.ndarray | None = None,
    censored: np.ndarray | None = None,
    pred_contact_break: np.ndarray | None = None,
    true_contact_break: np.ndarray | None = None,
    pred_lead_times: np.ndarray | None = None,
    true_lead_times: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        pred_log_koff: Predicted log k_off values.
        true_log_koff: True log k_off values.
        pred_hazard: Predicted hazard rates [T, B].
        event_times: Event time bins [B].
        censored: Censoring indicators [B].
        pred_contact_break: Predicted contact break probabilities.
        true_contact_break: True contact break labels.
        pred_lead_times: Predicted lead times before rupture.
        true_lead_times: True lead times.

    Returns:
        Dict of metric name -> value.
    """
    metrics = {}

    # Filter NaN
    valid = ~(np.isnan(pred_log_koff) | np.isnan(true_log_koff))
    if valid.sum() >= 2:
        pred_v = pred_log_koff[valid]
        true_v = true_log_koff[valid]

        # Kinetics regression metrics
        metrics["rmse"] = float(np.sqrt(mean_squared_error(true_v, pred_v)))
        metrics["spearman_rho"] = float(stats.spearmanr(true_v, pred_v).correlation)
        metrics["pearson_r"] = float(stats.pearsonr(true_v, pred_v)[0])
        metrics["mae"] = float(np.mean(np.abs(true_v - pred_v)))

    # Concordance index
    if event_times is not None and censored is not None:
        c_index = concordance_index(pred_log_koff, event_times, censored)
        metrics["c_index"] = c_index

    # Integrated Brier Score
    if pred_hazard is not None and event_times is not None:
        ibs = integrated_brier_score(pred_hazard, event_times, censored)
        metrics["integrated_brier_score"] = ibs

    # Contact-break prediction metrics
    if pred_contact_break is not None and true_contact_break is not None:
        valid_cb = ~(np.isnan(pred_contact_break) | np.isnan(true_contact_break))
        if valid_cb.sum() > 0 and len(np.unique(true_contact_break[valid_cb])) > 1:
            metrics["contact_auroc"] = float(
                roc_auc_score(true_contact_break[valid_cb],
                              pred_contact_break[valid_cb])
            )
            metrics["contact_auprc"] = float(
                average_precision_score(true_contact_break[valid_cb],
                                        pred_contact_break[valid_cb])
            )

    # Lead time analysis
    if pred_lead_times is not None and true_lead_times is not None:
        valid_lt = ~(np.isnan(pred_lead_times) | np.isnan(true_lead_times))
        if valid_lt.sum() >= 2:
            metrics["lead_time_correlation"] = float(
                stats.spearmanr(pred_lead_times[valid_lt],
                                true_lead_times[valid_lt]).correlation
            )
            metrics["mean_lead_time"] = float(np.mean(pred_lead_times[valid_lt]))

    return metrics


def concordance_index(
    predicted_risks: np.ndarray,
    event_times: np.ndarray,
    censored: np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    Higher risk prediction should correspond to shorter survival time.

    Args:
        predicted_risks: Higher value = higher risk (e.g., predicted log k_off).
        event_times: Time of event or last observation.
        censored: Boolean array, True if censored.

    Returns:
        C-index in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    n = len(predicted_risks)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider comparable pairs
            if censored[i] and censored[j]:
                continue  # Both censored: not comparable

            # Determine which event is earlier
            if event_times[i] < event_times[j]:
                if censored[i]:
                    continue  # i is censored before j: not comparable
                # i has earlier event; should have higher risk
                if predicted_risks[i] > predicted_risks[j]:
                    concordant += 1
                elif predicted_risks[i] < predicted_risks[j]:
                    discordant += 1
                else:
                    tied += 1
            elif event_times[j] < event_times[i]:
                if censored[j]:
                    continue
                if predicted_risks[j] > predicted_risks[i]:
                    concordant += 1
                elif predicted_risks[j] < predicted_risks[i]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5

    return (concordant + 0.5 * tied) / total


def integrated_brier_score(
    pred_hazard: np.ndarray,
    event_times: np.ndarray,
    censored: np.ndarray | None = None,
) -> float:
    """Compute Integrated Brier Score with IPCW weighting.

    Uses Inverse Probability of Censoring Weighting (IPCW) via
    Kaplan-Meier estimate of the censoring distribution, following
    Graf et al. (1999) and scikit-survival conventions.

    Args:
        pred_hazard: Predicted hazard rates [T, B].
        event_times: Event time bins [B].
        censored: Censoring indicators [B]. True = censored.

    Returns:
        IBS (lower is better).
    """
    T, B = pred_hazard.shape

    # Compute predicted survival function from hazard
    survival = np.cumprod(1.0 - pred_hazard, axis=0)  # [T, B]

    if censored is None:
        censored = np.zeros(B, dtype=bool)

    # Estimate censoring survival G(t) via Kaplan-Meier on the censoring distribution
    # (swap event/censoring: "events" are censorings, "censorings" are events)
    G = _kaplan_meier_censoring(event_times, censored, T)

    brier_scores = []
    for t in range(T):
        bs_t = 0.0
        weight_sum = 0.0

        G_t = max(G[t], 1e-8)  # G(t) for alive-at-t weighting

        for b in range(B):
            t_b = int(event_times[b])

            if t_b <= t and not censored[b]:
                # Event occurred at or before t: S(t) should be near 0
                # IPCW weight: 1 / G(t_b)
                G_tb = max(G[min(t_b, T - 1)], 1e-8)
                w = 1.0 / G_tb
                bs_t += w * survival[t, b] ** 2
                weight_sum += w
            elif t_b > t:
                # Still alive at t (event or censoring after t)
                # IPCW weight: 1 / G(t)
                w = 1.0 / G_t
                bs_t += w * (1.0 - survival[t, b]) ** 2
                weight_sum += w
            # else: censored before t — excluded (no info)

        if weight_sum > 0:
            brier_scores.append(bs_t / weight_sum)

    if not brier_scores:
        return 0.0

    return float(np.mean(brier_scores))


def _kaplan_meier_censoring(
    event_times: np.ndarray,
    censored: np.ndarray,
    T: int,
) -> np.ndarray:
    """Estimate censoring survival function G(t) via Kaplan-Meier.

    For IPCW, we treat censoring as the "event" and actual events as
    "censored" — i.e., we estimate the probability of NOT being censored.

    Args:
        event_times: Observation times [B].
        censored: True if censored [B].
        T: Number of time bins.

    Returns:
        G: Censoring survival function [T]. G[t] = P(not censored by t).
    """
    B = len(event_times)
    G = np.ones(T)

    for t in range(T):
        # Number at risk at time t
        at_risk = np.sum(event_times >= t)
        # Number of censorings at time t
        n_censored_at_t = np.sum((event_times == t) & censored)

        if at_risk > 0 and t > 0:
            G[t] = G[t - 1] * (1.0 - n_censored_at_t / at_risk)
        elif t > 0:
            G[t] = G[t - 1]

    return G
