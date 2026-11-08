"""Feature-distribution drift monitoring with population stability index (PSI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURE_COLUMNS


def _safe_proportions(counts: np.ndarray) -> np.ndarray:
    proportions = counts.astype(float) / max(float(counts.sum()), 1.0)
    return np.clip(proportions, 1e-6, None)


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Both reference and current numeric samples must contain values")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        low, high = float(np.min(ref)), float(np.max(ref))
        if low == high:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = np.linspace(low, high, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    expected = _safe_proportions(ref_counts)
    actual = _safe_proportions(cur_counts)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("__MISSING__").astype(str)
    cur = current.fillna("__MISSING__").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        raise ValueError("Both reference and current categorical samples must contain values")
    ref_counts = np.array([(ref == item).sum() for item in categories], dtype=float)
    cur_counts = np.array([(cur == item).sum() for item in categories], dtype=float)
    expected = _safe_proportions(ref_counts)
    actual = _safe_proportions(cur_counts)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def evaluate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.20,
) -> dict:
    """Compare a production sample with the training reference distributions."""
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    missing_ref = sorted(set(FEATURE_COLUMNS) - set(reference.columns))
    missing_cur = sorted(set(FEATURE_COLUMNS) - set(current.columns))
    if missing_ref or missing_cur:
        raise ValueError(
            f"Missing feature columns; reference={missing_ref}, current={missing_cur}"
        )
    if current.empty or reference.empty:
        raise ValueError("Reference and current data must not be empty")

    scores = {}
    for column in NUMERIC_FEATURES:
        scores[column] = numeric_psi(reference[column], current[column])
    for column in CATEGORICAL_FEATURES:
        scores[column] = categorical_psi(reference[column], current[column])

    drifted = sorted(name for name, score in scores.items() if score >= threshold)
    return {
        "status": "DRIFT_DETECTED" if drifted else "STABLE",
        "threshold": threshold,
        "features": {
            name: {
                "psi": round(score, 6),
                "status": "DRIFT" if score >= threshold else "STABLE",
            }
            for name, score in scores.items()
        },
        "drifted_features": drifted,
        "sample_sizes": {
            "reference": int(len(reference)),
            "current": int(len(current)),
        },
    }
