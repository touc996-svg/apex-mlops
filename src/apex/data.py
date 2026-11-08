"""Data ingestion helpers and reproducible synthetic customer-churn data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from apex.schema import FEATURE_COLUMNS, TARGET_COLUMN


def generate_dataset(rows: int = 1800, random_state: int = 42) -> pd.DataFrame:
    """Generate a reproducible, clearly synthetic churn-classification dataset."""
    if rows < 100:
        raise ValueError("rows must be at least 100 for a useful train/test split")

    raw, target = make_classification(
        n_samples=rows,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.76, 0.24],
        class_sep=1.0,
        flip_y=0.025,
        random_state=random_state,
    )
    frame = pd.DataFrame({
        "tenure_months": np.clip(raw[:, 0] * 10 + 30, 1, 72).round().astype(int),
        "monthly_charges": np.clip(raw[:, 1] * 17 + 75, 15, 180).round(2),
        "support_tickets": np.clip(np.abs(raw[:, 2] * 1.5).round(), 0, 15).astype(int),
        "usage_gb": np.clip(raw[:, 3] * 35 + 65, 1, 500).round(1),
        "contract_type": np.select(
            [raw[:, 4] > 0.55, raw[:, 4] < -0.55],
            ["monthly", "two_year"],
            default="annual",
        ),
        "payment_method": np.where(raw[:, 5] > 0.15, "electronic", "bank_transfer"),
        TARGET_COLUMN: target.astype(int),
    })
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_dataset(csv_path: str | Path | None = None, rows: int = 1800) -> pd.DataFrame:
    """Load a user CSV or generate synthetic training data when no file is supplied."""
    if csv_path:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        frame = pd.read_csv(path)
    else:
        frame = generate_dataset(rows=rows)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame[FEATURE_COLUMNS].isnull().all(axis=None):
        raise ValueError("All feature values are null")
    labels = set(frame[TARGET_COLUMN].dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError("churn must contain both binary labels 0 and 1")
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
