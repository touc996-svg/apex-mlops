"""Train, compare, evaluate, and persist candidate churn models."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from apex.features import build_candidates
from apex.schema import FEATURE_COLUMNS, TARGET_COLUMN


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Use SQLite by default; allow explicit CLI or environment overrides."""
    return tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"


def _evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train_models(frame: pd.DataFrame, output_dir: str | Path = "artifacts", *,
                 test_size: float = 0.20, random_state: int = 42,
                 track_mlflow: bool = False, tracking_uri: str | None = None) -> dict:
    """Compare two pipelines and select the best holdout ROC-AUC model."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns))
    if missing:
        raise ValueError("Training frame missing columns: " + ", ".join(missing))
    if len(frame) < 30:
        raise ValueError("Training frame must contain at least 30 rows")
    x, y = frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training target must contain both binary labels 0 and 1")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "reference_features.csv"
    x_train.to_csv(reference_path, index=False)
    results, fitted = {}, {}
    for name, candidate in build_candidates().items():
        cv_scores = cross_val_score(candidate, x_train, y_train, cv=3, scoring="roc_auc", n_jobs=1)
        candidate.fit(x_train, y_train)
        metrics = _evaluate(candidate, x_test, y_test)
        metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
        metrics["cv_roc_auc_std"] = float(cv_scores.std())
        results[name], fitted[name] = metrics, candidate
    selected_name = max(results, key=lambda name: results[name]["roc_auc"])
    model_path = output / "model.joblib"
    joblib.dump(fitted[selected_name], model_path)
    metadata = {
        "model_name": selected_name,
        "selected_model": selected_name,
        "model_version": "0.2.0",
        "problem_type": "binary_classification",
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "positive_class": 1,
        "selection_metric": "holdout_roc_auc",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
        "test_size": test_size,
        "candidate_metrics": results,
        "selected_metrics": results[selected_name],
        "synthetic_data": bool(frame.attrs.get("synthetic_data", False)),
    }
    metadata_path, metrics_path = output / "model_metadata.json", output / "metrics.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n")
    resolved_uri = resolve_tracking_uri(tracking_uri)
    if track_mlflow:
        import mlflow
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment("apex-customer-churn")
        with mlflow.start_run(run_name=selected_name):
            for key, value in {
                "selected_model": selected_name, "selection_metric": "holdout_roc_auc",
                "random_state": random_state, "test_size": test_size,
                "training_rows": len(x_train), "test_rows": len(x_test),
                "tracking_uri": resolved_uri,
            }.items():
                mlflow.log_param(key, value)
            for model_name, model_metrics in results.items():
                for metric_name, metric_value in model_metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            mlflow.log_metric("selected_holdout_roc_auc", results[selected_name]["roc_auc"])
            for artifact in (model_path, reference_path, metadata_path, metrics_path):
                mlflow.log_artifact(str(artifact))
    return {
        "selected_model": selected_name, "metrics": results,
        "model_path": str(model_path), "reference_path": str(reference_path),
        "metadata_path": str(metadata_path), "tracking_uri": resolved_uri,
        "metadata": metadata,
    }
