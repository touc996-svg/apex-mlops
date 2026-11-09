"""Train, compare, evaluate, version, track, and safely promote churn models."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from apex.features import build_candidates
from apex.schema import FEATURE_COLUMNS, TARGET_COLUMN


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    return tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _register_and_promote(mlflow, run_id: str, selected_name: str, score: float, tracking_uri: str) -> dict:
    """Register every candidate; move the champion alias only when holdout ROC-AUC improves."""
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException

    registry_name = os.getenv("APEX_REGISTERED_MODEL_NAME", "APEXCustomerChurn")
    min_improvement = float(os.getenv("APEX_PROMOTION_MIN_IMPROVEMENT", "0.002"))
    if min_improvement < 0 or not __import__("math").isfinite(min_improvement):
        raise ValueError("APEX_PROMOTION_MIN_IMPROVEMENT must be a finite non-negative number")
    client = MlflowClient(tracking_uri=tracking_uri)
    previous_score = None
    previous_version = None
    try:
        champion = client.get_model_version_by_alias(registry_name, "champion")
        previous_version = str(champion.version)
        run = client.get_run(champion.run_id)
        previous_score = run.data.metrics.get("selected_holdout_roc_auc")
    except MlflowException as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise

    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/selected_model",
        name=registry_name,
        await_registration_for=120,
    )
    promote = previous_score is None or score >= float(previous_score) + min_improvement
    client.set_model_version_tag(registry_name, registered.version, "holdout_roc_auc", str(score))
    client.set_model_version_tag(registry_name, registered.version, "candidate_model", selected_name)
    client.set_model_version_tag(registry_name, registered.version, "promotion_status", "champion" if promote else "candidate")
    if promote:
        client.set_registered_model_alias(registry_name, "champion", registered.version)
    return {
        "enabled": True,
        "registered_model": registry_name,
        "registered_version": str(registered.version),
        "alias": "champion" if promote else None,
        "promoted": promote,
        "previous_champion_version": previous_version,
        "previous_champion_roc_auc": previous_score,
        "candidate_roc_auc": score,
        "minimum_improvement": min_improvement,
    }


def train_models(frame: pd.DataFrame, output_dir: str | Path = "artifacts", *,
                 test_size: float = 0.20, random_state: int = 42,
                 track_mlflow: bool = False, tracking_uri: str | None = None) -> dict:
    """Compare two pipelines and persist the best holdout ROC-AUC candidate."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns))
    if missing:
        raise ValueError("Training frame missing columns: " + ", ".join(missing))
    if len(frame) < 30:
        raise ValueError("Training frame must contain at least 30 rows")
    clean = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        if column in {"tenure_months", "monthly_charges", "support_tickets", "usage_gb"}:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
            if clean[column].isna().any() or not pd.Series(clean[column]).map(lambda x: pd.notna(x) and abs(float(x)) != float("inf")).all():
                raise ValueError(f"Feature {column} contains missing or non-finite values")
        else:
            clean[column] = clean[column].astype("string").str.strip()
            if clean[column].isna().any() or (clean[column] == "").any():
                raise ValueError(f"Feature {column} contains missing or blank values")
    labels = set(pd.to_numeric(clean[TARGET_COLUMN], errors="coerce").dropna().unique())
    if clean[TARGET_COLUMN].isna().any() or labels != {0, 1}:
        raise ValueError("Training target must contain both binary labels 0 and 1")
    x, y = clean[FEATURE_COLUMNS].copy(), clean[TARGET_COLUMN].astype(int)
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
    trained_at = datetime.now(timezone.utc)
    version_stamp = trained_at.strftime("%Y%m%dT%H%M%SZ")
    version_dir = output / "versions" / version_stamp
    version_dir.mkdir(parents=True, exist_ok=True)
    versioned_model_path = version_dir / "model.joblib"
    shutil.copy2(model_path, versioned_model_path)
    checksum = _sha256(model_path)
    resolved_uri = resolve_tracking_uri(tracking_uri)
    metadata = {
        "model_name": selected_name,
        "selected_model": selected_name,
        "model_version": "0.3.0",
        "model_build": version_stamp,
        "problem_type": "binary_classification",
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "positive_class": 1,
        "selection_metric": "holdout_roc_auc",
        "trained_at": trained_at.isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
        "test_size": test_size,
        "candidate_metrics": results,
        "selected_metrics": results[selected_name],
        "synthetic_data": bool(frame.attrs.get("synthetic_data", False)),
        "model_sha256": checksum,
        "versioned_model_path": str(versioned_model_path),
        "tracking_uri": resolved_uri,
        "model_registry": {"enabled": False},
    }
    metadata_path, metrics_path = output / "model_metadata.json", output / "metrics.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model_file": "model.joblib",
        "versioned_model_file": str(versioned_model_path.relative_to(output)),
        "sha256": checksum,
        "model_build": version_stamp,
        "model_version": metadata["model_version"],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    registry_info = {"enabled": False}
    if track_mlflow:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "apex-customer-churn"))
        with mlflow.start_run(run_name=selected_name) as active_run:
            for key, value in {
                "selected_model": selected_name,
                "selection_metric": "holdout_roc_auc",
                "random_state": random_state,
                "test_size": test_size,
                "training_rows": len(x_train),
                "test_rows": len(x_test),
                "tracking_uri": resolved_uri,
                "model_sha256": checksum,
                "synthetic_data": bool(frame.attrs.get("synthetic_data", False)),
            }.items():
                mlflow.log_param(key, value)
            for model_name, model_metrics in results.items():
                for metric_name, metric_value in model_metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            mlflow.log_metric("selected_holdout_roc_auc", results[selected_name]["roc_auc"])
            for artifact in (model_path, versioned_model_path, reference_path, metadata_path, metrics_path, output / "artifact_manifest.json"):
                mlflow.log_artifact(str(artifact))
            mlflow.sklearn.log_model(fitted[selected_name], artifact_path="selected_model")
            run_id = active_run.info.run_id
        if os.getenv("APEX_ENABLE_MODEL_REGISTRY", "true").strip().lower() not in {"0", "false", "no"}:
            registry_info = _register_and_promote(mlflow, run_id, selected_name, results[selected_name]["roc_auc"], resolved_uri)
            metadata["model_registry"] = registry_info
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "selected_model": selected_name,
        "metrics": results,
        "model_path": str(model_path),
        "versioned_model_path": str(versioned_model_path),
        "reference_path": str(reference_path),
        "metadata_path": str(metadata_path),
        "tracking_uri": resolved_uri,
        "model_registry": metadata.get("model_registry", registry_info),
        "metadata": metadata,
    }
