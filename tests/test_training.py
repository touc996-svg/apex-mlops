import json
import joblib
import pytest
from apex.data import generate_dataset
from apex.training import resolve_tracking_uri, train_models


def test_training_compares_models_and_persists_selected_pipeline(tmp_path):
    frame = generate_dataset(rows=500, random_state=11)
    result = train_models(frame, tmp_path, track_mlflow=False)
    assert result["selected_model"] in {"logistic_regression", "random_forest"}
    assert set(result["metrics"]) == {"logistic_regression", "random_forest"}
    for metrics in result["metrics"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"} <= set(metrics)
        assert 0 <= metrics["roc_auc"] <= 1
    model = joblib.load(tmp_path / "model.joblib")
    assert len(model.predict(frame.iloc[:5].drop(columns=["churn"]))) == 5
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert metadata["selected_model"] == result["selected_model"]
    assert metadata["model_name"] == result["selected_model"]
    assert metadata["model_version"] == "0.3.0"
    assert (tmp_path / "reference_features.csv").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_mlflow_defaults_to_sqlite_and_allows_overrides(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri() == "sqlite:///mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    assert resolve_tracking_uri() == "sqlite:///custom.db"
    assert resolve_tracking_uri("sqlite:///explicit.db") == "sqlite:///explicit.db"


def test_training_rejects_too_small_dataset(tmp_path):
    frame = generate_dataset(rows=100, random_state=14).head(20)
    with pytest.raises(ValueError, match="at least 30"):
        train_models(frame, tmp_path, track_mlflow=False)
