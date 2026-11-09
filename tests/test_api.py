from fastapi.testclient import TestClient
from apex.api import create_app
from apex.data import generate_dataset
from apex.training import train_models


def sample_record():
    return {"tenure_months": 24, "monthly_charges": 89.50,
            "support_tickets": 2, "usage_gb": 75.0,
            "contract_type": "monthly", "payment_method": "electronic"}


def test_api_health_prediction_and_drift_endpoints(tmp_path):
    train_models(generate_dataset(rows=400, random_state=21), tmp_path, track_mlflow=False)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["model_loaded"] is True
        assert health.json()["model_version"] == "0.3.0"
        assert client.get("/health/live").json()["status"] == "alive"
        assert client.get("/health/ready").json()["status"] == "ready"
        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["model_name"] in {"logistic_regression", "random_forest"}
        prediction = client.post("/predict", json={"records": [sample_record()]})
        assert prediction.status_code == 200
        predicted = prediction.json()["predictions"][0]
        assert 0 <= predicted["churn_probability"] <= 1
        assert predicted["risk_band"] in {"low", "medium", "high"}
        drift = client.post("/monitor/drift", json={"records": [sample_record(), sample_record()]})
        assert drift.status_code == 200
        assert drift.json()["status"] in {"STABLE", "DRIFT_DETECTED"}
        assert drift.json()["evaluated_at"]
        latest = client.get("/monitor/drift/latest")
        assert latest.status_code == 200
        assert latest.json()["status"] == drift.json()["status"]


def test_api_rejects_invalid_feature_ranges():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [{**sample_record(), "tenure_months": -10}]})
        assert response.status_code == 422


def test_api_reports_not_ready_without_model(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"records": [sample_record()]}).status_code == 503



def test_api_performance_monitoring_persists_metrics(tmp_path):
    train_models(generate_dataset(rows=400, random_state=31), tmp_path, track_mlflow=False)
    app = create_app(tmp_path)
    payload = {
        "records": [sample_record(), {**sample_record(), "tenure_months": 4, "support_tickets": 8}],
        "actual_churn": [0, 1],
    }
    with TestClient(app) as client:
        response = client.post("/monitor/performance", json=payload)
        assert response.status_code == 200
        report = response.json()
        assert report["sample_size"] == 2
        assert set(report["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert client.get("/monitor/performance/latest").json() == report
        assert (tmp_path / "performance_latest.json").is_file()


def test_api_performance_monitoring_rejects_unbalanced_or_mismatched_labels(tmp_path):
    train_models(generate_dataset(rows=400, random_state=32), tmp_path, track_mlflow=False)
    with TestClient(create_app(tmp_path)) as client:
        mismatched = client.post("/monitor/performance", json={
            "records": [sample_record(), sample_record()], "actual_churn": [1, 0, 1]
        })
        assert mismatched.status_code == 422
        one_class = client.post("/monitor/performance", json={
            "records": [sample_record(), sample_record()], "actual_churn": [1, 1]
        })
        assert one_class.status_code == 422
