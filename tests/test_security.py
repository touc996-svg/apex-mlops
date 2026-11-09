import pytest
from fastapi.testclient import TestClient

from apex.api import create_app


def test_development_mode_allows_local_requests_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_ENV", "development")
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/predict", json={"records": [{
            "tenure_months": 10, "monthly_charges": 50, "support_tickets": 1,
            "usage_gb": 20, "contract_type": "monthly", "payment_method": "electronic"
        }]})
        assert response.status_code == 503  # Auth passed; no model was provisioned.


def test_configured_key_protects_inference_route(tmp_path, monkeypatch):
    key = "test-only-apex-key-0123456789-abcdef"
    monkeypatch.setenv("APEX_ENV", "development")
    monkeypatch.setenv("APEX_API_KEY", key)
    with TestClient(create_app(tmp_path)) as client:
        payload = {"records": [{"tenure_months": 10, "monthly_charges": 50,
                   "support_tickets": 1, "usage_gb": 20,
                   "contract_type": "monthly", "payment_method": "electronic"}]}
        assert client.post("/predict", json=payload).status_code == 401
        assert client.post("/predict", json=payload, headers={"X-API-Key": key}).status_code == 503
        assert client.get("/health/live").status_code == 200


def test_production_requires_a_strong_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_ENV", "production")
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required"):
        with TestClient(create_app(tmp_path)):
            pass


def test_weak_api_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_ENV", "development")
    monkeypatch.setenv("APEX_API_KEY", "too-short")
    with pytest.raises(RuntimeError, match="32 characters"):
        with TestClient(create_app(tmp_path)):
            pass
