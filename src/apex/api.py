"""Secure FastAPI inference, model metadata, drift, and performance monitoring."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from apex.monitoring import evaluate_drift
from apex.security import PUBLIC_PATHS, api_key_matches, security_settings


class CustomerRecord(BaseModel):
    tenure_months: float = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_tickets: int = Field(ge=0, le=100)
    usage_gb: float = Field(ge=0, le=10000)
    contract_type: str = Field(min_length=1, max_length=40)
    payment_method: str = Field(min_length=1, max_length=60)


class PredictionRequest(BaseModel):
    records: list[CustomerRecord] = Field(min_length=1, max_length=500)


class DriftRequest(BaseModel):
    records: list[CustomerRecord] = Field(min_length=2, max_length=5000)
    threshold: float = Field(default=0.20, gt=0, le=5)


class PerformanceRequest(BaseModel):
    records: list[CustomerRecord] = Field(min_length=2, max_length=5000)
    actual_churn: list[int] = Field(min_length=2, max_length=5000)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    resolved_dir = Path(artifact_dir or os.getenv("APEX_ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        security_settings()
        model_path = resolved_dir / "model.joblib"
        reference_path = resolved_dir / "reference_features.csv"
        metadata_path = resolved_dir / "model_metadata.json"
        manifest_path = resolved_dir / "artifact_manifest.json"
        application.state.artifact_dir = resolved_dir
        if model_path.is_file() and manifest_path.is_file():
            manifest = _read_json(manifest_path) or {}
            expected_hash = manifest.get("sha256")
            if expected_hash:
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if not hmac.compare_digest(digest, expected_hash):
                    raise RuntimeError("Model artifact checksum does not match artifact_manifest.json")
        application.state.model = joblib.load(model_path) if model_path.is_file() else None
        application.state.reference = pd.read_csv(reference_path) if reference_path.is_file() else None
        application.state.model_metadata = _read_json(metadata_path) or {}
        application.state.latest_drift = _read_json(resolved_dir / "drift_latest.json")
        application.state.latest_performance = _read_json(resolved_dir / "performance_latest.json")
        yield

    application = FastAPI(
        title="APEX MLOps Prediction API",
        description="Versioned churn inference with API-key protection, feature drift, and live model-performance monitoring.",
        version="0.3.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def protect_private_routes(request: Request, call_next):
        if request.url.path not in PUBLIC_PATHS and not request.url.path.startswith(("/docs/", "/redoc/")):
            try:
                _, expected_key = security_settings()
            except RuntimeError as exc:
                return JSONResponse(status_code=503, content={"detail": str(exc)})
            if expected_key and not api_key_matches(request.headers.get("X-API-Key"), expected_key):
                return JSONResponse(status_code=401, content={"detail": "A valid X-API-Key header is required"})
        return await call_next(request)

    @application.get("/")
    def root():
        return {"service": "APEX MLOps", "version": "0.3.0", "docs": "/docs"}

    @application.get("/health")
    def health():
        loaded = getattr(application.state, "model", None) is not None
        metadata = getattr(application.state, "model_metadata", {})
        return {
            "status": "healthy" if loaded else "model_unavailable",
            "model_loaded": loaded,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "registry": metadata.get("model_registry", {}),
        }

    @application.get("/health/live")
    def liveness():
        return {"status": "alive"}

    @application.get("/health/ready")
    def readiness():
        model_loaded = getattr(application.state, "model", None) is not None
        reference_loaded = getattr(application.state, "reference", None) is not None
        if not model_loaded or not reference_loaded:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "model_loaded": model_loaded, "reference_loaded": reference_loaded})
        return {"status": "ready", "model_loaded": True, "reference_loaded": True}

    @application.get("/model/info")
    def model_info():
        metadata = getattr(application.state, "model_metadata", {})
        if not metadata:
            raise HTTPException(status_code=503, detail="Model metadata is not available")
        return metadata

    @application.post("/predict")
    def predict(request: PredictionRequest):
        model = getattr(application.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available")
        frame = pd.DataFrame([record.model_dump() for record in request.records])
        probabilities = model.predict_proba(frame)[:, 1]
        predictions = model.predict(frame)
        metadata = getattr(application.state, "model_metadata", {})
        rows = []
        for label, probability in zip(predictions, probabilities):
            score = float(probability)
            rows.append({"churn_prediction": int(label), "churn_probability": round(score, 6),
                         "risk_band": "high" if score >= 0.70 else "medium" if score >= 0.40 else "low"})
        return {"model": metadata.get("model_name", "APEX customer churn classifier"),
                "model_version": metadata.get("model_version", "unknown"), "predictions": rows}

    @application.post("/monitor/drift")
    def monitor_drift(request: DriftRequest):
        reference = getattr(application.state, "reference", None)
        if reference is None:
            raise HTTPException(status_code=503, detail="Training reference data is unavailable")
        current = pd.DataFrame([record.model_dump() for record in request.records])
        try:
            report = evaluate_drift(reference, current, threshold=request.threshold)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        report["evaluated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        report_path = Path(application.state.artifact_dir) / "drift_latest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = report_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(report_path)
        application.state.latest_drift = report
        return report

    @application.get("/monitor/drift/latest")
    def latest_drift():
        report = getattr(application.state, "latest_drift", None)
        if report is None:
            raise HTTPException(status_code=404, detail="No drift evaluation has been recorded")
        return report

    @application.post("/monitor/performance")
    def monitor_performance(request: PerformanceRequest):
        if len(request.records) != len(request.actual_churn):
            raise HTTPException(status_code=422, detail="records and actual_churn must contain the same number of items")
        if any(label not in (0, 1) for label in request.actual_churn) or len(set(request.actual_churn)) != 2:
            raise HTTPException(status_code=422, detail="actual_churn must contain both binary labels 0 and 1")
        model = getattr(application.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not available")
        frame = pd.DataFrame([record.model_dump() for record in request.records])
        probabilities = model.predict_proba(frame)[:, 1]
        predictions = model.predict(frame)
        actual = request.actual_churn
        report = {
            "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "sample_size": len(actual),
            "metrics": {
                "accuracy": float(accuracy_score(actual, predictions)),
                "precision": float(precision_score(actual, predictions, zero_division=0)),
                "recall": float(recall_score(actual, predictions, zero_division=0)),
                "f1": float(f1_score(actual, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(actual, probabilities)),
            },
        }
        report_path = Path(application.state.artifact_dir) / "performance_latest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = report_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(report_path)
        application.state.latest_performance = report
        return report

    @application.get("/monitor/performance/latest")
    def latest_performance():
        report = getattr(application.state, "latest_performance", None)
        if report is None:
            raise HTTPException(status_code=404, detail="No model performance evaluation has been recorded")
        return report

    return application


app = create_app()
