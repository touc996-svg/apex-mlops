"""FastAPI inference and data-drift service."""
from __future__ import annotations
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from apex.monitoring import evaluate_drift


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


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    resolved_dir = Path(artifact_dir or os.getenv("APEX_ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_path = resolved_dir / "model.joblib"
        reference_path = resolved_dir / "reference_features.csv"
        metadata_path = resolved_dir / "model_metadata.json"
        drift_path = resolved_dir / "drift_latest.json"
        app.state.artifact_dir = resolved_dir
        app.state.model = joblib.load(model_path) if model_path.is_file() else None
        app.state.reference = pd.read_csv(reference_path) if reference_path.is_file() else None
        app.state.model_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        app.state.latest_drift = json.loads(drift_path.read_text()) if drift_path.is_file() else None
        yield

    application = FastAPI(
        title="APEX MLOps Prediction API",
        description="Customer churn predictions and feature-drift monitoring.",
        version="0.2.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health():
        loaded = getattr(application.state, "model", None) is not None
        metadata = getattr(application.state, "model_metadata", {})
        return {"status": "healthy" if loaded else "model_unavailable",
                "model_loaded": loaded, "model_name": metadata.get("model_name"),
                "model_version": metadata.get("model_version")}

    @application.get("/health/live")
    def liveness():
        return {"status": "alive"}

    @application.get("/health/ready")
    def readiness():
        model_loaded = getattr(application.state, "model", None) is not None
        reference_loaded = getattr(application.state, "reference", None) is not None
        if not model_loaded or not reference_loaded:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready", "model_loaded": model_loaded,
                "reference_loaded": reference_loaded})
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
        probabilities, predictions = model.predict_proba(frame)[:, 1], model.predict(frame)
        metadata = getattr(application.state, "model_metadata", {})
        rows = []
        for label, probability in zip(predictions, probabilities):
            score = float(probability)
            rows.append({
                "churn_prediction": int(label),
                "churn_probability": round(score, 6),
                "risk_band": "high" if score >= 0.70 else "medium" if score >= 0.40 else "low",
            })
        return {"model": metadata.get("model_name", "APEX customer churn classifier"),
                "model_version": metadata.get("model_version", "unknown"),
                "predictions": rows}

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
        temp_path.write_text(json.dumps(report, indent=2) + "\n")
        temp_path.replace(report_path)
        application.state.latest_drift = report
        return report

    @application.get("/monitor/drift/latest")
    def latest_drift():
        report = getattr(application.state, "latest_drift", None)
        if report is None:
            raise HTTPException(status_code=404, detail="No drift evaluation has been recorded")
        return report

    return application


app = create_app()
