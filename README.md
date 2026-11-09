# APEX — End-to-End MLOps Platform

**Platform release: 0.3.0**

APEX demonstrates a reproducible machine-learning lifecycle for binary customer-churn classification: data ingestion, feature preprocessing, model comparison, cross-validation, holdout evaluation, MLflow experiment tracking, model persistence, API inference, and feature-drift monitoring.

> **Data notice:** The built-in dataset is synthetic and is intended for engineering demonstrations only. It is not real customer data and must not be used for business decisions.

## Architecture

CSV / Synthetic Data → Contract Validation → Feature Preprocessing → Logistic Regression + Random Forest → Cross-validation + Holdout Evaluation → MLflow Tracking (SQLite) → Saved Model → FastAPI Prediction Service → PSI Feature Drift Monitoring

## Stack

Python 3.12, pandas, NumPy, scikit-learn, MLflow Tracking + Model Registry, FastAPI, Pydantic, Uvicorn, PSI drift monitoring, outcome-based performance monitoring, API-key security, Pytest, Docker, GitHub Actions.

## Setup

```bash
cd ~/Projects/apex-mlops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Train and track experiments

SQLite is the default local MLflow tracking backend:

```bash
python -m apex.train --rows 1800 --output artifacts
```

The workflow compares Logistic Regression and Random Forest on the same stratified holdout set, performs three-fold ROC-AUC cross-validation on the training split, selects the best holdout ROC-AUC candidate, persists the full pipeline and reference features, and logs both candidates' metrics and artifacts to MLflow.

Optional tracking URI overrides:

```bash
MLFLOW_TRACKING_URI=sqlite:///custom-mlflow.db python -m apex.train
python -m apex.train --tracking-uri sqlite:///another-mlflow.db
```

CSV input must contain `tenure_months`, `monthly_charges`, `support_tickets`, `usage_gb`, `contract_type`, `payment_method`, and `churn`. The target must contain both binary labels 0 and 1.

## MLflow UI

```bash
cd ~/Projects/apex-mlops
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `apex-customer-churn`.

## Prediction API

```bash
cd ~/Projects/apex-mlops
source .venv/bin/activate
uvicorn apex.api:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model availability and version |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Model and reference readiness |
| `GET /model/info` | Training metadata and model metrics |
| `POST /predict` | Batch prediction, probability, risk band |
| `POST /monitor/drift` | Compare current distributions with training reference |
| `GET /monitor/drift/latest` | Most recently persisted drift report |
| `POST /monitor/performance` | Evaluate predictions against newly available ground-truth labels |
| `GET /monitor/performance/latest` | Latest persisted model-quality report |

## Tests and commands

```bash
make test
make train
make api
make mlflow
make docker-build
```

## Drift monitoring

APEX computes Population Stability Index (PSI) for numeric and categorical input features against the saved training reference. The default alert threshold is 0.20. The latest report is persisted to `artifacts/drift_latest.json` and served by `GET /monitor/drift/latest`.

PSI is a distribution-shift heuristic, not a universal statistical guarantee. The built-in test deliberately shifts input distributions to demonstrate detection. Production monitoring should use larger, representative time windows and separately track model quality once ground-truth labels become available.

## Docker and CI/CD

The Docker image trains a demonstration model from synthetic data during image build:

```bash
docker build -t apex-mlops:local .
docker run --rm -p 8000:8000 apex-mlops:local
```

GitHub Actions runs compilation, pytest, dependency consistency checks, and a Docker build on pushes and pull requests to `main`.

## Production limitations

- The built-in dataset is synthetic unless a user-provided CSV is supplied.
- Retraining is initiated explicitly from a newly labeled CSV; champion promotion is controlled by the MLflow registry quality gate.
- API-key authentication is included; production deployments should still add authorization scopes, rate limiting, TLS termination, and managed secret storage.
- Feature drift does not prove model performance degradation or concept drift.
- Local MLflow uses SQLite; shared team deployments need secured tracking and managed database/artifact storage.


## APEX 0.3.0 — model governance and production hardening

- **API security:** `/predict`, `/model/info`, and monitoring endpoints accept `X-API-Key`. Local development remains frictionless when no key is configured. Production startup requires `APEX_API_KEY` with at least 32 characters. Liveness and readiness endpoints remain public for container orchestration.
- **Model lineage:** every training run writes a timestamped immutable model snapshot, a SHA-256 checksum, and an artifact manifest alongside the stable `artifacts/model.joblib` serving path.
- **MLflow Model Registry:** tracked runs register the selected pipeline as `APEXCustomerChurn`. The `champion` alias moves only when holdout ROC-AUC beats the existing champion by `APEX_PROMOTION_MIN_IMPROVEMENT`; otherwise the new version remains a candidate.
- **Performance monitoring:** when ground-truth labels become available, `POST /monitor/performance` calculates accuracy, precision, recall, F1, and ROC-AUC and persists the latest report. Drift and model quality are separate signals.
- **Controlled retraining:** provide a newly labeled CSV that follows the feature contract; APEX trains a candidate, tracks it, and applies the champion promotion gate:

```bash
python -m apex.retrain --csv data/new_labeled_churn.csv
```

Local configuration is documented in `.env.example`. Copy it to `.env`, generate a private key for deployments, and never commit `.env`. This release does not claim autonomous scheduled retraining: retraining is an explicit, auditable operation after labeled data is available.

### API key setup

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Put the generated value in `APEX_API_KEY`. For a local protected API, set the same key in the shell before starting Uvicorn. Send it as the `X-API-Key` header. For production set `APEX_ENV=production`; the service will refuse to start without a strong key.

---

## Engineering Portfolio

This repository is part of a four-project engineering portfolio spanning data engineering, machine learning operations, cloud infrastructure automation, and applied AI engineering.

| Project | Engineering Focus | Technology |
|---|---|---|
| [ATLAS](https://github.com/touc996-svg/atlas) | Event-driven data engineering and lakehouse analytics | Python, Kafka, PostgreSQL, Spark, Parquet, dbt, FastAPI |
| [APEX MLOps](https://github.com/touc996-svg/apex-mlops) | Reproducible ML lifecycle, model governance and monitoring | Python, scikit-learn, MLflow, FastAPI, Docker |
| [ORBIT](https://github.com/touc996-svg/orbit-cloud-automation) | AWS infrastructure automation and security validation | Terraform, AWS, Python, GitHub Actions |
| [NEXUS AI](https://github.com/touc996-svg/nexus-ai) | Multi-agent AI, RAG architecture and intelligent orchestration | FastAPI, Qwen3, Ollama, Qdrant, Nomic Embeddings, SQL |

### Portfolio Verification Highlights

- **ATLAS:** Verified local end-to-end pipeline execution, 3/3 dbt models and 9/9 dbt data-quality tests.
- **APEX MLOps 0.3.0:** Model registry, artifact integrity, controlled retraining, protected inference API and drift monitoring.
- **ORBIT:** Terraform validation, security scanning, infrastructure preflight and CI. This is an infrastructure baseline; no live AWS deployment is claimed.
- **NEXUS AI 0.7.0:** 70 passing local tests, 12/12 reported live integration checks, successful Qwen3 inference, 768-dimensional Nomic embeddings, and successful GitHub CI for commit `5c6a949`. RAG endpoints passed HTTP smoke tests; retrieval quality remains a separate validation item.

Each project has its own architecture, setup instructions, tests and documented limitations. The repositories can be reviewed and run independently.
