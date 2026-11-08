PYTHON := .venv/bin/python
.PHONY: install test train api mlflow docker-build clean
install:
	$(PYTHON) -m pip install -r requirements-dev.txt
test:
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m pytest -q
	$(PYTHON) -m pip check
train:
	$(PYTHON) -m apex.train --rows 1800 --output artifacts
api:
	.venv/bin/uvicorn apex.api:app --reload --host 127.0.0.1 --port 8000
mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
docker-build:
	docker build -t apex-mlops:local .
clean:
	rm -rf .pytest_cache mlruns
