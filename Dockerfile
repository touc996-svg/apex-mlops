FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APEX_ARTIFACT_DIR=/app/artifacts

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY . .
RUN python -m apex.train --rows 1600 --output /app/artifacts --disable-mlflow

EXPOSE 8000
CMD ["uvicorn", "apex.api:app", "--host", "0.0.0.0", "--port", "8000"]
