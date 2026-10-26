"""FastAPI inference service for the baseline model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from text_utils import clean_text

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/baseline.joblib"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-mlops-pipeline")
app = FastAPI(title="Sentiment MLOps Pipeline", version="1.0.0")
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _mlflow_client():
    """Create the MLflow client on the server; the URI never reaches the browser."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="MLflow is not installed for the serving API") from exc
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _metric_group(metrics: dict[str, float], split: str) -> dict[str, float]:
    prefix = f"{split}_"
    return {key.removeprefix(prefix): value for key, value in metrics.items() if key.startswith(prefix)}


def _artifact_links(client: Any, run_id: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    pending = [""]
    while pending:
        path = pending.pop()
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                pending.append(item.path)
            else:
                links.append({"path": item.path, "url": f"/mlflow/runs/{run_id}/artifacts/{item.path}"})
    return links


def _serialize_run(client: Any, run: Any) -> dict[str, Any]:
    info = run.info
    data = run.data
    metrics = dict(data.metrics)
    params = dict(data.params)
    tags = dict(data.tags)
    duration_ms = None
    if info.start_time is not None and info.end_time is not None:
        duration_ms = max(0, info.end_time - info.start_time)
    return {
        "run_id": info.run_id,
        "run_name": info.run_name or tags.get("mlflow.runName") or info.run_id[:8],
        "status": info.status,
        "start_time": info.start_time,
        "end_time": info.end_time,
        "duration_ms": duration_ms,
        "params": params,
        "metrics": metrics,
        "validation": _metric_group(metrics, "validation"),
        "test": _metric_group(metrics, "test"),
        "tags": tags,
        "model_type": params.get("model", "unknown"),
        "epochs": params.get("epochs"),
        "artifacts": _artifact_links(client, info.run_id),
        "evaluation_samples": None,
        "confusion_matrix": None,
    }


@app.get("/mlflow/runs")
def mlflow_runs():
    """Return sanitized run metadata for the dashboard from the configured MLflow store."""
    client = _mlflow_client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return {"experiment": MLFLOW_EXPERIMENT_NAME, "runs": []}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=500,
    )
    return {"experiment": experiment.name, "experiment_id": experiment.experiment_id, "runs": [_serialize_run(client, run) for run in runs]}


@app.get("/mlflow/runs/{run_id}")
def mlflow_run(run_id: str):
    client = _mlflow_client()
    try:
        return _serialize_run(client, client.get_run(run_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail="MLflow run was not found") from exc


@app.get("/mlflow/runs/{run_id}/artifacts/{artifact_path:path}")
def mlflow_artifact(run_id: str, artifact_path: str):
    """Stream a known MLflow artifact without exposing the tracking store path."""
    from fastapi.responses import FileResponse

    if not artifact_path or Path(artifact_path).is_absolute() or ".." in Path(artifact_path).parts:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    client = _mlflow_client()
    try:
        local_path = Path(client.download_artifacts(run_id, artifact_path))
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Artifact was not found") from exc
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact is not a file")
    return FileResponse(local_path)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available")
    probability = float(model.predict_proba([clean_text(request.text)])[0, 1])
    return {"sentiment": "positive" if probability >= 0.5 else "negative", "positive_probability": probability}


# The production container serves the React build from the same origin. Keeping
# this optional makes the API usable by itself during local development too.
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

