"""FastAPI inference service for the baseline model."""
from __future__ import annotations

import os
import logging
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from text_utils import clean_text
from serving.database import DatabaseUnavailable, database_status, list_predictions, record_prediction
from serving.monitoring import MonitoringUnavailable, build_monitoring_report

LOGGER = logging.getLogger("sentiment_api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str) -> Path:
    """Resolve relative paths from the project root, regardless of cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

MODEL_PATH = _resolve_project_path(os.getenv("MODEL_PATH", "artifacts/baseline.joblib"))
DATA_PATH = _resolve_project_path(os.getenv("DATA_PATH", "data/processed/imdb.json"))
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


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Log request outcomes without logging review text or credentials."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    LOGGER.info(
        "request method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _load_model(path: Path):
    """Load the serving artifact once and degrade safely when it is unavailable."""
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        LOGGER.exception("Could not load model artifact: %s", path)
        return None


model = _load_model(MODEL_PATH)


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
        try:
            artifacts = client.list_artifacts(run_id, path)
        except Exception:
            LOGGER.warning("Artifacts are unavailable for MLflow run %s", run_id)
            return links
        for item in artifacts:
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
    try:
        client = _mlflow_client()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="MLflow tracking data is unavailable") from exc
    if experiment is None:
        return {
            "experiment": MLFLOW_EXPERIMENT_NAME,
            "available": False,
            "message": "The configured MLflow experiment is not available in this deployment.",
            "runs": [],
        }
    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=500,
        )
        serialized = [_serialize_run(client, run) for run in runs]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="MLflow tracking data is unavailable") from exc
    return {"experiment": experiment.name, "experiment_id": experiment.experiment_id, "available": True, "runs": serialized}


@app.get("/dataset/insights")
def dataset_insights():
    """Return reproducible statistics from the existing processed dataset artifact."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Processed dataset is not available; run preprocessing first")
    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        splits = {}
        all_texts: list[str] = []
        all_labels: list[int] = []
        for split in ("train", "validation", "test"):
            texts = [str(item) for item in payload[split]["text"]]
            labels = [int(item) for item in payload[split]["label"]]
            lengths = [len(text.split()) for text in texts]
            splits[split] = {
                "samples": len(texts),
                "positive": sum(label == 1 for label in labels),
                "negative": sum(label == 0 for label in labels),
                "empty_reviews": sum(not text.strip() for text in texts),
                "average_words": round(mean(lengths), 2) if lengths else 0,
                "median_words": median(lengths) if lengths else 0,
            }
            all_texts.extend(texts)
            all_labels.extend(labels)
        return {
            "source": "processed dataset artifact",
            "seed": payload.get("seed"),
            "total_samples": len(all_texts),
            "duplicate_texts": len(all_texts) - len(set(all_texts)),
            "label_distribution": {
                "positive": sum(label == 1 for label in all_labels),
                "negative": sum(label == 0 for label in all_labels),
            },
            "splits": splits,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Processed dataset artifact is invalid") from exc


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


class PredictionResponse(BaseModel):
    sentiment: str
    positive_probability: float = Field(ge=0, le=1)
    history_saved: bool


class PredictionHistoryItem(BaseModel):
    id: int
    text: str
    sentiment: Literal["positive", "negative"]
    positive_probability: float = Field(ge=0, le=1)
    created_at: Any


class PredictionHistoryResponse(BaseModel):
    items: list[PredictionHistoryItem]
    page: int
    page_size: int
    total: int
    total_pages: int
    sentiment: Literal["positive", "negative"] | None = None


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "database": database_status(),
    }


@app.get("/ready")
def ready():
    """Report readiness only when the model and PostgreSQL are available."""
    if model is None:
        raise HTTPException(status_code=503, detail="UNREADY: Model is not available")

    db_status = database_status()
    if not db_status.get("available", False):
        message = db_status.get("message", "Database is not available")
        raise HTTPException(status_code=503, detail=f"UNREADY: {message}")

    return {
        "status": "ready",
        "model_loaded": True,
        "database": "connected",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available")
    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text must contain at least one usable character")
    probability = float(model.predict_proba([cleaned])[0, 1])
    sentiment = "positive" if probability >= 0.5 else "negative"
    history_saved = False
    try:
        record_prediction(cleaned, sentiment, probability)
        history_saved = True
    except DatabaseUnavailable:
        LOGGER.warning("Prediction succeeded but history persistence is unavailable")
    return {
        "sentiment": sentiment,
        "positive_probability": probability,
        "history_saved": history_saved,
    }


@app.get("/predictions/history", response_model=PredictionHistoryResponse)
def prediction_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sentiment: Literal["positive", "negative"] | None = Query(default=None),
):
    try:
        return list_predictions(page=page, page_size=page_size, sentiment=sentiment)
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/monitoring/report")
def monitoring_report():
    """Return input and prediction drift monitoring without claiming accuracy."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not available")
    try:
        return build_monitoring_report(model)
    except MonitoringUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# The production container serves the React build from the same origin. Keeping
# this optional makes the API usable by itself during local development too.
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

