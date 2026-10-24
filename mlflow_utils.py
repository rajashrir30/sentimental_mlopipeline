"""Small MLflow helpers shared by training jobs."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow


def configure_mlflow(experiment: str = "sentiment-mlops-pipeline") -> None:
    # MLflow 3 requires an explicit opt-in for the local file backend used by
    # this small local project. Deployments can override this with a server URI.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment)


def log_metrics_and_artifact(metrics: dict, params: dict, artifact: str | Path | None = None) -> None:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if artifact:
        mlflow.log_artifact(str(artifact))

