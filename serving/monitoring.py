"""Input and prediction monitoring built on stored prediction history."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from serving.database import DatabaseUnavailable, list_predictions
from text_utils import clean_text

MONITORING_REFERENCE_PATH = Path(os.getenv("MONITORING_REFERENCE_PATH", "data/processed/imdb.json"))
MONITORING_MIN_ROWS = int(os.getenv("MONITORING_MIN_ROWS", "30"))
MONITORING_CURRENT_ROWS = int(os.getenv("MONITORING_CURRENT_ROWS", "500"))


class MonitoringUnavailable(RuntimeError):
    """Raised when monitoring cannot produce a meaningful result."""


def _reference_texts(path: Path, limit: int) -> list[str]:
    if not path.exists():
        raise MonitoringUnavailable("Monitoring reference data is not available")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "train" in payload:
            texts = payload["train"]["text"]
        elif isinstance(payload, list):
            texts = [item.get("text", "") if isinstance(item, dict) else item for item in payload]
        else:
            raise ValueError("Unsupported reference format")
        return [clean_text(str(text)) for text in texts[:limit] if clean_text(str(text))]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MonitoringUnavailable("Monitoring reference data is invalid") from exc


def _features(texts: list[str], sentiments: list[str], probabilities: list[float]) -> pd.DataFrame:
    rows = []
    for text, sentiment, probability in zip(texts, sentiments, probabilities):
        words = text.split()
        rows.append(
            {
                "text_length_chars": len(text),
                "text_length_words": len(words),
                "unique_word_ratio": len(set(words)) / len(words) if words else 0.0,
                "punctuation_rate": sum(char in "!?.," for char in text) / len(text) if text else 0.0,
                "sentiment": sentiment,
                "positive_probability": float(probability),
            }
        )
    return pd.DataFrame(rows)


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    counts = Counter(frame["sentiment"].tolist())
    total = len(frame)
    return {
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "positive_share": round(counts.get("positive", 0) / total, 4) if total else 0,
        "negative_share": round(counts.get("negative", 0) / total, 4) if total else 0,
        "average_positive_probability": round(float(frame["positive_probability"].mean()), 4) if total else 0,
    }


def _evidently_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return {"available": False, "message": "Evidently is not installed"}

    numerical_columns = [
        "text_length_chars",
        "text_length_words",
        "unique_word_ratio",
        "punctuation_rate",
        "positive_probability",
    ]
    definition = DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=["sentiment"],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)
    snapshot = Report([DataDriftPreset()], include_tests=True).run(current_dataset, reference_dataset)
    raw = snapshot.dict()
    drifted_columns = 0
    drift_share = 0.0
    for metric in raw.get("metrics", []):
        metric_id = str(metric.get("metric_id", ""))
        value = metric.get("value")
        if "DriftedColumnsCount" in metric_id and isinstance(value, dict):
            drifted_columns = int(value.get("count", 0))
            drift_share = float(value.get("share", 0.0))
    return {
        "available": True,
        "drift_detected": drifted_columns > 0,
        "drifted_columns": drifted_columns,
        "drift_share": round(drift_share, 4),
        "report_generated": True,
    }


def build_monitoring_report(model: Any) -> dict[str, Any]:
    try:
        history = list_predictions(page=1, page_size=MONITORING_CURRENT_ROWS)
    except DatabaseUnavailable as exc:
        raise MonitoringUnavailable(str(exc)) from exc
    current_items = history["items"]
    reference_texts = _reference_texts(MONITORING_REFERENCE_PATH, MONITORING_CURRENT_ROWS)
    if len(current_items) < MONITORING_MIN_ROWS or len(reference_texts) < MONITORING_MIN_ROWS:
        return {
            "status": "insufficient_data",
            "message": f"At least {MONITORING_MIN_ROWS} current and reference rows are required for drift monitoring.",
            "current_rows": len(current_items),
            "reference_rows": len(reference_texts),
            "accuracy_monitored": False,
            "evidently": {"available": False, "report_generated": False},
        }

    current_texts = [item["text"] for item in current_items]
    current_sentiments = [item["sentiment"] for item in current_items]
    current_probabilities = [item["positive_probability"] for item in current_items]
    reference_probabilities = model.predict_proba(reference_texts)[:, 1].tolist()
    reference_sentiments = ["positive" if probability >= 0.5 else "negative" for probability in reference_probabilities]
    reference = _features(reference_texts, reference_sentiments, reference_probabilities)
    current = _features(current_texts, current_sentiments, current_probabilities)
    return {
        "status": "ok",
        "current_rows": len(current),
        "reference_rows": len(reference),
        "accuracy_monitored": False,
        "input_features": ["text_length_chars", "text_length_words", "unique_word_ratio", "punctuation_rate"],
        "reference_prediction_distribution": _distribution(reference),
        "current_prediction_distribution": _distribution(current),
        "evidently": _evidently_drift(reference, current),
    }
