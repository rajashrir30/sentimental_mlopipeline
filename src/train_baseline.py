"""Train and log a TF-IDF + Logistic Regression sentiment baseline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import classification_metrics
from mlflow_utils import configure_mlflow, log_metrics_and_artifact
from preprocess import DEFAULT_OUTPUT, load_processed


def train(data_path=DEFAULT_OUTPUT, model_path="artifacts/baseline.joblib"):
    started_at = time.perf_counter()
    data = load_processed(data_path)
    def make_model(c):
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=300, C=c, random_state=42)),
        ])

    # Select hyperparameters using validation only; the test split remains untouched.
    candidates = []
    for c in (0.5, 1.0, 2.0, 4.0):
        candidate = make_model(c)
        candidate.fit(data["train"]["text"], data["train"]["label"])
        validation_metrics = classification_metrics(
            data["validation"]["label"], candidate.predict(data["validation"]["text"])
        )
        candidates.append((validation_metrics["f1"], c, validation_metrics))
    _, best_c, validation_metrics = max(candidates, key=lambda item: item[0])

    # Retrain the selected model on train + validation before the final test evaluation.
    fit_text = data["train"]["text"] + data["validation"]["text"]
    fit_labels = data["train"]["label"] + data["validation"]["label"]
    model = make_model(best_c)
    model.fit(fit_text, fit_labels)
    metrics = classification_metrics(data["test"]["label"], model.predict(data["test"]["text"]))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    configure_mlflow()
    with mlflow.start_run(run_name="tfidf-logistic-regression"):
        log_metrics_and_artifact(
            {**{f"validation_{k}": v for k, v in validation_metrics.items()},
             **{f"test_{k}": v for k, v in metrics.items()}},
            {"model": "logistic_regression", "features": "tfidf", "best_c": best_c,
             "seed": 42, "train_samples": len(data["train"]["text"]),
             "validation_samples": len(data["validation"]["text"]),
             "test_samples": len(data["test"]["text"]),
             "training_seconds": round(time.perf_counter() - started_at, 3)},
            model_path,
        )
    result = {"validation": validation_metrics, "test": metrics, "best_c": best_c}
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="artifacts/baseline.joblib")
    args = parser.parse_args()
    train(args.data, args.model)
