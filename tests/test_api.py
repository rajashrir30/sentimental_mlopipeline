import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pathlib import Path

import serving.app as api
from serving.app import PredictionRequest, dataset_insights, health, predict, ready


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["database"]["configured"] is False


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_reports_loaded_model():
    payload = health()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["model_loaded"] is True


def test_ready_reports_available_model():
    assert ready() == {"status": "ready"}


def test_dataset_insights_uses_the_processed_artifact(monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "processed_imdb.json"
    monkeypatch.setattr(api, "DATA_PATH", fixture)
    payload = dataset_insights()
    assert payload["total_samples"] == 4
    assert set(payload["splits"]) == {"train", "validation", "test"}
    assert payload["duplicate_texts"] >= 0


def test_predict_returns_a_real_model_prediction():
    result = predict(PredictionRequest(text="A warm, funny and beautifully acted film."))
    assert result["sentiment"] in {"positive", "negative"}
    assert 0 <= result["positive_probability"] <= 1
    assert result["history_saved"] is False


def test_prediction_endpoint_response_structure():
    response = client.post("/predict", json={"text": "A warm, funny and beautifully acted film."})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"sentiment", "positive_probability", "history_saved"}
    assert payload["sentiment"] in {"positive", "negative"}
    assert isinstance(payload["positive_probability"], float)
    assert payload["history_saved"] is False


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "x" * 20_001}])
def test_prediction_endpoint_rejects_invalid_input(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_rejects_whitespace_input():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_prediction_history_requires_database_configuration():
    response = client.get("/predictions/history")
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_prediction_history_supports_pagination_and_filtering(monkeypatch):
    observed = {}

    def fake_list_predictions(page, page_size, sentiment):
        observed.update(page=page, page_size=page_size, sentiment=sentiment)
        return {
            "items": [
                {
                    "id": 1,
                    "text": "great film",
                    "sentiment": "positive",
                    "positive_probability": 0.95,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "page": page,
            "page_size": page_size,
            "total": 1,
            "total_pages": 1,
            "sentiment": sentiment,
        }

    monkeypatch.setattr(api, "list_predictions", fake_list_predictions)
    response = client.get("/predictions/history?page=2&page_size=5&sentiment=positive")
    assert response.status_code == 200
    assert observed == {"page": 2, "page_size": 5, "sentiment": "positive"}
    assert response.json()["items"][0]["sentiment"] == "positive"


def test_monitoring_reports_database_unavailable_without_configuration():
    response = client.get("/monitoring/report")
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_monitoring_returns_insufficient_data_without_claiming_accuracy(monkeypatch):
    monkeypatch.setattr(
        api,
        "build_monitoring_report",
        lambda model: {
            "status": "insufficient_data",
            "message": "At least 30 current and reference rows are required.",
            "current_rows": 2,
            "reference_rows": 4,
            "accuracy_monitored": False,
            "evidently": {"available": False, "report_generated": False},
        },
    )
    response = client.get("/monitoring/report")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_data"
    assert payload["accuracy_monitored"] is False


def test_monitoring_feature_frame_contains_text_and_prediction_signals():
    from serving.monitoring import _features

    frame = _features(["great film!", "bad film."], ["positive", "negative"], [0.9, 0.1])
    assert list(frame["sentiment"]) == ["positive", "negative"]
    assert list(frame["text_length_words"]) == [2, 2]
    assert "punctuation_rate" in frame


def test_prediction_endpoint_returns_503_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = client.post("/predict", json={"text": "A review"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model artifact is not available"


def test_readiness_returns_503_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = client.get("/ready")
    assert response.status_code == 503


def test_missing_model_file_is_handled():
    assert api._load_model(Path("artifacts") / "missing-model.joblib") is None


def test_corrupt_model_file_is_handled(monkeypatch):
    model_path = Path("artifacts") / "baseline.joblib"
    monkeypatch.setattr(api.joblib, "load", lambda path: (_ for _ in ()).throw(ValueError("corrupt")))
    assert api._load_model(model_path) is None


def test_whitespace_review_is_rejected():
    with pytest.raises(HTTPException) as error:
        predict(PredictionRequest(text="   "))
    assert error.value.status_code == 422


def test_invalid_review_lengths_are_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(text="")
    with pytest.raises(ValidationError):
        PredictionRequest(text="x" * 20_001)


def test_mlflow_unavailable_returns_service_error(monkeypatch):
    def unavailable():
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(api, "_mlflow_client", unavailable)
    with pytest.raises(HTTPException) as error:
        api.mlflow_runs()
    assert error.value.status_code == 503


def test_missing_mlflow_experiment_is_reported_as_unavailable(monkeypatch):
    class EmptyClient:
        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(api, "_mlflow_client", lambda: EmptyClient())
    payload = api.mlflow_runs()
    assert payload["available"] is False
    assert payload["runs"] == []
