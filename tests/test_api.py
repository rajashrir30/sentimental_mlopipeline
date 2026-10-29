import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import serving.app as api
from serving.app import PredictionRequest, health, predict, ready


def test_health_reports_loaded_model():
    payload = health()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["model_loaded"] is True


def test_ready_reports_available_model():
    assert ready() == {"status": "ready"}


def test_predict_returns_a_real_model_prediction():
    result = predict(PredictionRequest(text="A warm, funny and beautifully acted film."))
    assert result["sentiment"] in {"positive", "negative"}
    assert 0 <= result["positive_probability"] <= 1


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
