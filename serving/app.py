"""FastAPI inference service for the baseline model."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from text_utils import clean_text

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/baseline.joblib"))
app = FastAPI(title="Sentiment MLOps Pipeline", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


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

