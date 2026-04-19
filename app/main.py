from __future__ import annotations

from fastapi import FastAPI

from app.monitoring import log_prediction
from app.predictor import predictor
from app.schemas import PatientInput, PredictionOutput
from src.config import settings
from src.train import train_models

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="FastAPI service for a 30-day hospital readmission prediction demo",
    version="1.0.0",
)


@app.on_event("startup")
def ensure_assets() -> None:
    if not settings.primary_model_path.exists():
        train_models()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_path": str(settings.primary_model_path), "dataset_version": settings.dataset_version}


@app.get("/")
def root() -> dict:
    return {
        "message": "Hospital Readmission Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PatientInput) -> PredictionOutput:
    prediction = predictor.predict(payload.model_dump())
    log_prediction(payload.model_dump(), prediction)
    return PredictionOutput(**prediction)
