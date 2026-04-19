from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.features import FEATURE_COLUMNS
from src.train import train_models


class Predictor:
    def __init__(self) -> None:
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        model_path = Path(settings.primary_model_path)
        if not model_path.exists():
            train_models()
        return joblib.load(model_path)

    def _payload_to_frame(self, payload: dict) -> pd.DataFrame:
        diagnosis = payload.get("diagnosis_group", "other")
        disposition = payload.get("discharge_disposition", "home")

        row = {
            "age": payload["age"],
            "prior_admissions": payload["prior_admissions"],
            "medication_complexity": payload["medication_complexity"],
            "length_of_stay": payload["length_of_stay"],
            "comorbidity_score": payload["comorbidity_score"],
            "diagnosis_circulatory": 1 if diagnosis == "circulatory" else 0,
            "diagnosis_injury": 1 if diagnosis == "injury" else 0,
            "diagnosis_metabolic": 1 if diagnosis == "metabolic" else 0,
            "diagnosis_other": 1 if diagnosis == "other" else 0,
            "diagnosis_respiratory": 1 if diagnosis == "respiratory" else 0,
            "disposition_home": 1 if disposition == "home" else 0,
            "disposition_home_health": 1 if disposition == "home_health" else 0,
            "disposition_rehab": 1 if disposition == "rehab" else 0,
            "disposition_snf": 1 if disposition == "snf" else 0,
        }
        return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    def predict(self, payload: dict) -> dict:
        frame = self._payload_to_frame(payload)
        probability = float(self.model.predict_proba(frame)[0][1])
        risk_label = (
            "High Risk"
            if probability >= settings.positive_threshold
            else "Moderate Risk"
            if probability >= 0.30
            else "Low Risk"
        )
        contributions = {
            "prior_admissions": round(min(1.0, payload["prior_admissions"] / 10) * 0.40, 3),
            "medication_complexity": round(min(1.0, payload["medication_complexity"] / 10) * 0.30, 3),
            "age": round(min(1.0, payload["age"] / 100) * 0.15, 3),
            "length_of_stay": round(min(1.0, payload["length_of_stay"] / 20) * 0.10, 3),
            "comorbidity_score": round(min(1.0, payload["comorbidity_score"] / 12) * 0.25, 3),
        }
        sorted_factors = dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:3])
        return {
            "readmission_risk": round(probability, 4),
            "risk_label": risk_label,
            "threshold": settings.positive_threshold,
            "top_risk_factors": sorted_factors,
        }


predictor = Predictor()
