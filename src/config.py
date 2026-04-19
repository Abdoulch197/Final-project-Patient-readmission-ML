from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils import MODELS_DIR, PROCESSED_DIR, SAMPLE_DIR


@dataclass(frozen=True)
class Settings:
    sample_dataset_path: Path = SAMPLE_DIR / "hospital_readmissions_sample.csv"
    processed_dataset_path: Path = PROCESSED_DIR / "readmission_features_v1.csv"
    baseline_model_path: Path = MODELS_DIR / "baseline_logreg.pkl"
    primary_model_path: Path = MODELS_DIR / "xgb_readmit_v1.pkl"
    metadata_path: Path = MODELS_DIR / "metadata.json"
    prediction_log_path: Path = MODELS_DIR / "prediction_logs.csv"
    dataset_version: str = "readmission_features_v1"
    positive_threshold: float = 0.65


settings = Settings()
