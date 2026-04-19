from __future__ import annotations

import pandas as pd

from src.config import settings
from src.ingest import load_raw_data
from src.validate import validate_raw_data


FEATURE_COLUMNS = [
    "age",
    "prior_admissions",
    "medication_complexity",
    "length_of_stay",
    "comorbidity_score",
    "diagnosis_circulatory",
    "diagnosis_injury",
    "diagnosis_metabolic",
    "diagnosis_other",
    "diagnosis_respiratory",
    "disposition_home",
    "disposition_home_health",
    "disposition_rehab",
    "disposition_snf",
]



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_raw_data(df)

    encoded = pd.get_dummies(
        df,
        columns=["diagnosis_group", "discharge_disposition"],
        prefix=["diagnosis", "disposition"],
        dtype=int,
    )

    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0

    result = encoded[["patient_id", *FEATURE_COLUMNS, "readmitted_30d"]].copy()
    return result



def build_and_save_processed_dataset() -> pd.DataFrame:
    raw_df = load_raw_data()
    processed_df = build_features(raw_df)
    settings.processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(settings.processed_dataset_path, index=False)
    return processed_df


if __name__ == "__main__":
    processed = build_and_save_processed_dataset()
    print(f"Saved processed dataset to {settings.processed_dataset_path} with shape {processed.shape}")
