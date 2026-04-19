from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "patient_id",
    "age",
    "prior_admissions",
    "medication_complexity",
    "length_of_stay",
    "comorbidity_score",
    "diagnosis_group",
    "discharge_disposition",
    "readmitted_30d",
}


def validate_raw_data(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if not df["age"].between(0, 120).all():
        raise ValueError("Age contains invalid values")

    if not df["length_of_stay"].between(0, 365).all():
        raise ValueError("Length of stay contains invalid values")

    if not set(df["readmitted_30d"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Label column must be binary")

    if df["patient_id"].isna().any():
        raise ValueError("Patient IDs cannot be null")
