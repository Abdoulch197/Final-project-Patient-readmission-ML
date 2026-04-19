from __future__ import annotations

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    prior_admissions: int = Field(..., ge=0, le=20)
    medication_complexity: int = Field(..., ge=1, le=10)
    length_of_stay: float = Field(..., ge=0, le=365)
    comorbidity_score: int = Field(..., ge=0, le=20)
    diagnosis_group: str = Field(default="other")
    discharge_disposition: str = Field(default="home")


class PredictionOutput(BaseModel):
    readmission_risk: float
    risk_label: str
    threshold: float
    top_risk_factors: dict[str, float]
