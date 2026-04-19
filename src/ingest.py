from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings


def generate_sample_dataset(n_rows: int = 500, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    age = rng.integers(18, 91, size=n_rows)
    prior_admissions = rng.poisson(lam=1.8, size=n_rows).clip(0, 10)
    medication_complexity = rng.integers(1, 6, size=n_rows)
    length_of_stay = rng.normal(loc=4.5, scale=2.0, size=n_rows).clip(1, 20)
    comorbidity_score = rng.poisson(lam=2.5, size=n_rows).clip(0, 12)
    diagnosis_group = rng.choice(["circulatory", "respiratory", "metabolic", "injury", "other"], size=n_rows)
    discharge_disposition = rng.choice(["home", "rehab", "snf", "home_health"], size=n_rows)

    diagnosis_weight = pd.Series(diagnosis_group).map(
        {
            "circulatory": 0.55,
            "respiratory": 0.45,
            "metabolic": 0.40,
            "injury": 0.20,
            "other": 0.15,
        }
    ).to_numpy()
    disposition_weight = pd.Series(discharge_disposition).map(
        {"home": 0.0, "home_health": 0.1, "rehab": 0.28, "snf": 0.35}
    ).to_numpy()

    raw_score = (
        0.018 * age
        + 0.45 * prior_admissions
        + 0.38 * medication_complexity
        + 0.12 * length_of_stay
        + 0.30 * comorbidity_score
        + diagnosis_weight
        + disposition_weight
        - 3.7
    )
    probs = 1.0 / (1.0 + np.exp(-raw_score))
    readmitted_30d = (rng.random(n_rows) < probs).astype(int)

    return pd.DataFrame(
        {
            "patient_id": [f"MVP-{1000+i}" for i in range(n_rows)],
            "age": age,
            "prior_admissions": prior_admissions,
            "medication_complexity": medication_complexity,
            "length_of_stay": length_of_stay.round(1),
            "comorbidity_score": comorbidity_score,
            "diagnosis_group": diagnosis_group,
            "discharge_disposition": discharge_disposition,
            "readmitted_30d": readmitted_30d,
        }
    )



def save_sample_dataset(path: Path | None = None) -> Path:
    target_path = path or settings.sample_dataset_path
    df = generate_sample_dataset()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path



def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    target_path = path or settings.sample_dataset_path
    if not target_path.exists():
        save_sample_dataset(target_path)
    return pd.read_csv(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sample readmission dataset")
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()
    df = generate_sample_dataset(n_rows=args.rows)
    settings.sample_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.sample_dataset_path, index=False)
    print(f"Saved sample dataset to {settings.sample_dataset_path}")
