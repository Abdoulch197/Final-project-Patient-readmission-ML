from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import settings
from src.evaluate import evaluate_predictions, metrics_to_frame
from src.features import FEATURE_COLUMNS, build_and_save_processed_dataset
from src.utils import save_json

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None


@contextmanager
def tracking_run(run_name: str):
    if mlflow is not None:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("hospital-readmission-prediction")
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        yield



def _log_param(key: str, value):
    if mlflow is not None:
        mlflow.log_param(key, value)



def _log_metric(key: str, value):
    if mlflow is not None:
        mlflow.log_metric(key, value)



def _log_dict(payload: dict, artifact_file: str):
    if mlflow is not None:
        mlflow.log_dict(payload, artifact_file)



def train_models() -> dict:
    processed_df = build_and_save_processed_dataset()
    X = processed_df[FEATURE_COLUMNS]
    y = processed_df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results: dict[str, dict] = {}

    model_specs = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "xgboost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
    }

    for name, model in model_specs.items():
        with tracking_run(name):
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            _log_param("dataset_version", settings.dataset_version)
            for key, value in model.get_params().items():
                if isinstance(value, (str, int, float, bool)):
                    _log_param(key, value)
            for metric_name, metric_value in metrics.items():
                if metric_name != "confusion_matrix":
                    _log_metric(metric_name, metric_value)
            _log_dict(metrics["confusion_matrix"], f"{name}_confusion_matrix.json")

            results[name] = {
                "model": model,
                "metrics": metrics,
            }

    baseline_model = results["logistic_regression"]["model"]
    primary_model = results["xgboost"]["model"]
    joblib.dump(baseline_model, settings.baseline_model_path)
    joblib.dump(primary_model, settings.primary_model_path)

    metrics_by_model = {k: v["metrics"] for k, v in results.items()}
    metrics_df = metrics_to_frame(metrics_by_model)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv("reports/figures/model_metrics.csv", index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": settings.dataset_version,
        "tracking_backend": "mlflow" if mlflow is not None else "local-metadata",
        "feature_columns": FEATURE_COLUMNS,
        "models": {
            "baseline": {
                "name": "logistic_regression",
                "path": str(settings.baseline_model_path),
                "metrics": metrics_by_model["logistic_regression"],
            },
            "primary": {
                "name": "xgboost",
                "path": str(settings.primary_model_path),
                "metrics": metrics_by_model["xgboost"],
            },
        },
    }
    save_json(settings.metadata_path, metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train readmission models")
    _ = parser.parse_args()
    metadata = train_models()
    print(pd.DataFrame([
        {"model": "logistic_regression", **{k: v for k, v in metadata['models']['baseline']['metrics'].items() if k != 'confusion_matrix'}},
        {"model": "xgboost", **{k: v for k, v in metadata['models']['primary']['metrics'].items() if k != 'confusion_matrix'}},
    ]))
