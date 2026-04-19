from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score



def evaluate_predictions(y_true, y_pred, y_prob) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }



def metrics_to_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = {"model": model_name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
        rows.append(row)
    return pd.DataFrame(rows)
