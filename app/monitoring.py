from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import settings



def log_prediction(payload: dict, prediction: dict) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
        **prediction,
    }
    path = Path(settings.prediction_log_path)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
