from __future__ import annotations

from src.config import settings
from src.utils import load_json, save_json


def promote_primary_model(stage: str = "staging") -> dict:
    metadata = load_json(settings.metadata_path)
    metadata["models"]["primary"]["stage"] = stage
    save_json(settings.metadata_path, metadata)
    return metadata


if __name__ == "__main__":
    updated = promote_primary_model()
    print(f"Primary model promoted to {updated['models']['primary'].get('stage')}")
