from __future__ import annotations

import functools
import json
import os

_MODEL_TYPES = ("muse_spark",)


@functools.cache
def _read_model_type(name_or_path: str) -> str:
    if not name_or_path:
        return ""
    config_path = os.path.join(name_or_path, "config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return (config.get("model_type", "") or "") if isinstance(config, dict) else ""


def is_muse_spark_omni_checkpoint(name_or_path: str) -> bool:
    """A Muse Spark omni checkpoint carries the NaViT vision tower next to the text model."""
    return _read_model_type(name_or_path) in _MODEL_TYPES
