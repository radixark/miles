from __future__ import annotations

import functools
import json
import os
from typing import Any

_MODEL_TYPES = ("inkling_mm_model", "inkling_mm_model")


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
    if not isinstance(config, dict):
        return ""
    return config.get("model_type", "") or ""


def is_inkling(tokenizer: Any) -> bool:
    return _read_model_type(getattr(tokenizer, "name_or_path", "")) in _MODEL_TYPES


def is_inkling_checkpoint(name_or_path: str) -> bool:
    return _read_model_type(name_or_path) in _MODEL_TYPES


@functools.cache
def fixed_chat_template() -> str:
    """The bundled Inkling chat template (the checkpoint's, with the upstream fixes)."""
    # imported lazily: tito_tokenizer imports template.py, which imports this module
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR

    return (TEMPLATE_DIR / "inkling_fixed.jinja").read_text(encoding="utf-8")
