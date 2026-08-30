"""Training processor loader for GLM-5.3 on Transformers 5.12.1."""

import json
from pathlib import Path


def is_glm5_next_checkpoint(name_or_path: str) -> bool:
    """Return whether a local checkpoint declares ``model_type=glm5_next``."""
    config_path = Path(name_or_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "glm5_next"


def load_glm5_next_processor(name_or_path: str, **processor_kwargs):
    """Build the official image processor without upgrading Transformers."""
    # Keep the vision dependencies lazy: ``load_processor`` calls the lightweight
    # checkpoint predicate for text-only model families too. SGLang owns the
    # Transformers-pin compatibility processor used by both training and rollout.
    from sglang.srt.configs.glm5_next_processing import Glm5NextProcessor

    processor_kwargs.setdefault("trust_remote_code", True)
    return Glm5NextProcessor.from_pretrained(name_or_path, **processor_kwargs)
