from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

from sglang.srt.entrypoints.openai import encoding_dsv32

logger = logging.getLogger(__name__)

_MODEL_TYPE = "deepseek_v32"

_KNOWN_KWARGS = frozenset(
    {
        "thinking_mode",
        "drop_thinking",
        "add_default_bos_token",
        "context",
    }
)


@functools.cache
def _read_model_type(name_or_path: str) -> str:
    """Read ``model_type`` from a checkpoint's ``config.json`` (cached per path)."""
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


def is_deepseek_v32(tokenizer: Any) -> bool:
    """Return True when *tokenizer* is a DeepSeek V3.2 checkpoint."""
    return _read_model_type(tokenizer.name_or_path) == _MODEL_TYPE


def _build_deepseek_encode_config(kwargs: dict) -> dict:
    # reject unknown kwargs to avoid silent config drop
    unknown = set(kwargs) - _KNOWN_KWARGS
    if unknown:
        raise ValueError(
            f"apply_chat_template_kwargs has unsupported kwargs {sorted(unknown)} "
            f"for the DeepSeek encoder. Known keys: {sorted(_KNOWN_KWARGS)}"
        )
    cfg = {"thinking_mode": "thinking", "drop_thinking": True, "add_default_bos_token": True}
    for key in _KNOWN_KWARGS:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def render_messages(messages: list[dict[str, Any]], *, tools: list[dict] | None = None, **kwargs: Any) -> str:
    """Render *messages* into a DeepSeek V3.2 prompt via sglang ``encode_messages``.

    Assume input messages tool_call ``arguments`` are already JSON strings.
    """
    if tools:
        raise ValueError(
            "DeepSeek V3.2 chat template does not support tools def in apply chat template, plz inject it in system message."
        )
    encode_config = _build_deepseek_encode_config(kwargs)
    return encoding_dsv32.encode_messages(messages, **encode_config)
