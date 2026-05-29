from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_TYPE = "deepseek_v4"

_KNOWN_KWARGS = frozenset(
    {
        "thinking",
        "thinking_mode",
        "drop_thinking",
        "add_default_bos_token",
        "context",
        "reasoning_effort",
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


def is_deepseek_v4(tokenizer: Any) -> bool:
    """Return True when *tokenizer* is a DeepSeek V4 checkpoint."""
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
    # HF passes `thinking` (bool); the DeepSeek encoder takes `thinking_mode` ("thinking"/"chat").
    if "thinking" in kwargs:
        cfg["thinking_mode"] = "thinking" if kwargs["thinking"] else "chat"
    for key in ("thinking_mode", "drop_thinking", "add_default_bos_token", "context", "reasoning_effort"):
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def render_messages(messages: list[dict[str, Any]], *, tools: list[dict] | None = None, **kwargs: Any) -> str:
    """Render *messages* into a DeepSeek V4 prompt via sglang ``encode_messages``.

    Assume input messages tool_call ``arguments`` are already JSON strings.
    """
    if tools:
        raise ValueError(
            "DeepSeek V4 chat template does not support tools def in apply chat template, plz inject it in system message."
        )
    # Imported lazily: the canonical sglang-miles build may not ship encoding_dsv4,
    # and a module-level import would break every chat_template_utils import there.
    from sglang.srt.entrypoints.openai import encoding_dsv4

    encode_config = _build_deepseek_encode_config(kwargs)
    return encoding_dsv4.encode_messages(messages, **encode_config)
