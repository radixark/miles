from __future__ import annotations

import copy
import functools
import json
import logging
import os
from typing import Any

from sglang.srt.entrypoints.openai import encoding_dsv32
from sglang.srt.entrypoints.openai.protocol import Tool

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
    kwargs = dict(kwargs)
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking_mode", "thinking" if enable_thinking else "chat")
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


def _inject_tools_into_system(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put *tools* in the system message, where ``encode_messages`` reads them.

    The encoder serializes each tool dict verbatim into ``<functions>``, so they
    must round-trip through ``Tool.model_dump()`` (fills defaults / orders fields)
    or the token ids drift from what sglang serves.
    """
    out = copy.deepcopy(messages)
    if not out or out[0].get("role") != "system":
        out.insert(0, {"role": "system", "content": ""})
    out[0]["tools"] = [Tool.model_validate(t).model_dump() for t in tools]
    return out


def render_messages(messages: list[dict[str, Any]], *, tools: list[dict] | None = None, **kwargs: Any) -> str:
    """Render *messages* into a DeepSeek V3.2 prompt via sglang ``encode_messages``.

    Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
    injected into the system message (see ``_inject_tools_into_system``).
    """
    encode_config = _build_deepseek_encode_config(kwargs)
    if tools:
        messages = _inject_tools_into_system(messages, tools)
    return encoding_dsv32.encode_messages(messages, **encode_config)
