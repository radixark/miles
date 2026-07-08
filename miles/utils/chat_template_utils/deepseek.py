"""Shared bridge for the DeepSeek official-encoder families (V3.2, V4).

Neither family ships a jinja chat_template: sglang renders their prompts
through per-family ``encoding_dsv*`` modules that share one calling
convention, and miles' ``apply_chat_template`` routes any matching tokenizer
here so training-side renders stay byte-aligned with what the runtime
serves.  Family deltas are confined to the encoder module and its
known-kwargs set inside the two ``render_messages_v*`` functions;
everything else is shared.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
from typing import Any

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32
from sglang.srt.entrypoints.openai.protocol import Tool

logger = logging.getLogger(__name__)

_KNOWN_KWARGS_V32 = frozenset(
    {
        "thinking_mode",
        "drop_thinking",
        "add_default_bos_token",
        "context",
    }
)
_KNOWN_KWARGS_V4 = _KNOWN_KWARGS_V32 | {"reasoning_effort"}


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


def _build_encode_config(kwargs: dict, known_kwargs: frozenset[str]) -> dict:
    kwargs = dict(kwargs)
    if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
        kwargs.setdefault("thinking_mode", "thinking" if enable_thinking else "chat")
    # sglang can accept thinking as a kwarg to set thinking_mode, like dsv3.1
    if (thinking := kwargs.pop("thinking", None)) is not None:
        kwargs.setdefault("thinking_mode", "thinking" if thinking else "chat")
    # reject unknown kwargs to avoid silent config drop
    unknown = set(kwargs) - known_kwargs
    if unknown:
        raise ValueError(
            f"apply_chat_template_kwargs has unsupported kwargs {sorted(unknown)} "
            f"for the DeepSeek encoder. Known keys: {sorted(known_kwargs)}"
        )
    # reasoning_effort has no default: like context, it is only forwarded when the
    # caller supplies it, and its value is validated by the encoder (not here).
    cfg = {"thinking_mode": "thinking", "drop_thinking": True, "add_default_bos_token": True}
    for key in known_kwargs:
        if key in kwargs:
            cfg[key] = kwargs[key]
    return cfg


def render_thinking_enabled(chat_template_kwargs: dict[str, Any]) -> bool:
    """Whether *chat_template_kwargs* resolve to thinking mode, through the
    same resolution path the render functions use."""
    return _build_encode_config(chat_template_kwargs, _KNOWN_KWARGS_V4)["thinking_mode"] == "thinking"


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


def render_messages_v32(messages: list[dict[str, Any]], *, tools: list[dict] | None = None, **kwargs: Any) -> str:
    """Render *messages* into a DeepSeek V3.2 prompt via sglang ``encode_messages``.

    Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
    injected into the system message (see ``_inject_tools_into_system``).
    """
    encode_config = _build_encode_config(kwargs, _KNOWN_KWARGS_V32)
    if tools:
        messages = _inject_tools_into_system(messages, tools)
    return encoding_dsv32.encode_messages(messages, **encode_config)


def render_messages_v4(messages: list[dict[str, Any]], *, tools: list[dict] | None = None, **kwargs: Any) -> str:
    """Render *messages* into a DeepSeek V4 prompt via sglang ``encode_messages``.

    Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
    injected into the system message (see ``_inject_tools_into_system``).
    """
    encode_config = _build_encode_config(kwargs, _KNOWN_KWARGS_V4)
    if tools:
        messages = _inject_tools_into_system(messages, tools)
    return encoding_dsv4.encode_messages(messages, **encode_config)


_RENDERERS = {
    "deepseek_v32": render_messages_v32,
    "deepseek_v4": render_messages_v4,
}


def model_type(tokenizer: Any) -> str | None:
    """The DeepSeek family ``model_type`` for *tokenizer*, or ``None`` when the
    checkpoint is not a DeepSeek official-encoder family."""
    mt = _read_model_type(tokenizer.name_or_path)
    return mt if mt in _RENDERERS else None


def apply_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict] | None = None,
    tokenize: bool = False,
    **kwargs: Any,
) -> str | list[int]:
    """Render *messages* for *tokenizer*'s DeepSeek family, optionally encoding.

    Tool_call ``arguments`` must already be JSON strings (the caller
    normalizes).  Raises ``ValueError`` for a non-DeepSeek tokenizer — guard
    call sites with ``model_type``.
    """
    mt = model_type(tokenizer)
    if mt is None:
        raise ValueError(f"not a DeepSeek official-encoder checkpoint: {tokenizer.name_or_path!r}")
    rendered = _RENDERERS[mt](messages, tools=tools, **kwargs)
    return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered
