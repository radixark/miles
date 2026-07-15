from __future__ import annotations

import functools
import json
import os
from typing import Any

_MODEL_TYPES = ("inkling_mm_model", "inkling_mm_model")  # v0 + final (Inkling) release

_MESSAGE = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}
_MESSAGE_MODEL = "<|message_model|>"
_CONTENT_TEXT = "<|content_text|>"
_CONTENT_THINKING = "<|content_thinking|>"
_CONTENT_XML = "<|content_xml|>"
_END_MESSAGE = "<|end_message|>"


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


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                t = p.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
            else:
                raise TypeError(
                    f"Inkling text-only render: unsupported content part {p!r}; "
                    "image/audio parts require the Inkling multimodal processor "
                    "(--inkling-mm-towers, no --apply-chat-template)."
                )
        return "".join(parts)
    raise TypeError(f"Inkling render: content must be str or list of parts, got {type(content).__name__}")


def render_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> str:
    """Render *messages* into a Inkling prompt string (text-only path)."""
    parts: list[str] = []

    if tools:
        parts.append(
            f"{_MESSAGE['system']}tool_declare{_CONTENT_XML}" f"{json.dumps(tools, ensure_ascii=False)}{_END_MESSAGE}"
        )

    for msg in messages:
        role = msg.get("role")
        if role not in _MESSAGE:
            raise ValueError(f"unsupported Inkling message role {role!r}; expected one of {sorted(_MESSAGE)}")
        if role == "assistant" and msg.get("reasoning_content"):
            rc = msg["reasoning_content"]
            if not isinstance(rc, str):
                raise TypeError("assistant reasoning_content must be a string for Inkling rendering")
            parts.append(f"{_MESSAGE[role]}{_CONTENT_THINKING}{rc}{_END_MESSAGE}")
        parts.append(f"{_MESSAGE[role]}{_CONTENT_TEXT}{_text_content(msg.get('content', ''))}{_END_MESSAGE}")

    if add_generation_prompt:
        parts.append(_MESSAGE_MODEL)
    return "".join(parts)
