"""Shared bridge for the DeepSeek official-encoder families (V3.2, V4).

Neither family ships a jinja chat_template: sglang renders their prompts
through per-family ``encoding_dsv*`` modules that share one calling
convention, and miles' ``apply_chat_template`` routes any matching tokenizer
here so training-side renders stay byte-aligned with what the runtime
serves.  Each family is one ``DeepSeekFamily`` instance wrapping its encoder
module; everything else is shared.
"""

from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32
from sglang.srt.entrypoints.openai.protocol import Tool

_ASSISTANT_SP_TOKEN = "<｜Assistant｜>"


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


class DeepSeekFamily:
    """Shared behavior for the DeepSeek official-encoder families."""

    template: Any

    def _build_encode_config(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if (enable_thinking := kwargs.pop("enable_thinking", None)) is not None:
            kwargs.setdefault("thinking_mode", "thinking" if enable_thinking else "chat")
        # sglang can accept thinking as a kwarg to set thinking_mode, like dsv3.1
        if (thinking := kwargs.pop("thinking", None)) is not None:
            kwargs.setdefault("thinking_mode", "thinking" if thinking else "chat")
        # encode_messages requires thinking_mode; thinking-on matches the server default
        kwargs.setdefault("thinking_mode", "thinking")
        return kwargs

    def render_thinking_enabled(self, chat_template_kwargs: dict[str, Any]) -> bool:
        """Whether *chat_template_kwargs* resolve to thinking mode, through the
        same resolution path ``render_messages`` uses."""
        return self._build_encode_config(chat_template_kwargs)["thinking_mode"] == "thinking"

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        raise NotImplementedError

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        """Render *messages* into this family's prompt via sglang ``encode_messages``.

        Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
        injected into the system message (see ``_inject_tools_into_system``).  The
        encoder appends a family- and role-specific next-assistant suffix after
        generation-ready tails; ``add_generation_prompt=False`` strips it.
        """
        encode_config = self._build_encode_config(kwargs)
        if tools:
            messages = _inject_tools_into_system(messages, tools)
        rendered = self.template.encode_messages(messages, **encode_config)
        if add_generation_prompt or not messages:
            return rendered

        thinking_token = (
            self.template.thinking_start_token
            if encode_config["thinking_mode"] == "thinking"
            else self.template.thinking_end_token
        )
        suffix = self._generation_prompt_suffix(messages[-1].get("role"), thinking_token)
        if suffix is None or not rendered.endswith(suffix):
            return rendered
        return rendered[: -len(suffix)]


class DeepSeekV32Family(DeepSeekFamily):
    template = encoding_dsv32

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        if tail_role in {"user", "developer"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        if tail_role == "tool":
            return "\n\n" + thinking_token
        return None


class DeepSeekV4Family(DeepSeekFamily):
    template = encoding_dsv4

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        if tail_role in {"user", "developer", "tool"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        return None


V32 = DeepSeekV32Family()
V4 = DeepSeekV4Family()

_FAMILIES = {
    "deepseek_v32": V32,
    "deepseek_v4": V4,
}


def model_type(tokenizer: Any) -> str | None:
    """The DeepSeek family ``model_type`` for *tokenizer*, or ``None`` when the
    checkpoint is not a DeepSeek official-encoder family."""
    mt = _read_model_type(tokenizer.name_or_path)
    return mt if mt in _FAMILIES else None


def apply_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict] | None = None,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
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
    rendered = _FAMILIES[mt].render_messages(
        messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs
    )
    return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered
