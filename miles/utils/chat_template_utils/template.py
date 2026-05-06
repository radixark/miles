"""Core chat template operations: load from HuggingFace and render from string.

``load_hf_chat_template`` fetches original (unmodified) chat templates via
``hf_hub_download``.  Files are cached locally after the first download —
subsequent calls read from disk without network access.

``apply_chat_template_from_str`` renders a Jinja2 chat template string
without depending on a HuggingFace tokenizer, equivalent to
``tokenizer.apply_chat_template(..., tokenize=False)``.

``apply_chat_template`` applies via an HF tokenizer object (returns
``str`` or ``list[int]``).  Both functions normalize tool arguments,
canonicalize tool definitions, and fall back between tool dict formats.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from huggingface_hub import hf_hub_download
from jinja2 import TemplateError
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from transformers.utils.chat_template_utils import render_jinja_template


def load_hf_chat_template(model_id: str) -> str:
    """Load an original chat template from HuggingFace (cached locally).

    Handles two layouts:
    - ``chat_template`` field in ``tokenizer_config.json`` (most models)
    - Separate ``chat_template.jinja`` file (e.g. GLM-5)
    """
    config_path = hf_hub_download(model_id, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
    template = config.get("chat_template", "")
    if template:
        if isinstance(template, list):
            for t in template:
                if t.get("name") == "default" or not t.get("name"):
                    return t["template"]
            return template[0]["template"]
        return template

    jinja_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(jinja_path) as f:
        return f.read()


def _normalize_tool_arguments(messages: list[dict]) -> list[dict]:
    """Deep-copy messages and normalize for template rendering.

    Normalizations:
    - Parse JSON-string tool_call arguments to dicts.  Matches SGLang's
      ``_apply_jinja_template`` normalization in ``serving_chat.py``.
    - Convert ``content: None`` to ``content: ""`` for assistant messages with
      tool_calls.  The OpenAI API returns ``content: null`` for tool-call-only
      responses; Jinja2 renders Python ``None`` as the literal string "None".
    """
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("role") == "assistant":
            if msg.get("content") is None and msg.get("tool_calls"):
                msg["content"] = ""
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for item in msg["tool_calls"]:
                    func = item.get("function")
                    if func and "arguments" in func and isinstance(func["arguments"], str):
                        func["arguments"] = json.loads(func["arguments"])
    return normalized


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Canonicalize tools via Pydantic, returning full Tool model dumps.

    Matches SGLang's ``_process_messages`` (``serving_chat.py`` lines 343-344):
    ``tools = [item.model_dump() for item in request.tools]`` — each tool is
    a full ``Tool`` model dump (``{"type": "function", "function": {...}}``).
    """
    if not tools:
        return None

    wrapped = [t if isinstance(t, dict) and "function" in t else {"type": "function", "function": t} for t in tools]
    validated = TypeAdapter(list[Tool]).validate_python(wrapped)
    return [tool.model_dump() for tool in validated]


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False, no tokenizer needed).

    Calls HF transformers' ``render_jinja_template`` directly — the same
    function that ``tokenizer.apply_chat_template`` uses internally.  Both
    SGLang and our ``apply_chat_template`` go through that same HF code path.

    Applies SGLang-style normalizations (tool argument parsing, tool dict
    canonicalization, tool format fallback).
    """

    def _render(tool_defs):
        rendered, _ = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tools=tool_defs,
            **kwargs,
        )
        return rendered[0]

    messages = _normalize_tool_arguments(messages)
    tool_defs = extract_tool_dicts(tools)
    try:
        return _render(tool_defs)
    except TemplateError as e:
        if tool_defs is not None:
            try:
                return _render([t["function"] if "function" in t else t for t in tool_defs])
            except TemplateError as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e


_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def _normalize_value(value: Any) -> Any:
    """Normalize falsy sentinels that produce identical Jinja2 output.

    None, "" and [] are all falsy in Jinja2 and render the same way,
    but client libraries may interchange them (e.g. content: null vs ""
    for tool-call-only responses, or tool_calls: null vs []).

    Only collapses falsy values — non-falsy content (including whitespace
    like trailing newlines) is returned as-is.  Message boundary characters
    must be preserved exactly so they tokenize identically across turns.
    """
    if value is None or value == "" or value == []:
        return None
    return value


def message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare only the fields that affect chat-template tokenization.

    External client libraries (e.g. litellm) may inject extra keys like
    ``provider_specific_fields`` into messages.  These have no effect on
    the Jinja2 chat template output, so we only compare the keys that
    templates actually read: role, content, reasoning_content, tool_calls.
    """
    for key in _TEMPLATE_RELEVANT_KEYS:
        if _normalize_value(stored.get(key)) != _normalize_value(new.get(key)):
            return False
    return True


_DEFAULT_APPEND_ROLES: list[str] = ["tool"]


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: list[str] = _DEFAULT_APPEND_ROLES,
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*.

    The stored prefix must match exactly (compared by template-relevant keys),
    and any appended messages must have a role in *allowed_append_roles*
    (default: ``{'tool'}``).
    """
    if not stored_messages:
        return

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not message_matches(stored_msg, new_messages[i]):
            diffs = {
                key: {"stored": repr(stored_msg.get(key))[:200], "new": repr(new_messages[i].get(key))[:200]}
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    for j, msg in enumerate(new_messages[len(stored_messages) :]):
        if msg.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={allowed_append_roles}"
            )


def apply_chat_template(
    messages: list[dict],
    *,
    tokenizer,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
) -> str | list[int]:
    """Apply chat template via HF tokenizer in SGLang style.

    Passes ``return_dict=False`` to match SGLang's ``serving_chat.py``,
    ensuring the result is ``str`` (tokenize=False) or ``list[int]``
    (tokenize=True), not a ``BatchEncoding`` or ``dict``.
    """
    messages = _normalize_tool_arguments(messages)
    tool_defs = extract_tool_dicts(tools)
    render_kwargs = dict(add_generation_prompt=add_generation_prompt, **kwargs)

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, tools=tool_defs, return_dict=False, **render_kwargs
        )
    except TemplateError as e:
        if tool_defs is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    tools=[t["function"] if "function" in t else t for t in tool_defs],
                    return_dict=False,
                    **render_kwargs,
                )
            except TemplateError as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e
