"""Shared bridge for the DeepSeek families without a jinja chat_template (V3.2, V4, V4.1).

V3.2 renders through miles' vendored official encoder (``templates.encoding_dsv32``).
V4 and V4.1 render through deepseek-recipe via sglang's ``DeepSeekRecipe``, the
renderer sglang's chat endpoint uses for a request without input_ids.  miles'
``apply_chat_template`` routes any matching tokenizer here; each family is one
instance, and the thinking-mode kwargs resolve the same way for all of them.
"""

from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.parser.deepseek_recipe_renderer import (
    ASSISTANT_TOKEN,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    DeepSeekRecipe,
    load_deepseek_recipe,
    parse_effort_budget,
)

from miles.utils.chat_template_utils.templates import encoding_dsv32

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
    must round-trip through ``Tool.model_dump()`` as sglang's dsv32 serving path
    does, or the token ids drift.
    """
    out = copy.deepcopy(messages)
    if not out or out[0].get("role") != "system":
        out.insert(0, {"role": "system", "content": ""})
    out[0]["tools"] = [Tool.model_validate(t).model_dump() for t in tools]
    return out


class DeepSeekFamily:
    """Thinking-mode resolution shared by the DeepSeek families."""

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
        same resolution path rendering uses."""
        return self._build_encode_config(chat_template_kwargs)["thinking_mode"] == "thinking"


class DeepSeekV32Family(DeepSeekFamily):
    template = encoding_dsv32

    def _generation_prompt_suffix(self, tail_role: str | None, thinking_token: str) -> str | None:
        if tail_role in {"user", "developer"}:
            return _ASSISTANT_SP_TOKEN + thinking_token
        if tail_role == "tool":
            return "\n\n" + thinking_token
        return None

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        """Render *messages* into this family's prompt via its encoder's ``encode_messages``.

        Tool_call ``arguments`` must already be JSON strings; *tools*, if given, are
        injected into the system message (see ``_inject_tools_into_system``).  The
        encoder appends a role-specific next-assistant suffix after generation-ready
        tails; ``add_generation_prompt=False`` strips it.
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

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Any,
        *,
        tools: list[dict] | None,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str | list[int]:
        rendered = self.render_messages(messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered


# The V4-Flash / V4-Pro checkpoints' own encoder uses the "preview" effort profile: high
# renders no effort line and max renders the line recipe calls high.  recipe renders the
# "official" profile; sglang v0.5.22 picked the profile per checkpoint, v0.5.23 does not.
_DSV4_PREVIEW_EFFORT = {"high": "low", "max": "high"}


@functools.cache
def _deepseek_recipe(spec: str, tokenizer: Any) -> DeepSeekRecipe:
    return DeepSeekRecipe.from_hf_tokenizer(spec, tokenizer)


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    # sglang's chat endpoint renders a null content as "".
    return message if message.get("content") is not None else {**message, "content": ""}


def _tool_payload(tool: dict[str, Any]) -> dict[str, Any]:
    # As sglang's chat endpoint passes tools: the client's fields only, without None values.
    payload = Tool.model_validate(tool).model_dump()
    function = {k: v for k, v in payload["function"].items() if v is not None}
    return {k: v for k, v in payload.items() if v is not None} | {"function": function}


def _placeholder_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Empty results for the tool calls a trailing assistant turn still awaits."""
    tail = messages[-1] if messages else {}
    if tail.get("role") != "assistant":
        return []
    return [{"role": "tool", "tool_call_id": call["id"], "content": ""} for call in tail.get("tool_calls") or []]


class DeepSeekRecipeFamily(DeepSeekFamily):
    """V4 / V4.1: deepseek-recipe renders the prompt as sglang's chat endpoint does,
    except for the V4 effort profile (``_DSV4_PREVIEW_EFFORT``).  Text renders decode
    the ids, which round-trip to recipe's prompt text."""

    def __init__(self, spec: str):
        self.spec = spec

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Any,
        *,
        tools: list[dict] | None,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str | list[int]:
        recipe = _deepseek_recipe(self.spec, tokenizer)
        ids = self._encode(recipe, messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        return ids if tokenize else recipe.hf_tokenizer.decode(ids, skip_special_tokens=False)

    def _encode(
        self,
        recipe: DeepSeekRecipe,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> list[int]:
        config = self._build_encode_config(kwargs)
        thinking = config.pop("thinking_mode") == "thinking"
        effort, budget = self._resolve_effort(config.pop("reasoning_effort", None))
        if config:
            raise TypeError(f"{self.spec} renders through deepseek-recipe, which takes no {sorted(config)}")
        body: dict[str, Any] = {"messages": [_wire_message(m) for m in messages], "thinking": thinking}
        if tools:
            body["tools"] = [_tool_payload(t) for t in tools]
        if effort is not None:
            body["reasoning_effort"] = effort

        # recipe converts only a history whose tool calls are answered; a prefix of a
        # converted conversation renders as a prefix of its prompt.
        placeholders = _placeholder_tool_results(messages)
        conversation = recipe.convert({**body, "messages": body["messages"] + placeholders}, effort_budget=budget)
        if placeholders:
            conversation = load_deepseek_recipe().Conversation(
                conversation.messages[: len(messages)],
                thinking_mode=conversation.thinking_mode,
                tools=conversation.tools,
                tool_choice=conversation.tool_choice,
                reasoning_effort=conversation.reasoning_effort,
            )
        ids = recipe.encode_conversation(conversation, effort_budget=budget)

        # recipe always ends with the next assistant turn's header.
        header = [
            recipe.hf_tokenizer.token_to_id(ASSISTANT_TOKEN),
            recipe.hf_tokenizer.token_to_id(THINK_START_TOKEN if thinking else THINK_END_TOKEN),
        ]
        if not add_generation_prompt and ids[-len(header) :] == header:
            return ids[: -len(header)]
        return ids

    def _resolve_effort(self, effort: Any) -> tuple[str | None, int | None]:
        """The recipe effort tier and the V4.1 integer budget for *effort*."""
        if self.spec == "dsv4":
            if effort is None:
                effort = "high"
            if effort not in _DSV4_PREVIEW_EFFORT:
                raise ValueError(
                    f"DeepSeek-V4 reasoning_effort must be one of {list(_DSV4_PREVIEW_EFFORT)}, got {effort!r}"
                )
            return _DSV4_PREVIEW_EFFORT[effort], None
        if effort is None:
            # The default sglang's chat endpoint applies, so a request without one renders alike.
            raw = os.environ.get("SGLANG_DSV41_REASONING_EFFORT", "").strip()
            effort = int(raw) if raw.isdigit() else raw or None
        if isinstance(effort, (int, float)) and not isinstance(effort, bool):
            return None, parse_effort_budget(effort)
        return effort, None


V32 = DeepSeekV32Family()
V4 = DeepSeekRecipeFamily("dsv4")
V41 = DeepSeekRecipeFamily("dsv41")

_FAMILIES = {
    "deepseek_v32": V32,
    "deepseek_v4": V4,
    "deepseek_v41": V41,
}


def model_type(tokenizer: Any) -> str | None:
    """The DeepSeek family ``model_type`` for *tokenizer*, or ``None`` when the
    checkpoint is not one of these families."""
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
    return _FAMILIES[mt].apply_chat_template(
        messages, tokenizer, tools=tools, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
    )
