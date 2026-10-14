"""DeepSeek TITO tokenizers: V3.2, V4, and V4.1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from miles.utils.chat_template_utils import deepseek
from miles.utils.chat_template_utils.message_matcher_hub import assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer

_DEEPSEEK_MODE_KWARG_ALIASES = frozenset({"thinking_mode", "enable_thinking", "thinking"})


class DeepSeekV32TITOTokenizer(TITOTokenizer):
    """DeepSeek V3.2 — miles' vendored copy of the official ``encoding_dsv32``.

    V3.2 ships no jinja chat_template; prompts render through
    ``templates.encoding_dsv32.encode_messages``, and miles'
    ``apply_chat_template`` routes any V3.2 tokenizer to the thin
    ``chat_template_utils.deepseek`` bridge.  TITO incremental tokenization
    rides that same bridge.

    Upstream ``encoding_dsv32`` gates every thinking block on
    ``last_user_idx``: appending a *user* turn re-classifies every prior
    assistant as "before last user" and strips its thinking block, which is
    not append-only.  The vendored copy honors ``drop_thinking=False`` at the
    render level (like ``encoding_dsv4``), so every surface pins it and the
    ``{tool, user}`` surface becomes legal.
    """

    reasoning_parser = "deepseek-v3"
    tool_call_parser = "deepseekv32"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"drop_thinking": False},
        consistant_kwargs=["add_default_bos_token", "context"],
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # V3.2 has no jinja template, so assistant_start_str can't be sniffed
        # from one; pin it explicitly.  The comparator keys off the User /
        # Assistant sentinels to find assistant-content boundaries.
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
        )

    def request_arg_rules(self) -> list[Callable[..., None]]:
        rules = super().request_arg_rules()
        rules.append(self.resolve_thinking)
        return rules

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        super().resolve_template_kwargs(request_args, request_source=request_source, turn_args=turn_args)
        # Thinking aliases are one setting: history overrides the request and launch defaults.
        for source in (
            (turn_args or {}).get("chat_template_kwargs") or {},
            request_source.get("chat_template_kwargs") or {},
            self.chat_template_kwargs,
        ):
            mode = {key: source[key] for key in _DEEPSEEK_MODE_KWARG_ALIASES if key in source}
            if mode:
                break
        kwargs = request_args["chat_template_kwargs"]
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        kwargs.update(mode)

    def resolve_thinking(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        kwargs = request_args["chat_template_kwargs"]
        thinking = deepseek.V32.render_thinking_enabled(kwargs)
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        # SGLang's reasoning parser reads the canonical thinking flag.
        kwargs["thinking"] = thinking


class DeepSeekV4TITOTokenizer(TITOTokenizer):
    """DeepSeek V4 — official encoder via sglang's ``encoding_dsv4``.

    Like V3.2, V4 ships no jinja chat_template; miles' ``apply_chat_template``
    routes any V4 tokenizer to the ``chat_template_utils.deepseek`` bridge, and
    TITO incremental tokenization rides that same bridge to stay byte-aligned
    with what the runtime serves.
    """

    reasoning_parser = "deepseek-v4"
    tool_call_parser = "deepseekv4"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"drop_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_default_bos_token", "context", "reasoning_effort"],
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
        )
        self._assistant_id: int = tokenizer.convert_tokens_to_ids("<｜Assistant｜>")
        self._think_bracket_ids: set[int] = {
            tokenizer.convert_tokens_to_ids("<think>"),
            tokenizer.convert_tokens_to_ids("</think>"),
        }
        self.trailing_token_ids = frozenset({self._assistant_id} | self._think_bracket_ids)

    def request_arg_rules(self) -> list[Callable[..., None]]:
        rules = super().request_arg_rules()
        rules.append(self.resolve_thinking)
        return rules

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        super().resolve_template_kwargs(request_args, request_source=request_source, turn_args=turn_args)
        # Thinking aliases are one setting: history overrides the request and launch defaults.
        for source in (
            (turn_args or {}).get("chat_template_kwargs") or {},
            request_source.get("chat_template_kwargs") or {},
            self.chat_template_kwargs,
        ):
            mode = {key: source[key] for key in _DEEPSEEK_MODE_KWARG_ALIASES if key in source}
            if mode:
                break
        kwargs = request_args["chat_template_kwargs"]
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        kwargs.update(mode)

    def resolve_thinking(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        kwargs = request_args["chat_template_kwargs"]
        thinking = deepseek.V4.render_thinking_enabled(kwargs)
        for alias in _DEEPSEEK_MODE_KWARG_ALIASES:
            kwargs.pop(alias, None)
        # SGLang's reasoning parser reads the canonical thinking flag.
        kwargs["thinking"] = thinking

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        """Diff real-history renders because V4 folds adjacent ``tool``/``user`` turns."""
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        text_old = self.apply_chat_template(old_messages, add_generation_prompt=False, template_args=template_args)
        text_new = self.apply_chat_template(new_messages, add_generation_prompt=True, template_args=template_args)
        if not text_new.startswith(text_old):
            raise ValueError(
                "deepseek_v4 render is not append-only for the appended messages "
                "(prefix render changed; check drop_thinking and tool-result ordering)"
            )
        return self._encode_text(text_new[len(text_old) :])


class DeepSeekV41TITOTokenizer(DeepSeekV4TITOTokenizer):
    """DeepSeek V4.1 — official encoder via sglang's ``encoding_dsv41``; the
    V4 incremental-render machinery applies unchanged."""

    reasoning_parser = "deepseek-v41"
    tool_call_parser = "deepseekv41"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        super().__init__(tokenizer, chat_template_kwargs=chat_template_kwargs, assistant_start_str=assistant_start_str)
        self.chat_template_kwargs = {
            **self.chat_template_kwargs,
            "thinking": deepseek.V41.render_thinking_enabled(self.chat_template_kwargs),
        }
