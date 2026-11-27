"""DeepSeek TITO tokenizers: V3.2, V4, and V4.1."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from miles.utils.chat_template_utils import deepseek
from miles.utils.chat_template_utils.message_matcher_hub import assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer, _from_last_assistant
from miles.utils.chat_template_utils.truncation import MarkerPair, TruncationGrammar
from miles.utils.reasoning import REASONING_DELIMITERS


_DEEPSEEK_MODE_KWARG_ALIASES = frozenset({"thinking_mode", "enable_thinking", "thinking"})


def _dsml_truncation(block: str, invoke: str, parameter: str) -> TruncationGrammar:
    """DeepSeek closing grammar; DSML tool-call tags are fixed text around ``｜DSML｜``.

    The invoke and parameter openers are tag prefixes: their attributes follow.
    """
    return TruncationGrammar(
        think=MarkerPair(*REASONING_DELIMITERS["think"]),
        tool_blocks=(
            MarkerPair(f"<｜DSML｜{block}>", f"\n</｜DSML｜{block}>"),
            MarkerPair(f"<｜DSML｜{invoke}", f"\n</｜DSML｜{invoke}>"),
            MarkerPair(f"<｜DSML｜{parameter}", f"</｜DSML｜{parameter}>"),
        ),
        end_of_turn="<｜end▁of▁sentence｜>",
    )


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
    render level, so every surface pins it and the
    ``{tool, user}`` surface becomes legal.  The encoder refuses a ``tool``
    message after an assistant without ``tool_calls``.
    """

    reasoning_parser = "deepseek-v3"
    tool_call_parser = "deepseekv32"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"drop_thinking": False},
        consistant_kwargs=["add_default_bos_token", "context"],
        roles_after_no_tool_call=frozenset({"user", "system", "assistant"}),
        truncation_closer=_dsml_truncation("function_calls", "invoke", "parameter"),
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
    """DeepSeek V4 — deepseek-recipe via sglang's ``DeepSeekRecipe``.

    Like V3.2, V4 ships no jinja chat_template; miles' ``apply_chat_template``
    routes any V4 tokenizer to the ``chat_template_utils.deepseek`` bridge, and
    TITO incremental tokenization rides that same bridge to stay byte-aligned
    with what the session server sends as TITO input ids.

    deepseek-recipe converts only a history whose tool calls are all answered,
    so ``tool`` messages are the only continuation of an assistant with
    ``tool_calls``, and a ``tool`` message needs a preceding call.
    """

    reasoning_parser = "deepseek-v4"
    tool_call_parser = "deepseekv4"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["reasoning_effort"],
        roles_after_tool_call=frozenset({"tool"}),
        roles_after_no_tool_call=frozenset({"user", "assistant"}),
        truncation_closer=_dsml_truncation("tool_calls", "invoke", "parameter"),
    )
    SERVER_MESSAGE_KEYS = (deepseek.CONTENT_LEADING_WHITESPACE_KEY,)

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
            tokenizer.convert_tokens_to_ids(token) for token in REASONING_DELIMITERS["think"]
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
        """Diff real-history renders because V4 folds adjacent ``tool``/``user`` turns.

        Both renders start at the history's last assistant turn.  recipe folds, sorts
        and encodes nothing across an assistant turn, so they end as renders of the
        whole history do, at a cost that follows the appended messages.

        The diff is over recipe's ids where it can be: recipe encodes message text
        without the added vocabulary, so re-encoding text could turn content into
        special tokens.
        """
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        context = _from_last_assistant(old_messages)
        ids_old = self.apply_chat_template(
            context, add_generation_prompt=False, tokenize=True, template_args=template_args
        )
        ids_new = self.apply_chat_template(
            context + new_messages[len(old_messages) :],
            add_generation_prompt=True,
            tokenize=True,
            template_args=template_args,
        )
        if ids_new[: len(ids_old)] == ids_old:
            return ids_new[len(ids_old) :]
        # A plain-text junction, such as a user turn folded after a tool result, can merge
        # the prefix's last token with the appended text.  recipe encodes the text between
        # special tokens on its own, so only the appended text up to the next one changes.
        added = set(self.tokenizer.added_tokens_decoder)
        diverged = next(
            (i for i, (old, new) in enumerate(zip(ids_old, ids_new, strict=False)) if old != new),
            min(len(ids_old), len(ids_new)),
        )
        special = next((i for i in range(diverged, len(ids_new)) if ids_new[i] in added), len(ids_new))
        text_old = self.tokenizer.decode(ids_old, skip_special_tokens=False)
        junction = self.tokenizer.decode(ids_new[:special], skip_special_tokens=False)
        if not junction.startswith(text_old):
            raise ValueError(
                "deepseek_v4 render is not append-only for the appended messages "
                "(prefix render changed; check tool-result ordering)"
            )
        return deepseek.encode_plain_text(self.tokenizer, junction[len(text_old) :]) + ids_new[special:]

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        """Store the whitespace recipe's parser stripped before the content; the client's message stays as is."""
        text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=False)
        whitespace = deepseek.stripped_content_whitespace(text, assistant_message.get("content") or "")
        if not whitespace:
            return assistant_message
        return {**assistant_message, deepseek.CONTENT_LEADING_WHITESPACE_KEY: whitespace}


class DeepSeekV41TITOTokenizer(DeepSeekV4TITOTokenizer):
    """DeepSeek V4.1 — deepseek-recipe's V4.1 encoding; the V4 incremental-render
    machinery applies unchanged.

    V4.1 renders a ``system`` message where it stands, behind ``<｜System｜>``, so
    one appended after an answer or a tool result keeps the prefix.  V4 hoists
    every system message into the preamble.
    """

    reasoning_parser = "deepseek-v41"
    tool_call_parser = "deepseekv41"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        consistant_kwargs=["reasoning_effort"],
        roles_after_tool_call=frozenset({"tool"}),
        roles_after_no_tool_call=frozenset({"user", "system", "assistant"}),
        # V4.1 tag names carry a leading space.
        truncation_closer=_dsml_truncation(" calls", " invoke", " parameter"),
    )

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
