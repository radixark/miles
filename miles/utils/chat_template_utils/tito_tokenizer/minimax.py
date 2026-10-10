"""MiniMax M2 TITO tokenizers.

M2.5 and M2.7 share tokenizer/arch and stop-token semantics; only their
default system identity strings differ.
"""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer


class MinimaxM25TITOTokenizer(TITOTokenizer):
    """MiniMax-M2.5 family: bespoke ``]~!b[`` / ``[e~[`` / ``]~b]`` tag set.

    Shares tokenizer.json (sha256) and architecture (MiniMaxM2ForCausalLM)
    with M2.7 — only the chat template's default system identity string
    differs (``MiniMax-M2.5`` vs ``MiniMax-M2.7``).  Stop-token handling
    (``[e~[`` / trailing newline) is identical to M2.7.

    Reasoning is gated by a per-message ``last_user_index`` check:
    ``reasoning_content`` is only rendered for assistant turns *after* the
    last ``user`` — appending a new ``user`` therefore strips prior assistant
    ``<think>`` blocks and breaks append-only.  Only ``{tool}`` surface is
    registered on HF-native template for that reason; multi-user-turn
    requires the fixed jinja with ``clear_thinking=False`` to always
    preserve history reasoning.  The fixed Jinja renders tool, user, and
    assistant appends, but ignores mid-session system messages, so that role
    is excluded from its capability.
    """

    reasoning_parser = "minimax-append-think"
    tool_call_parser = "minimax-m2"

    FIXED_TEMPLATE = FixedTemplate(
        template="minimax_m25_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
    )

    _default_assistant_start_str: str = "]~b]ai"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._eos_id: int = tokenizer.convert_tokens_to_ids("[e~[")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
        return prefix + incremental


class MinimaxM27TITOTokenizer(MinimaxM25TITOTokenizer):
    """MiniMax-M2.7 family: tokenizer / arch / stop-token semantics identical
    to M2.5; the chat template only differs by default system identity string.

    Inherits parsers, ``__init__``, ``merge_tokens``, and
    ``_default_assistant_start_str`` from M2.5; only ``FIXED_TEMPLATE``
    is rebound to ``minimax_m27_fixed.jinja`` so the fixed-template lookup
    points at the M2.7-derived jinja.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template="minimax_m27_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
    )
