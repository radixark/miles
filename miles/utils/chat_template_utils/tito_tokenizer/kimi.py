"""Moonshot Kimi K2 TITO tokenizers: K2.5 and K2.6."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator


def _kimi_segment_special_token_ids(tokenizer: Any) -> set[int]:
    """Kimi specials minus ``<|im_middle|>`` (intra-turn role-name/body
    separator, not a role boundary; must not be a segment boundary)."""
    return TokenSeqComparator.collect_special_ids(tokenizer) - {tokenizer.convert_tokens_to_ids("<|im_middle|>")}


class Kimi25TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.5: ``<|im_end|>`` boundary (no trailing newline).

    K2.5 has no kwarg escape hatch for the "drop reasoning of prior assistants
    once a new non-tool-call assistant arrives" behavior.  Ships a
    bundled fixed jinja that wraps the ``last_non_tool_call_assistant_msg``
    loop in ``{%- if not preserve_thinking -%}`` so multi-user-turn rollout
    can pass ``preserve_thinking=True`` to keep history append-only.  Only the
    ``{tool, user}`` surface is registered (per current onboarding scope).
    """

    FIXED_TEMPLATE = FixedTemplate(
        template="kimi_k25_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        consistant_kwargs=["thinking", "tools_ts_str"],
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

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
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
        )


class Kimi26TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.6: same boundary as K2.5 + native ``preserve_thinking`` kwarg.

    K2.6's HF-native template already carries the ``preserve_thinking`` gate
    that K2.5 needs patched in.  No bundled fixed
    template required; ``{tool, user}`` row registers ``template=None`` and
    auto-merges ``preserve_thinking=True`` for multi-user-turn rollout.

    Tool-call parser is bound to ``kimi_k2_raw_id`` rather than ``kimi_k2``:
    RL trajectories need the model-emitted ``tool_call_id`` to round-trip
    verbatim across turns (no ``history_tool_calls_cnt`` renumbering), and
    miles is the primary consumer of this TITO family.
    """

    reasoning_parser = "kimi_k2"
    tool_call_parser = "kimi_k2_raw_id"

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"preserve_thinking": True},
        consistant_kwargs=["tools_ts_str"],
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

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
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
        )
