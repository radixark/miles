"""Inkling TITO tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer
from miles.utils.reasoning import REASONING_DELIMITERS

# Control tokens that finish the message a cut left open (``ParsedAssistantCompletion.open_part``)
# and end the turn; an unfinished content block of any kind only needs its end.
_END_BLOCK = ("end_message", "end_sampling")
_CLOSERS = {
    None: (),
    "between": ("end_sampling",),
    "header": ("content_text", *_END_BLOCK),
    "recipient": ("content_invoke_tool_json", *_END_BLOCK),
}
_CUT_KINDS = {"thinking": "reasoning", "recipient": "tool_call", "tool_json": "tool_call", "tool_text": "tool_call"}


@dataclass(frozen=True)
class InklingTruncation:
    """Closes an Inkling turn cut by ``finish_reason='length'`` (``TruncationCloser``).

    A turn is a run of messages, each ``<|message_model|>``, an optional tool
    recipient, a content token, a body and ``<|end_message|>``, ended by
    ``<|content_model_end_sampling|>``.  Every ``<|message_model|>`` is a
    comparator boundary, so the stored message must hold the unfinished message
    too; ``InklingResponseParser`` keeps it for a completion cut by length.
    """

    @property
    def reasoning_answer_delimiter(self) -> str:
        return REASONING_DELIMITERS["inkling"][1]

    def close_truncated_turn(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[int]:
        parser = InklingResponseParser(tokenizer)
        open_part = parser.parse(token_ids[completion_start:], finish_reason="length").open_part
        return [parser.ids[name] for name in _CLOSERS.get(open_part, _END_BLOCK)]

    def check_turn_structure(
        self, token_ids: list[int], span: tuple[int, int], *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[str]:
        parsed = InklingResponseParser(tokenizer).parse(token_ids[span[0] : span[1]], finish_reason="length")
        # A message's content is not structure: an unfinished tool call stays malformed after closing.
        if parsed.frame_error is not None:
            return [parsed.frame_error]
        if parsed.open_part == "between":
            return ["turn does not end with <|content_model_end_sampling|>"]
        return [] if parsed.open_part is None else [f"unclosed: {parsed.open_part}"]

    def cut_kind(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> str:
        parsed = InklingResponseParser(tokenizer).parse(token_ids[completion_start:], finish_reason="length")
        return _CUT_KINDS.get(parsed.open_part, "answer")


class InklingTITOTokenizer(TITOTokenizer):
    """Inkling family (Inkling / Inkling-Small).

    The runtime serves Inkling through sglang's token-level renderer
    (``chat_encoding_spec == "inkling"``).  The fixed template matches its
    empty scalar-content behavior: no empty text block, and no bare assistant
    terminator when the turn contains no rendered blocks.  All four message-role
    sentinels remain comparator boundaries so non-assistant mismatches are hard
    failures after an assistant turn.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template="inkling_fixed.jinja", consistant_kwargs=["reasoning_effort"], truncation_closer=InklingTruncation()
    )
    SERVER_MESSAGE_KEYS = ("content_blocks",)

    _DEFAULT_ASSISTANT_START = "<|message_model|>"

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
                tokenizer.convert_tokens_to_ids("<|message_user|>"),
                tokenizer.convert_tokens_to_ids("<|message_model|>"),
                tokenizer.convert_tokens_to_ids("<|message_system|>"),
                tokenizer.convert_tokens_to_ids("<|message_tool|>"),
            },
        )
        self._response_parser = None

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        if self._response_parser is None:
            self._response_parser = InklingResponseParser(self.tokenizer)
        parsed = self._response_parser.parse(
            completion_token_ids,
            finish_reason=choice.get("finish_reason"),
        )
        choice["message"] = parsed.client_message
        meta_info = choice.setdefault("meta_info", {})
        meta_info["miles_response_parser"] = parsed.parser_name
        if parsed.parse_error is not None:
            meta_info["miles_response_parse_error"] = parsed.parse_error
        elif parsed.client_message.get("tool_calls") and choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"
        return parsed.stored_message
