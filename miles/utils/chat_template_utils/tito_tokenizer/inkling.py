"""Inkling TITO tokenizer."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.chat_template_utils.message_matcher_hub import strict_message_matches
from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer


class InklingTITOTokenizer(TITOTokenizer):
    """Inkling family (Inkling / Inkling-Small).

    The runtime serves Inkling through sglang's token-level renderer
    (``chat_encoding_spec == "inkling"``).  The fixed template matches its
    empty scalar-content behavior: no empty text block, and no bare assistant
    terminator when the turn contains no rendered blocks.  All four message-role
    sentinels remain comparator boundaries so non-assistant mismatches are hard
    failures after an assistant turn.
    """

    FIXED_TEMPLATE = FixedTemplate(template="inkling_fixed.jinja", consistant_kwargs=["reasoning_effort"])

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

    def preserve_server_message_state(
        self,
        stored_messages: list[dict[str, Any]],
        request_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        preserved = list(request_messages)
        for index, request_message in enumerate(preserved):
            if index >= len(stored_messages):
                break
            stored_message = stored_messages[index]
            if stored_message.get("role") != "assistant" or not strict_message_matches(
                stored_message, request_message
            ):
                continue
            if "content_blocks" in stored_message:
                preserved[index] = {**request_message, "content_blocks": stored_message["content_blocks"]}
        return preserved
