"""Token-aware parsing of raw Inkling assistant completions.

SGLang returns the generated token IDs in ``meta_info.output_token_logprobs``.
The session server uses those IDs as the source of truth and exposes ordinary
OpenAI ``reasoning_content``/``content``/``tool_calls`` fields to the agent.
Parsing is stateless and linear in the completion length so every session
server process can use its own tokenizer instance without coordination.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from miles.utils.reasoning import REASONING_DELIMITERS


@dataclass(frozen=True)
class ParsedAssistantCompletion:
    """Optional model-family projection of one raw assistant completion."""

    client_message: dict[str, Any]
    stored_message: dict[str, Any]
    parser_name: str
    parse_error: str | None = None
    # What the completion leaves open, or None when the turn ended: "header" (a message
    # with nothing after its start), "recipient" (a tool recipient with no content token),
    # the kind of an unfinished content block ("thinking", "text", "tool_json",
    # "tool_text"), or "between" (every message ended, the turn did not).
    open_part: str | None = None
    # The error that stopped the scan of the turn's messages, as opposed to one in
    # a message's content (a malformed tool call); ``open_part`` is then where it stopped.
    frame_error: str | None = None


class InklingResponseParser:
    """Parse Inkling control-token IDs without interpreting marker-like text."""

    _TOKEN_TEXT = {
        "message_model": "<|message_model|>",
        "content_thinking": REASONING_DELIMITERS["inkling"][0],
        "content_text": REASONING_DELIMITERS["inkling"][1],
        "content_invoke_tool_json": "<|content_invoke_tool_json|>",
        "content_invoke_tool_text": "<|content_invoke_tool_text|>",
        "end_message": "<|end_message|>",
        "end_sampling": "<|content_model_end_sampling|>",
    }

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.ids = {name: self._single_token_id(literal) for name, literal in self._TOKEN_TEXT.items()}
        # Treat every tokenizer-native special ID as framing.  This rejects a
        # real control token in payload while preserving marker-looking text
        # that the model spelled with ordinary BPE tokens.
        self._control_ids = frozenset(
            {*self.ids.values(), *getattr(tokenizer, "all_special_ids", [])}
        )  # config-access-exempt: custom tokenizers may omit special-token metadata

    def _single_token_id(self, literal: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(literal)
        if token_id is None or token_id == getattr(
            self.tokenizer, "unk_token_id", None
        ):  # config-access-exempt: custom tokenizers may omit special-token metadata
            encoded = self.tokenizer.encode(literal, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Inkling control token {literal!r} did not encode to one ID: {encoded}")
            token_id = encoded[0]
        return int(token_id)

    def _decode_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def _tool_call_id(completion_token_ids: list[int], index: int) -> str:
        digest = hashlib.sha256()
        digest.update(index.to_bytes(4, "big"))
        for token_id in completion_token_ids:
            digest.update(int(token_id).to_bytes(4, "big", signed=True))
        return f"call_{digest.hexdigest()[:24]}"

    def parse(
        self,
        completion_token_ids: list[int],
        *,
        finish_reason: str | None,
    ) -> ParsedAssistantCompletion:
        """Parse one completion.

        A completion cut by ``finish_reason='length'`` is not malformed: its
        unfinished message stays in ``content_blocks`` as sampled, so the
        stored message renders the cut turn followed by its closer
        (``InklingTruncation``).  An unfinished tool call is stored with its
        recipient and partial JSON and is not a call.
        """
        blocks: list[dict[str, Any]] = []
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        error: str | None = None
        frame_error: str | None = None
        ended = False

        # The generation prompt already supplied the first MESSAGE_MODEL token.
        state = "header"
        header_ids: list[int] = []
        payload_ids: list[int] = []
        block_kind: str | None = None

        def fail(reason: str) -> None:
            nonlocal error
            if error is None:
                error = reason

        def fail_frame(reason: str) -> None:
            nonlocal frame_error
            frame_error = reason
            fail(reason)

        def close_block(*, cut: bool = False) -> None:
            nonlocal block_kind, header_ids, payload_ids
            if block_kind is None:
                fail("end_message_without_content")
                return
            text = self._decode_text(payload_ids)
            recipient = self._decode_text(header_ids)
            if block_kind == "thinking":
                if recipient:
                    fail("unexpected_thinking_recipient")
                blocks.append({"type": "thinking", "text": text})
                reasoning_parts.append(text)
            elif block_kind == "text":
                if recipient:
                    fail("unexpected_text_recipient")
                blocks.append({"type": "text", "text": text})
                visible_parts.append(text)
            elif block_kind == "tool_json" and cut:
                blocks.append({"type": "tool_call", "recipient": recipient, "raw_json": text})
            elif block_kind == "tool_json":
                try:
                    payload = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    fail("malformed_tool_json")
                    payload = None
                if not isinstance(payload, dict):
                    fail("tool_payload_not_object")
                else:
                    name = payload.get("name")
                    arguments = payload.get("args")
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        fail("invalid_tool_payload_shape")
                    else:
                        call_index = len(tool_calls)
                        tool_calls.append(
                            {
                                "id": self._tool_call_id(completion_token_ids, call_index),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                                },
                            }
                        )
                        blocks.append(
                            {
                                "type": "tool_call",
                                "recipient": recipient,
                                "name": name,
                                "arguments": arguments,
                                "raw_json": text,
                            }
                        )
            else:
                fail("unsupported_text_tool_call")
            block_kind = None
            header_ids = []
            payload_ids = []

        for token_id in completion_token_ids:
            token_id = int(token_id)
            if ended:
                fail_frame("tokens_after_end_sampling")
                break

            if state == "payload":
                if token_id == self.ids["end_message"]:
                    close_block()
                    state = "between"
                elif token_id in self._control_ids:
                    fail_frame("control_token_inside_open_block")
                    break
                else:
                    payload_ids.append(token_id)
                continue

            if state == "between":
                if token_id == self.ids["message_model"]:
                    state = "header"
                elif token_id == self.ids["end_sampling"]:
                    ended = True
                else:
                    fail_frame("missing_message_model_boundary")
                    break
                continue

            # header: ordinary IDs are the model-emitted invocation recipient.
            if token_id == self.ids["message_model"] and not header_ids:
                continue
            if token_id == self.ids["content_thinking"]:
                block_kind = "thinking"
            elif token_id == self.ids["content_text"]:
                block_kind = "text"
            elif token_id == self.ids["content_invoke_tool_json"]:
                block_kind = "tool_json"
            elif token_id == self.ids["content_invoke_tool_text"]:
                block_kind = "tool_text"
            elif token_id == self.ids["end_sampling"] and not header_ids:
                ended = True
                state = "between"
                continue
            elif token_id in self._control_ids:
                fail_frame("unexpected_control_token_in_header")
                break
            else:
                header_ids.append(token_id)
                continue
            payload_ids = []
            state = "payload"

        open_part: str | None = None
        if state == "payload":
            open_part = block_kind
        elif state == "header":
            open_part = "recipient" if header_ids else "header"
        elif not ended:
            open_part = "between"

        # An earlier message's content error does not stop the scan, so the cut message is still kept.
        if open_part is not None and frame_error is None and finish_reason == "length":
            # The closer ends a bare header as an empty text message and a recipient as an empty tool call.
            if state == "header":
                block_kind = "tool_json" if header_ids else "text"
            if open_part != "between":
                close_block(cut=True)
        elif state == "payload":
            fail("unterminated_content_block")
        elif state == "header" and header_ids:
            fail("unterminated_message_header")
        elif not ended:
            fail("missing_end_sampling")

        parse_error = error
        client_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(visible_parts),
        }
        if reasoning_parts:
            client_message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls and parse_error is None:
            client_message["tool_calls"] = tool_calls
        stored_message = {
            **client_message,
            "content_blocks": blocks,
        }
        return ParsedAssistantCompletion(
            client_message=client_message,
            stored_message=stored_message,
            parser_name="inkling",
            parse_error=parse_error,
            open_part=open_part,
            frame_error=frame_error,
        )
