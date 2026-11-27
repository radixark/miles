"""Muse Spark TITO tokenizer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator
from miles.utils.reasoning import REASONING_DELIMITERS

_START, _MESSAGE, _EOM, _EOT = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"
# The generation prompt is "<|start|>assistant", so the turn's first header continues after the role.
_ROLE = "assistant"
_HEADER = "assistant to="
# Recipients whose body is text; any other recipient is a tool whose body is an ATEM block.
_TEXT_RECIPIENTS = ("self", "user", "commentary")
# Recipients whose message may end the turn with <|eom|>, as the template renders them.
_EOM_ENDS_TURN = ("self", "commentary")
# An empty final message: tail-only closing cannot turn a sampled <|eom|> into <|eot|>.
_FINAL_HEADER = f"{_ROLE} {REASONING_DELIMITERS['muse_spark'][1]}"
_FINAL_MESSAGE = f"{_START}{_FINAL_HEADER}{_EOT}"

# Complete ATEM tags; a partial opening tag is sampled text.  A sampled opening tag may hold
# more than the name="..." the template writes.
_ATEM_TAG = re.compile(
    r"<atem:(function_calls)>|<atem:(invoke|parameter)\b[^<>]*>|</atem:(function_calls|invoke|parameter)>"
)
# element -> (closing tag, text the template writes after it, whether the template starts it on a new line)
_ATEM_CLOSE = {
    "parameter": ("</atem:parameter>", "\n", False),
    "invoke": ("</atem:invoke>", "\n", True),
    "function_calls": ("</atem:function_calls>", "", True),
}

# Server-only key of a stored assistant message cut by finish_reason='length': the turn as
# sampled plus its closer, one {"header", "body", "end"} per message.  ``header`` is the
# text after the message's <|start|>; the first message's follows the generation prompt.
TRUNCATED_TURN_KEY = "truncated_turn"


@dataclass(frozen=True)
class _Message:
    header: str
    first: bool
    body: str | None = None  # None while the header is open
    end: str | None = None  # None while the message is open

    @property
    def full_header(self) -> str:
        return (_ROLE if self.first else "") + self.header

    @property
    def recipient(self) -> str | None:
        """The recipient, or None for a header that is not ``assistant to=<recipient>`` (yet)."""
        return self.full_header[len(_HEADER) :] if self.full_header.startswith(_HEADER) else None

    @property
    def is_tool(self) -> bool:
        return self.recipient is not None and self.recipient not in _TEXT_RECIPIENTS


@dataclass(frozen=True)
class _Turn:
    messages: tuple[_Message, ...]  # every message, the open one last
    is_open: bool
    frame_error: str | None = None


def _scan(tokenizer: Any, completion_ids: list[int]) -> _Turn:
    """Split a turn's sampled tokens into messages at its control tokens."""
    start, message, eom, eot = tokenizer.convert_tokens_to_ids([_START, _MESSAGE, _EOM, _EOT])
    ends = {eom: _EOM, eot: _EOT}
    messages: list[_Message] = []
    header_ids: list[int] = []
    body_ids: list[int] | None = None
    state = "header"

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    def stop(reason: str) -> _Turn:
        return _Turn(tuple(messages), is_open=False, frame_error=reason)

    for token_id in completion_ids:
        if state == "header":
            if token_id == message:
                state, body_ids = "body", []
            elif token_id in (start, eom, eot):
                return stop(f"{decode([token_id])} in a message header")
            else:
                header_ids.append(token_id)
        elif state == "body":
            if token_id in ends:
                messages.append(
                    _Message(header=decode(header_ids), first=not messages, body=decode(body_ids), end=ends[token_id])
                )
                state = "ended" if token_id == eot else "between"
            elif token_id == start:
                return stop(f"{_START} in a message body")
            else:
                body_ids.append(token_id)
        elif state == "between" and token_id == start:
            state, header_ids = "header", []
        else:
            return stop(f"{decode([token_id])} after the end of a message")

    if state == "header":
        messages.append(_Message(header=decode(header_ids), first=not messages))
    elif state == "body":
        messages.append(_Message(header=decode(header_ids), first=not messages, body=decode(body_ids)))
    return _Turn(tuple(messages), is_open=state in ("header", "body"))


def _open_atem(body: str) -> list[str]:
    """ATEM elements *body* opens and leaves open, innermost last."""
    stack: list[str] = []
    for match in _ATEM_TAG.finditer(body):
        opened = match.group(1) or match.group(2)
        if opened:
            stack.append(opened)
        elif match.group(3) in stack:
            # A closing tag closes its element and every element opened inside it.
            del stack[len(stack) - 1 - stack[::-1].index(match.group(3)) :]
    return stack


def _close_atem(body: str) -> str:
    """Text that closes every open ATEM element of *body*, innermost first, as the template renders it."""
    stack = _open_atem(body)
    closer = ""
    if stack:
        tag, after, _ = _ATEM_CLOSE[stack[-1]]
        # A closing tag cut after at least "</" is completed; a lone "<" may begin an opening tag.
        started = next((size for size in range(len(tag) - 1, 1, -1) if body.endswith(tag[:size])), 0)
        if started:
            closer = tag[started:] + after
            stack.pop()
    for element in reversed(stack):
        tag, after, on_new_line = _ATEM_CLOSE[element]
        lead = "\n" if on_new_line and not (body + closer).endswith("\n") else ""
        closer += lead + tag + after
    return closer


def _end_token(message: _Message) -> str:
    return _EOM if message.recipient in _EOM_ENDS_TURN else _EOT


def _closing_text(turn: _Turn) -> str:
    """Text that finishes the open message, if any, and ends the turn; nothing after a broken frame."""
    if turn.frame_error is not None:
        return ""
    last = turn.messages[-1]
    if not turn.is_open:
        return "" if last.end == _EOT or last.recipient in _EOM_ENDS_TURN else _FINAL_MESSAGE
    if last.body is None:
        if _HEADER.startswith(last.full_header):
            return _FINAL_HEADER[len(last.full_header) :] + _EOT
        return _MESSAGE + _end_token(last)
    if last.is_tool:
        return _close_atem(last.body) + _EOT
    return _end_token(last)


@dataclass(frozen=True)
class MuseSparkTruncation:
    """Closes a Muse Spark turn cut by ``finish_reason='length'`` (``TruncationCloser``).

    A turn is a run of messages ``<|start|>assistant to=<recipient><|message|>body<|eom|>``;
    the generation prompt ends at ``<|start|>assistant``.  Recipient ``self`` is
    reasoning, ``user`` and ``commentary`` are text, any other is a tool call
    whose body is an ``<atem:function_calls>`` block.  The template ends the
    turn's last message with ``<|eot|>`` unless it is ``self`` or ``commentary``.

    ``<|start|>``, ``<|eom|>`` and ``<|eot|>`` are comparator boundaries, so the
    stored message must hold the cut turn too: ``closed_turn_messages`` is the
    stored view ``MuseSparkTITOTokenizer.postprocess_completion`` keeps.
    """

    @property
    def reasoning_answer_delimiter(self) -> str:
        return REASONING_DELIMITERS["muse_spark"][1]

    def close_truncated_turn(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[int]:
        closing_text = _closing_text(_scan(tokenizer, token_ids[completion_start:]))
        return tokenizer.encode(closing_text, add_special_tokens=False)

    def check_turn_structure(
        self, token_ids: list[int], span: tuple[int, int], *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[str]:
        turn = _scan(tokenizer, token_ids[span[0] : span[1]])
        if turn.frame_error is not None:
            return [turn.frame_error]
        problems = []
        for message in turn.messages:
            header = message.full_header
            if message.recipient is None and not (message.body is None and _HEADER.startswith(header)):
                problems.append(f"header {header!r} is not '{_HEADER}<recipient>'")
            open_atem = _open_atem(message.body) if message.end is not None and message.is_tool else []
            if open_atem:
                problems.append(f"to={message.recipient} message ends with <atem:{open_atem[-1]}> open")
        last = turn.messages[-1]
        if turn.is_open:
            part = "message header" if last.body is None else f"to={last.recipient} message"
            problems.append(f"unclosed: {part}")
        elif last.end == _EOM and last.recipient not in _EOM_ENDS_TURN:
            problems.append(f"turn does not end with {_EOT}")
        return problems

    def cut_kind(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> str:
        turn = _scan(tokenizer, token_ids[completion_start:])
        if not turn.is_open:
            return "answer"
        last = turn.messages[-1]
        if last.body is not None:
            return "reasoning" if last.recipient == "self" else "tool_call" if last.is_tool else "answer"
        # A partial recipient that may still become a text recipient is not a tool call yet.
        recipient = last.recipient
        is_text = not recipient or any(text.startswith(recipient) for text in _TEXT_RECIPIENTS)
        return "answer" if is_text else "tool_call"

    def closed_turn_messages(self, completion_ids: list[int], *, tokenizer: Any) -> list[dict[str, str]] | None:
        """The cut turn's sampled tokens followed by its closer, one dict per message; None after a broken frame."""
        closer = self.close_truncated_turn(completion_ids, 0, tokenizer=tokenizer, message_boundary_ids=set())
        turn = _scan(tokenizer, completion_ids + closer)
        if turn.frame_error is not None:
            return None
        return [{"header": m.header, "body": m.body, "end": m.end} for m in turn.messages]


class MuseSparkTITOTokenizer(TITOTokenizer):
    FIXED_TEMPLATE = FixedTemplate(template="muse_spark_fixed.jinja", truncation_closer=MuseSparkTruncation())
    SERVER_MESSAGE_KEYS = (TRUNCATED_TURN_KEY,)
    reasoning_parser = "muse"
    tool_call_parser = "muse"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        # Keep the recipient header and body in one comparator segment; the
        # message separator is internal to a turn, not a role boundary.
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or _START + _ROLE,
            special_token_ids=TokenSeqComparator.collect_special_ids(tokenizer)
            - {tokenizer.convert_tokens_to_ids(_MESSAGE)},
        )

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        """Store a turn cut by length as sampled plus its closer; the client's message is not changed.

        The parsed message has no place for an unfinished message and the
        template cannot render a partial one, so the template renders
        ``TRUNCATED_TURN_KEY`` as stored instead.
        """
        if choice.get("finish_reason") != "length":
            return assistant_message
        closer = self.FIXED_TEMPLATE.truncation_closer
        messages = closer.closed_turn_messages(list(completion_token_ids), tokenizer=self.tokenizer)
        if messages is None:
            return assistant_message
        return {**assistant_message, TRUNCATED_TURN_KEY: messages}
