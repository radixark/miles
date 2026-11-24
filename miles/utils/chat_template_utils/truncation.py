"""Closing assistant turns cut by ``finish_reason='length'``.

A length-truncated turn can stop inside reasoning or a tool-call block and
never emits its end-of-turn token.  Continuing it as is would place the next
message inside an unclosed structure.  A family declares a ``TruncationCloser``
on its ``FixedTemplate``; the closer returns the tokens that close the turn
after the sampled tokens, without changing any of them.

``TruncationGrammar`` is the closer for families whose assistant turn is one
message with paired text markers.  Its markers are matched in decoded text, the
way SGLang's reasoning and tool-call parsers see them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# Where a turn can stop, as ``TruncationCloser.cut_kind`` names it.
CUT_KINDS = ("reasoning", "tool_call", "answer")


class TruncationCloser(Protocol):
    """How one family closes, and checks, an assistant turn.

    *token_ids* is the prompt followed by the turn's sampled tokens.
    *message_boundary_ids* are the token IDs that delimit messages: the
    family's comparator segment boundaries.
    """

    @property
    def reasoning_answer_delimiter(self) -> str:
        """Marker separating reasoning from the final answer in decoded text."""

    def close_truncated_turn(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[int]:
        """Tokens that close the turn whose sampled tokens start at *completion_start*."""

    def check_turn_structure(
        self, token_ids: list[int], span: tuple[int, int], *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[str]:
        """Structure problems of the turn at *span*, ``(completion_start, turn_end)``; empty when well formed."""

    def cut_kind(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> str:
        """Which of ``CUT_KINDS`` the turn stops in: an open tool call, open reasoning, or neither."""


@dataclass(frozen=True)
class MarkerPair:
    """An opening marker and the text the template renders to close it.

    ``close`` is the closing marker with the template's surrounding whitespace;
    the marker itself is ``close.strip()``.  ``open`` may be the fixed prefix of
    a longer tag, e.g. DSML ``<｜DSML｜invoke`` before ``name="..."``.
    """

    open: str
    close: str

    @property
    def close_marker(self) -> str:
        return self.close.strip()


@dataclass(frozen=True)
class TruncationGrammar:
    """The structures a family's assistant turn opens and closes.

    ``think.close`` is the template's text for closing reasoning when nothing
    follows it.  ``tool_blocks`` are the tool-call structures the template
    renders as single tokens or fixed text, outermost first.  The inner ones
    count only inside the outermost block, where the template renders them;
    elsewhere the same text is ordinary content.  ``end_of_turn`` is ``None``
    when the next message's role token ends the turn.
    """

    think: MarkerPair
    tool_blocks: tuple[MarkerPair, ...]
    end_of_turn: str | None
    _roles: dict[str, tuple[MarkerPair, bool]] = field(init=False, repr=False, compare=False)
    _pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        roles: dict[str, tuple[MarkerPair, bool]] = {}
        for pair in (self.think, *self.tool_blocks):
            roles[pair.open] = (pair, True)
            roles[pair.close_marker] = (pair, False)
        # Longest first, so a marker never matches as a prefix of a longer one.
        pattern = re.compile("|".join(re.escape(marker) for marker in sorted(roles, key=len, reverse=True)))
        object.__setattr__(self, "_roles", roles)
        object.__setattr__(self, "_pattern", pattern)

    @property
    def reasoning_answer_delimiter(self) -> str:
        return self.think.close_marker

    def close_truncated_turn(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[int]:
        """Close each open structure, innermost first, then end the turn.

        Partial tool-call arguments are left as sampled.
        """
        prompt_text, completion_text = _decode_turn(tokenizer, token_ids, completion_start, message_boundary_ids)
        stack, _ = self.scan(completion_text, self.opened_by(prompt_text))
        # A closing marker written as several tokens can be cut partway; a single-token one cannot.
        marker_is_text = bool(stack) and len(tokenizer.encode(stack[-1].close_marker, add_special_tokens=False)) > 1
        closer_text = self.closing_text(prompt_text + completion_text, stack, complete_innermost=marker_is_text)
        return tokenizer.encode(closer_text, add_special_tokens=False)

    def cut_kind(
        self, token_ids: list[int], completion_start: int, *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> str:
        prompt_text, completion_text = _decode_turn(tokenizer, token_ids, completion_start, message_boundary_ids)
        stack, _ = self.scan(completion_text, self.opened_by(prompt_text))
        if any(pair in self.tool_blocks for pair in stack):
            return "tool_call"
        return "reasoning" if stack else "answer"

    def check_turn_structure(
        self, token_ids: list[int], span: tuple[int, int], *, tokenizer: Any, message_boundary_ids: set[int]
    ) -> list[str]:
        """A well-formed turn closes every structure it opens, innermost first, and ends with the
        end-of-turn token at ``turn_end - 1``; a family without one ends the turn at ``turn_end``.
        """
        start, end = span
        prompt_text, turn_text = _decode_turn(tokenizer, token_ids[:end], start, message_boundary_ids)
        stack, problems = self.scan(turn_text, self.opened_by(prompt_text))
        if stack:
            problems.append("unclosed: " + ", ".join(repr(pair.close_marker) for pair in reversed(stack)))
        if self.end_of_turn is not None:
            end_of_turn_ids = tokenizer.encode(self.end_of_turn, add_special_tokens=False)
            if token_ids[start:end][-1:] != end_of_turn_ids:
                problems.append(f"turn does not end with {self.end_of_turn!r}")
        return problems

    def opened_by(self, prompt_text: str) -> tuple[MarkerPair, ...]:
        """Structures the current message's prompt text leaves open: reasoning or nothing."""
        if prompt_text.rfind(self.think.open) > prompt_text.rfind(self.think.close_marker):
            return (self.think,)
        return ()

    def scan(self, text: str, opened: tuple[MarkerPair, ...]) -> tuple[list[MarkerPair], list[str]]:
        """Track the structures *text* opens and closes after *opened*.

        Returns the structures still open, innermost last, and the nesting
        problems found.  A closing marker that is not innermost closes its
        structure and every structure opened inside it; one with no open
        structure is ignored.
        """
        stack = list(opened)
        problems: list[str] = []
        for match in self._pattern.finditer(text):
            pair, is_open = self._roles[match.group()]
            if pair in self.tool_blocks[1:] and self.tool_blocks[0] not in stack:
                continue
            if is_open:
                stack.append(pair)
            elif stack and stack[-1] == pair:
                stack.pop()
            else:
                problems.append(
                    f"{pair.close_marker!r} at offset {match.start()} does not close the innermost structure"
                )
                if pair in stack:
                    del stack[len(stack) - 1 - stack[::-1].index(pair) :]
        return stack, problems

    def closing_text(self, text: str, stack: list[MarkerPair], *, complete_innermost: bool = False) -> str:
        """Text that closes every structure in *stack*, innermost first, then ends the turn after *text*.

        With *complete_innermost*, a closing marker of the innermost structure
        that *text* already starts is completed instead of written again.  A
        start that could as well begin an opening marker, such as a lone ``<``,
        is left as sampled text.
        """
        openers = [pair.open for pair in (self.think, *self.tool_blocks)]
        closer = ""
        for depth, pair in enumerate(reversed(stack)):
            started = _started_marker(text, pair.close_marker, openers) if complete_innermost and depth == 0 else 0
            if started:
                closer += pair.close[pair.close.index(pair.close_marker) + started :]
            else:
                closer += _skip_whitespace_overlap(text + closer, pair.close)
        return closer + (self.end_of_turn or "")


def _decode_turn(
    tokenizer: Any, token_ids: list[int], completion_start: int, message_boundary_ids: set[int]
) -> tuple[str, str]:
    """Decode the turn's prompt text after its last message boundary, and its sampled text.

    The prompt text holds the generation prompt, which may open reasoning.
    """
    boundary = next(
        (i for i in range(completion_start - 1, -1, -1) if token_ids[i] in message_boundary_ids),
        0,
    )
    return (
        tokenizer.decode(token_ids[boundary:completion_start]),
        tokenizer.decode(token_ids[completion_start:]),
    )


def _started_marker(text: str, marker: str, openers: list[str]) -> int:
    """Length of the longest proper prefix of *marker* that *text* ends with and that starts no opener."""
    size = next((size for size in range(len(marker) - 1, 0, -1) if text.endswith(marker[:size])), 0)
    return 0 if any(opener.startswith(marker[:size]) for opener in openers) else size


def _skip_whitespace_overlap(text: str, fragment: str) -> str:
    """Drop the part of *fragment*'s leading whitespace that *text* already ends with."""
    lead = fragment[: len(fragment) - len(fragment.lstrip())]
    for size in range(len(lead), 0, -1):
        if text.endswith(lead[:size]):
            return fragment[size:]
    return fragment
