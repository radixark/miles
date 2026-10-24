"""Closing grammar for assistant turns cut by ``finish_reason='length'``.

A length-truncated turn can stop inside reasoning or a tool-call block and
never emits its end-of-turn token.  Continuing it as is would place the next
message inside an unclosed structure.  A family's ``TruncationGrammar`` lists
the structures its template opens and closes, so the closer can append the
missing closing text after the sampled tokens without changing any of them.

Structure markers are matched in decoded text, the way SGLang's reasoning and
tool-call parsers see them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


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
    follows it.  ``tool_blocks`` are the tool-call structures, outermost first;
    only markers that are single tokens or fixed template text are tracked, not
    inner text such as ``<function=``.  ``end_of_turn`` is ``None`` when the next
    message's role token ends the turn.
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

    def closing_text(self, text: str, stack: list[MarkerPair]) -> str:
        """Text that closes every structure in *stack*, innermost first, then ends the turn after *text*."""
        closer = ""
        for pair in reversed(stack):
            closer += _skip_whitespace_overlap(text + closer, pair.close)
        return closer + (self.end_of_turn or "")


def _skip_whitespace_overlap(text: str, fragment: str) -> str:
    """Drop the part of *fragment*'s leading whitespace that *text* already ends with."""
    lead = fragment[: len(fragment) - len(fragment.lstrip())]
    for size in range(len(lead), 0, -1):
        if text.endswith(lead[:size]):
            return fragment[size:]
    return fragment
