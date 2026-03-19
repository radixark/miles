"""TokenSeqComparator: segment token IDs by special-token boundaries and compare sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A contiguous slice of token IDs — either a special token or plain tokens."""

    token_ids: list[int] = field(default_factory=list)
    is_special: bool = False


class MismatchType(Enum):
    # Segment count or structure (special/content pattern) differs between
    # expected and actual.  When this happens, segments can't be aligned so
    # no per-segment text/JSON comparison is possible.
    SPECIAL_TOKEN_COUNT = "special_token_count"

    # A special-token segment has the same position in both sequences but
    # contains a different token ID.
    SPECIAL_TOKEN_TYPE = "special_token_type"

    JSON = "json"
    TEXT = "text"


@dataclass
class Mismatch:
    """A single difference found between two token sequences."""

    type: MismatchType
    segment_index: int
    expected_text: str
    actual_text: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "segment_index": self.segment_index,
            "expected_text": self.expected_text,
            "actual_text": self.actual_text,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class TokenSeqComparator:
    """Segment token sequences by special tokens and compare them.

    Parameters
    ----------
    tokenizer
        A HuggingFace tokenizer (``PreTrainedTokenizerBase``).
    special_token_ids : set[int] | None
        Token IDs treated as segment boundaries.
        Default: ``tokenizer.all_special_ids``.
    tool_start_ids : set[int] | None
        Special-token IDs that mark the **start** of a tool-call or
        tool-response block (e.g. ``<tool_call>``, ``<tool_response>``).
        Content segments immediately following one of these are compared
        via JSON parsing instead of raw text.
    tool_end_ids : set[int] | None
        Special-token IDs that mark the **end** of a tool-call or
        tool-response block (e.g. ``</tool_call>``, ``</tool_response>``).
    """

    def __init__(
        self,
        tokenizer,
        special_token_ids: set[int] | None = None,
        tool_start_ids: set[int] | None = None,
        tool_end_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self._special_ids: set[int] = self._collect_special_ids(tokenizer)
        if special_token_ids is not None:
            self._special_ids |= set(special_token_ids)
        self._tool_start_ids: set[int] = set(tool_start_ids) if tool_start_ids else set()
        self._tool_end_ids: set[int] = set(tool_end_ids) if tool_end_ids else set()

    @staticmethod
    def _collect_special_ids(tokenizer) -> set[int]:
        """Collect all added-token IDs from the tokenizer.

        All tokens in ``added_tokens_decoder`` are used as segment
        boundaries, regardless of their ``special`` flag.  Tokens like
        ``<arg_value>``, ``</arg_value>`` etc. may have ``special=False``
        but still function as structural delimiters for comparison.
        """
        ids = set(tokenizer.all_special_ids)
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if decoder:
            ids |= set(decoder.keys())
        return ids

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_by_special_tokens(self, token_ids: list[int]) -> list[Segment]:
        """Split *token_ids* into segments delimited by special tokens.

        Consecutive non-special tokens are grouped into a single plain segment.
        Each special token becomes its own segment (``is_special=True``).
        Empty input returns an empty list.
        """
        if not token_ids:
            return []

        segments: list[Segment] = []
        current: list[int] = []

        for tid in token_ids:
            if tid in self._special_ids:
                if current:
                    segments.append(Segment(token_ids=current, is_special=False))
                    current = []
                segments.append(Segment(token_ids=[tid], is_special=True))
            else:
                current.append(tid)

        if current:
            segments.append(Segment(token_ids=current, is_special=False))

        return segments

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_sequences(
        self,
        expected_ids: list[int],
        actual_ids: list[int],
        trim_trailing_ids: set[int] | None = None,
    ) -> list[Mismatch]:
        """Compare two token-ID sequences and return a list of mismatches.

        Algorithm:
        1. Segment both sequences by special tokens.
        2. Verify that the segment structures match (same count, same
           ``is_special`` pattern).  Any difference is reported as
           ``MismatchType.SPECIAL_TOKEN_COUNT``.
        3. For each pair of aligned segments:
           - **Special-token segments**: token IDs must be identical —
             ``MismatchType.SPECIAL_TOKEN_TYPE`` on mismatch.
           - **Tool-call / tool-response content segments** (preceded by a
             tool-start token and, when end tokens exist, followed by a
             tool-end token): decode both, parse as JSON, compare parsed
             structures — ``MismatchType.JSON`` on mismatch.
           - **Other content segments**: decode both, compare text —
             ``MismatchType.TEXT`` on mismatch.
        """
        if trim_trailing_ids:
            while expected_ids and expected_ids[-1] in trim_trailing_ids:
                expected_ids = expected_ids[:-1]
            while actual_ids and actual_ids[-1] in trim_trailing_ids:
                actual_ids = actual_ids[:-1]

        expected_segs = self.segment_by_special_tokens(expected_ids)
        actual_segs = self.segment_by_special_tokens(actual_ids)

        # --- structural check ---
        # When segment count or structure pattern differs, segments can't be
        # aligned — no per-segment comparison is possible.
        if len(expected_segs) != len(actual_segs):
            return [
                Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_COUNT,
                    segment_index=-1,
                    expected_text=self._describe_structure(expected_segs),
                    actual_text=self._describe_structure(actual_segs),
                    detail=(f"segment count differs: " f"expected {len(expected_segs)}, got {len(actual_segs)}"),
                )
            ]

        pattern_expected = [s.is_special for s in expected_segs]
        pattern_actual = [s.is_special for s in actual_segs]
        if pattern_expected != pattern_actual:
            return [
                Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_COUNT,
                    segment_index=-1,
                    expected_text=self._describe_structure(expected_segs),
                    actual_text=self._describe_structure(actual_segs),
                    detail="segment structure (special/content pattern) differs",
                )
            ]

        # --- per-segment comparison ---
        mismatches: list[Mismatch] = []
        for idx, (exp, act) in enumerate(zip(expected_segs, actual_segs, strict=False)):
            if exp.is_special:
                if exp.token_ids != act.token_ids:
                    mismatches.append(
                        Mismatch(
                            type=MismatchType.SPECIAL_TOKEN_TYPE,
                            segment_index=idx,
                            expected_text=self._decode(exp.token_ids),
                            actual_text=self._decode(act.token_ids),
                        )
                    )
            else:
                # Chat-template assembly and natural model output may differ
                # in leading/trailing whitespace (\n, spaces) at segment
                # boundaries — strip so these are not reported as mismatches.
                exp_text = self._decode(exp.token_ids).strip()
                act_text = self._decode(act.token_ids).strip()
                if exp_text == act_text:
                    continue

                if self._is_tool_segment(expected_segs, idx):
                    mismatch = self._compare_tool_content(idx, exp_text, act_text)
                    if mismatch is not None:
                        mismatches.append(mismatch)
                else:
                    mismatches.append(
                        Mismatch(
                            type=MismatchType.TEXT,
                            segment_index=idx,
                            expected_text=exp_text,
                            actual_text=act_text,
                        )
                    )

        return mismatches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _is_tool_segment(self, segments: list[Segment], content_idx: int) -> bool:
        """Determine if the content segment at *content_idx* is a tool block.

        A content segment is a tool block when:
        - The immediately preceding segment is a special token whose ID is
          in ``tool_start_ids``.
        - If ``tool_end_ids`` is configured, the immediately following
          segment must be a special token whose ID is in ``tool_end_ids``.
        """
        if not self._tool_start_ids:
            return False

        # Check preceding segment is a tool-start token.
        if content_idx == 0:
            return False
        prev = segments[content_idx - 1]
        if not prev.is_special or prev.token_ids[0] not in self._tool_start_ids:
            return False

        # If end tokens are configured, the following segment must be a tool-end token.
        if self._tool_end_ids:
            if content_idx + 1 >= len(segments):
                return False
            nxt = segments[content_idx + 1]
            if not nxt.is_special or nxt.token_ids[0] not in self._tool_end_ids:
                return False

        return True

    _TOOL_CALL_RELEVANT_KEYS = ("name", "arguments")

    def _compare_tool_content(self, idx: int, exp_text: str, act_text: str) -> Mismatch | None:
        """Compare two tool-segment texts via JSON parsing.

        Only the keys listed in ``_TOOL_CALL_RELEVANT_KEYS`` (``name``,
        ``arguments``) are compared.  Extra fields (e.g. ``id``, ``type``)
        are ignored.  Returns ``None`` when all relevant fields match.
        """
        try:
            exp_parsed = json.loads(exp_text.strip())
            act_parsed = json.loads(act_text.strip())
        except json.JSONDecodeError:
            return Mismatch(
                type=MismatchType.TEXT,
                segment_index=idx,
                expected_text=exp_text,
                actual_text=act_text,
                detail="tool segment but JSON parsing failed, falling back to text compare",
            )

        for key in self._TOOL_CALL_RELEVANT_KEYS:
            if exp_parsed.get(key) != act_parsed.get(key):
                return Mismatch(
                    type=MismatchType.JSON,
                    segment_index=idx,
                    expected_text=exp_text,
                    actual_text=act_text,
                    detail=f"tool call field '{key}' differs",
                )

        return None

    def _describe_structure(self, segments: list[Segment]) -> str:
        """Human-readable summary of segment structure for error messages."""
        parts = []
        for s in segments:
            if s.is_special:
                parts.append(f"[{self._decode(s.token_ids)}]")
            else:
                parts.append(f"({len(s.token_ids)} tokens)")
        return " ".join(parts)
