"""TokenSeqComparator: segment token IDs by special-token boundaries and compare sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Segment:
    """A contiguous run of token IDs — either a special token or a content segment."""

    token_ids: list[int] = field(default_factory=list)
    is_special: bool = False


class MismatchType(Enum):
    # Segment count or structure (special/content pattern) differs between
    # expected and actual.  When this happens, segments can't be aligned so
    # no per-segment comparison is possible.
    SPECIAL_TOKEN_COUNT = "special_token_count"

    # A special-token segment has the same position in both sequences but
    # contains a different token ID.
    SPECIAL_TOKEN_TYPE = "special_token_type"

    # Non-assistant content (user, system, tool, etc.) differs.  This indicates
    # a bug in the TITO algorithm — these regions should match exactly.
    NON_ASSISTANT_TEXT = "non_assistant_text"

    # Assistant content differs.  Expected and non-severe: assistant tokens
    # are inherited directly from the pretokenized prefix across turns,
    # so they may not match the chat template's canonical tokenization.
    ASSISTANT_TEXT = "assistant_text"


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


class TokenSeqComparator:
    """Segment token sequences by special tokens and compare them.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    special_token_ids : set[int] | None
        Extra IDs treated as segment boundaries (merged with auto-detected ones).
    assistant_start_str : str
        Decoded text prefix identifying assistant content segments, e.g.
        ``"<|im_start|>assistant"`` (Qwen3) or ``"<|assistant|>"`` (GLM).
        Used to classify content mismatches as assistant vs non-assistant.
    trim_trailing_ids : set[int] | None
        Token IDs to strip from both sequence tails before comparison
        (see :func:`_trim_trailing`).  Stored as a default; callers of
        :meth:`compare_sequences` may supply additional IDs that are
        **unioned** with this set.
    """

    def __init__(
        self,
        tokenizer,
        assistant_start_str: str,
        special_token_ids: set[int] | None = None,
        trim_trailing_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self._special_ids = self._collect_special_ids(tokenizer)
        if special_token_ids is not None:
            self._special_ids |= set(special_token_ids)
        self._assistant_start_str = assistant_start_str
        self._trim_trailing_ids: set[int] | None = set(trim_trailing_ids) if trim_trailing_ids else None

    @staticmethod
    def _collect_special_ids(tokenizer) -> set[int]:
        """Collect token IDs with ``special=True`` from the tokenizer.

        Special tokens are structural markers added by the chat template to
        delimit messages, roles, and control flow — for example
        ``<|im_start|>``, ``<|im_end|>``, ``<|endoftext|>``, ``<s>``,
        ``</s>``, and ``<|assistant|>``.

        Tokens that encode *content* produced by a role are **not** special,
        even if they look "special" to a human.  For instance, ``<think>`` and
        ``</think>`` in reasoning models are regular content tokens generated
        by the assistant — the tokenizer does not flag them as
        ``special=True``, so they are not collected here.
        """
        ids = set(tokenizer.all_special_ids)
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if decoder:
            ids |= {k for k, v in decoder.items() if v.special}
        return ids

    def segment_by_special_tokens(self, token_ids: list[int]) -> list[Segment]:
        """Split *token_ids* into segments at special-token boundaries.

        Each special token becomes its own single-ID segment with
        ``is_special=True``.  Consecutive non-special tokens are grouped
        into content segments.  Example for Qwen3::

            [<|im_start|>, "assistant", "\\n", "Hi", <|im_end|>, "\\n"]
            → [special(<|im_start|>), content("assistant\\nHi"), special(<|im_end|>), content("\\n")]
        """
        if not token_ids:
            return []

        segments: list[Segment] = []
        current: list[int] = []
        for tid in token_ids:
            if tid in self._special_ids:
                if current:
                    segments.append(Segment(token_ids=current))
                    current = []
                segments.append(Segment(token_ids=[tid], is_special=True))
            else:
                current.append(tid)
        if current:
            segments.append(Segment(token_ids=current))
        return segments

    def compare_sequences(
        self,
        expected_ids: list[int],
        actual_ids: list[int],
        trim_trailing_ids: set[int] | None = None,
    ) -> list[Mismatch]:
        """Compare two token-ID sequences and return mismatches.

        Parameters
        ----------
        trim_trailing_ids : set[int] | None
            Additional token IDs to strip from both sequence tails before
            comparison.  **Unioned** with the IDs passed at construction time.
        """
        trim = self._trim_trailing_ids or set()
        if trim_trailing_ids:
            trim = trim | trim_trailing_ids
        if trim:
            expected_ids = _trim_trailing(expected_ids, trim)
            actual_ids = _trim_trailing(actual_ids, trim)

        exp_segs = self.segment_by_special_tokens(expected_ids)
        act_segs = self.segment_by_special_tokens(actual_ids)

        structural = self._check_segment_structure(exp_segs, act_segs)
        if structural:
            return [structural]

        mismatches: list[Mismatch] = []
        for idx, (exp, act) in enumerate(zip(exp_segs, act_segs, strict=True)):
            is_assistant_content = self._is_assistant_content(exp_segs, idx) and self._is_assistant_content(
                act_segs, idx
            )
            m = self._compare_single_segment(idx, exp, act, is_assistant_content=is_assistant_content)
            if m is not None:
                mismatches.append(m)
        return mismatches

    def _check_segment_structure(
        self,
        exp_segs: list[Segment],
        act_segs: list[Segment],
    ) -> Mismatch | None:
        """Pre-check that expected and actual segment lists have the same count
        and the same special/content pattern before per-segment comparison."""
        if len(exp_segs) != len(act_segs):
            detail = f"segment count differs: expected {len(exp_segs)}, got {len(act_segs)}"
        elif [s.is_special for s in exp_segs] != [s.is_special for s in act_segs]:
            detail = "segment structure (special/content pattern) differs"
        else:
            return None
        return Mismatch(
            type=MismatchType.SPECIAL_TOKEN_COUNT,
            segment_index=-1,
            expected_text=self._describe_structure(exp_segs),
            actual_text=self._describe_structure(act_segs),
            detail=detail,
        )

    def _compare_single_segment(
        self,
        idx: int,
        exp: Segment,
        act: Segment,
        *,
        is_assistant_content: bool,
    ) -> Mismatch | None:
        """Compare a single aligned segment pair and return a mismatch if they differ.

        Special segments are compared by token ID.  Content segments are decoded
        and compared as stripped text — leading/trailing whitespace (``\\n``,
        spaces) is ignored because chat templates may insert boundary newlines
        that differ from the TITO prefix.  This whitespace-only difference does
        not cause meaningful misalignment with the chat template, so we strip
        to avoid noisy false positives.
        """
        if exp.is_special:
            if exp.token_ids != act.token_ids:
                return Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_TYPE,
                    segment_index=idx,
                    expected_text=self._decode(exp.token_ids),
                    actual_text=self._decode(act.token_ids),
                )
            return None

        exp_text = self._decode(exp.token_ids).strip()
        act_text = self._decode(act.token_ids).strip()
        if exp_text == act_text:
            return None

        return Mismatch(
            type=MismatchType.ASSISTANT_TEXT if is_assistant_content else MismatchType.NON_ASSISTANT_TEXT,
            segment_index=idx,
            expected_text=exp_text,
            actual_text=act_text,
        )

    def _is_assistant_content(self, segments: list[Segment], idx: int) -> bool:
        """Check if the content segment at *idx* belongs to an assistant message.

        Decodes the preceding special-token segment and the first few tokens of
        the current segment *separately*, then concatenates the decoded strings.
        If the result starts with ``assistant_start_str`` (e.g.
        ``"<|im_start|>assistant"``), this segment is classified as assistant
        content — mismatches there are expected and non-severe.

        """
        if self._assistant_start_str is None:
            return False
        if segments[idx].is_special:
            return False
        if idx == 0:
            return False
        prev = segments[idx - 1]
        if not prev.is_special:
            return False
        special_text = self._decode(prev.token_ids)
        # Decode enough prefix tokens to capture the role label (e.g. "assistant\n").
        content_prefix = self._decode(segments[idx].token_ids[:20])
        return (special_text + content_prefix).startswith(self._assistant_start_str)

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _describe_structure(self, segments: list[Segment]) -> str:
        return " ".join(
            f"[{self._decode(s.token_ids)}]" if s.is_special else f"({len(s.token_ids)} tokens)" for s in segments
        )


def _trim_trailing(ids: list[int], to_remove: set[int]) -> list[int]:
    """Strip trailing token IDs that belong to *to_remove*.

    The model's generated output typically ends with a stop token (e.g.
    ``<|observation|>`` for GLM, ``<|im_end|>`` for Qwen) that won't appear
    at the same position in the template-rendered expected sequence.  Stripping
    these trailing tokens from both sides before comparison avoids false
    structural mismatches.
    """
    end = len(ids)
    while end > 0 and ids[end - 1] in to_remove:
        end -= 1
    return ids[:end]
