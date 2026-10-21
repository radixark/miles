"""A sample's merge plan over the replay arrays SGLang stores in the output store.

With the output store, SGLang writes a training response's replay arrays (routed experts, indexer
top-k, sampling mask) as one object and returns its ref; the rollout never reads an object. Every
event on a sample edits its ``StorePlan`` only: which rows of which objects each field takes, and
where forced rows go. The trainer reads the objects once and has the plan materialize the training
columns of the inline path.

Three things, and nothing else:

- ``Seg``: rows ``[start, end)`` of one object, or ``end - start`` forced rows;
- ``Segments``: one field's segments plus how events edit them and how its training columns are
  built from the objects. One subclass per stored field;
- ``StorePlan``: the sample's fields and the objects it owns. It holds no rule of its own: every
  operation is the same call on each field.

What a row of an object means is SGLang's definition: a forward field (routed experts, indexer
top-k) stores a row per forward position, and may start at ``routed_experts_start_len``; the
sampling mask stores a row per output token. A sample's fields count rows like the inline path:
forward fields ``len(tokens) - 1``, the mask ``response_length``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
import torch

from miles.utils.retry_utils import NonRetryableError

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

# Stored array names, as radix_transfer_plugins' TokenOutputStash names them.
MASK_LENGTHS = "output_token_sampling_mask_lengths"
MASK_TOKEN_IDS = "output_token_sampling_mask_token_ids"
# Support log-prob mode only: a log-prob per support entry.
MASK_LOGPROBS = "output_token_sampling_logprobs"


class StorePlanError(NonRetryableError, ValueError):
    """A plan edit, a response's ref or an object breaks the plan's contract.

    Retrying cannot fix it. A ``ValueError``, so the session server answers it with 422.
    """


class Seg(NamedTuple):
    """Rows ``[start, end)`` of one object; ``ref`` None means forced rows."""

    ref: StoreRef | None
    start: int
    end: int


# ============================== one field's segments ==============================


@dataclass(frozen=True)
class Segments:
    """One field's rows as segments of objects. Subclasses define the field."""

    segs: tuple[Seg, ...] = ()

    name: ClassVar[str]
    # Stored array names; the first one's leading dimension counts the object's rows.
    stored: ClassVar[tuple[str, ...]]
    # The request field that makes SGLang return this field.
    request_key: ClassVar[str]
    # The launch flag that makes the field required on every training sample.
    required_by: ClassVar[str | None] = None

    def num_rows(self) -> int:
        return sum(seg.end - seg.start for seg in self.segs)

    def prefix(self, n: int) -> Segments:
        """The first ``n`` rows."""
        kept, left = [], n
        for seg in self.segs:
            if left <= 0:
                break
            take = min(left, seg.end - seg.start)
            kept.append(seg._replace(end=seg.start + take))
            left -= take
        return type(self)(tuple(kept))

    @classmethod
    def from_ref(
        cls,
        ref: StoreRef | None,
        specs: Mapping[str, Any],
        *,
        expected: int,
        allow_extra_row: bool,
        no_output_abort: bool,
    ) -> Segments:
        """The field's rows of one response: all ``expected`` rows of its object, checked against the ref.

        ``specs`` maps stored array names to the ref's field specs. ``allow_extra_row`` accepts an
        object with one row more than expected (a stop-edge row) and leaves it out.
        """
        spec = specs.get(cls.stored[0])
        if spec is None:
            if no_output_abort and expected == 0:
                return cls()
            raise StorePlanError(f"the request asked for {cls.name}, but the response's output store ref has none")
        rows = spec.shape[0]
        if rows != expected and not (allow_extra_row and rows == expected + 1):
            raise StorePlanError(f"stored {cls.stored[0]} has {rows} rows; expected {expected}")
        return cls((Seg(ref, 0, expected),) if expected else ())

    def append(self, new: Segments, *, seq_start: int) -> Segments:
        """A later response of the same sample, whose forward started at sequence position ``seq_start``."""
        raise NotImplementedError

    def insert(self, n: int) -> Segments:
        """``n`` tokens Miles inserted into the response, such as a tool observation."""
        raise NotImplementedError

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        raise NotImplementedError

    def materialize(
        self, objects: Mapping[str, Mapping[str, np.ndarray]], *, response_tokens: np.ndarray, args: Any
    ) -> dict[str, Any]:
        """The field's training columns of this sample.

        ``objects`` maps a store key to the object's stored arrays; ``response_tokens`` is the
        sample's response.
        """
        raise NotImplementedError


class ForwardSegments(Segments):
    """A forward output: a row per sequence position but the last. Never forced."""

    column: ClassVar[str]

    def append(self, new: Segments, *, seq_start: int) -> Segments:
        # The forward covered the sequence from seq_start on; seq_start 0 replaces every row.
        return type(self)(self.prefix(seq_start).segs + new.segs)

    def insert(self, n: int) -> Segments:
        # The next forward covers the inserted tokens.
        return self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return num_tokens - 1

    def materialize(
        self, objects: Mapping[str, Mapping[str, np.ndarray]], *, response_tokens: np.ndarray, args: Any
    ) -> dict[str, Any]:
        (name,) = self.stored
        rows = np.concatenate([objects[seg.ref.key][name][seg.start : seg.end] for seg in self.segs])
        self.check(rows, args)
        return {self.column: rows}

    def check(self, rows: np.ndarray, args: Any) -> None:
        """The field's check of this sample's rows."""


class RoutedExpertsSegments(ForwardSegments):
    """Routed experts, ``[rows, layers, topk]``."""

    name = "routed_experts"
    stored = ("routed_experts",)
    request_key = "return_routed_experts"
    required_by = "use_rollout_routing_replay"
    column = "rollout_routed_experts"

    def check(self, rows: np.ndarray, args: Any) -> None:
        # As on the inline path: all zeros means the engine did not capture routed experts.
        if rows.size and not rows.any():
            raise StorePlanError(
                "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
                "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
            )


class IndexerTopkSegments(ForwardSegments):
    """Indexer top-k, ``[rows, streams, topk]``: a stream per DSA indexer layer, by position."""

    name = "indexer_topk"
    stored = ("indexer_topk",)
    request_key = "return_indexer_topk"
    required_by = "use_rollout_indexer_replay"
    column = "rollout_indexer_topk"

    def check(self, rows: np.ndarray, args: Any) -> None:
        expected = getattr(args, "rollout_indexer_topk_num_streams", None)
        if expected is not None and rows.shape[1] != expected:
            raise StorePlanError(
                f"stored indexer_topk has {rows.shape[1]} streams but the model has {expected} indexer layers; "
                "replaying it would map streams to the wrong layers."
            )


class SamplingMaskSegments(Segments):
    """The sampling mask: a row per response token, each row the CSR support the token was sampled from.

    A forced row (a token Miles inserted) is filled with its own token from ``response_tokens``,
    only so the mask stays valid: an empty support makes the loss NaN even under loss mask 0.
    """

    name = "sampling_mask"
    stored = (MASK_LENGTHS, MASK_TOKEN_IDS, MASK_LOGPROBS)
    request_key = "return_sampling_mask"

    def append(self, new: Segments, *, seq_start: int) -> Segments:
        return type(self)(self.segs + new.segs)

    def insert(self, n: int) -> Segments:
        return type(self)(self.segs + (Seg(None, 0, n),)) if n else self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return response_length

    def materialize(
        self, objects: Mapping[str, Mapping[str, np.ndarray]], *, response_tokens: np.ndarray, args: Any
    ) -> dict[str, Any]:
        support_mode = args.rollout_sampling_logprobs_mode == "support"
        response_tokens = np.asarray(response_tokens)
        # Typed empties, so a mask with no rows (a no-output abort) concatenates to empty columns.
        lengths = [np.empty(0, dtype=np.int32)]
        ids = [np.empty(0, dtype=np.int32)]
        logprobs = [np.empty(0, dtype=np.float32)]
        forced = [np.empty(0, dtype=bool)]
        position = 0
        for seg in self.segs:
            n = seg.end - seg.start
            if seg.ref is None:
                lengths.append(np.ones(n, dtype=np.int32))
                ids.append(response_tokens[position : position + n].astype(np.int32))
                logprobs.append(np.zeros(n, dtype=np.float32))
            else:
                arrays = objects[seg.ref.key]
                entry_starts = np.concatenate(([0], np.cumsum(arrays[MASK_LENGTHS])))
                a, b = int(entry_starts[seg.start]), int(entry_starts[seg.end])
                lengths.append(arrays[MASK_LENGTHS][seg.start : seg.end])
                ids.append(arrays[MASK_TOKEN_IDS][a:b])
                if support_mode:
                    logprobs.append(arrays[MASK_LOGPROBS][a:b])
            forced.append(np.full(n, seg.ref is None))
            position += n

        row_lengths = np.concatenate(lengths).astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(row_lengths)))
        ids = np.concatenate(ids).astype(np.int32, copy=False)
        forced = np.concatenate(forced)
        check_support(ids, offsets, response_tokens, exactly_once=support_mode)
        columns = {
            "rollout_sampling_mask_ids": torch.from_numpy(ids),
            "rollout_sampling_mask_offsets": torch.from_numpy(offsets),
        }
        if support_mode and (k := args.rollout_top_logprobs_num):
            topk_ids, topk_logprobs = support_topk(ids, offsets, np.concatenate(logprobs), forced, k)
            columns["rollout_topk_token_ids"] = topk_ids
            columns["rollout_topk_log_probs"] = topk_logprobs
        return columns


# The order is the wire's field index (StorePlan.to_table).
FIELDS: tuple[type[Segments], ...] = (RoutedExpertsSegments, IndexerTopkSegments, SamplingMaskSegments)
_FIELD_INDEX = {cls.name: i for i, cls in enumerate(FIELDS)}


def check_support(ids: np.ndarray, offsets: np.ndarray, sampled_tokens: np.ndarray, *, exactly_once: bool) -> None:
    """Every support is non-empty and holds its sampled token (exactly once in support mode)."""
    row_lengths = np.diff(offsets)
    if (empty := row_lengths == 0).any():
        raise StorePlanError(f"sampling support must contain at least one token (row {int(np.argmax(empty))})")
    entry_row = np.repeat(np.arange(len(row_lengths)), row_lengths)
    hits = np.bincount(entry_row[ids == sampled_tokens[entry_row]], minlength=len(row_lengths))
    bad = hits != 1 if exactly_once else hits == 0
    if bad.any():
        row = int(np.argmax(bad))
        what = "must occur exactly once in" if exactly_once else "is absent from"
        raise StorePlanError(f"sampled token {int(sampled_tokens[row])} {what} its sampling support (row {row})")


def support_topk(
    ids: np.ndarray, offsets: np.ndarray, logprobs: np.ndarray, forced: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Support-mode top-k candidates ``[rows, k]``: each row's support and its log-probs.

    Unused slots and forced rows hold -1 / -inf, as the inline path pads them.
    """
    row_lengths = np.diff(offsets)
    if (too_long := (row_lengths > k) & ~forced).any():
        raise StorePlanError(
            f"Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num (row {int(np.argmax(too_long))})"
        )
    out_ids = np.full((len(row_lengths), k), -1, dtype=np.int32)
    out_logprobs = np.full((len(row_lengths), k), -np.inf, dtype=np.float32)
    entry_row = np.repeat(np.arange(len(row_lengths)), row_lengths)
    entry_col = np.arange(len(entry_row)) - offsets[entry_row]
    keep = ~forced[entry_row]
    out_ids[entry_row[keep], entry_col[keep]] = ids[keep]
    out_logprobs[entry_row[keep], entry_col[keep]] = logprobs[keep]
    return out_ids, out_logprobs


# ============================== one sample's plan ==============================


@dataclass(frozen=True)
class StorePlan:
    """One sample's plan: its requested fields, and every object it owns.

    ``owned_refs`` holds every ref the sample received, used by its segments or not: edits never
    drop one and merges take the union, so the objects a sample brought in can be deleted with it.
    """

    fields: dict[str, Segments]
    owned_refs: tuple[StoreRef, ...]

    def field(self, name: str) -> Segments | None:
        return self.fields.get(name)

    def with_field(self, name: str, segments: Segments) -> StorePlan:
        return replace(self, fields={**self.fields, name: segments})

    @classmethod
    def from_response(
        cls,
        ref: StoreRef | None,
        *,
        requested: Collection[str],
        num_tokens: int,
        response_length: int,
        allow_extra_seq_row: bool = False,
        no_output_abort: bool = False,
    ) -> StorePlan:
        """The plan of one response; reads only the ref.

        ``num_tokens`` is the length of the sequence the response's forward covered (prompt plus
        output), ``response_length`` its output length. A requested field must be in the ref,
        except an empty sampling mask of a no-output abort. ``allow_extra_seq_row`` drops the
        stop-edge row the legacy rollout's forward fields may carry.
        """
        specs = {spec.name: spec for spec in ref.fields} if ref is not None else {}
        fields = {
            field.name: field.from_ref(
                ref,
                specs,
                expected=field.expected_rows(num_tokens=num_tokens, response_length=response_length),
                allow_extra_row=allow_extra_seq_row and issubclass(field, ForwardSegments),
                no_output_abort=no_output_abort,
            )
            for field in FIELDS
            if field.name in requested
        }
        return cls(fields=fields, owned_refs=() if ref is None else (ref,))

    def append_response(self, new: StorePlan, *, seq_start: int = 0) -> StorePlan:
        """Add a later response of the same sample."""
        return StorePlan(
            fields=_zip_fields(self, new, lambda a, b: a.append(b, seq_start=seq_start)),
            owned_refs=_union_refs(self.owned_refs, new.owned_refs),
        )

    def adopt_refs(self, refs: Iterable[StoreRef]) -> StorePlan:
        """Own more objects without using their rows, so they are deleted with the sample."""
        return replace(self, owned_refs=_union_refs(self.owned_refs, refs))

    def insert_tokens(self, n: int) -> StorePlan:
        """Tokens Miles inserted into the response, such as a tool observation."""
        return replace(self, fields={name: segments.insert(n) for name, segments in self.fields.items()})

    def merge_turn(self, later: StorePlan, *, observation_tokens: int) -> StorePlan:
        """A session's next turn after ``observation_tokens`` forced tokens."""
        return self.insert_tokens(observation_tokens).append_response(later)

    def strip_tail(self, n: int) -> StorePlan:
        """Drop the rows of the last ``n`` tokens."""
        fields = {name: segments.prefix(segments.num_rows() - n) for name, segments in self.fields.items()}
        return replace(self, fields=fields)

    def validate(self, *, num_tokens: int, response_length: int) -> None:
        for name, segments in self.fields.items():
            expected = segments.expected_rows(num_tokens=num_tokens, response_length=response_length)
            if segments.num_rows() != expected:
                raise StorePlanError(f"{name} has {segments.num_rows()} rows; expected {expected}")

    def materialize(
        self, objects: Mapping[str, Mapping[str, np.ndarray]], *, response_tokens: np.ndarray, args: Any
    ) -> dict[str, Any]:
        """Every field's training columns of this sample."""
        columns = {}
        for segments in self.fields.values():
            columns.update(segments.materialize(objects, response_tokens=response_tokens, args=args))
        return columns

    def to_table(self) -> tuple[np.ndarray, list[StoreRef]]:
        """int32 ``[n, 4]`` rows of ``(field, obj, start, end)``; ``obj`` indexes ``owned_refs``.

        Forced rows have ``obj = -1``; a field present with no rows gets one forced ``(0, 0)`` row.
        """
        index = {ref.key: i for i, ref in enumerate(self.owned_refs)}
        table = []
        for name, segments in self.fields.items():
            f = _FIELD_INDEX[name]
            if not segments.segs:
                table.append((f, -1, 0, 0))
            for seg in segments.segs:
                table.append((f, -1 if seg.ref is None else index[seg.ref.key], seg.start, seg.end))
        return np.asarray(table, dtype=np.int32).reshape(-1, 4), list(self.owned_refs)

    @classmethod
    def from_table(cls, table: np.ndarray, refs: Sequence[StoreRef]) -> StorePlan:
        segs: dict[int, list[Seg]] = {}
        for f, obj, start, end in table.tolist():
            rows = segs.setdefault(f, [])
            if obj >= 0 or start != end:
                rows.append(Seg(None if obj < 0 else refs[obj], start, end))
        fields = {FIELDS[f].name: FIELDS[f](tuple(rows)) for f, rows in sorted(segs.items())}
        return cls(fields=fields, owned_refs=tuple(refs))


def _zip_fields(a: StorePlan, b: StorePlan, combine) -> dict[str, Segments]:
    if a.fields.keys() != b.fields.keys():
        raise StorePlanError(f"cannot combine a sample with fields {sorted(a.fields)} and one with {sorted(b.fields)}")
    return {name: combine(segments, b.fields[name]) for name, segments in a.fields.items()}


def _union_refs(a: Iterable[StoreRef], b: Iterable[StoreRef]) -> tuple[StoreRef, ...]:
    return tuple({ref.key: ref for ref in (*a, *b)}.values())


def combine_plans(a: StorePlan | None, b: StorePlan | None, combine) -> StorePlan | None:
    """Apply a two-plan edit under the strict None rule: both or neither sample has a plan."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise StorePlanError("cannot combine a sample with output store data and one without")
    return combine(a, b)
