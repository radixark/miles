"""A sample's merge plan over the replay arrays SGLang stores in the output store.

With the output store, SGLang writes a training response's replay arrays (routed experts, indexer
top-k, sampling mask) as one object and returns its ref; the rollout never reads an object. Every
event on a sample edits its ``StorePlan`` only: which rows of which objects each field takes, and
where forced rows go. The trainer reads the objects once and has the plan materialize the training
columns of the inline path.

Three things, and nothing else:

- ``Seg``: rows ``[start, end)`` of one object, or ``end - start`` rows no object of this
  response holds (``ref`` None; each field says what they are);
- ``Segments``: one field's segments plus how events edit them and how its training columns are
  built from the objects. One leaf subclass per field; the two forward fields share ``ForwardSegments``;
- ``StorePlan``: the sample's fields and the objects it owns. It holds no rule of its own: every
  operation is the same call on each field.

What a row of an object means is SGLang's definition: a forward field (routed experts, indexer
top-k) stores a row per forward position; routed experts may start at ``routed_experts_start_len``
(an incremental forward); the sampling mask stores a row per output token. A sample's fields count
rows like the inline path: forward fields ``len(tokens) - 1``, the mask ``response_length``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
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
# Support log-prob mode only: a log-prob per support entry. In selected mode SGLang's meta_info
# uses the same name for the sampled tokens' log-probs, one per token, which are never stored.
MASK_SUPPORT_LOGPROBS = "output_token_sampling_logprobs"

# The request field that has SGLang store the response's replay arrays, and the meta_info key of the ref.
OUTPUT_STORE_REQUEST_KEY = "return_outputs_via_store"
OUTPUT_STORE_REF_KEY = "output_store_ref"


class StorePlanError(NonRetryableError, ValueError):
    """A plan edit, a response's ref or inline sampled log-probs, or an object breaks the output store contract.

    Retrying cannot fix it. The session server answers it with ``OutputStoreContractError`` (424), which
    the session client raises as this error again.
    """


class Seg(NamedTuple):
    """Rows ``[start, end)`` of one object; with ``ref`` None, ``end - start`` rows no object holds."""

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
        request: Mapping[str, Any],
        num_tokens: int,
        response_length: int,
        no_output_abort: bool,
    ) -> Segments:
        """The field's rows of one response: every row of its object; ``specs`` is the ref's specs by stored name."""
        rows = cls.expected_rows(num_tokens=num_tokens, response_length=response_length)
        return cls(cls._whole_object(ref, specs, rows, no_output_abort=no_output_abort))

    @classmethod
    def _whole_object(
        cls, ref: StoreRef | None, specs: Mapping[str, Any], rows: int, *, no_output_abort: bool
    ) -> tuple[Seg, ...]:
        """All rows of the response's object, which must hold exactly ``rows``."""
        spec = specs.get(cls.stored[0])
        if spec is None:
            # Aborted in SGLang's waiting queue: no ref; aborted after prefill: no mask.
            if no_output_abort and (ref is None or rows == 0):
                return ()
            raise StorePlanError(f"the request asked for {cls.name}, but the response's output store ref has none")
        if spec.shape[0] != rows:
            raise StorePlanError(f"stored {cls.stored[0]} has {spec.shape[0]} rows; expected {rows}")
        return (Seg(ref, 0, rows),) if rows else ()

    def append(self, new: Segments) -> Segments:
        """A later response of the same sample."""
        raise NotImplementedError

    def insert(self, n: int) -> Segments:
        """``n`` tokens Miles inserted into the response, such as a tool observation."""
        raise NotImplementedError

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        raise NotImplementedError

    def materialize(
        self,
        objects: Mapping[str, Mapping[str, np.ndarray]],
        *,
        response_tokens: np.ndarray,
        support_topk: int,
        indexer_streams: int | None,
    ) -> dict[str, Any]:
        """The field's training columns of this sample.

        ``objects`` maps a store key to the object's stored arrays; ``response_tokens`` is the
        sample's response. ``support_topk`` is k when the requests asked SGLang for support
        log-probs (support mode with top-k collection), else 0; ``indexer_streams`` is the model's
        indexer layer count, or None.
        """
        raise NotImplementedError

    def needed(self, *, support_topk: int) -> tuple[str, ...]:
        """The stored arrays ``materialize`` reads from this field's objects."""
        return self.stored


class ForwardSegments(Segments):
    """A forward output: a row per sequence position but the last.

    The request's start position (``start_key``) alone decides how a response's rows combine
    with the earlier ones; ``from_ref`` reads it once:

    - 0: the object holds every row, and ``append`` replaces all earlier rows;
    - ``start`` > 0: the object holds rows from ``start`` on, preceded by a ref-less segment of
      ``start`` rows, which ``append`` fills with the earlier plan's first ``start`` rows.
    """

    column: ClassVar[str]
    # The request field holding the field's start position; None means always 0.
    start_key: ClassVar[str | None] = None

    @classmethod
    def from_ref(
        cls,
        ref: StoreRef | None,
        specs: Mapping[str, Any],
        *,
        request: Mapping[str, Any],
        num_tokens: int,
        response_length: int,
        no_output_abort: bool,
    ) -> Segments:
        start = (request.get(cls.start_key) or 0) if cls.start_key else 0
        rows = cls.expected_rows(num_tokens=num_tokens, response_length=response_length)
        covered = (Seg(None, 0, start),) if start else ()
        return cls(covered + cls._whole_object(ref, specs, rows - start, no_output_abort=no_output_abort))

    def append(self, new: Segments) -> Segments:
        first = new.segs[0] if new.segs else None
        covered = first.end - first.start if first is not None and first.ref is None else 0
        if self.num_rows() < covered:
            # Otherwise the later rows would sit at the wrong sequence positions.
            raise StorePlanError(
                f"{self.name} continues from row {covered}, but only {self.num_rows()} rows precede it"
            )
        return type(self)(self.prefix(covered).segs + new.segs[1 if covered else 0 :])

    def insert(self, n: int) -> Segments:
        # The next forward covers the inserted tokens.
        return self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return num_tokens - 1

    def materialize(
        self,
        objects: Mapping[str, Mapping[str, np.ndarray]],
        *,
        response_tokens: np.ndarray,
        support_topk: int,
        indexer_streams: int | None,
    ) -> dict[str, Any]:
        (name,) = self.stored
        for key in dict.fromkeys(seg.ref.key for seg in self.segs):
            self.check(objects[key][name], indexer_streams=indexer_streams)
        return {self.column: np.concatenate([objects[seg.ref.key][name][seg.start : seg.end] for seg in self.segs])}

    def check(self, rows: np.ndarray, *, indexer_streams: int | None) -> None:
        """The field's check of every row one response stored."""


class RoutedExpertsSegments(ForwardSegments):
    """Routed experts, ``[rows, layers, topk]``."""

    name = "routed_experts"
    stored = ("routed_experts",)
    request_key = "return_routed_experts"
    required_by = "use_rollout_routing_replay"
    column = "rollout_routed_experts"
    start_key = "routed_experts_start_len"

    def check(self, rows: np.ndarray, *, indexer_streams: int | None) -> None:
        # As on the inline path, per response: all zeros means the engine did not capture routed experts.
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
    # No start_key: SGLang stores every indexer row of each forward, so a response replaces them all.

    def check(self, rows: np.ndarray, *, indexer_streams: int | None) -> None:
        if indexer_streams is not None and rows.shape[1] != indexer_streams:
            raise StorePlanError(
                f"stored indexer_topk has {rows.shape[1]} streams but the model has {indexer_streams} indexer layers; "
                "replaying it would map streams to the wrong layers."
            )


class SamplingMaskSegments(Segments):
    """The sampling mask: a row per response token, each row the CSR support the token was sampled from.

    A forced row (a token Miles inserted) is filled with its own token from ``response_tokens``,
    only so the mask stays valid: an empty support makes the loss NaN even under loss mask 0.
    """

    name = "sampling_mask"
    stored = (MASK_LENGTHS, MASK_TOKEN_IDS, MASK_SUPPORT_LOGPROBS)
    request_key = "return_sampling_mask"

    def append(self, new: Segments) -> Segments:
        return type(self)(self.segs + new.segs)

    def insert(self, n: int) -> Segments:
        return type(self)(self.segs + (Seg(None, 0, n),)) if n else self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return response_length

    def needed(self, *, support_topk: int) -> tuple[str, ...]:
        # Support log-probs exist only for support requests.
        return self.stored if support_topk else (MASK_LENGTHS, MASK_TOKEN_IDS)

    def materialize(
        self,
        objects: Mapping[str, Mapping[str, np.ndarray]],
        *,
        response_tokens: np.ndarray,
        support_topk: int,
        indexer_streams: int | None,
    ) -> dict[str, Any]:
        support_mode = support_topk > 0
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
                    logprobs.append(arrays[MASK_SUPPORT_LOGPROBS][a:b])
            forced.append(np.full(n, seg.ref is None))
            position += n

        row_lengths = np.concatenate(lengths).astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(row_lengths)))
        ids = np.concatenate(ids).astype(np.int32, copy=False)
        forced = np.concatenate(forced)
        _check_support(ids=ids, offsets=offsets, sampled_tokens=response_tokens, exactly_once=support_mode)
        columns = {
            "rollout_sampling_mask_ids": torch.from_numpy(ids),
            "rollout_sampling_mask_offsets": torch.from_numpy(offsets),
        }
        if support_mode:
            topk_ids, topk_logprobs = _support_topk(
                ids=ids, offsets=offsets, logprobs=np.concatenate(logprobs), forced=forced, k=support_topk
            )
            columns["rollout_topk_token_ids"] = topk_ids
            columns["rollout_topk_log_probs"] = topk_logprobs
        return columns


FIELDS: tuple[type[Segments], ...] = (RoutedExpertsSegments, IndexerTopkSegments, SamplingMaskSegments)


def _check_support(ids: np.ndarray, offsets: np.ndarray, sampled_tokens: np.ndarray, *, exactly_once: bool) -> None:
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


def _support_topk(
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

    @classmethod
    def from_response(
        cls,
        ref: StoreRef | None,
        *,
        request: Mapping[str, Any],
        num_tokens: int,
        response_length: int,
        no_output_abort: bool = False,
    ) -> StorePlan:
        """The plan of one response; reads only the ref.

        ``request`` is what Miles sent: the fields it asked for and where an incremental forward
        starts. ``num_tokens`` is the length of the sequence the response's forward covered (prompt
        plus output), ``response_length`` its output length. A requested field must be in the ref,
        unless the response is a no-output abort with no ref, or with no rows of that field to store;
        that field then gets none of the response's rows.
        """
        specs = {spec.name: spec for spec in ref.fields} if ref is not None else {}
        fields = {
            field.name: field.from_ref(
                ref,
                specs,
                request=request,
                num_tokens=num_tokens,
                response_length=response_length,
                no_output_abort=no_output_abort,
            )
            for field in FIELDS
            if request.get(field.request_key)
        }
        return cls(fields=fields, owned_refs=() if ref is None else (ref,))

    def append_response(self, new: StorePlan) -> StorePlan:
        """Add a later response of the same sample."""
        return StorePlan(
            fields=_zip_fields(self, new, lambda a, b: a.append(b)),
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
        self,
        objects: Mapping[str, Mapping[str, np.ndarray]],
        *,
        response_tokens: np.ndarray,
        support_topk: int,
        indexer_streams: int | None,
    ) -> dict[str, Any]:
        """Every field's training columns of this sample; see ``Segments.materialize``."""
        columns = {}
        for segments in self.fields.values():
            columns.update(
                segments.materialize(
                    objects,
                    response_tokens=response_tokens,
                    support_topk=support_topk,
                    indexer_streams=indexer_streams,
                )
            )
        return columns

    def to_dict(self) -> dict[str, Any]:
        """The plan in JSON types, for the session wire and the train dict.

        Each field maps to its segments as ``[store key, start, end]``, ``[None, start, end]``
        for rows no object holds; a field with no rows is an empty list. ``refs`` holds every owned ref.
        """
        return {
            "fields": {
                name: [[None if seg.ref is None else seg.ref.key, seg.start, seg.end] for seg in segments.segs]
                for name, segments in self.fields.items()
            },
            # StoreRef.to_builtins keeps tuples, which the train dict's msgpack codec rejects.
            "refs": [json.loads(json.dumps(ref.to_builtins())) for ref in self.owned_refs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StorePlan:
        refs = tuple(store_ref_from_dict(ref) for ref in value["refs"])
        by_key = {ref.key: ref for ref in refs}
        by_name = {field.name: field for field in FIELDS}
        return cls(
            fields={
                name: by_name[name](
                    tuple(Seg(None if key is None else by_key[key], start, end) for key, start, end in segs)
                )
                for name, segs in value["fields"].items()
            },
            owned_refs=refs,
        )


def store_reads(plans: Iterable[StorePlan], *, support_topk: int) -> dict[str, tuple[StoreRef, list[str]]]:
    """What ``materialize`` reads for these plans, by store key: the ref, and the arrays it needs from the object.

    Each object and array once; ``support_topk`` as in ``Segments.materialize``. An array the ref
    lacks is still asked for, so the read, not ``materialize``, fails on it. Objects a plan owns
    without using, such as replaced forward rows, are not read.
    """
    reads: dict[str, tuple[StoreRef, list[str]]] = {}
    for plan in plans:
        for segments in plan.fields.values():
            needed = segments.needed(support_topk=support_topk)
            for seg in segments.segs:
                if seg.ref is None:
                    continue
                names = reads.setdefault(seg.ref.key, (seg.ref, []))[1]
                names += [name for name in needed if name not in names]
    return reads


def store_ref_from_dict(value: Mapping[str, Any] | None) -> StoreRef | None:
    """A ref as SGLang returns it or ``StorePlan.to_dict`` writes it; None (no ref) stays None."""
    if value is None:
        return None
    # Not a Miles dependency: only output store runs install it.
    from radix_transfer_plugins import StoreRef

    return StoreRef.from_builtins(value)


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
