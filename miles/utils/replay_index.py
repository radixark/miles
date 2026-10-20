"""Per-token replay data of one sample: routed experts (R3), indexer top-k and the sampling mask.

A ``ReplayIndex`` holds one ``ReplaySegments`` per field. A field is an ordered tuple of
row segments; each segment is a row range of one source:

- inline: arrays that came in the response's ``meta_info``;
- store: an object of the output store, named by the ``StoreRef`` in ``meta_info["output_store_ref"]``;
- forced (sampling mask only): rows of tokens Miles inserted itself, such as tool observations.

Rollout-side edits (truncation, multi-turn merge, additional R3) only rearrange segments and never
read data. ``miles.utils.replay_assembly`` materializes the arrays, in the rollout executor or, with
``--enable-delay-token-replay-assembly``, on the trainer. Only the trainer reads store objects
(``miles.backends.training_utils.data.replay``), and inline and store data then take the same path.

Rows: R3 and indexer top-k count prompt and response tokens minus the last token
(``len(tokens) - 1``); the sampling mask counts response tokens.

The sampling mask holds each row's support and, in support log-prob mode, a log-prob per support
entry. The sampled tokens' log-probs, one per token, are not replay data: the rollout reads them from
each response as ``rollout_log_probs`` (``sampled_logprobs_from_meta_info``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64

from miles.utils.retry_utils import NonRetryableError

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

ROUTED_EXPERTS = "routed_experts"
INDEXER_TOPK = "indexer_topk"
SAMPLING_MASK = "sampling_mask"

# Stored field names, as radix_transfer_plugins' TokenOutputStash names them.
MASK_LENGTHS = "output_token_sampling_mask_lengths"
MASK_TOKEN_IDS = "output_token_sampling_mask_token_ids"
# Support log-prob mode only: a log-prob per support entry.
MASK_LOGPROBS = "output_token_sampling_logprobs"

OUTPUT_STORE_REF_KEY = "output_store_ref"
# Support log-prob mode: each sampled token's entry of its support row, picked by SGLang.
SELECTED_LOGPROBS_KEY = "output_token_sampling_selected_logprobs"
# The only backend of phase 1: every engine Miles launches writes to the run's own Mooncake.
SUPPORTED_STORE_BACKEND = "mooncake"

# Request fields that ask SGLang for the replay outputs a ReplayIndex holds.
REPLAY_REQUEST_KEYS = ("return_routed_experts", "return_indexer_topk", "return_sampling_mask")
VIA_STORE_REQUEST_KEY = "return_outputs_via_store"


def set_outputs_via_store(request: dict[str, Any], *, use_output_store: bool, evaluation: bool) -> None:
    """Ask SGLang to return a training request's replay outputs through the output store.

    Miles owns the field: a caller's own value is replaced, and evaluation requests never set it.
    """
    if use_output_store and not evaluation and any(request.get(key) for key in REPLAY_REQUEST_KEYS):
        request[VIA_STORE_REQUEST_KEY] = True
    else:
        request.pop(VIA_STORE_REQUEST_KEY, None)


class ReplayDataError(NonRetryableError, ValueError):
    """Replay data is missing, inconsistent, or has the wrong shape.

    Retrying cannot fix it: the data a retry would read is the same. A ``ValueError``, so the session
    server answers a malformed response with 422 as before.
    """


@dataclass(frozen=True)
class _FieldLayout:
    name: str
    # Stored field names this field may have; the first one's leading dimension counts its rows.
    # A segment has the ones its source carries (the mask has MASK_LOGPROBS in support mode only).
    stored: tuple[str, ...]


_LAYOUTS: tuple[_FieldLayout, ...] = (
    _FieldLayout(ROUTED_EXPERTS, (ROUTED_EXPERTS,)),
    _FieldLayout(INDEXER_TOPK, (INDEXER_TOPK,)),
    _FieldLayout(SAMPLING_MASK, (MASK_LENGTHS, MASK_TOKEN_IDS, MASK_LOGPROBS)),
)
_LAYOUT_BY_NAME = {layout.name: layout for layout in _LAYOUTS}

INLINE = "inline"
STORE = "store"
FORCED = "forced"


@dataclass(frozen=True, eq=False)
class ReplaySegment:
    """Rows ``[start, stop)`` of one source; at most one of ``inline`` and ``ref`` is set."""

    start: int
    stop: int
    # Stored field name -> the whole inline array the rows index into.
    inline: Mapping[str, np.ndarray] | None = field(default=None, repr=False)
    ref: StoreRef | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReplaySegment):
            return NotImplemented
        if (self.start, self.stop, self.ref) != (other.start, other.stop, other.ref):
            return False
        if self.inline is None or other.inline is None:
            return self.inline is other.inline
        return self.inline.keys() == other.inline.keys() and all(
            np.array_equal(self.inline[name], other.inline[name]) for name in self.inline
        )

    __hash__ = None

    @property
    def kind(self) -> str:
        if self.inline is not None:
            return INLINE
        return STORE if self.ref is not None else FORCED

    @property
    def num_rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ReplaySegments:
    """One field of one sample: segments in row order."""

    name: str
    segments: tuple[ReplaySegment, ...] = ()

    @property
    def num_rows(self) -> int:
        return sum(segment.num_rows for segment in self.segments)

    @property
    def source_kind(self) -> str | None:
        """``inline`` or ``store``, or None when the field has only forced rows or none."""
        kinds = {segment.kind for segment in self.segments} - {FORCED}
        return next(iter(kinds), None)

    def prefix(self, num_rows: int) -> ReplaySegments:
        if not 0 <= num_rows <= self.num_rows:
            raise ReplayDataError(f"{self.name} prefix of {num_rows} rows is outside [0, {self.num_rows}]")
        segments, left = [], num_rows
        for segment in self.segments:
            if left == 0 and segments:
                break
            take = min(left, segment.num_rows)
            # An empty prefix keeps one empty segment, so the rows keep their trailing shape.
            segments.append(replace(segment, stop=segment.start + take))
            left -= take
        return ReplaySegments(self.name, tuple(segments))

    def concat(self, *others: ReplaySegments) -> ReplaySegments:
        segments = self.segments + tuple(chain.from_iterable(other.segments for other in others))
        _check_single_source(self.name, segments)
        if self.name != SAMPLING_MASK:
            _check_row_shapes(self.name, segments)
        return ReplaySegments(self.name, segments)

    def refs(self) -> list[StoreRef]:
        return [segment.ref for segment in self.segments if segment.ref is not None]

    def materialize_rows(self, fetched: Mapping[tuple[str, str], np.ndarray]) -> np.ndarray:
        """R3 or indexer top-k rows: ``[num_rows, num_layers, topk]``.

        ``fetched`` maps (store key, stored field name) to the array read for it.
        """
        layout = _LAYOUT_BY_NAME[self.name]
        (stored,) = layout.stored
        parts = [
            _source_arrays(segment, layout, fetched)[stored][segment.start : segment.stop] for segment in self.segments
        ]
        if not parts:
            raise ReplayDataError(f"{self.name} has no rows to materialize")
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def mask_pieces(self, fetched: Mapping[tuple[str, str], np.ndarray]) -> list[MaskPiece]:
        """The sampling mask as pieces in response order."""
        pieces = []
        for segment in self.segments:
            if segment.kind == FORCED:
                pieces.append(MaskPiece(num_rows=segment.num_rows))
                continue
            arrays = _slice_mask(_source_arrays(segment, _LAYOUT_BY_NAME[SAMPLING_MASK], fetched), segment)
            pieces.append(
                MaskPiece(
                    num_rows=segment.num_rows,
                    lengths=arrays[MASK_LENGTHS],
                    token_ids=arrays[MASK_TOKEN_IDS],
                    logprobs=arrays.get(MASK_LOGPROBS),
                )
            )
        return pieces


@dataclass(frozen=True, eq=False)
class MaskPiece:
    """Sampling-mask rows of one segment; forced rows carry no arrays."""

    num_rows: int
    lengths: np.ndarray | None = field(default=None, repr=False)
    token_ids: np.ndarray | None = field(default=None, repr=False)
    # Per support entry; None in selected log-prob mode.
    logprobs: np.ndarray | None = field(default=None, repr=False)

    @property
    def forced(self) -> bool:
        return self.lengths is None


@dataclass(frozen=True)
class ReplayIndex:
    """The replay fields of one sample; a field is None when the sample has none."""

    routed_experts: ReplaySegments | None = None
    indexer_topk: ReplaySegments | None = None
    sampling_mask: ReplaySegments | None = None

    def fields(self) -> Iterator[ReplaySegments]:
        for layout in _LAYOUTS:
            if (segments := getattr(self, layout.name)) is not None:
                yield segments

    def is_empty(self) -> bool:
        return next(self.fields(), None) is None

    def store_refs(self) -> list[StoreRef]:
        return [ref for segments in self.fields() for ref in segments.refs()]

    def store_reads(self) -> list[tuple[StoreRef, tuple[str, ...]]]:
        """Each store object with the stored fields this sample reads from it."""
        return [
            (segment.ref, _segment_stored_names(_LAYOUT_BY_NAME[segments.name], segment))
            for segments in self.fields()
            for segment in segments.segments
            if segment.ref is not None
        ]

    # ----------------------------- rollout-side edits -----------------------------

    def strip_tail(self, num_tokens: int) -> ReplayIndex:
        """Drop the last ``num_tokens`` tokens' rows of every field."""
        return ReplayIndex(
            **{segments.name: segments.prefix(_rows_after_strip(segments, num_tokens)) for segments in self.fields()}
        )

    def append_response(self, new: ReplayIndex, *, response_length: int) -> ReplayIndex:
        """Add a later generation call on the same token sequence.

        R3 and indexer top-k cover the whole sequence in every call, so ``new``'s replace them;
        sampling-mask rows of ``new`` follow the ``response_length`` rows already recorded.
        """
        sampling_mask = self.sampling_mask
        if new.sampling_mask is not None:
            sampling_mask = _append_mask_rows(self.sampling_mask, new.sampling_mask, response_length=response_length)
        return ReplayIndex(
            routed_experts=new.routed_experts, indexer_topk=new.indexer_topk, sampling_mask=sampling_mask
        )

    def append_forced_tokens(self, num_tokens: int, *, response_length: int) -> ReplayIndex:
        """Add ``num_tokens`` tokens Miles inserted (tool observations) after ``response_length`` rows."""
        if self.sampling_mask is None:
            return self
        forced = ReplaySegments(SAMPLING_MASK, (ReplaySegment(start=0, stop=num_tokens),))
        return replace(
            self, sampling_mask=_append_mask_rows(self.sampling_mask, forced, response_length=response_length)
        )

    def merge_turn(self, later: ReplayIndex, *, observation_tokens: int) -> ReplayIndex:
        """Merge ``later``, the next turn of the same trajectory, after ``observation_tokens`` inserted tokens."""
        first, second = self.sampling_mask, later.sampling_mask
        if (first is None) != (second is None):
            raise ReplayDataError("cannot merge samples unless both turns carry a complete rollout sampling mask")
        sampling_mask = None
        if first is not None:
            forced = ReplaySegments(SAMPLING_MASK, (ReplaySegment(start=0, stop=observation_tokens),))
            sampling_mask = first.concat(forced, second)
        return ReplayIndex(
            routed_experts=later.routed_experts, indexer_topk=later.indexer_topk, sampling_mask=sampling_mask
        )

    @classmethod
    def from_arrays(
        cls, *, routed_experts: np.ndarray | None = None, indexer_topk: np.ndarray | None = None
    ) -> ReplayIndex:
        """An index of inline ``[rows, layers, topk]`` int32 arrays, for producers that decode them themselves."""
        fields = {}
        for name, array in ((ROUTED_EXPERTS, routed_experts), (INDEXER_TOPK, indexer_topk)):
            if array is not None:
                if array.ndim != 3 or array.dtype != np.int32:
                    raise ReplayDataError(f"{name} must be an int32 [rows, layers, topk] array")
                fields[name] = ReplaySegments(name, (ReplaySegment(start=0, stop=len(array), inline={name: array}),))
        return cls(**fields)

    # ------------------------------- parse a response -------------------------------

    @classmethod
    def from_meta_info(
        cls,
        meta_info: Mapping[str, Any],
        *,
        via_store: bool,
        routed_experts_rows: int | None,
        num_layers: int | None,
        indexer_topk_rows: int | None,
        indexer_num_streams: int | None,
        return_sampling_mask: bool,
        support_logprobs: bool,
        num_output_tokens: int,
        aborted: bool,
        allow_extra_routed_experts_row: bool = False,
    ) -> ReplayIndex:
        """The only parser of a response's replay fields, inline or through the output store.

        ``via_store`` is whether the request set ``return_outputs_via_store``. A field whose expected
        rows are None is not parsed. Structural checks that need no data (rows, layers, the
        sampling-mask length) run here, so a malformed response fails on arrival; checks that read
        data run at assembly (``miles.utils.replay_assembly``).
        ``allow_extra_routed_experts_row`` accepts R3 with one row past ``routed_experts_rows``
        (the legacy rollout's stop edge) and drops it.
        """
        ref = _parse_store_ref(meta_info, via_store=via_store)
        stored = {spec.name: spec for spec in ref.fields} if ref is not None else {}
        routed_experts = indexer_topk = None
        if routed_experts_rows is not None:
            routed_experts = _parse_rows_field(
                meta_info,
                ROUTED_EXPERTS,
                ref=ref,
                stored=stored,
                rows=routed_experts_rows,
                num_layers=num_layers,
                allow_extra_row=allow_extra_routed_experts_row,
            )
        if indexer_topk_rows is not None:
            indexer_topk = _parse_rows_field(
                meta_info,
                INDEXER_TOPK,
                ref=ref,
                stored=stored,
                rows=indexer_topk_rows,
                num_layers=_indexer_num_layers(meta_info, stored, expected=indexer_num_streams),
                allow_extra_row=False,
            )
        sampling_mask = None
        if return_sampling_mask:
            sampling_mask = _parse_sampling_mask(
                meta_info,
                ref=ref,
                stored=stored,
                support_logprobs=support_logprobs,
                num_output_tokens=num_output_tokens,
                aborted=aborted,
            )
        return cls(routed_experts=routed_experts, indexer_topk=indexer_topk, sampling_mask=sampling_mask)

    # --------------------------------- wire format ---------------------------------

    def to_wire(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
        """Split into a builtins descriptor and the inline arrays.

        Inline rows of each stored field are concatenated in segment order; the descriptor names
        each segment's kind and row count, and store segments carry their ref and row range.
        """
        descriptor: dict[str, list[dict[str, Any]]] = {}
        arrays: dict[str, np.ndarray] = {}
        for segments in self.fields():
            layout = _LAYOUT_BY_NAME[segments.name]
            entries, inline_parts = [], []
            for segment in segments.segments:
                if segment.kind == INLINE:
                    entries.append({INLINE: segment.num_rows})
                    inline_parts.append(_slice_inline(layout, segment))
                elif segment.kind == STORE:
                    entries.append({STORE: _ref_to_plain(segment.ref), "start": segment.start, "stop": segment.stop})
                else:
                    entries.append({FORCED: segment.num_rows})
            descriptor[segments.name] = entries
            for stored in inline_parts[0] if inline_parts else ():
                parts = [part[stored] for part in inline_parts]
                arrays[stored] = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return descriptor, arrays

    @classmethod
    def from_wire(
        cls, descriptor: Mapping[str, Sequence[Mapping[str, Any]]], arrays: Mapping[str, np.ndarray]
    ) -> ReplayIndex:
        fields = {}
        for name, entries in descriptor.items():
            layout = _LAYOUT_BY_NAME[name]
            inline = {stored: arrays[stored] for stored in layout.stored if stored in arrays} or None
            segments, cursor = [], 0
            for entry in entries:
                if INLINE in entry:
                    if inline is None:
                        raise ReplayDataError(f"{name} descriptor names inline rows but the payload has no arrays")
                    rows = int(entry[INLINE])
                    segments.append(ReplaySegment(start=cursor, stop=cursor + rows, inline=inline))
                    cursor += rows
                elif STORE in entry:
                    ref = _store_ref_from_builtins(entry[STORE])
                    segments.append(ReplaySegment(start=int(entry["start"]), stop=int(entry["stop"]), ref=ref))
                else:
                    segments.append(ReplaySegment(start=0, stop=int(entry[FORCED])))
            if inline is not None and cursor != _num_source_rows(layout, inline):
                raise ReplayDataError(
                    f"{name} descriptor uses {cursor} inline rows of {_num_source_rows(layout, inline)}"
                )
            fields[name] = ReplaySegments(name, tuple(segments))
        return cls(**fields)


def sampled_logprobs_from_meta_info(
    meta_info: Mapping[str, Any], *, output_token_ids: Sequence[int], support_logprobs: bool, aborted: bool
) -> list[float]:
    """The sampled tokens' log-probs under the sampling distribution, one per output token.

    With a sampling mask they replace ``output_token_logprobs`` as ``rollout_log_probs``. SGLang
    returns ``output_token_sampling_logprobs``: one per output token in selected mode; in support
    mode a row per support, from which each sampled token's entry is picked.
    """
    # A response whose sampling mask is in the output store has no support rows inline: in support mode
    # its sampled log-probs are SGLang's own pick from them, output_token_sampling_selected_logprobs.
    via_store = OUTPUT_STORE_REF_KEY in meta_info
    key = SELECTED_LOGPROBS_KEY if support_logprobs and via_store else "output_token_sampling_logprobs"
    supports = meta_info.get("output_token_sampling_mask")
    log_probs = meta_info.get(key)
    if log_probs is None or (supports is None and not via_store):
        if _is_no_output_abort(meta_info, aborted=aborted, num_output_tokens=len(output_token_ids)):
            return []
        raise ValueError(
            f"SGLang response is missing the sampling mask or {key}; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    if len(log_probs) != len(output_token_ids):
        raise ValueError(f"sampling log-prob length {len(log_probs)} != output token length {len(output_token_ids)}")

    if support_logprobs and not via_store:
        selected = []
        for token_id, support, row in zip(output_token_ids, supports, log_probs, strict=True):
            if not isinstance(row, (list, tuple)) or len(row) != len(support):
                raise ValueError("SGLang support log-probs must align with the sampling support IDs")
            matches = [i for i, candidate in enumerate(support) if candidate == token_id]
            if len(matches) != 1:
                raise ValueError("Sampled token must occur exactly once in its sampling support")
            selected.append(float(row[matches[0]]))
    else:
        if any(isinstance(row, (list, tuple)) for row in log_probs):
            raise ValueError("SGLang selected sampling log-probs must be scalar")
        selected = [float(value) for value in log_probs]
    return selected


def response_has_routed_experts(meta_info: Mapping[str, Any]) -> bool:
    """Whether a response carries R3 rows, inline or in its output store object; reads no data."""
    if ROUTED_EXPERTS in meta_info:
        return True
    ref = meta_info.get(OUTPUT_STORE_REF_KEY)
    return ref is not None and any(spec["name"] == ROUTED_EXPERTS for spec in ref["fields"])


# ============================== train dict columns ==============================

# One builtins descriptor per sample (``ReplayIndex.to_wire``); inline arrays go in typed columns.
REPLAY_INDEX_COLUMN = "replay_index"
INLINE_TRAIN_COLUMNS: dict[str, tuple[str, np.dtype]] = {
    ROUTED_EXPERTS: ("rollout_routed_experts", np.dtype(np.int32)),
    INDEXER_TOPK: ("rollout_indexer_topk", np.dtype(np.int32)),
    MASK_LENGTHS: ("rollout_sampling_mask_lengths", np.dtype(np.int32)),
    MASK_TOKEN_IDS: ("rollout_sampling_mask_token_ids", np.dtype(np.int32)),
    MASK_LOGPROBS: ("rollout_sampling_mask_logprobs", np.dtype(np.float32)),
}


def compute_train_columns(indexes: Sequence[ReplayIndex]) -> dict[str, list[Any]]:
    """Train dict columns of a batch's replay indexes; empty when no sample has replay data.

    A sample without inline rows of a stored field gets an empty array in that column, so a step
    may mix inline and store samples.
    """
    wires = [index.to_wire() for index in indexes]
    if not any(descriptor for descriptor, _ in wires):
        return {}
    columns: dict[str, list[Any]] = {REPLAY_INDEX_COLUMN: [descriptor for descriptor, _ in wires]}
    for stored, (column, dtype) in INLINE_TRAIN_COLUMNS.items():
        if not any(stored in arrays for _, arrays in wires):
            continue
        values = []
        for _, arrays in wires:
            array = arrays.get(stored, np.empty(0, dtype=dtype))
            if array.dtype != dtype:
                raise ReplayDataError(f"inline {stored} must have dtype {dtype}, got {array.dtype}")
            values.append(array)
        columns[column] = values
    return columns


def pop_train_columns(rollout_data: dict[str, Any]) -> list[ReplayIndex] | None:
    """Rebuild the replay indexes of a train dict shard, removing their columns."""
    descriptors = rollout_data.pop(REPLAY_INDEX_COLUMN, None)
    if descriptors is None:
        return None
    inline = {
        stored: rollout_data.pop(column)
        for stored, (column, _) in INLINE_TRAIN_COLUMNS.items()
        if column in rollout_data
    }
    return [
        ReplayIndex.from_wire(descriptor, {stored: np.asarray(values[i]) for stored, values in inline.items()})
        for i, descriptor in enumerate(descriptors)
    ]


def stored_field_names(name: str) -> tuple[str, ...]:
    return _LAYOUT_BY_NAME[name].stored


# ================================ helpers ================================


def _check_single_source(name: str, segments: Sequence[ReplaySegment]) -> None:
    """A field of one sample comes from one source; Miles' forced rows do not count (MA9)."""
    sources = {
        INLINE if segment.kind == INLINE else f"{STORE}:{segment.ref.backend}"
        for segment in segments
        if segment.kind != FORCED
    }
    if len(sources) > 1:
        raise ReplayDataError(
            f"{name} of one sample mixes replay data sources {sorted(sources)}; "
            "every turn of a sample must return its replay data the same way"
        )


def _check_row_shapes(name: str, segments: Sequence[ReplaySegment]) -> None:
    """R3 and indexer rows of one sample share their [layers, topk] shape; known without reading data."""
    shapes = set()
    for segment in segments:
        if segment.inline is not None:
            shapes.add(tuple(segment.inline[name].shape[1:]))
        elif segment.ref is not None:
            shapes.update(tuple(spec.shape[1:]) for spec in segment.ref.fields if spec.name == name)
    if len(shapes) > 1:
        raise ReplayDataError(f"{name} segments of one sample have different row shapes {sorted(shapes)}")


def _append_mask_rows(current: ReplaySegments | None, new: ReplaySegments, *, response_length: int) -> ReplaySegments:
    if current is None:
        if response_length != 0:
            raise ReplayDataError("cannot initialize a sampling mask after response tokens have already been appended")
        return new
    if current.num_rows != response_length:
        raise ReplayDataError(
            f"sampling mask length {current.num_rows} is not aligned with response_length {response_length} "
            "before appending"
        )
    return current.concat(new)


def _rows_after_strip(segments: ReplaySegments, num_tokens: int) -> int:
    rows = segments.num_rows - num_tokens
    if rows < 0:
        raise ReplayDataError(f"cannot strip {num_tokens} tokens from {segments.num_rows} {segments.name} rows")
    return rows


def _num_source_rows(layout: _FieldLayout, arrays: Mapping[str, np.ndarray]) -> int:
    return int(arrays[layout.stored[0]].shape[0])


def _segment_stored_names(layout: _FieldLayout, segment: ReplaySegment) -> tuple[str, ...]:
    """The stored fields of ``layout`` that the segment's source carries."""
    present = segment.inline.keys() if segment.inline is not None else {spec.name for spec in segment.ref.fields}
    return tuple(name for name in layout.stored if name in present)


def _source_arrays(
    segment: ReplaySegment, layout: _FieldLayout, fetched: Mapping[tuple[str, str], np.ndarray]
) -> Mapping[str, np.ndarray]:
    if segment.inline is not None:
        return segment.inline
    key = segment.ref.key
    try:
        return {name: fetched[(key, name)] for name in _segment_stored_names(layout, segment)}
    except KeyError as e:
        raise ReplayDataError(f"output store object {key} was not read for field {e.args[0][1]}") from None


def _slice_inline(layout: _FieldLayout, segment: ReplaySegment) -> dict[str, np.ndarray]:
    if layout.name == SAMPLING_MASK:
        return _slice_mask(segment.inline, segment)
    (stored,) = layout.stored
    return {stored: segment.inline[stored][segment.start : segment.stop]}


def _slice_mask(arrays: Mapping[str, np.ndarray], segment: ReplaySegment) -> dict[str, np.ndarray]:
    """The segment's rows of a sampling mask in CSR form; support log-probs follow the token IDs."""
    lengths = arrays[MASK_LENGTHS]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    id_start, id_stop = int(offsets[segment.start]), int(offsets[segment.stop])
    sliced = {MASK_LENGTHS: lengths[segment.start : segment.stop]}
    for name in (MASK_TOKEN_IDS, MASK_LOGPROBS):
        if name in arrays:
            sliced[name] = arrays[name][id_start:id_stop]
    return sliced


def _ref_to_plain(ref: StoreRef) -> dict[str, Any]:
    """The ref as JSON types only: the train dict's msgpack codec rejects the tuples of ``to_builtins``."""
    import msgspec

    return msgspec.json.decode(msgspec.json.encode(ref))


def _store_ref_from_builtins(value: Any) -> StoreRef:
    # The package is needed only when a ref is present: runs without the output store do not install it.
    from radix_transfer_plugins import StoreRef

    ref = StoreRef.from_builtins(value)
    if ref.backend != SUPPORTED_STORE_BACKEND:
        raise ReplayDataError(
            f"output store backend {ref.backend!r} is not supported; Miles reads only {SUPPORTED_STORE_BACKEND!r} refs"
        )
    return ref


def _parse_store_ref(meta_info: Mapping[str, Any], *, via_store: bool) -> StoreRef | None:
    value = meta_info.get(OUTPUT_STORE_REF_KEY)
    inline_fields = [key for key in (ROUTED_EXPERTS, INDEXER_TOPK, "output_token_sampling_mask") if key in meta_info]
    if not via_store:
        if value is not None:
            raise ReplayDataError("response carries an output_store_ref, but its request did not ask for one")
        return None
    if value is None:
        if inline_fields:
            raise ReplayDataError(
                f"request set return_outputs_via_store but the response carries inline {inline_fields} and no "
                "output_store_ref; a router between Miles and SGLang may have dropped return_outputs_via_store"
            )
        return None
    if inline_fields:
        raise ReplayDataError(f"response carries both an output_store_ref and inline {inline_fields}")
    return _store_ref_from_builtins(value)


def _decode_inline_rows(info: str, *, rows: int, num_layers: int, allow_extra_row: bool) -> np.ndarray:
    # validate=True: SGLang sends plain base64; pybase64 skips its slow non-alphabet filtering pass.
    x = np.frombuffer(pybase64.b64decode(info, validate=True), dtype=np.int32)
    if rows <= 0:
        if len(x):
            raise ReplayDataError(f"inline buffer of {len(x)} values for zero rows")
        return np.empty((0, num_layers, 0), dtype=np.int32)
    topk = len(x) // (rows * num_layers)
    if topk == 0:
        raise ReplayDataError(f"inline buffer of {len(x)} values cannot hold {rows} rows of {num_layers} layers")
    if allow_extra_row and topk and len(x) == (rows + 1) * num_layers * topk:
        x = x[: rows * num_layers * topk]
    if len(x) != rows * num_layers * topk:
        raise ReplayDataError(
            f"inline buffer of {len(x)} values does not hold {rows} rows of {num_layers} layers x {topk}"
        )
    return x.reshape(rows, num_layers, topk)


def _parse_rows_field(
    meta_info: Mapping[str, Any],
    name: str,
    *,
    ref: StoreRef | None,
    stored: Mapping[str, Any],
    rows: int,
    num_layers: int | None,
    allow_extra_row: bool,
) -> ReplaySegments | None:
    if name in stored:
        spec = stored[name]
        if len(spec.shape) != 3:
            raise ReplayDataError(f"stored {name} has shape {spec.shape}; expected [rows, layers, topk]")
        if num_layers is not None and spec.shape[1] != num_layers:
            raise ReplayDataError(f"stored {name} has {spec.shape[1]} layers; expected {num_layers}")
        stored_rows = spec.shape[0]
        if stored_rows != rows and not (allow_extra_row and stored_rows == rows + 1):
            raise ReplayDataError(f"stored {name} has {stored_rows} rows; expected {rows}")
        return ReplaySegments(name, (ReplaySegment(start=0, stop=rows, ref=ref),))
    if (info := meta_info.get(name)) is None:
        return None
    if num_layers is None:
        raise ReplayDataError(f"cannot decode inline {name} without its layer count")
    array = _decode_inline_rows(info, rows=rows, num_layers=num_layers, allow_extra_row=allow_extra_row)
    return ReplaySegments(name, (ReplaySegment(start=0, stop=rows, inline={name: array}),))


def _indexer_num_layers(
    meta_info: Mapping[str, Any], stored: Mapping[str, Any], *, expected: int | None
) -> int | None:
    if INDEXER_TOPK in stored:
        num_layers = stored[INDEXER_TOPK].shape[1] if len(stored[INDEXER_TOPK].shape) == 3 else None
    elif INDEXER_TOPK in meta_info:
        num_layers = meta_info.get("indexer_topk_num_layers")
        if num_layers is None:
            raise ReplayDataError(
                "Server returned indexer_topk without indexer_topk_num_layers; "
                "sglang-miles must include the layer count in meta_info."
            )
    else:
        return None
    if expected is not None and num_layers != expected:
        raise ReplayDataError(
            f"Server returned indexer_topk with {num_layers} streams but the model has "
            f"{expected} indexer layers; replaying it would map streams to the wrong layers."
        )
    return num_layers


def _parse_sampling_mask(
    meta_info: Mapping[str, Any],
    *,
    ref: StoreRef | None,
    stored: Mapping[str, Any],
    support_logprobs: bool,
    num_output_tokens: int,
    aborted: bool,
) -> ReplaySegments:
    if MASK_LENGTHS in stored:
        segment = _stored_mask_segment(
            meta_info, ref=ref, stored=stored, support_logprobs=support_logprobs, num_output_tokens=num_output_tokens
        )
    elif "output_token_sampling_mask" in meta_info:
        segment = _inline_mask_segment(
            meta_info, support_logprobs=support_logprobs, num_output_tokens=num_output_tokens
        )
    elif _is_no_output_abort(meta_info, aborted=aborted, num_output_tokens=num_output_tokens):
        return ReplaySegments(SAMPLING_MASK)
    else:
        raise ReplayDataError(
            "SGLang response is missing output_token_sampling_mask; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    return ReplaySegments(SAMPLING_MASK, (segment,))


def _is_no_output_abort(meta_info: Mapping[str, Any], *, aborted: bool, num_output_tokens: int) -> bool:
    finish_reason = meta_info.get("finish_reason") or {}
    return (aborted or finish_reason.get("type") == "abort") and num_output_tokens == 0


def _stored_mask_segment(
    meta_info: Mapping[str, Any],
    *,
    ref: StoreRef,
    stored: Mapping[str, Any],
    support_logprobs: bool,
    num_output_tokens: int,
) -> ReplaySegment:
    required = (MASK_TOKEN_IDS, MASK_LOGPROBS) if support_logprobs else (MASK_TOKEN_IDS,)
    missing = [name for name in required if name not in stored]
    if missing:
        raise ReplayDataError(f"output store ref carries {MASK_LENGTHS} without {missing}")
    rows = stored[MASK_LENGTHS].shape[0]
    if rows != num_output_tokens:
        raise ReplayDataError(f"sampling support length {rows} != token length {num_output_tokens}")
    if (length := meta_info.get("output_token_sampling_mask_length")) is not None and length != rows:
        raise ReplayDataError(f"output_token_sampling_mask_length {length} != stored mask rows {rows}")
    entries = stored[MASK_TOKEN_IDS].shape[0]
    if support_logprobs and (logprobs := stored[MASK_LOGPROBS].shape[0]) != entries:
        raise ReplayDataError(f"stored support log-probs have {logprobs} values for {entries} support entries")
    return ReplaySegment(start=0, stop=rows, ref=ref)


def _inline_mask_segment(
    meta_info: Mapping[str, Any], *, support_logprobs: bool, num_output_tokens: int
) -> ReplaySegment:
    supports = meta_info["output_token_sampling_mask"]
    if len(supports) != num_output_tokens:
        raise ReplayDataError(f"sampling support length {len(supports)} != token length {num_output_tokens}")
    lengths = np.fromiter((len(support) for support in supports), dtype=np.int32, count=len(supports))
    token_ids = np.fromiter(chain.from_iterable(supports), dtype=np.int32, count=int(lengths.sum()))
    inline = {MASK_LENGTHS: lengths, MASK_TOKEN_IDS: token_ids}
    if support_logprobs:
        log_probs = meta_info.get("output_token_sampling_logprobs")
        if log_probs is None or len(log_probs) != num_output_tokens:
            raise ReplayDataError("support mode needs output_token_sampling_logprobs with a row per output token")
        for support, row in zip(supports, log_probs, strict=True):
            if not isinstance(row, (list, tuple)) or len(row) != len(support):
                raise ReplayDataError("SGLang support log-probs must align with the sampling support IDs")
        inline[MASK_LOGPROBS] = np.fromiter(chain.from_iterable(log_probs), dtype=np.float32, count=len(token_ids))
    return ReplaySegment(start=0, stop=num_output_tokens, inline=inline)
