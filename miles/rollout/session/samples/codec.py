"""Encode server-computed fields and merge them into driver input samples.

Only fields in `SAMPLES_VALUE_SPEC` cross the wire. `encode_samples` packs them
into safetensors; `decode_samples_and_merge_input_sample` overlays them onto a
deepcopy of the input sample and merges server metadata.

Tree-serving v2 selects `SAMPLES_VALUE_SPEC_V2` to carry `Sample.reward`.
"""

import dataclasses
import json
from copy import copy, deepcopy

import numpy as np
import safetensors.numpy
import torch

from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


@dataclasses.dataclass(frozen=True)
class ValueSpec:
    """Wire contract of one computed field."""

    codec: str  # "tensor" | "tensor_list" | "sampling_mask" | "json"
    dtype: np.dtype | None = None  # tensor codecs: pinned on both sides; a mismatch raises instead of converting
    strict: bool = False  # encode never converts, only validates (R3 replay tensors must arrive as int32)
    null: object = None  # decoded value for a null-marked field; copied per sample so no instance is shared


# tokens/loss_mask/logprobs are re-materialized as Python lists on decode,
# exactly like the legacy JSON path (int64/f64 round-trips are lossless).
SAMPLES_VALUE_SPEC: dict[str, ValueSpec] = {
    "tokens": ValueSpec("tensor_list", np.dtype(np.int64), null=[]),
    "response": ValueSpec("json"),
    "response_length": ValueSpec("json"),
    "loss_mask": ValueSpec("tensor_list", np.dtype(np.uint8)),
    "rollout_log_probs": ValueSpec("tensor_list", np.dtype(np.float64)),
    "rollout_sampling_mask": ValueSpec("sampling_mask"),
    "rollout_routed_experts": ValueSpec("tensor", np.dtype(np.int32), strict=True),
    "rollout_indexer_topk": ValueSpec("tensor", np.dtype(np.int32), strict=True),
    "status": ValueSpec("json"),
    "weight_versions": ValueSpec("json"),
    "start_weight_version": ValueSpec("json"),
    "prefix_cache_info": ValueSpec("json"),
    "metadata": ValueSpec("json"),
}

# Tree-serving v2 adds `Sample.reward` to the wire.
SAMPLES_VALUE_SPEC_V2: dict[str, ValueSpec] = {
    **SAMPLES_VALUE_SPEC,
    "reward": ValueSpec("json"),
}

ROLLOUT_SAMPLING_MASK_FIELDS = ("rollout_sampling_mask",)

# The wire allowlists, derived: only table fields cross the samples wire.
COMPUTED_FIELDS = tuple(field for field in SAMPLES_VALUE_SPEC if field not in ROLLOUT_SAMPLING_MASK_FIELDS)
COMPUTED_FIELDS_V2 = tuple(field for field in SAMPLES_VALUE_SPEC_V2 if field not in ROLLOUT_SAMPLING_MASK_FIELDS)

assert all(
    spec.codec in ("tensor", "tensor_list", "sampling_mask", "json") for spec in SAMPLES_VALUE_SPEC_V2.values()
), "unknown codec in SAMPLES_VALUE_SPEC"

_TENSOR_FIELDS = frozenset(field for field, spec in SAMPLES_VALUE_SPEC_V2.items() if spec.codec != "json")
assert _TENSOR_FIELDS <= set(COMPUTED_FIELDS + ROLLOUT_SAMPLING_MASK_FIELDS)

_SAMPLES_META_KEY = "_samples_meta"
_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


@dataclasses.dataclass
class SamplesReply:
    """Decoded `POST /sessions/{id}/samples` reply."""

    samples: list[Sample]
    session_metadata: dict
    empty_reason: str | None


def _asarray_wire(field: str, value, dtype: np.dtype) -> np.ndarray:
    """Convert to the wire dtype, refusing any conversion that changes values
    (e.g. a negative loss-mask entry wrapping into uint8)."""
    arr = np.asarray(value)
    if arr.dtype == dtype:
        return arr
    converted = arr.astype(dtype)
    if not np.array_equal(converted, arr):
        raise ValueError(f"{field} values do not fit wire dtype {dtype}")
    return converted


def encode_samples(
    samples: list[Sample],
    session_metadata: dict,
    empty_reason: str | None = None,
    *,
    fields: tuple[str, ...] = COMPUTED_FIELDS,
) -> bytes:
    """Server side: pack assembled samples into one safetensors payload.

    ``fields`` selects the wire allowlist (v1 default; the v2 server passes
    ``COMPUTED_FIELDS_V2``) — with the default, the payload is byte-identical
    to the pre-parameterized codec.
    """
    tensors: dict[str, np.ndarray] = {}
    sample_metas = []
    for sample_index, sample in enumerate(samples):
        sample_meta: dict = {}
        nulls: list[str] = []
        for field in fields:
            spec = SAMPLES_VALUE_SPEC_V2[field]
            value = getattr(sample, field)
            if spec.codec == "json":
                if field == "status":
                    value = value.value
                elif field == "prefix_cache_info":
                    value = value.to_dict()
                sample_meta[field] = value
                continue
            if value is None:
                nulls.append(field)
                continue
            if spec.codec == "sampling_mask":
                if not isinstance(value, RolloutSamplingMask):
                    raise TypeError(f"{field} must be a RolloutSamplingMask, got {type(value).__name__}")
                ids, offsets = value._as_tensors()
                tensors[f"{field}.ids.{sample_index}"] = np.ascontiguousarray(ids.numpy())
                tensors[f"{field}.offsets.{sample_index}"] = np.ascontiguousarray(offsets.numpy())
                continue
            if spec.strict:
                arr = np.asarray(value)
                if arr.dtype != spec.dtype:
                    raise ValueError(f"{field} must have dtype {spec.dtype}, got {arr.dtype}")
            else:
                arr = _asarray_wire(field, value, spec.dtype)
            # ascontiguousarray is a correctness requirement: the numpy adapter
            # serializes some non-contiguous views without raising, with wrong values.
            tensors[f"{field}.{sample_index}"] = np.ascontiguousarray(arr)
        sample_meta["nulls"] = nulls
        sample_metas.append(sample_meta)
    meta = {"samples": sample_metas, "session_metadata": session_metadata, "empty_reason": empty_reason}
    # Compact separators: no reason to ship JSON padding on every reply.
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tensors[_SAMPLES_META_KEY] = np.frombuffer(meta_bytes, dtype=np.uint8)
    return safetensors.numpy.save(tensors)


def decode_samples_and_merge_input_sample(
    payload: bytes, input_sample: Sample, *, fields: tuple[str, ...] = COMPUTED_FIELDS
) -> SamplesReply:
    """Driver side: overlay each wire sample's computed fields onto a deepcopy of `input_sample`.

    ``fields`` must match the server's encode allowlist (v1 default); extra
    keys a newer server sent are ignored, so a v1 decode of a v2 payload
    keeps exactly the v1 overlay semantics.
    """
    tensors = safetensors.numpy.load(payload)  # SafetensorError propagates: invalid container
    meta_arr = tensors.pop(_SAMPLES_META_KEY)  # KeyError propagates: missing meta is malformed
    if meta_arr.ndim != 1 or meta_arr.dtype != np.uint8:
        raise ValueError(
            f"{_SAMPLES_META_KEY} must be a rank-one uint8 tensor, got {meta_arr.dtype} rank {meta_arr.ndim}"
        )
    meta = json.loads(meta_arr.tobytes().decode("utf-8"))
    if meta["samples"]:
        assert_input_sample_defaults(input_sample)
    samples = []
    for sample_index, sample_meta in enumerate(meta["samples"]):
        sample = deepcopy(input_sample)
        nulls = set(sample_meta["nulls"])  # KeyError propagates: missing null markers are malformed
        if nulls - _TENSOR_FIELDS:
            raise ValueError(f"null markers reference non-tensor fields: {sorted(nulls - _TENSOR_FIELDS)}")
        for field in fields:
            spec = SAMPLES_VALUE_SPEC_V2[field]
            if spec.codec == "json":
                value = sample_meta[field]
                if field == "status":
                    value = Sample.Status(value)
                elif field == "weight_versions":
                    value = list(value)
                elif field == "prefix_cache_info":
                    value = Sample.PrefixCacheInfo.from_dict(value)
                elif field == "reward":
                    # Server reward is authoritative only when assigned; a null
                    # keeps the driver input's local reward.
                    if value is None:
                        continue
                elif field == "metadata":
                    if not isinstance(value, dict):
                        raise ValueError(f"metadata must be a JSON object, got {type(value).__name__}")
                    sample.metadata.update(value)
                    continue
                setattr(sample, field, value)
                continue
            if field in nulls:
                setattr(sample, field, copy(spec.null))
                continue
            if spec.codec == "sampling_mask":
                ids = tensors.pop(f"{field}.ids.{sample_index}")
                offsets = tensors.pop(f"{field}.offsets.{sample_index}")
                if ids.dtype != np.int32 or offsets.dtype != np.int64:
                    raise ValueError(
                        f"{field} must use int32 ids and int64 offsets, got {ids.dtype} and {offsets.dtype}"
                    )
                setattr(
                    sample,
                    field,
                    RolloutSamplingMask(ids=torch.from_numpy(ids), offsets=torch.from_numpy(offsets)),
                )
                continue
            arr = tensors.pop(f"{field}.{sample_index}")  # KeyError propagates: a promised tensor must exist
            if arr.dtype != spec.dtype:
                raise ValueError(f"{field} must have dtype {spec.dtype}, got {arr.dtype}")
            setattr(sample, field, arr.tolist() if spec.codec == "tensor_list" else arr)
        samples.append(sample)
    if tensors:
        raise ValueError(f"payload carries unreferenced tensors: {sorted(tensors)}")
    return SamplesReply(samples=samples, session_metadata=meta["session_metadata"], empty_reason=meta["empty_reason"])


def assert_input_sample_defaults(input_sample: Sample) -> None:
    """Require input-sample defaults; otherwise merging server fields can corrupt existing sample state."""
    assert input_sample.weight_versions == [], (
        f"input sample must not carry weight_versions (got {input_sample.weight_versions}); "
        "the legacy pipeline appended to it, the samples-wire overlay replaces it"
    )
    assert (
        input_sample.prefix_cache_info.to_dict() == Sample.PrefixCacheInfo().to_dict()
    ), f"input sample must carry a default prefix_cache_info (got {input_sample.prefix_cache_info.to_dict()})"
    assert (
        input_sample.spec_info.to_dict() == Sample.SpecInfo().to_dict()
    ), f"input sample must carry a default spec_info (got {input_sample.spec_info.to_dict()})"
    assert input_sample.teacher_log_probs is None and input_sample.opd_reverse_kl is None, (
        "input sample must not carry teacher_log_probs/opd_reverse_kl; "
        "the legacy pipeline trimmed them per turn, the samples-wire overlay carries them verbatim"
    )
    assert _OPD_STUDENT_TOP_LOGPROBS_KEY not in (input_sample.metadata or {}), (
        f"input sample metadata must not carry {_OPD_STUDENT_TOP_LOGPROBS_KEY!r}; "
        "merge_samples gives it per-token semantics that only hold for per-turn values"
    )
