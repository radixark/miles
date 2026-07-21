"""Wire codec for the samples reply: assembled `Sample`s <-> one safetensors payload.

The operation-primitive pair `encode_samples_reply` / `decode_samples_reply` (`bytes` <-> `SamplesReply`) plus its wire contract, kept free of HTTP, session state, and assembly (`samples.py`) so the contract depends only on `Sample`, NumPy, and safetensors.

- The payload is exactly one safetensors buffer: one named tensor per non-`None` `_TENSOR_SPECS` field, plus the scalar JSON stored as the rank-one uint8 tensor `_samples_meta` (the in-memory `safetensors.numpy.load` API does not expose header metadata, and `__metadata__` is reserved by the format).
- Assembly builds each per-turn `Sample` from a blank template and populates only the COMPUTED_FIELDS; the driver's input-sample (template) fields never cross the wire — the driver overlays the computed fields onto deepcopies of its local input sample (`decode_samples_reply`). Overlay equivalence with the legacy driver-side pipeline requires the input sample to carry dataclass defaults on the fields that pipeline evolved in place — see `_assert_overlay_template_defaults`.
- Decoded arrays are writable copies (`safetensors.numpy.load`), unlike the read-only `np.frombuffer` views of the removed envelope; values and Python/NumPy types are contractual, array flags are not.
"""

import dataclasses
import json
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import safetensors.numpy

from miles.utils.types import Sample

# Every Sample field is either COMPUTED by assembly from the records (crosses
# the wire) or belongs to the driver's input-sample TEMPLATE (never crosses;
# the driver overlay keeps its local deepcopy's value). Adding a Sample field
# without classifying it here fails at import time, not silently at training.
COMPUTED_FIELDS = (
    "tokens",
    "response",
    "response_length",
    "loss_mask",
    "rollout_log_probs",
    "rollout_routed_experts",
    "rollout_indexer_topk",
    "status",
    "weight_versions",
    "prefix_cache_info",
)
TEMPLATE_FIELDS = (
    "group_index",
    "index",
    "prompt",
    "multimodal_inputs",
    "multimodal_train_inputs",
    "label",
    "reward",
    "remove_sample",
    "teacher_log_probs",
    "opd_reverse_kl",
    "metadata",
    "generate_function_path",
    "train_metadata",
    "routing_key",
    "non_generation_time",
    "spec_info",
)

_SAMPLE_FIELDS = {f.name for f in dataclasses.fields(Sample)}
assert set(COMPUTED_FIELDS) | set(TEMPLATE_FIELDS) == _SAMPLE_FIELDS and not set(COMPUTED_FIELDS) & set(
    TEMPLATE_FIELDS
), (
    "Sample fields drifted: every field must be classified as COMPUTED (crosses the samples wire) "
    "or TEMPLATE (stays on the driver's input sample). "
    f"Unclassified: {sorted(_SAMPLE_FIELDS - set(COMPUTED_FIELDS) - set(TEMPLATE_FIELDS))}, "
    f"unknown: {sorted((set(COMPUTED_FIELDS) | set(TEMPLATE_FIELDS)) - _SAMPLE_FIELDS)}, "
    f"overlap: {sorted(set(COMPUTED_FIELDS) & set(TEMPLATE_FIELDS))}"
)


@dataclasses.dataclass(frozen=True)
class _TensorSpec:
    """Wire contract of one COMPUTED tensor field; the codec is a loop over `_TENSOR_SPECS`."""

    normalize_dtype: np.dtype | None  # np.asarray target on encode; None keeps the caller's dtype
    wire_dtype: np.dtype  # pinned on both sides; a mismatch raises instead of silently converting
    restore_list: bool  # decode returns .tolist() (legacy JSON-path types) instead of the ndarray
    null_factory: Callable[[], object]  # decode value for JSON null; factory so no instance is shared


# Token ids and logprobs are re-materialized as Python lists on decode, exactly
# like the legacy JSON path (int64/f64 round-trips are lossless for both). The
# R3 replay fields must arrive as int32 and are never converted.
_TENSOR_SPECS = {
    "tokens": _TensorSpec(np.dtype(np.int64), np.dtype(np.int64), True, list),
    "rollout_log_probs": _TensorSpec(np.dtype(np.float64), np.dtype(np.float64), True, lambda: None),
    "rollout_routed_experts": _TensorSpec(None, np.dtype(np.int32), False, lambda: None),
    "rollout_indexer_topk": _TensorSpec(None, np.dtype(np.int32), False, lambda: None),
}
_SCALAR_FIELDS = ("response", "response_length", "loss_mask", "status", "weight_versions", "prefix_cache_info")

assert set(_TENSOR_SPECS) | set(_SCALAR_FIELDS) == set(COMPUTED_FIELDS) and not set(_TENSOR_SPECS) & set(
    _SCALAR_FIELDS
), (
    "every COMPUTED field needs exactly one wire representation (tensor spec or scalar JSON); "
    f"uncovered: {sorted(set(COMPUTED_FIELDS) - set(_TENSOR_SPECS) - set(_SCALAR_FIELDS))}, "
    f"unknown: {sorted((set(_TENSOR_SPECS) | set(_SCALAR_FIELDS)) - set(COMPUTED_FIELDS))}, "
    f"overlap: {sorted(set(_TENSOR_SPECS) & set(_SCALAR_FIELDS))}"
)

_SAMPLES_META_KEY = "_samples_meta"
_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


def _tensor_name(sample_index: int, field: str) -> str:
    return f"sample.{sample_index}.{field}"


@dataclasses.dataclass
class SamplesReply:
    """Decoded `POST /sessions/{id}/samples` reply."""

    samples: list[Sample]
    session_metadata: dict
    empty_reason: str | None


def encode_samples_reply(samples: list[Sample], session_metadata: dict, empty_reason: str | None = None) -> bytes:
    """Worker side: pack assembled samples into one safetensors payload."""
    tensors: dict[str, np.ndarray] = {}
    sample_metas = []
    for sample_index, sample in enumerate(samples):
        tensor_meta = {}
        for field, spec in _TENSOR_SPECS.items():
            value = getattr(sample, field)
            if value is None:
                tensor_meta[field] = None
                continue
            arr = np.asarray(value, dtype=spec.normalize_dtype)
            if arr.dtype != spec.wire_dtype:
                raise ValueError(f"{field} must have dtype {spec.wire_dtype}, got {arr.dtype}")
            name = _tensor_name(sample_index, field)
            # ascontiguousarray is a correctness requirement: the numpy adapter
            # serializes some non-contiguous views without raising, with wrong values.
            tensors[name] = np.ascontiguousarray(arr)
            tensor_meta[field] = name
        scalar_meta = {}
        for field in _SCALAR_FIELDS:
            value = getattr(sample, field)
            if field == "status":
                value = value.value
            elif field == "prefix_cache_info":
                value = value.to_dict()
            scalar_meta[field] = value
        sample_metas.append({**scalar_meta, "tensors": tensor_meta})
    meta = {"samples": sample_metas, "session_metadata": session_metadata, "empty_reason": empty_reason}
    # Compact separators are load-bearing: the default ", "/": " padding costs
    # ~1 byte per loss_mask/token entry, ~100KB on production-sized replies.
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tensors[_SAMPLES_META_KEY] = np.frombuffer(meta_bytes, dtype=np.uint8)
    return safetensors.numpy.save(tensors)


def decode_samples_reply(payload: bytes, input_sample: Sample) -> SamplesReply:
    """Driver side: overlay each wire sample's computed fields onto a deepcopy of `input_sample`."""
    tensors = safetensors.numpy.load(payload)  # SafetensorError propagates: invalid container
    meta_arr = tensors.pop(_SAMPLES_META_KEY)  # KeyError propagates: missing meta is malformed
    if meta_arr.ndim != 1 or meta_arr.dtype != np.uint8:
        raise ValueError(
            f"{_SAMPLES_META_KEY} must be a rank-one uint8 tensor, got {meta_arr.dtype} rank {meta_arr.ndim}"
        )
    meta = json.loads(meta_arr.tobytes().decode("utf-8"))
    if meta["samples"]:
        _assert_overlay_template_defaults(input_sample)
    samples = []
    for sample_index, sample_meta in enumerate(meta["samples"]):
        sample = deepcopy(input_sample)
        tensor_meta = sample_meta["tensors"]
        for field, spec in _TENSOR_SPECS.items():
            name = tensor_meta[field]
            if name is None:
                setattr(sample, field, spec.null_factory())
                continue
            expected = _tensor_name(sample_index, field)
            if name != expected:
                raise ValueError(f"{field} references tensor {name!r}, expected {expected!r}")
            arr = tensors.pop(name)  # KeyError propagates: a referenced tensor must exist
            if arr.dtype != spec.wire_dtype:
                raise ValueError(f"{field} must have dtype {spec.wire_dtype}, got {arr.dtype}")
            setattr(sample, field, arr.tolist() if spec.restore_list else arr)
        for field in _SCALAR_FIELDS:
            value = sample_meta[field]
            if field == "status":
                value = Sample.Status(value)
            elif field == "weight_versions":
                value = list(value)
            elif field == "prefix_cache_info":
                value = Sample.PrefixCacheInfo.from_dict(value)
            setattr(sample, field, value)
        samples.append(sample)
    if tensors:
        raise ValueError(f"payload carries unreferenced tensors: {sorted(tensors)}")
    return SamplesReply(samples=samples, session_metadata=meta["session_metadata"], empty_reason=meta["empty_reason"])


def _assert_overlay_template_defaults(input_sample: Sample) -> None:
    """Overlay equivalence precondition (fail-loud).

    The legacy driver-side pipeline EVOLVED some fields of the input sample in
    place (`weight_versions` append, `prefix_cache_info` accumulate, merge sums
    `spec_info` across turns, `strip_last_output_tokens` trims
    `teacher_log_probs`/`opd_reverse_kl`/`metadata["opd_student_top_logprobs"]`),
    while the overlay REPLACES the computed fields and carries the template
    verbatim. The two agree exactly when the input sample holds dataclass
    defaults on those fields — true for every sample fresh from the data loader
    (and `reset_for_retry` restores it on framework retries).
    """
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
