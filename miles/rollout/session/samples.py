"""Training-sample assembly and its wire codec: session records -> `Sample` -> reply bytes -> driver overlay.

Owned by the session package so the assembly runs on the owning instance (records never have to leave the session server); the driver decodes the reply without the registry/backend stack.

- Depends on `generate_utils.generate_endpoint_utils` for the R3 replay decoders (accepted utils-level dependency: the decoders have other consumers on the single-turn `/generate` path and must not fork).
- Order contract: `truncate_samples_by_total_tokens` runs BEFORE `merge_samples` — truncation is a turn-level budget decision (which turns survive; the overflowing turn is cut at a turn boundary, later turns are dropped) and the turn structure only exists pre-merge.
- Assembly builds each per-turn `Sample` from a blank template and populates only the COMPUTED_FIELDS; the driver's input-sample (template) fields never cross the wire — the driver overlays the computed fields onto deepcopies of its local input sample (`decode_samples_reply`). Overlay equivalence with the legacy driver-side pipeline requires the input sample to carry dataclass defaults on the fields that pipeline evolved in place — see `_assert_overlay_template_defaults`.
"""

import dataclasses
import json
import struct
from argparse import Namespace
from copy import deepcopy

import numpy as np

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample


def compute_samples_from_openai_records(
    args: Namespace,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Each record carries its own ``prompt_token_ids`` and ``output_token_ids``
    (with logprobs).  We want to reuse those per-turn logprobs directly
    instead of re-decoding, but we must first trim "trailing tokens" — stop
    tokens the model emitted that the chat template also renders as the next
    turn's delimiter — to avoid double-counting.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/session/test_samples.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.request["input_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            # Step 1: position cursor right after this turn's prompt
            cursor = len(prompt_ids)

            # Step 2: greedily match output_ids against accumulated[cursor:]
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break

            # Step 3: unmatched trailing tokens were consumed by the next
            # turn's template rendering (e.g. stop tokens that double as
            # the next message delimiter) — strip them from the sample.
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={accumulated_token_ids[cursor:cursor+3]}"
            )

            # Step 4: advance cursor past matched output to the next turn
            cursor += matched

        sample = _compute_sample_from_openai_record(args, record, tokenizer, trim_count)
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace, record: SessionRecord, tokenizer, trim_count: int = 0
) -> Sample:
    choice = record.response["choices"][0]

    prompt_token_ids = record.request.get("input_ids")
    if prompt_token_ids is None:
        raise ValueError("input_ids not found in request — the session server should populate it")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = Sample()
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, choice, sample)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, choice, sample)

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    sample.prefix_cache_info.add(choice.get("meta_info", {}))
    if "weight_version" in choice["meta_info"]:
        sample.weight_versions.append(choice["meta_info"]["weight_version"])

    return sample


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        sample.strip_last_output_tokens(overshoot, tokenizer)
        sample.status = Sample.Status.TRUNCATED
        result.append(sample)
        break

    return result


# Wire envelope of the samples reply: u64-length-prefixed JSON meta, then the
# raw binary segments (no base64).
_LEN = struct.Struct(">Q")


def encode_envelope(meta: dict, body: bytes) -> bytes:
    """Pack `meta` (JSON) + raw `body` (no base64) into one buffer."""
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _LEN.pack(len(meta_bytes)) + meta_bytes + body


def decode_envelope(buf: bytes) -> tuple[dict, bytes]:
    """Inverse of :func:`encode_envelope`."""
    (meta_len,) = _LEN.unpack_from(buf, 0)
    start = _LEN.size
    meta = json.loads(buf[start : start + meta_len])
    return meta, buf[start + meta_len :]


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
    "session_id",
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

# Fields carried as raw binary segments (dtype + shape in the JSON meta); the
# scalar/list computed fields ride in the JSON meta directly. Token ids and
# logprobs are re-materialized as Python lists on decode, exactly like the
# legacy JSON path (int64/f64 round-trips are lossless for both).
_SEGMENT_DTYPES = {"tokens": np.int64, "rollout_log_probs": np.float64}
_SEGMENT_FIELDS = ("tokens", "rollout_log_probs", "rollout_routed_experts", "rollout_indexer_topk")

_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


@dataclasses.dataclass
class SamplesReply:
    """Decoded `POST /sessions/{id}/samples` reply."""

    samples: list[Sample]
    session_metadata: dict
    empty_reason: str | None


def encode_samples_reply(samples: list[Sample], session_metadata: dict, empty_reason: str | None = None) -> bytes:
    """Worker side: pack assembled samples into one envelope (JSON meta + binary body)."""
    sample_metas = []
    segments: list[bytes] = []
    offset = 0
    for sample in samples:
        segment_meta = {}
        for name in _SEGMENT_FIELDS:
            value = getattr(sample, name)
            if value is None:
                segment_meta[name] = None
                continue
            arr = np.asarray(value, dtype=_SEGMENT_DTYPES.get(name))
            data = arr.tobytes()
            segment_meta[name] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": len(data),
            }
            segments.append(data)
            offset += len(data)
        sample_metas.append(
            {
                "response": sample.response,
                "response_length": sample.response_length,
                "loss_mask": sample.loss_mask,
                "status": sample.status.value,
                "weight_versions": sample.weight_versions,
                "prefix_cache_info": sample.prefix_cache_info.to_dict(),
                "segments": segment_meta,
            }
        )
    meta = {"samples": sample_metas, "session_metadata": session_metadata, "empty_reason": empty_reason}
    return encode_envelope(meta, b"".join(segments))


def decode_samples_reply(payload: bytes, input_sample: Sample) -> SamplesReply:
    """Driver side: overlay each wire sample's computed fields onto a deepcopy of `input_sample`."""
    meta, body = decode_envelope(payload)
    if meta["samples"]:
        _assert_overlay_template_defaults(input_sample)
    samples = []
    for sample_meta in meta["samples"]:
        sample = deepcopy(input_sample)
        segment_meta = sample_meta["segments"]
        tokens = _read_segment(body, segment_meta["tokens"])
        log_probs = _read_segment(body, segment_meta["rollout_log_probs"])
        sample.tokens = tokens.tolist() if tokens is not None else []
        sample.rollout_log_probs = log_probs.tolist() if log_probs is not None else None
        sample.rollout_routed_experts = _read_segment(body, segment_meta["rollout_routed_experts"])
        sample.rollout_indexer_topk = _read_segment(body, segment_meta["rollout_indexer_topk"])
        sample.response = sample_meta["response"]
        sample.response_length = sample_meta["response_length"]
        sample.loss_mask = sample_meta["loss_mask"]
        sample.status = Sample.Status(sample_meta["status"])
        sample.weight_versions = list(sample_meta["weight_versions"])
        sample.prefix_cache_info = Sample.PrefixCacheInfo.from_dict(sample_meta["prefix_cache_info"])
        samples.append(sample)
    return SamplesReply(samples=samples, session_metadata=meta["session_metadata"], empty_reason=meta["empty_reason"])


def _read_segment(body: bytes, segment_meta: dict | None) -> np.ndarray | None:
    if segment_meta is None:
        return None
    start = segment_meta["offset"]
    arr = np.frombuffer(body[start : start + segment_meta["nbytes"]], dtype=segment_meta["dtype"])
    return arr.reshape(segment_meta["shape"])


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
