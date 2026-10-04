"""Completed GPU-delta timing summaries, without additional distributed work.

Receiver clocks are compared only within the same original process. Nested
spans remain separate. Engine-host work is sampled once from each arena creator,
not from the zero counters on ranks that attach to its arena.
"""

import logging
import math
from statistics import median

from miles.utils.gpu_delta_publication import canonical_json
from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)
_PREFIX = "perf/gpu_delta/"
_RANK_TIMINGS = (
    "host_prepare_s",
    "paused_apply_host_wall_s",
    "host_shared_register_s",
    "host_payload_cache_wait_s",
    "host_manifest_read_parse_s",
    "host_plan_validate_s",
    "host_tensor_prepare_s",
    "host_raw_pack_s",
    "host_decoder_prepare_s",
    "host_ready_wait_s",
    "host_raw_enqueue_s",
    "host_matrix_enqueue_s",
    "host_derived_enqueue_s",
    "host_apply_completion_wait_s",
)
_HOST_TIMINGS = (
    "host_frames_validate_s",
    "host_payload_read_s",
    "host_payload_sha256_s",
    "host_payload_decode_hash_s",
    "host_payload_hash_wait_s",
    "host_outer_zstd_validate_s",
    "host_outer_zstd_decode_s",
    "host_outer_zstd_worker_decode_sum_s",
    "host_shared_build_s",
    "host_shared_allocation_s",
    "host_encoded_allocation_s",
)
_RANK_REGISTRATION = (
    "host_plan_cache_reused",
    "host_shared_register_calls",
    "host_shared_registered_bytes",
    "host_shared_registration_reused",
    "host_shared_mapping_reused",
    "host_shared_registration_capacity_bytes",
)
_HOST_CAPACITIES = (
    "host_shared_arena_bytes",
    "host_shared_capacity_bytes",
    "host_encoded_capacity_bytes",
)
_HOST_WORK_TOTALS = (
    "host_frames_validations",
    "host_payload_hash_bytes",
    "host_outer_zstd_encoded_bytes",
    "host_outer_zstd_decoded_bytes",
    "host_shared_allocation_calls",
    "host_shared_allocation_bytes",
    "host_encoded_allocation_calls",
    "host_encoded_allocation_bytes",
)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("GPU-delta timing must be a finite nonnegative number")
    return value


def _distribution(metrics, name, values):
    if values:
        values = [_number(value) for value in values]
        for statistic, value in (("min", min(values)), ("p50", median(values)), ("max", max(values))):
            metrics[_PREFIX + name + "/" + statistic] = value


def _joined_receipts(activation):
    applied = {canonical_json(r["identity"]): r for r in activation["receipts"]}
    resumed = {canonical_json(r["identity"]): r for r in activation["resumed_receipts"]}
    if (
        not applied
        or applied.keys() != resumed.keys()
        or len(applied) != len(activation["receipts"])
        or len(resumed) != len(activation["resumed_receipts"])
    ):
        raise ValueError("GPU-delta timing requires all original participants exactly once")
    for identity, before in applied.items():
        after = resumed[identity]
        if before["state"] != "APPLIED" or after["state"] != "RESUMED":
            raise ValueError("GPU-delta timing requires completed APPLIED/RESUMED receipts")
        for key in (
            "session_id",
            "cohort_digest",
            "manifest_sha256",
            "stream_id",
            "base_version",
            "target_version",
            "plan_digest",
        ):
            if before[key] != after[key]:
                raise ValueError(f"GPU-delta timing receipt {key} differs")
        first, last = before["scheduler_timing"], after["scheduler_timing"]
        if first["clock"] != "monotonic_ns" or last["clock"] != "monotonic_ns":
            raise ValueError("GPU-delta scheduler timing clock differs")
        start, fence, end = (last[key] for key in ("pause_started_ns", "reader_fence_completed_ns", "resumed_ns"))
        if (
            any(type(value) is not int for value in (start, fence, end))
            or not 0 <= start <= fence <= end
            or first["resumed_ns"] is not None
            or first["pause_started_ns"] != start
            or first["reader_fence_completed_ns"] != fence
            or not math.isclose(_number(last["blocked_s"]), (end - start) / 1e9, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("GPU-delta scheduler timing interval is incomplete or inconsistent")
        yield before, (fence - start) / 1e9, (end - start) / 1e9


def activation_metrics(activation):
    """Reduce already-validated RPC evidence; never poll or synchronize devices."""
    rows = list(_joined_receipts(activation))
    metrics = {
        _PREFIX + "receiver_ranks": len(rows),
        _PREFIX + "base_version": rows[0][0]["base_version"],
        _PREFIX + "target_version": rows[0][0]["target_version"],
    }
    _distribution(metrics, "receiver_reader_fence_s", [row[1] for row in rows])
    _distribution(metrics, "receiver_scheduler_pause_s", [row[2] for row in rows])
    timings = [row[0]["result"]["timings"] for row in rows]
    for name in _RANK_TIMINGS + _RANK_REGISTRATION:
        # Optional profiling spans are emitted only with complete rank coverage.
        if all(name in timing for timing in timings):
            _distribution(metrics, "receiver_" + name, [timing[name] for timing in timings])
    arenas = {}
    for (receipt, _, _), timing in zip(rows, timings, strict=True):
        arena = (receipt["identity"]["engine_id"], receipt["identity"]["host_cache_id"])
        arenas.setdefault(arena, []).append(timing)
    creators = []
    for host_rows in arenas.values():
        created = [row for row in host_rows if row["host_payload_cache_created"] == 1]
        if len(created) > 1:
            raise ValueError("Multiple payload creators in one engine-host arena")
        creators.extend(created)
    metrics.update({_PREFIX + "receiver_host_arenas": len(arenas), _PREFIX + "host_cache_creators": len(creators)})
    for name in _HOST_TIMINGS:
        if all(name in row for row in creators):
            _distribution(metrics, "creator_" + name, [row[name] for row in creators])
    _distribution(metrics, "creator_cpu_workers", [row["host_outer_zstd_cpu_workers"] for row in creators])
    for name in _HOST_WORK_TOTALS:
        if all(name in row for row in creators):
            metrics[_PREFIX + "creator_" + name + "/sum"] = sum(_number(row[name]) for row in creators)
    # Engine-local ranks map one host arena. Count capacity once per arena, including
    # reattachment with no creator; per-rank CUDA registrations are not additive
    # physical storage. Two independent engines may hold duplicate physical bytes.
    for name in _HOST_CAPACITIES:
        if not all(name in row for row in timings):
            continue
        capacities = []
        for host_rows in arenas.values():
            values = {_number(row[name]) for row in host_rows}
            if len(values) != 1:
                raise ValueError(f"Shared host capacity differs between ranks: {name}")
            capacities.append(values.pop())
        _distribution(metrics, name, capacities)
        metrics[_PREFIX + name + "/sum"] = sum(capacities)
    engines = {row[0]["identity"]["engine_id"] for row in rows}
    engine_timings = activation["engine_timings"]
    if {row["engine_id"] for row in engine_timings} != engines or len(engine_timings) != len(engines):
        raise ValueError("Coordinator timings do not cover the original engines exactly once")
    metrics[_PREFIX + "receiver_engines"] = len(engines)
    for name in ("prepare_s", "apply_s", "resume_s", "activation_s"):
        _distribution(metrics, "engine_coordinator_" + name, [row[name] for row in engine_timings])
    metrics[_PREFIX + "coordinator_activation_s"] = _number(activation["coordinator_timings"]["activation_s"])
    return metrics


def producer_metrics(owners):
    """The existing owner gather observes a prefix, before publication/activation."""
    metrics = {_PREFIX + "producer_ranks": len(owners)}
    for name in ("producer_wall_s", "export_staging_wait_s", "encoding_tail_wait_s", "owner_seal_s"):
        _distribution(metrics, "producer_prefix_" + name, [owner[name] for owner in owners])
    return metrics


def log_completed_update(args, metrics, rollout_id, is_primary_rank):
    """Log completed metrics, including a final update with no subsequent train call."""
    if not is_primary_rank:
        return
    if rollout_id is None:
        # Initial baseline capture has no completed update metrics. Never assign
        # an unexpected pre-training publication to an invented rollout step.
        logger.warning("GPU-delta metrics have no trained rollout; not submitting to tracking: %s", metrics)
        return
    log_dict, step_key = namespace_metrics(
        metrics,
        trainer_model_id=args.trainer_model_id,
        step_name="rollout/step",
        step=compute_rollout_step(args, rollout_id),
    )
    logger.info("GPU-delta completed update metrics at rollout %s: %s", rollout_id, metrics)
    try:
        tracking.log(args, log_dict, step_key=step_key)
    except Exception:
        # The engines have already resumed. A reporting failure must not cause
        # the actor controller to retry an already-applied weight publication.
        logger.exception(
            "Tracking failed for completed GPU-delta target version %s", metrics.get(_PREFIX + "target_version")
        )
