"""Completed GPU-delta timing summaries, without additional distributed work.

Receiver clocks are compared only within the same original process. Nested
spans remain separate. Shared-host work is sampled once from each cache creator,
not from the zero counters on ranks that attach to its arena.
"""

import math
from statistics import median

from miles.utils.gpu_delta_publication import canonical_json

_PREFIX = "perf/gpu_delta/"
_RANK_TIMINGS = (
    "host_prepare_s",
    "paused_apply_host_wall_s",
    "host_shared_register_s",
    "host_payload_cache_wait_s",
    "host_manifest_read_parse_s",
    "host_plan_validate_s",
    "host_frames_validate_s",
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
    "host_payload_read_s",
    "host_payload_sha256_s",
    "host_outer_zstd_validate_s",
    "host_outer_zstd_decode_s",
    "host_outer_zstd_worker_decode_sum_s",
    "host_shared_build_s",
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
    metrics = {_PREFIX + "receiver_ranks": len(rows)}
    _distribution(metrics, "receiver_reader_fence_s", [row[1] for row in rows])
    _distribution(metrics, "receiver_scheduler_pause_s", [row[2] for row in rows])
    timings = [row[0]["result"]["timings"] for row in rows]
    for name in _RANK_TIMINGS:
        # Optional profiling spans are emitted only with complete rank coverage.
        if all(name in timing for timing in timings):
            _distribution(metrics, "receiver_" + name, [timing[name] for timing in timings])
    hosts = {}
    for (receipt, _, _), timing in zip(rows, timings, strict=True):
        hosts.setdefault(receipt["identity"]["host_cache_id"], []).append(timing)
    creators = []
    for host_rows in hosts.values():
        created = [row for row in host_rows if row["host_payload_cache_created"] == 1]
        if len(created) > 1:
            raise ValueError("Multiple shared payload creators on one host")
        creators.extend(created)
    metrics.update({_PREFIX + "receiver_hosts": len(hosts), _PREFIX + "host_cache_creators": len(creators)})
    for name in _HOST_TIMINGS:
        _distribution(metrics, "creator_" + name, [row[name] for row in creators])
    _distribution(metrics, "creator_cpu_workers", [row["host_outer_zstd_cpu_workers"] for row in creators])
    for name in ("host_payload_hash_bytes", "host_outer_zstd_encoded_bytes", "host_outer_zstd_decoded_bytes"):
        metrics[_PREFIX + "creator_" + name + "/sum"] = sum(_number(row[name]) for row in creators)
    for name, value in activation["coordinator_timings"].items():
        metrics[_PREFIX + "coordinator_" + name] = _number(value)
    return metrics


def producer_metrics(owners):
    """The existing owner gather observes a prefix, before publication/activation."""
    metrics = {_PREFIX + "producer_ranks": len(owners)}
    for name in ("producer_wall_s", "export_staging_wait_s", "encoding_tail_wait_s", "owner_seal_s"):
        _distribution(metrics, "producer_prefix_" + name, [owner[name] for owner in owners])
    return metrics
