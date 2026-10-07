"""Completed GPU-delta timing summaries, without additional distributed work.

Receiver clocks are compared only within the same original process. Nested
spans remain separate. Encoded-cache work is sampled once per engine-host;
outer decode and original host arenas are counted for every receiver rank.
"""

import logging
from statistics import median

from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)
_PREFIX = "perf/gpu_delta/"
_RANK_TIMINGS = (
    "host_prepare_s",
    "paused_apply_host_wall_s",
    "host_encoded_cache_wait_s",
    "host_rank_prepare_s",
    "host_rank_outer_zstd_decode_s",
    "host_rank_outer_zstd_worker_decode_sum_s",
    "host_rank_allocation_s",
    "host_manifest_read_parse_s",
    "host_plan_validate_s",
    "host_tensor_prepare_s",
    "host_raw_pack_s",
    "host_metadata_prepare_s",
    "host_metadata_wait_s",
    "paused_setup_host_s",
    "paused_apply_tune_s",
    "host_matrix_enqueue_s",
    "host_apply_completion_wait_s",
)
_CACHE_TIMINGS = (
    "host_encoded_cache_read_worker_sum_s",
    "host_encoded_cache_sha256_worker_sum_s",
    "host_encoded_cache_read_hash_s",
    "host_encoded_cache_build_s",
    "host_encoded_cache_allocation_s",
)
_RANK_COUNTERS = (
    "host_plan_cache_reused",
    "host_batch_plan_reused",
    "host_rank_mapping_reused",
    "host_rank_capacity_generation",
    "host_rank_cpu_workers",
    "host_encoded_cache_created",
    "host_encoded_cache_reused",
    "compressed_batches",
    "compressed_tensors",
    "layers_per_batch",
    "de_host_input_bytes",
    "raw_h2d_bytes",
    "decoded_buffers",
    "decoded_scratch_bytes",
    "decoder_workspace_bytes",
    "decoder_metadata_uploads",
    "decoder_metadata_h2d_bytes",
    "apply_metadata_h2d_bytes",
    "decoded_zero_ranges",
    "decoded_zero_bytes",
    "apply_tuned_batches",
    "apply_tune_cache_hits",
    "apply_tune_skipped_batches",
    "apply_tune_bytes",
)
_RANK_WORK_TOTALS = (
    "host_rank_outer_zstd_encoded_bytes",
    "host_rank_outer_zstd_decoded_bytes",
    "host_rank_outer_zstd_tensors",
    "host_rank_outer_zstd_frames",
    "host_rank_allocation_calls",
    "host_rank_allocation_bytes",
    "host_rank_arena_bytes",
    "host_rank_capacity_bytes",
)
_CACHE_WORK_TOTALS = (
    "host_encoded_cache_hash_bytes",
    "host_encoded_cache_hash_files",
    "host_encoded_cache_allocation_calls",
    "host_encoded_cache_allocation_bytes",
)


def _distribution(metrics, name, values):
    if values:
        for statistic, value in (("min", min(values)), ("p50", median(values)), ("max", max(values))):
            metrics[_PREFIX + name + "/" + statistic] = value


def activation_metrics(activation):
    """Reduce completed resume receipts; never poll or synchronize devices."""
    rows = activation["resumed_receipts"]
    metrics = {
        _PREFIX + "receiver_ranks": len(rows),
        _PREFIX + "base_version": rows[0]["base_version"],
        _PREFIX + "target_version": rows[0]["target_version"],
    }
    fences, pauses = [], []
    for receipt in rows:
        timing = receipt["scheduler_timing"]
        start, fence, end = (timing[key] for key in ("pause_started_ns", "reader_fence_completed_ns", "resumed_ns"))
        if not 0 <= start <= fence <= end:
            raise ValueError("GPU-delta scheduler timing interval is incomplete or inconsistent")
        fences.append((fence - start) / 1e9)
        pauses.append((end - start) / 1e9)
    _distribution(metrics, "receiver_reader_fence_s", fences)
    _distribution(metrics, "receiver_scheduler_pause_s", pauses)
    timings = [row["result"]["timings"] for row in rows]
    for name in _RANK_TIMINGS + _RANK_COUNTERS + _RANK_WORK_TOTALS:
        # Optional profiling spans are emitted only with complete rank coverage.
        if all(name in timing for timing in timings):
            values = [timing[name] for timing in timings]
            _distribution(metrics, "receiver_" + name, values)
            if name in _RANK_WORK_TOTALS:
                metrics[_PREFIX + "receiver_" + name + "/sum"] = sum(values)
    if all("h2d_bytes" in row["result"] for row in rows):
        _distribution(metrics, "receiver_h2d_bytes", [row["result"]["h2d_bytes"] for row in rows])
    caches = {}
    for receipt, timing in zip(rows, timings, strict=True):
        cache = (receipt["identity"]["engine_id"], receipt["identity"]["host_cache_id"])
        caches.setdefault(cache, timing)
    creators = [row for row in timings if row["host_encoded_cache_created"] == 1]
    metrics.update(
        {_PREFIX + "receiver_host_encoded_caches": len(caches), _PREFIX + "host_encoded_cache_creators": len(creators)}
    )
    for name in _CACHE_TIMINGS:
        if all(name in row for row in creators):
            _distribution(metrics, "creator_" + name, [row[name] for row in creators])
    for name in _CACHE_WORK_TOTALS:
        if all(name in row for row in creators):
            metrics[_PREFIX + "creator_" + name + "/sum"] = sum(row[name] for row in creators)
    # Encoded files are physically shared within each engine-host cache; their
    # retained capacity still counts when this publication has no new creator.
    # Original DE arenas above are separate physical storage on every rank.
    for name in ("host_encoded_cache_capacity_bytes", "host_encoded_cache_capacity_generation"):
        if all(name in row for row in timings):
            values = [timing[name] for timing in caches.values()]
            _distribution(metrics, name, values)
            if name == "host_encoded_cache_capacity_bytes":
                metrics[_PREFIX + name + "/sum"] = sum(values)
    engine_timings = activation["engine_timings"]
    metrics[_PREFIX + "receiver_engines"] = len(engine_timings)
    for name in ("prepare_s", "apply_s", "resume_s", "activation_s"):
        _distribution(metrics, "engine_coordinator_" + name, [row[name] for row in engine_timings])
    metrics[_PREFIX + "coordinator_activation_s"] = activation["coordinator_timings"]["activation_s"]
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
        # An opt-in initial delta precedes any trained rollout. Report it without
        # assigning an invented rollout step to the tracking series.
        logger.info("GPU-delta initial sync completed before the first rollout: %s", metrics)
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
