"""Completed pause and physical/shared host-storage accounting without CUDA."""

import copy

import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta import metrics


def _activation():
    resumed = []
    for rank in range(4):
        start = (10 + rank * 10_000) * 10**9  # Independent process clock origins.
        blocked = rank + 1
        creator = rank % 2 == 0
        timings = {name: float(rank + 1) for name in metrics._RANK_TIMINGS}
        timings.update({name: float(rank + 10) if creator else 0.0 for name in metrics._CACHE_TIMINGS})
        timings.update(
            host_plan_cache_reused=1,
            host_encoded_cache_created=int(creator),
            host_encoded_cache_reused=int(not creator),
            host_encoded_cache_hash_bytes=100 if creator else 0,
            host_encoded_cache_hash_files=int(creator),
            host_encoded_cache_read_hash_s=1 + rank if creator else 0,
            host_encoded_cache_read_worker_sum_s=3 + rank if creator else 0,
            host_encoded_cache_sha256_worker_sum_s=5 + rank if creator else 0,
            host_rank_cpu_workers=2 * (rank + 1),
            host_rank_outer_zstd_encoded_bytes=40 * (rank + 1),
            host_rank_outer_zstd_decoded_bytes=200 * (rank + 1),
            host_rank_outer_zstd_tensors=rank + 1,
            host_rank_outer_zstd_frames=2 * (rank + 1),
        )
        receipt = {
            "identity": {
                "rank_id": str(rank),
                "engine_id": f"engine-{rank // 2}",
                "host_cache_id": f"host-{rank // 2}",
            },
            "base_version": 0,
            "target_version": 1,
            "result": {"timings": timings},
            "scheduler_timing": {
                "pause_started_ns": start,
                "reader_fence_completed_ns": start + 100_000_000,
                "resumed_ns": start + blocked * 10**9,
            },
        }
        resumed.append(receipt)
    return {
        "resumed_receipts": list(reversed(resumed)),
        "coordinator_timings": {"activation_s": 10.1},
        "engine_timings": [
            {"engine_id": "engine-0", "prepare_s": 3, "apply_s": 2, "resume_s": 0.1, "activation_s": 5.1},
            {"engine_id": "engine-1", "prepare_s": 6, "apply_s": 4, "resume_s": 0.1, "activation_s": 10.1},
        ],
    }


def test_original_rank_pause_and_creator_only_cache_metrics_remain_separate():
    activation = _activation()
    original = copy.deepcopy(activation)
    result = metrics.activation_metrics(activation)
    prefix = "perf/gpu_delta/"
    assert (result[prefix + "base_version"], result[prefix + "target_version"]) == (0, 1)
    assert result[prefix + "receiver_scheduler_pause_s/p50"] == 2.5
    assert result[prefix + "receiver_reader_fence_s/max"] == 0.1
    assert result[prefix + "receiver_host_rank_outer_zstd_decode_s/p50"] == 2.5
    assert result[prefix + "receiver_host_rank_outer_zstd_worker_decode_sum_s/p50"] == 2.5
    assert result[prefix + "receiver_host_rank_cpu_workers/p50"] == 5
    assert result[prefix + "receiver_host_rank_outer_zstd_encoded_bytes/sum"] == 400
    assert result[prefix + "receiver_host_rank_outer_zstd_decoded_bytes/sum"] == 2000
    assert result[prefix + "receiver_host_rank_outer_zstd_tensors/sum"] == 10
    assert result[prefix + "receiver_host_rank_outer_zstd_frames/sum"] == 20
    assert result[prefix + "creator_host_encoded_cache_hash_bytes/sum"] == 200
    assert result[prefix + "creator_host_encoded_cache_hash_files/sum"] == 2
    assert result[prefix + "receiver_host_metadata_prepare_s/p50"] == 2.5
    assert result[prefix + "receiver_host_encoded_cache_wait_s/p50"] == 2.5
    assert result[prefix + "receiver_host_rank_prepare_s/p50"] == 2.5
    assert result[prefix + "coordinator_activation_s"] == 10.1
    assert result[prefix + "engine_coordinator_prepare_s/p50"] == 4.5
    assert result[prefix + "engine_coordinator_activation_s/max"] == 10.1
    assert result[prefix + "receiver_engines"] == 2
    assert prefix + "coordinator_apply_barrier_s" not in result
    assert result[prefix + "creator_host_encoded_cache_read_worker_sum_s/p50"] == 4
    assert result[prefix + "creator_host_encoded_cache_sha256_worker_sum_s/p50"] == 6
    assert result[prefix + "creator_host_encoded_cache_read_hash_s/p50"] == 2
    assert result[prefix + "creator_host_encoded_cache_build_s/p50"] == 11
    assert prefix + "host_encoded_cache_capacity_bytes/sum" not in result
    assert prefix + "receiver_decoded_scratch_bytes/max" not in result
    assert result[prefix + "receiver_host_plan_cache_reused/min"] == 1
    assert activation == original
    # Reusing encoded bytes removes creator work, but ranks still prepare local payloads.
    for receipt in activation["resumed_receipts"]:
        receipt["result"]["timings"]["host_encoded_cache_created"] = 0
        receipt["result"]["timings"]["host_encoded_cache_reused"] = 1
    result = metrics.activation_metrics(activation)
    assert result[prefix + "host_encoded_cache_creators"] == 0
    assert result[prefix + "receiver_host_rank_outer_zstd_decode_s/p50"] == 2.5
    assert result[prefix + "receiver_host_rank_outer_zstd_encoded_bytes/sum"] == 400


@pytest.mark.parametrize("warm", [False, True])
def test_encoded_capacity_counts_once_while_rank_arenas_and_gpu_scratch_remain_per_rank(warm):
    activation = _activation()
    for receipt in activation["resumed_receipts"]:
        rank = int(receipt["identity"]["rank_id"])
        host = rank // 2 + 1
        creator = rank % 2 == 0
        timings = receipt["result"]["timings"]
        timings.update(
            host_rank_arena_bytes=600 * (rank + 1),
            host_rank_capacity_bytes=1024 * (rank + 1),
            host_rank_capacity_generation=rank + 1,
            host_encoded_cache_capacity_bytes=512 * host,
            host_encoded_cache_capacity_generation=host,
            de_host_input_bytes=1200 * (rank + 1),
            decoded_buffers=2,
            decoded_scratch_bytes=4096 * (rank + 1),
            decoder_workspace_bytes=128 * (rank + 1),
            decoder_metadata_uploads=2,
            decoder_metadata_h2d_bytes=64,
            apply_metadata_h2d_bytes=32,
            raw_h2d_bytes=16,
            host_rank_mapping_reused=int(warm),
            host_rank_allocation_s=0 if warm else 0.5 * (rank + 1),
            host_encoded_cache_allocation_s=0 if warm or not creator else 0.25 * host,
            host_rank_allocation_calls=int(not warm),
            host_rank_allocation_bytes=1024 * (rank + 1) if not warm else 0,
            host_encoded_cache_allocation_calls=int(creator and not warm),
            host_encoded_cache_allocation_bytes=512 * host if creator and not warm else 0,
        )
        receipt["result"]["h2d_bytes"] = 112
    result = metrics.activation_metrics(activation)
    prefix = "perf/gpu_delta/"
    assert result[prefix + "receiver_host_encoded_caches"] == 2
    assert result[prefix + "host_encoded_cache_creators"] == 2
    assert result[prefix + "receiver_host_rank_arena_bytes/sum"] == 6000
    assert result[prefix + "receiver_host_rank_capacity_bytes/sum"] == 10240
    assert result[prefix + "receiver_host_rank_capacity_bytes/p50"] == 2560
    assert result[prefix + "receiver_host_rank_capacity_generation/p50"] == 2.5
    assert result[prefix + "host_encoded_cache_capacity_bytes/sum"] == 1536
    assert result[prefix + "host_encoded_cache_capacity_generation/p50"] == 1.5
    assert prefix + "host_encoded_cache_capacity_generation/sum" not in result
    assert result[prefix + "receiver_de_host_input_bytes/p50"] == 3000
    assert result[prefix + "receiver_decoded_scratch_bytes/p50"] == 10240
    assert prefix + "receiver_decoded_scratch_bytes/sum" not in result
    assert result[prefix + "receiver_decoded_buffers/min"] == 2
    assert result[prefix + "receiver_decoder_workspace_bytes/max"] == 512
    assert result[prefix + "receiver_decoder_metadata_h2d_bytes/min"] == 64
    assert result[prefix + "receiver_apply_metadata_h2d_bytes/max"] == 32
    assert result[prefix + "receiver_h2d_bytes/p50"] == 112
    assert result[prefix + "receiver_host_rank_mapping_reused/max"] == int(warm)
    assert result[prefix + "receiver_host_rank_allocation_s/p50"] == (0 if warm else 1.25)
    assert result[prefix + "creator_host_encoded_cache_allocation_s/p50"] == (0 if warm else 0.375)
    assert result[prefix + "receiver_host_rank_allocation_calls/sum"] == (0 if warm else 4)
    assert result[prefix + "creator_host_encoded_cache_allocation_calls/sum"] == (0 if warm else 2)
    assert result[prefix + "receiver_host_rank_allocation_bytes/sum"] == (0 if warm else 10240)
    assert result[prefix + "creator_host_encoded_cache_allocation_bytes/sum"] == (0 if warm else 1536)

    # Retained capacities count even without new shared-cache construction.
    for receipt in activation["resumed_receipts"]:
        receipt["result"]["timings"]["host_encoded_cache_created"] = 0
    result = metrics.activation_metrics(activation)
    assert result[prefix + "host_encoded_cache_capacity_bytes/sum"] == 1536
    assert result[prefix + "receiver_host_rank_capacity_bytes/sum"] == 10240
    assert prefix + "creator_host_encoded_cache_allocation_s/p50" not in result

    # Both engines share prepared pages; DE additionally keeps private inputs.
    # Only one owner per engine reports outer-decode work and holds a pool.
    for receipt in activation["resumed_receipts"]:
        rank = int(receipt["identity"]["rank_id"])
        creator = rank % 2 == 0
        shared_capacity = 8192 if rank < 2 else 6144
        receipt["result"]["timings"].update(
            host_rank_capacity_bytes=8192 if rank < 2 else 1024 * (rank + 1),
            host_rank_private_capacity_bytes=0 if rank < 2 else 1024 * (rank + 1),
            host_rank_cpu_workers=32 if creator else 0,
            host_rank_outer_zstd_decoded_bytes=4096 if creator else 0,
            host_prepared_cache_created=int(creator),
            host_prepared_cache_capacity_bytes=shared_capacity,
            host_prepared_cache_capacity_generation=1,
            host_prepared_cache_arena_bytes=4096,
            host_prepared_cache_layout_s=0.1 if creator else 0,
            host_prepared_cache_build_s=0.2,
            host_prepared_cache_allocation_s=0.3 if not warm and creator else 0,
            host_prepared_cache_allocation_calls=int(not warm and creator),
            host_prepared_cache_allocation_bytes=shared_capacity if not warm and creator else 0,
        )
    result = metrics.activation_metrics(activation)
    assert result[prefix + "receiver_host_rank_capacity_bytes/sum"] == 23552
    assert result[prefix + "receiver_host_rank_private_capacity_bytes/sum"] == 7168
    assert result[prefix + "host_prepared_cache_capacity_bytes/sum"] == 14336
    assert result[prefix + "host_input_capacity_bytes/sum"] == 21504
    assert result[prefix + "creator_host_prepared_cache_layout_s/min"] == 0.1
    assert result[prefix + "creator_host_prepared_cache_allocation_bytes/sum"] == (0 if warm else 14336)
    assert result[prefix + "receiver_host_rank_outer_zstd_decoded_bytes/sum"] == 8192
    del activation["resumed_receipts"][0]["result"]["timings"]["host_prepared_cache_capacity_bytes"]
    assert prefix + "host_input_capacity_bytes/sum" not in metrics.activation_metrics(activation)


def test_kernel_preparation_and_metadata_require_complete_rank_coverage():
    activation = _activation()
    for receipt in activation["resumed_receipts"]:
        receipt["result"]["timings"].update(
            apply_kernel_cache_misses=0,
            apply_kernel_cache_reused=1,
            apply_descriptor_h2d_bytes=128,
            apply_metadata_h2d_bytes=512,
        )
    result = metrics.activation_metrics(activation)
    prefix = "perf/gpu_delta/receiver_"
    assert result[prefix + "host_apply_kernel_prepare_s/max"] == 4
    assert result[prefix + "apply_kernel_cache_misses/max"] == 0
    assert result[prefix + "apply_kernel_cache_reused/min"] == 1
    assert result[prefix + "apply_descriptor_h2d_bytes/max"] == 128
    assert result[prefix + "apply_metadata_h2d_bytes/max"] == 512
    # An older/missing rank cannot be silently reported as a zero module-cache miss count.
    del activation["resumed_receipts"][0]["result"]["timings"]["apply_kernel_cache_misses"]
    result = metrics.activation_metrics(activation)
    assert prefix + "apply_kernel_cache_misses/max" not in result
    assert result[prefix + "apply_kernel_cache_reused/min"] == 1
