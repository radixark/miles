"""Completed original-process pause and host-once metrics, without CUDA or RPCs."""

import copy
from argparse import Namespace

import pytest

from miles.backends.training_utils.weight_update import gpu_delta_metrics as metrics
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols import gpu_delta


def _activation():
    applied, resumed = [], []
    for rank in range(4):
        start = (10 + rank * 10_000) * 10**9  # Independent process clock origins.
        blocked = rank + 1
        creator = rank % 2 == 0
        timings = {name: float(rank + 1) for name in metrics._RANK_TIMINGS}
        timings.update({name: float(rank + 10) if creator else 0.0 for name in metrics._HOST_TIMINGS})
        timings.update(
            host_payload_cache_created=int(creator),
            host_outer_zstd_cpu_workers=4,
            host_payload_hash_bytes=100 if creator else 0,
            host_outer_zstd_encoded_bytes=40 if creator else 0,
            host_outer_zstd_decoded_bytes=200 if creator else 0,
        )
        receipt = {
            "identity": {
                "rank_id": str(rank),
                "pid": 100 + rank,
                "start_ticks": 12,
                "host_cache_id": f"host-{rank // 2}",
            },
            "session_id": "s",
            "cohort_digest": "c",
            "manifest_sha256": "m",
            "stream_id": "stream",
            "base_version": 0,
            "target_version": 1,
            "plan_digest": "p",
            "state": "APPLIED",
            "result": {"timings": timings},
            "scheduler_timing": {
                "clock": "monotonic_ns",
                "pause_started_ns": start,
                "reader_fence_completed_ns": start + 100_000_000,
                "resumed_ns": None,
                "blocked_s": None,
            },
        }
        applied.append(receipt)
        resumed.append(
            receipt
            | {
                "state": "RESUMED",
                "scheduler_timing": receipt["scheduler_timing"]
                | {
                    "resumed_ns": start + blocked * 10**9,
                    "blocked_s": float(blocked),
                },
            }
        )
    return {
        "receipts": applied,
        "resumed_receipts": list(reversed(resumed)),
        "coordinator_timings": {"prepare_s": 6, "apply_barrier_s": 4, "resume_barrier_s": 0.1, "activation_s": 10.1},
    }


def test_original_rank_pause_and_creator_only_host_metrics_remain_separate():
    activation = _activation()
    original = copy.deepcopy(activation)
    result = metrics.activation_metrics(activation)
    prefix = "perf/gpu_delta/"
    assert (result[prefix + "base_version"], result[prefix + "target_version"]) == (0, 1)
    assert result[prefix + "receiver_scheduler_pause_s/p50"] == 2.5
    assert result[prefix + "receiver_reader_fence_s/max"] == 0.1
    assert result[prefix + "creator_host_outer_zstd_decode_s/p50"] == 11
    assert result[prefix + "creator_host_payload_hash_bytes/sum"] == 200
    assert result[prefix + "receiver_host_shared_register_s/p50"] == 2.5
    assert result[prefix + "coordinator_activation_s"] == 10.1
    assert activation == original
    # A reused publication has no new creator work, not a zero-duration decode.
    for receipt in activation["receipts"]:
        receipt["result"]["timings"]["host_payload_cache_created"] = 0
    result = metrics.activation_metrics(activation)
    assert result[prefix + "host_cache_creators"] == 0
    assert prefix + "creator_host_outer_zstd_decode_s/p50" not in result


@pytest.mark.parametrize("corruption", ["incarnation", "open", "clock", "host_duplicate"])
def test_partial_or_mismatched_receipts_never_become_completed_pause_metrics(corruption):
    activation = _activation()
    receipt = activation["resumed_receipts"][0]
    if corruption == "incarnation":
        receipt["identity"] = receipt["identity"] | {"start_ticks": 999}
    elif corruption == "open":
        receipt["scheduler_timing"]["resumed_ns"] = None
    elif corruption == "clock":
        receipt["scheduler_timing"]["pause_started_ns"] += 1
    else:
        activation["receipts"][1]["result"]["timings"]["host_payload_cache_created"] = 1
    with pytest.raises(ValueError):
        metrics.activation_metrics(activation)


def test_gpu_delta_factory_admits_protocol_and_drains_completed_logging_rank_metrics(monkeypatch):
    protocol = get_weight_transfer_protocol(
        Namespace(
            update_weight_transfer_mode="gpu-delta",
            colocate=False,
            custom_update_weight_post_write_path=None,
        )
    )
    assert isinstance(protocol, gpu_delta.UpdateWeightFromGpuDelta)
    protocol._started = 10.0
    monkeypatch.setattr(gpu_delta.time, "monotonic", lambda: 14.0)
    monkeypatch.setattr(gpu_delta.dist, "get_rank", lambda: 7)
    protocol.after_engines_resumed()
    assert protocol.pop_metrics() == {
        "perf/gpu_delta/trainer_logging_rank": 7,
        "perf/gpu_delta/trainer_logging_rank_blocked_s": 4.0,
    }
    assert protocol.pop_metrics() == {}
