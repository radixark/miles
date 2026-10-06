"""Completed pause, shared encoded-cache and rank-owned arena metrics without CUDA."""

import ast
import copy
import logging
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.training_utils.weight_update import gpu_delta_metrics as metrics
from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol, get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols import gpu_delta
from miles.utils.timer import Timer


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
            host_encoded_cache_frames_validations=int(creator),
            host_encoded_cache_hash_bytes=100 if creator else 0,
            host_encoded_cache_hash_files=int(creator),
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
                "pid": 100 + rank,
                "start_ticks": 12,
                "host_cache_id": f"host-{rank // 2}",
            },
            "session_id": "s",
            "manifest_sha256": "m",
            "stream_id": "stream",
            "base_version": 0,
            "target_version": 1,
            "plan_digest": "p",
            "state": "RESUMED",
            "result": {"timings": timings},
            "scheduler_timing": {
                "clock": "monotonic_ns",
                "pause_started_ns": start,
                "reader_fence_completed_ns": start + 100_000_000,
                "resumed_ns": start + blocked * 10**9,
                "blocked_s": float(blocked),
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
    assert result[prefix + "receiver_host_rank_outer_zstd_validate_s/p50"] == 2.5
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
    assert result[prefix + "creator_host_encoded_cache_frames_validate_s/p50"] == 11
    assert result[prefix + "creator_host_encoded_cache_frames_validations/sum"] == 2
    assert result[prefix + "creator_host_encoded_cache_sha256_s/p50"] == 11
    assert result[prefix + "creator_host_encoded_cache_read_sha256_s/p50"] == 11
    assert result[prefix + "creator_host_encoded_cache_build_s/p50"] == 11
    assert prefix + "receiver_host_encoded_cache_frames_validate_s/p50" not in result
    assert prefix + "host_encoded_cache_capacity_bytes/sum" not in result
    assert prefix + "receiver_decoded_scratch_bytes/max" not in result
    assert result[prefix + "receiver_host_plan_cache_reused/min"] == 1
    assert activation == original
    # Reusing encoded bytes removes creator work, but every rank still decodes.
    for receipt in activation["resumed_receipts"]:
        receipt["result"]["timings"]["host_encoded_cache_created"] = 0
        receipt["result"]["timings"]["host_encoded_cache_reused"] = 1
    result = metrics.activation_metrics(activation)
    assert result[prefix + "host_encoded_cache_creators"] == 0
    assert prefix + "creator_host_encoded_cache_frames_validate_s/p50" not in result
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


class _OrdinaryProtocol(WeightTransferProtocol):
    def connect(self, *args):
        raise AssertionError("already connected")

    def send_bucket(self, bucket):
        raise AssertionError("empty fixture export")


@pytest.fixture
def actor_update(monkeypatch):
    """Execute the exact actor entry point without importing native Megatron."""
    source = Path(gpu_delta.__file__).parents[3] / "megatron_utils" / "actor.py"
    actor = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "update_weights")
    method.decorator_list = []
    parallel = SimpleNamespace(
        tp=SimpleNamespace(rank=0), is_pp_last_stage=True, effective_dp_cp=SimpleNamespace(rank=0)
    )
    namespace = {
        "UpdatableEngines": object,
        "nullcontext": nullcontext,
        "print_memory": lambda _: None,
        "get_parallel_state": lambda: parallel,
        "log_completed_update": metrics.log_completed_update,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    def run(mode="gpu-delta", rollout_id=7, primary=True, update_error=None, completed_metrics=None):
        parallel.is_pp_last_stage = primary
        args = Namespace(
            update_weight_transfer_mode=mode,
            colocate=False,
            custom_update_weight_post_write_path=None,
            debug_train_only=False,
            debug_rollout_only=False,
            offload_train=False,
            debug_skip_weight_update=False,
            ci_test=False,
            rematerialize_param_from_master_weight=False,
            trainer_model_id="actor",
            wandb_always_use_train_step=True,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            global_batch_size=8,
        )
        protocol = get_weight_transfer_protocol(args) if mode == "gpu-delta" else _OrdinaryProtocol(args)
        protocol.is_sender = False
        protocol._started = 10.0
        protocol.begin_sync = lambda *args: bool(completed_metrics) or update_error is not None
        protocol.after_base_weights = lambda: None

        def finish(_version):
            if update_error:
                raise update_error
            protocol.update_weight_metrics = dict(completed_metrics or {})

        protocol.finalize = finish
        # Exercise the real updater's final barrier, GPU-only drain and ordinary
        # None return, not a copied completion hook or synthetic updater result.
        updater = object.__new__(updater_module.WeightUpdater)
        updater.protocol = protocol
        updater.args = args
        updater.conn_status = SimpleNamespace(needs_reconnect=lambda _: False)
        updater.weight_version = 0
        updater.is_lora = False
        updater.weights_getter = lambda: {}
        updater._hf_weight_iterator = SimpleNamespace(iter_hf_weights=lambda *args, **kwargs: [])
        monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 7)
        monkeypatch.setattr(updater_module.dist, "barrier", lambda **kwargs: None)
        monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
        monkeypatch.setattr(gpu_delta.time, "monotonic", lambda: 14.0)
        instance = SimpleNamespace(
            args=args,
            weight_updater=updater,
            _last_rollout_id=rollout_id,
            _heartbeat=SimpleNamespace(bump=lambda: None),
        )
        result = namespace["update_weights"](
            instance, SimpleNamespace(rollout_engines=[], snapshot_cell_id_to_hashes={})
        )
        return result, updater

    return run


def test_final_update_logs_once_at_its_trained_rollout_without_resetting_timers(actor_update, monkeypatch):
    completed = metrics.activation_metrics(_activation())
    completed["perf/update_weights_wire_bytes"] = 123456
    wandb_submit = Mock()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(log=wandb_submit))
    monkeypatch.setattr(metrics.tracking._manager, "_backends", [metrics.tracking.WandbBackend()])
    submit = Mock(wraps=metrics.tracking.log)
    monkeypatch.setattr(metrics.tracking, "log", submit)
    reset = Mock(side_effect=AssertionError("completion logging must not reset training timers"))
    monkeypatch.setattr(Timer(), "reset", reset)

    # No train call follows this final evaluation publication.
    result, updater = actor_update(completed_metrics=completed)
    assert result == 1
    submit.assert_called_once()
    _, values = submit.call_args.args
    assert values["actor/rollout/step"] == 56  # Existing train-step conversion for rollout 7.
    assert values["actor/perf/gpu_delta/base_version"] == 0
    assert values["actor/perf/gpu_delta/target_version"] == 1
    assert values["actor/perf/update_weights_wire_bytes"] == 123456
    assert values["actor/perf/gpu_delta/trainer_logging_rank"] == 7
    assert values["actor/perf/gpu_delta/trainer_logging_rank_blocked_s"] == 4.0
    assert submit.call_args.kwargs == {"step_key": "actor/rollout/step"}
    wandb_submit.assert_called_once_with(values)
    assert updater.pop_metrics() == {}  # The next ordinary train drain cannot repeat it.
    reset.assert_not_called()


@pytest.mark.parametrize("case", ["nonprimary", "no_trained_rollout", "ordinary"])
def test_completion_logging_preserves_rank_startup_and_other_protocol_semantics(
    actor_update, monkeypatch, case, caplog
):
    completed = metrics.activation_metrics(_activation())
    submit = Mock()
    monkeypatch.setattr(metrics.tracking, "log", submit)
    overrides = {
        "nonprimary": {"primary": False},
        "no_trained_rollout": {"rollout_id": None},
        "ordinary": {"mode": "disk-delta"},
    }
    with caplog.at_level(logging.INFO):
        _, updater = actor_update(completed_metrics=completed, **overrides[case])
    submit.assert_not_called()
    assert updater.pop_metrics() == (completed if case == "ordinary" else {})
    if case == "no_trained_rollout":
        assert "initial sync completed before the first rollout" in caplog.text


def test_failed_update_is_not_reported_and_tracking_failure_cannot_retry_a_completed_update(
    actor_update, monkeypatch, caplog
):
    submit = Mock(side_effect=RuntimeError("tracking unavailable"))
    monkeypatch.setattr(metrics.tracking, "log", submit)
    with pytest.raises(RuntimeError, match="update failed"):
        actor_update(update_error=RuntimeError("update failed"))
    submit.assert_not_called()
    actor_update(rollout_id=None)  # Initial baseline capture has no completed summary.
    submit.assert_not_called()
    with caplog.at_level(logging.ERROR):
        result, updater = actor_update(completed_metrics=metrics.activation_metrics(_activation()))
    assert result == 1
    assert updater.pop_metrics() == {}
    assert "Tracking failed for completed GPU-delta target version 1" in caplog.text
