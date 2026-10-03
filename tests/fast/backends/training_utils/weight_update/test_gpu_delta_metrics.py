"""Completed original-process pause and host-once metrics, without CUDA or RPCs."""

import ast
import copy
import logging
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.training_utils.weight_update import gpu_delta_metrics as metrics
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols import gpu_delta
from miles.utils.timer import Timer


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


@pytest.fixture
def actor_update():
    """Execute the exact actor entry point without importing native Megatron."""
    source = Path(gpu_delta.__file__).parents[3] / "megatron_utils" / "actor.py"
    actor = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "update_weights")
    method.decorator_list = []
    parallel = SimpleNamespace(tp=SimpleNamespace(rank=0), is_pp_last_stage=True, effective_dp_cp=SimpleNamespace(rank=0))
    namespace = {
        "UpdatableEngines": object,
        "nullcontext": nullcontext,
        "print_memory": lambda _: None,
        "get_parallel_state": lambda: parallel,
        "log_completed_update": metrics.log_completed_update,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    def run(*, mode="gpu-delta", rollout_id=7, primary=True, update_error=None, completed_metrics=None):
        parallel.is_pp_last_stage = primary
        args = Namespace(
            update_weight_transfer_mode=mode,
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
        protocol = gpu_delta.UpdateWeightFromGpuDelta(args)

        def update():
            if update_error:
                raise update_error
            protocol.update_weight_metrics = dict(completed_metrics or {})

        updater = SimpleNamespace(
            conn_status=SimpleNamespace(needs_reconnect=lambda _: False),
            update_weights=update,
            pop_metrics=protocol.pop_metrics,
            weight_version=1,
        )
        instance = SimpleNamespace(
            args=args, weight_updater=updater, _last_rollout_id=rollout_id, _heartbeat=SimpleNamespace(bump=lambda: None)
        )
        result = namespace["update_weights"](
            instance, SimpleNamespace(rollout_engines=[], snapshot_cell_id_to_hashes={})
        )
        return result, updater

    return run


def test_final_update_logs_once_at_its_trained_rollout_without_resetting_timers(actor_update, monkeypatch):
    completed = metrics.activation_metrics(_activation())
    submit = Mock()
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
    assert submit.call_args.kwargs == {"step_key": "actor/rollout/step"}
    assert updater.pop_metrics() == {}  # The next ordinary train drain cannot repeat it.
    reset.assert_not_called()


@pytest.mark.parametrize("case", ["nonprimary", "no_trained_rollout", "ordinary"])
def test_completion_logging_preserves_rank_startup_and_other_protocol_semantics(actor_update, monkeypatch, case, caplog):
    completed = metrics.activation_metrics(_activation())
    submit = Mock()
    monkeypatch.setattr(metrics.tracking, "log", submit)
    overrides = {
        "nonprimary": {"primary": False},
        "no_trained_rollout": {"rollout_id": None},
        "ordinary": {"mode": "disk-delta"},
    }
    with caplog.at_level(logging.WARNING):
        _, updater = actor_update(completed_metrics=completed, **overrides[case])
    submit.assert_not_called()
    assert updater.pop_metrics() == (completed if case == "ordinary" else {})
    if case == "no_trained_rollout":
        assert "no trained rollout" in caplog.text


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
