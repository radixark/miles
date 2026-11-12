"""GPU-delta checkpoints, eval snapshots and completed updates share the actor's ordinary boundaries."""

import inspect
import logging
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.fast.backends.megatron_utils import test_shared_ppo_lifecycle
from tests.fast.backends.training_utils.weight_update.gpu_delta.test_gpu_delta_metrics import _activation
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol, get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols.gpu_delta import metrics
from miles.backends.training_utils.weight_update.protocols.gpu_delta import protocol as gpu_delta
from miles.ray.train_actor import WeightUpdateOutput
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.timer import Timer

actor_module = test_shared_ppo_lifecycle.actor_module


def _checkpoint_worker(actor_module, **overrides):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = make_trainer_args(
        **(
            dict(
                async_save=True,
                custom_megatron_post_save_hook_path=None,
                debug_rollout_only=False,
                save_hf=None,
                save="/checkpoints",
                update_weight_transfer_mode="gpu-delta",
                offload_train=False,
            )
            | overrides
        )
    )
    worker.role = "actor"
    worker._asleep = True
    worker._heartbeat = Mock()
    worker.model, worker.optimizer, worker.opt_param_scheduler = object(), object(), object()
    worker.snapshot_publisher = None
    worker.adapter_slots = None
    worker._schedule_first_rollout = 0
    worker.weight_updater = Mock()
    worker._get_actor_weight_version = Mock(return_value=42)
    return worker


@pytest.mark.parametrize("async_save,force_sync", [(False, False), (True, False), (True, True)])
def test_checkpoint_companion_becomes_ready_only_after_megatron_finishes(
    actor_module, monkeypatch, tmp_path, async_save, force_sync
):
    worker = _checkpoint_worker(actor_module, async_save=async_save, save=str(tmp_path))
    events, pending = [], []
    megatron_done = False
    checkpoint_dir = str(tmp_path / "iter_0000006")

    def save(*_args, **_kwargs):
        nonlocal megatron_done
        events.append("save")
        megatron_done = not async_save

    def prepare(path, rollout_id, weight_version):
        assert path == checkpoint_dir and rollout_id == 6 and weight_version == 42
        assert events[-1] == "save"
        pending.append(rollout_id)
        events.append("delta")

    def complete_megatron(**_kwargs):
        nonlocal megatron_done
        megatron_done = True

    def complete_delta():
        if pending:
            assert megatron_done
            events.append(("ready", pending.pop()))

    monkeypatch.setattr(actor_module, "save", save)
    monkeypatch.setattr(actor_module, "maybe_finalize_async_save", complete_megatron)
    monkeypatch.setattr(actor_module, "get_checkpoint_name", lambda *_args, **_kwargs: checkpoint_dir)
    worker.weight_updater.save_checkpoint_delta.side_effect = prepare
    worker.weight_updater.finish_checkpoint_delta.side_effect = complete_delta

    # The final checkpoint needs a companion even when no subsequent weight sync runs.
    worker.save_model(6, force_sync=force_sync)
    expected = ["save", "delta"]
    if force_sync or not async_save:
        expected.append(("ready", 6))
    assert events == expected
    worker._finalize_pending_async_save()
    assert events == ["save", "delta", ("ready", 6)]
    worker._get_actor_weight_version.assert_called_once_with()
    worker.weight_updater.update_weights.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_post_save_hook_finalizes_companions_on_all_ranks(actor_module, monkeypatch, rank):
    worker = _checkpoint_worker(
        actor_module, custom_megatron_post_save_hook_path=CustomFunctionConfig(path="test.hook")
    )
    events = []
    monkeypatch.setattr(actor_module, "save", lambda *_args, **_kwargs: events.append("save"))
    monkeypatch.setattr(actor_module, "maybe_finalize_async_save", lambda **_kwargs: events.append("finalize"))
    monkeypatch.setattr(actor_module, "get_checkpoint_name", lambda *_args, **_kwargs: "/checkpoints/iter_6")
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: rank)
    hook = Mock(side_effect=lambda *_args: events.append("hook"))
    monkeypatch.setattr("miles.utils.function_registry.load_function", lambda _path: hook)
    worker.weight_updater.save_checkpoint_delta.side_effect = lambda *_args: events.append("delta")
    worker.weight_updater.finish_checkpoint_delta.side_effect = lambda: events.append("finish_delta")

    worker.save_model(6)

    assert events == ["finalize", "finish_delta", "save", "delta", "finalize", "finish_delta"] + (
        ["hook"] if rank == 0 else []
    )


def test_failed_checkpoint_delta_releases_temporary_offload_runtime(actor_module, monkeypatch):
    worker = _checkpoint_worker(actor_module, offload_train=True)
    worker.weight_updater.save_checkpoint_delta.side_effect = RuntimeError("delta encode failed")
    events = []

    @contextmanager
    def disable():
        events.append("disable")
        try:
            yield
        finally:
            events.append("enable")

    saver = SimpleNamespace(disable=disable)
    monkeypatch.setattr(actor_module, "torch_memory_saver", saver)
    monkeypatch.setattr(actor_module, "save", Mock())
    monkeypatch.setattr(actor_module, "maybe_finalize_async_save", Mock())
    monkeypatch.setattr(actor_module, "get_checkpoint_name", lambda *_args, **_kwargs: "/checkpoints/iter_6")
    monkeypatch.setattr(actor_module, "reload_process_groups", lambda: events.append("reload"))
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda: events.append("destroy"))

    with pytest.raises(RuntimeError, match="delta encode failed"):
        worker.save_model(6, force_sync=True)

    assert events == ["disable", "reload", "destroy", "enable"]
    # Only the previous save was finalized; a failed companion is never announced ready.
    worker.weight_updater.finish_checkpoint_delta.assert_called_once_with()


@pytest.mark.parametrize("mode,eval_model", [("broadcast", True), ("gpu-delta", False)])
def test_other_eval_modes_keep_full_hf_export(actor_module, mode, eval_model):
    worker = _checkpoint_worker(
        actor_module,
        update_weight_transfer_mode=mode,
        sglang=SimpleNamespace(models=[SimpleNamespace(name="eval")] if eval_model else []),
    )
    worker.export_hf = Mock()

    inspect.unwrap(actor_module.MegatronTrainRayActor.export_eval_snapshot)(worker, 9, "/snapshot")

    worker.export_hf.assert_called_once_with(9, "/snapshot")


@pytest.mark.parametrize("offload_train,fails", [(False, False), (True, False), (True, True)])
def test_eval_snapshot_preserves_identity_completion_and_allocator_lifetime(
    actor_module, monkeypatch, tmp_path, offload_train, fails
):
    worker = _checkpoint_worker(
        actor_module,
        offload_train=offload_train,
        sglang=SimpleNamespace(models=[SimpleNamespace(name="eval")]),
    )
    worker._last_rollout_id = 7
    monkeypatch.setattr(actor_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(actor_module, "reload_process_groups", lambda: pytest.fail("dispatcher already onloaded"))
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda: pytest.fail("dispatcher owns offload"))
    events = []

    @contextmanager
    def disable():
        events.append("disable")
        try:
            yield
        finally:
            events.append("restore")

    monkeypatch.setattr(actor_module, "torch_memory_saver", SimpleNamespace(disable=disable))
    path = tmp_path / "snapshot"

    def publish(directory, rollout_id, version):
        assert (rollout_id, version) == (7, 42)
        assert events == (["disable"] if offload_train else [])
        assert not (directory / ".complete").exists()
        if fails:
            raise RuntimeError("publication failed")
        (directory / "gpu_delta").mkdir()
        (directory / "gpu_delta" / "READY.json").write_text("{}")

    worker.weight_updater.export_gpu_delta_snapshot.side_effect = publish
    export = inspect.unwrap(actor_module.MegatronTrainRayActor.export_eval_snapshot)
    if fails:
        with pytest.raises(RuntimeError, match="publication failed"):
            export(worker, 9, str(path))
        assert not path.exists()
    else:
        export(worker, 9, str(path))
        assert (path / "gpu_delta" / "READY.json").exists()
        assert (path / ".complete").exists()
    assert events == (["disable", "restore"] if offload_train else [])


class _OrdinaryProtocol(WeightTransferProtocol):
    def connect(self, *args):
        raise AssertionError("already connected")

    def send_bucket(self, bucket):
        raise AssertionError("empty fixture export")


@pytest.fixture
def actor_update(actor_module, monkeypatch):
    """Run the real actor entry point and updater; only the protocol's transfer phases are stubbed."""

    def run(mode="gpu-delta", rollout_id=7, primary=True, update_error=None, completed_metrics=None):
        worker = test_shared_ppo_lifecycle._weight_update_worker(actor_module, monkeypatch)
        worker.model[0].model_companion.weight_version.fill_(17)
        worker._last_rollout_id = rollout_id
        args = worker.args
        args.update_weight_transfer_mode = mode
        args.trainer_model_id = "actor"
        args.wandb_always_use_train_step = True
        args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size = 16, 4, 8
        protocol = get_weight_transfer_protocol(args) if mode == "gpu-delta" else _OrdinaryProtocol(args)
        protocol.is_sender = False
        protocol._started = 10.0
        protocol._is_metrics_rank = primary  # Stored by connect in production.

        def begin_sync(weight_version, iter_buckets, rollout_id=None):
            protocol._rollout_id = rollout_id
            return bool(completed_metrics) or update_error is not None

        def finish(_version):
            if update_error:
                raise update_error
            protocol.update_weight_metrics = dict(completed_metrics or {})

        protocol.begin_sync = begin_sync
        protocol.after_base_weights = lambda: None
        protocol.finalize = finish
        # Exercise the real updater's final barrier, the protocol's completed-update
        # logging and the ordinary bool return, not a copied hook or synthetic result.
        updater = object.__new__(updater_module.WeightUpdater)
        updater.protocol = protocol
        updater._uses_gpu_delta = mode == "gpu-delta"
        updater.args = args
        updater.conn_status = SimpleNamespace(needs_reconnect=lambda _: False)
        updater._receiver_pools = updater_module.ReceiverPools()
        updater._registered_adapters = set()
        updater.is_lora = False
        updater.weights_getter = lambda: {}
        updater._hf_weight_iterator = SimpleNamespace(iter_hf_weights=lambda *args, **kwargs: [])
        worker.weight_updater = updater
        monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 7)
        monkeypatch.setattr(updater_module.dist, "get_world_size", lambda **kwargs: 8)
        monkeypatch.setattr(
            updater_module.dist,
            "all_gather_object",
            lambda values, value, **kwargs: values.__setitem__(slice(None), [value] * len(values)),
        )
        monkeypatch.setattr(updater_module.dist, "barrier", lambda **kwargs: None)
        monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
        monkeypatch.setattr(gpu_delta.time, "monotonic", lambda: 14.0)
        result = worker.update_weights(
            SimpleNamespace(
                rollout_engines=[], engine_cell_ids=[], snapshot_cell_id_to_hashes={}, receiver_pool="rollout"
            ),
            debug_weight_update_id="metrics-update",
            rollout_id=rollout_id,
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

    # No train call follows this final rollout publication.
    result, updater = actor_update(completed_metrics=completed)
    assert result == WeightUpdateOutput(weight_version=17, failed_cell_ids=())
    submit.assert_called_once()
    _, values = submit.call_args.args
    assert values["actor/rollout/step"] == 56  # Existing train-step conversion for rollout 7.
    assert values["actor/perf/gpu_delta/base_version"] == 0
    assert values["actor/perf/gpu_delta/target_version"] == 1
    assert all(values[f"actor/{key}"] == value for key, value in completed.items())
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
    completed.update(
        {
            "perf/update_weights_wire_bytes": 128,
            "perf/gpu_delta/payload_raw_bytes": 7,
            "perf/gpu_delta/payload_matrix_payload_bytes": 101,
            "perf/gpu_delta/payload_matrix_inner_arena_bytes": 512,
            "perf/gpu_delta/payload_canonical_bytes": 2048,
            "perf/gpu_delta/payload_changed_bytes": 27,
        }
    )
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
        summary = next(record.args for record in caplog.records if "initial sync" in record.msg)
        assert summary == dict(
            wire_bytes=128,
            raw_bytes=7,
            matrix_payload_bytes=101,
            matrix_inner_arena_bytes=512,
            canonical_bytes=2048,
            changed_bytes=27,
            file_alignment_bytes=20,
        )
        assert "scheduler_pause" not in caplog.text
    else:
        assert "initial sync" not in caplog.text


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
    assert result == WeightUpdateOutput(weight_version=17, failed_cell_ids=())
    assert updater.pop_metrics() == {}
    assert "Tracking failed for completed GPU-delta target version 1" in caplog.text
