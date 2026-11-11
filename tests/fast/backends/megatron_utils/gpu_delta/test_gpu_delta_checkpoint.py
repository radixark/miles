"""GPU-delta checkpoints and eval snapshots share the ordinary completion boundary."""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.fast.backends.megatron_utils import test_shared_ppo_lifecycle
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.utils.args.custom_function import CustomFunctionConfig

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


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("force_sync", [False, True])
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
