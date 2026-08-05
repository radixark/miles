import importlib
import sys
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def actor_module():
    actor_module_name = "miles.backends.megatron_utils.actor"
    p2p_module_name = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p"
    actor_package = importlib.import_module("miles.backends.megatron_utils")
    p2p_package = importlib.import_module("miles.backends.megatron_utils.update_weight.update_weight_from_distributed")
    missing = object()
    saved_actor_module = sys.modules.get(actor_module_name, missing)
    saved_p2p_module = sys.modules.get(p2p_module_name, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_actor_package_attr = getattr(actor_package, "actor", missing)
    saved_p2p_package_attr = getattr(p2p_package, "p2p", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    p2p_module = ModuleType(p2p_module_name)
    p2p_module.UpdateWeightP2P = Mock(
        side_effect=AssertionError("shared PPO lifecycle tests must not construct UpdateWeightP2P")
    )
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules[p2p_module_name] = p2p_module
    p2p_package.p2p = p2p_module
    sys.modules.pop(actor_module_name, None)
    if saved_actor_package_attr is not missing:
        delattr(actor_package, "actor")

    try:
        yield importlib.import_module(actor_module_name)
    finally:
        sys.modules.pop(actor_module_name, None)
        if saved_actor_module is not missing:
            sys.modules[actor_module_name] = saved_actor_module
        if saved_actor_package_attr is missing:
            if hasattr(actor_package, "actor"):
                delattr(actor_package, "actor")
        else:
            actor_package.actor = saved_actor_package_attr
        sys.modules.pop(p2p_module_name, None)
        if saved_p2p_module is not missing:
            sys.modules[p2p_module_name] = saved_p2p_module
        if saved_p2p_package_attr is missing:
            if hasattr(p2p_package, "p2p"):
                delattr(p2p_package, "p2p")
        else:
            p2p_package.p2p = saved_p2p_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _worker(actor_module, role, *, asleep=True):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, debug_rollout_only=False)
    worker.role = role
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_critic_train_wakes_and_leaves_offload_to_driver(actor_module, monkeypatch):
    worker = _worker(actor_module, "critic")
    worker.train_critic = Mock(return_value={"values": ["cpu-value"]})
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    phases = []

    @contextmanager
    def capture_timer(name):
        phases.append(name)
        yield

    monkeypatch.setattr(actor_module, "timer", capture_timer)

    result = worker.train(3, object())

    worker.wake_up.assert_called_once_with()
    worker.train_critic.assert_called_once()
    worker.sleep.assert_not_called()
    assert result == {"values": ["cpu-value"]}
    assert phases == ["data_preprocess", "critic_train"]


def test_actor_receives_critic_payload_and_leaves_offload_to_driver(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    values = {"values": ["cpu-value"]}

    result = worker.train(4, object(), external_data=values)

    worker.wake_up.assert_called_once_with()
    worker.train_actor.assert_called_once()
    assert worker.train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_not_called()
    assert result is None


def test_train_keeps_model_resident(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor", asleep=False)
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )

    worker.train(5, object())

    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()


def test_save_model_does_not_manage_lifecycle(actor_module, monkeypatch):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        async_save=False,
        custom_megatron_post_save_hook_path=None,
        debug_rollout_only=False,
        save_hf=None,
    )
    worker.role = "actor"
    worker._heartbeat = Mock()
    worker.model = object()
    worker.optimizer = object()
    worker.opt_param_scheduler = object()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    save = Mock()
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "save", save)
    monkeypatch.setattr(actor_module, "is_multi_lora_enabled", lambda _args: False)
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)

    worker.save_model(6)

    save.assert_called_once_with(6, worker.model, worker.optimizer, worker.opt_param_scheduler)
    worker.wake_up.assert_not_called()
    worker.sleep.assert_not_called()
    reload_groups.assert_not_called()
    destroy_groups.assert_not_called()


@pytest.mark.parametrize("asleep", [False, True])
def test_update_weights_only_uses_temporary_process_groups_when_asleep(actor_module, monkeypatch, asleep):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(
        debug_rollout_only=False,
        debug_skip_weight_update=True,
        debug_train_only=False,
        offload_train=True,
    )
    worker._asleep = asleep
    worker._heartbeat = Mock()
    worker.weight_updater = Mock()
    worker.weight_updater.is_rollout_engines_fresh.return_value = True
    info = Namespace(
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
        has_new_engines=False,
        rollout_engine_lock=None,
        rollout_engines=[],
    )
    reload_groups = Mock()
    destroy_groups = Mock()
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy_groups)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)

    worker.update_weights(info)

    assert reload_groups.call_count == int(asleep)
    assert destroy_groups.call_count == int(asleep)


def _lifecycle_worker(actor_module, monkeypatch, asleep):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True)
    worker._asleep = asleep
    saver = Mock()
    reload_groups = Mock()
    monkeypatch.setattr(actor_module, "torch_memory_saver", saver)
    monkeypatch.setattr(actor_module, "clear_memory", Mock())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "destroy_process_groups", Mock())
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(actor_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(actor_module, "is_lora_enabled", lambda _args: False)
    return worker, saver, reload_groups


def test_sleep_is_idempotent(actor_module, monkeypatch):
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.sleep()
    worker.sleep()

    assert saver.pause.call_count == 1
    assert worker._asleep is True


def test_wake_up_when_resident_skips_resume_but_restores_groups(actor_module, monkeypatch):
    # A retried attempt can die between wake and sleep: memory stays resident but the
    # process groups may already be gone, so wake_up must restore groups without resuming.
    worker, saver, reload_groups = _lifecycle_worker(actor_module, monkeypatch, asleep=False)

    worker.wake_up()

    saver.resume.assert_not_called()
    reload_groups.assert_called_once_with()
    assert worker._asleep is False


def test_wake_up_resumes_offloaded_model_once(actor_module, monkeypatch):
    worker, saver, _ = _lifecycle_worker(actor_module, monkeypatch, asleep=True)

    worker.wake_up()
    worker.wake_up()

    assert saver.resume.call_count == 1
    assert worker._asleep is False
