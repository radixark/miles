import importlib
import sys
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def actor_module():
    module_name = "miles.backends.megatron_utils.actor"
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(module_name, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(module_name, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if saved_module is not missing:
            sys.modules[module_name] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


def _worker(actor_module, role):
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, debug_rollout_only=False)
    worker.role = role
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_critic_train_wakes_on_config_and_sleeps_on_options(actor_module, monkeypatch):
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

    result = worker.train(3, object(), options={"sleep_after_train": True})

    worker.wake_up.assert_called_once_with()
    worker.train_critic.assert_called_once()
    worker.sleep.assert_called_once_with()
    assert result == {"values": ["cpu-value"]}
    assert phases == ["data_preprocess", "critic_train"]


def test_actor_receives_critic_payload_between_wake_and_sleep(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    values = {"values": ["cpu-value"]}

    result = worker.train(4, object(), external_data=values, options={"sleep_after_train": True})

    worker.wake_up.assert_called_once_with()
    worker.train_actor.assert_called_once()
    assert worker.train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_called_once_with()
    assert result is None


def test_train_without_options_keeps_model_resident(actor_module, monkeypatch):
    worker = _worker(actor_module, "actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )

    worker.train(5, object())

    worker.wake_up.assert_called_once_with()
    worker.sleep.assert_not_called()


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
