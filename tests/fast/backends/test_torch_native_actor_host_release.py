"""A train step hands the host heap back, and the move to the host releases
the pinned cache (TorchNativeTrainRayActor)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from miles.backends.training_utils.torch_native import actor as actor_module
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor


def _bare_actor(monkeypatch, **args):
    actor = object.__new__(TorchNativeTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, debug_rollout_only=False, **args)
    actor.model_parts = []
    actor.optimizers = []
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor.output_store_read = None
    monkeypatch.setattr(TorchNativeTrainRayActor, "_fwd_tflops", property(lambda self: 0))
    monkeypatch.setattr(TorchNativeTrainRayActor, "_train_tflops", property(lambda self: 0))
    monkeypatch.setattr(TorchNativeTrainRayActor, "train_parallel_config", property(lambda self: None))
    monkeypatch.setattr(actor_module, "print_memory", lambda *a, **k: None)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
    return actor


def test_sleep_releases_the_host_cache_on_the_way_to_the_host(monkeypatch):
    actor = _bare_actor(monkeypatch)
    calls = []
    monkeypatch.setattr(
        actor_module,
        "move_train_state",
        lambda parts, opts, to_host: calls.append(("move", "host" if to_host else "device")),
    )
    monkeypatch.setattr(
        actor_module, "clear_memory", lambda clear_host_memory=False: calls.append(("clear", clear_host_memory))
    )

    actor.sleep()
    actor.wake_up()

    assert calls == [("move", "host"), ("clear", True), ("move", "device"), ("clear", False)]


def test_a_train_step_releases_the_host_memory_after_training(monkeypatch):
    actor = _bare_actor(monkeypatch)
    order = []
    monkeypatch.setattr(actor, "wake_up", lambda: order.append("wake"))
    monkeypatch.setattr(actor, "_train_core", lambda rollout_id, rollout_data: order.append(("train", rollout_id)))
    monkeypatch.setattr(
        actor_module,
        "get_rollout_data",
        lambda **kwargs: ({"tokens": []}, nullcontext()),
    )
    monkeypatch.setattr(actor_module, "release_host_memory", lambda: order.append("release"))
    monkeypatch.setattr(actor_module.perf, "log_perf_data_raw", lambda **k: order.append("perf"))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    # the step timers expect the rollout phase to have opened train_wait
    monkeypatch.setattr(actor_module, "timer", lambda name: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda name: nullcontext())

    actor.train(rollout_id=3, rollout_data_ref=object())

    assert order == ["wake", ("train", 3), "release", "perf"]


def test_release_host_memory_trims_the_heap(monkeypatch):
    from miles.utils import memory_utils

    seen = []
    monkeypatch.setattr(memory_utils, "trim_host_heap", lambda: seen.append("trim") or True)
    memory_utils.release_host_memory()
    assert seen == ["trim"]
