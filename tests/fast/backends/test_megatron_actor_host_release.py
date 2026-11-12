"""A Megatron train step hands the host heap back after training (MegatronTrainRayActor),
as the torch-native actor's does."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("megatron.core")

from miles.backends.megatron_utils import actor as actor_module
from miles.backends.megatron_utils.actor import MegatronTrainRayActor


def _bare_actor(role):
    actor = object.__new__(MegatronTrainRayActor)
    actor.args = SimpleNamespace(offload_train=False, debug_rollout_only=False)
    actor.role = role
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._asleep = False
    return actor


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_a_train_step_releases_the_host_memory_after_training(monkeypatch, role):
    actor = _bare_actor(role)
    order = []
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda args, rollout_data_ref, **kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(actor_module, "timer", lambda name: nullcontext())
    # upstream's train step reads the parallel state for the data loader's schedule config
    monkeypatch.setattr(
        actor_module,
        "get_parallel_state",
        lambda: SimpleNamespace(train_parallel_config=lambda **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        actor, "_train_actor", lambda rollout_id, data, **kwargs: order.append(("train", rollout_id)) or "out"
    )
    monkeypatch.setattr(actor, "_train_critic", lambda rollout_id, data: order.append(("train", rollout_id)) or "out")
    monkeypatch.setattr(actor_module, "release_host_memory", lambda: order.append("release"))
    # newer miles publishes companion info at the end of a step; absent on older bases
    monkeypatch.setattr(actor, "_publish_model_companion_info", lambda **kwargs: None, raising=False)

    assert actor.train(rollout_id=3, rollout_data_ref=object()) == "out"
    assert order == [("train", 3), "release"]
