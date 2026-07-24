from argparse import Namespace
from contextlib import contextmanager
from unittest.mock import Mock

from miles.backends.megatron_utils import actor as actor_module
from miles.backends.megatron_utils.actor import MegatronTrainRayActor


def _worker(role):
    worker = object.__new__(MegatronTrainRayActor)
    worker.args = Namespace(offload_train=True, use_critic=True, debug_rollout_only=False)
    worker.role = role
    worker._heartbeat = Mock()
    worker.wake_up = Mock()
    worker.sleep = Mock()
    return worker


def test_shared_critic_train_owns_wake_and_sleep(monkeypatch):
    worker = _worker("critic")
    worker.train_critic = Mock(return_value={"values": ["cpu-value"]})
    monkeypatch.setattr(actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: {"tokens": []})
    phases = []

    @contextmanager
    def capture_timer(name):
        phases.append(name)
        yield

    monkeypatch.setattr(actor_module, "timer", capture_timer)

    result = worker.train(3, object())

    worker.wake_up.assert_called_once_with()
    worker.train_critic.assert_called_once()
    worker.sleep.assert_called_once_with()
    assert result == {"values": ["cpu-value"]}
    assert phases == ["data_preprocess", "critic_train"]


def test_shared_actor_receives_critic_payload_between_wake_and_sleep(monkeypatch):
    worker = _worker("actor")
    worker.train_actor = Mock(return_value=None)
    monkeypatch.setattr(actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: {"tokens": []})
    values = {"values": ["cpu-value"]}

    result = worker.train(4, object(), external_data=values)

    worker.wake_up.assert_called_once_with()
    worker.train_actor.assert_called_once()
    assert worker.train_actor.call_args.kwargs["external_data"] is values
    worker.sleep.assert_called_once_with()
    assert result is None
