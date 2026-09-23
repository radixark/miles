from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput


@contextmanager
def _noop_timer(_name: str) -> Iterator[None]:
    yield


def test_fsdp_train_debug_rollout_only_returns_a_normal_output(monkeypatch):
    """A debug-rollout-only FSDP step trains nothing yet answers the driver with a NORMAL output."""
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(offload_train=False, debug_rollout_only=True)
    actor._heartbeat = Mock()
    actor._train_core = Mock()
    actor.wake_up = Mock()
    monkeypatch.setattr(
        actor_module, "get_rollout_data", lambda _args, _ref, **_kwargs: ({"tokens": []}, nullcontext())
    )
    monkeypatch.setattr(actor_module, "timer", _noop_timer)
    monkeypatch.setattr(actor_module, "inverse_timer", _noop_timer)

    result = actor.train(3, object())

    assert result == TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
    actor._train_core.assert_not_called()
