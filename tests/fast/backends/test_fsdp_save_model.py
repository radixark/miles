from types import SimpleNamespace
from unittest.mock import Mock

from miles.backends.experimental.fsdp_utils import actor as actor_module


def test_save_model_accepts_lifecycle_options(monkeypatch):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = SimpleNamespace(debug_rollout_only=False, save="/tmp/checkpoint", async_save=False)
    save = Mock()
    monkeypatch.setattr(actor_module.checkpoint, "save", save)

    actor.save_model(7, options={"wake_up_before_save": True})

    save.assert_called_once_with(actor, 7)
