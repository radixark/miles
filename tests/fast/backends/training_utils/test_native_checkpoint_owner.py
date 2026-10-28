from types import SimpleNamespace

import pytest

from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor


class TestNativeCheckpointOwner:
    @pytest.mark.parametrize("debug,save,expected", [(False, "/ckpt", [7]), (False, None, []), (True, "/ckpt", [])])
    def test_save_uses_backend_options_without_flat_duplicates(
        self, debug: bool, save: str | None, expected: list[int]
    ) -> None:
        """Only enabled native checkpoint saves reach the backend writer with the rollout identifier."""
        saved: list[int] = []
        actor = object.__new__(TorchNativeTrainRayActor)
        actor.args = SimpleNamespace(debug_rollout_only=debug, backend=SimpleNamespace(save=save, async_save=False))
        actor._save_checkpoint = saved.append

        actor.save_model(rollout_id=7)

        assert saved == expected

    def test_unsupported_async_save_is_rejected_before_writing(self) -> None:
        """Native async saving must fail rather than silently use the synchronous writer."""
        saved: list[int] = []
        actor = object.__new__(TorchNativeTrainRayActor)
        actor.args = SimpleNamespace(debug_rollout_only=False, backend=SimpleNamespace(save="/ckpt", async_save=True))
        actor._save_checkpoint = saved.append

        with pytest.raises(AssertionError, match="does not support async_save"):
            actor.save_model(rollout_id=7)
        assert saved == []
