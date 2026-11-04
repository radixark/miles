from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_executor import RolloutExecutor


class TestCheckpointRuntimeInput:
    @pytest.mark.parametrize("configured_load", [None, "/stale/config/path"])
    @pytest.mark.parametrize("requested_load", [None, "/runtime/path"])
    async def test_legacy_configuration_cannot_supply_or_hide_a_runtime_checkpoint_request(
        self, configured_load: str | None, requested_load: str | None
    ) -> None:
        """A rollout config carrying any load field is refused before reading checkpoint state."""
        executor = object.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(load=configured_load)

        with pytest.raises(AssertionError, match="checkpoint load path must be a runtime input"):
            await executor.load(0, load=requested_load)
