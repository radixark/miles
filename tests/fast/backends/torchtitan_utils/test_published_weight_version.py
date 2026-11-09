from pathlib import Path
from types import SimpleNamespace

import torch.distributed.checkpoint as dcp

from miles.backends.torchtitan_utils import published_weight_version
from miles.backends.torchtitan_utils.published_weight_version import (
    PUBLISHED_WEIGHT_VERSION,
    PublishedWeightVersionState,
)


class _TrainState:
    def state_dict(self) -> dict:
        return {"step": 4}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


class TestPublishedWeightVersionState:
    def test_a_saved_version_restores_into_a_fresh_actor(self, tmp_path: Path) -> None:
        """The torchtitan checkpoint state carries the published version across a DCP save and load."""
        saved = SimpleNamespace(_published_weight_version=6)
        dcp.save({PUBLISHED_WEIGHT_VERSION: PublishedWeightVersionState(saved)}, checkpoint_id=str(tmp_path))
        fresh = SimpleNamespace(_published_weight_version=0)

        dcp.load({PUBLISHED_WEIGHT_VERSION: PublishedWeightVersionState(fresh)}, checkpoint_id=str(tmp_path))

        assert fresh._published_weight_version == 6
        assert published_weight_version.saved_in(str(tmp_path))

    def test_a_checkpoint_saved_without_the_version_is_detected(self, tmp_path: Path) -> None:
        """Checkpoints saved before the version was persisted are recognized, so loading skips the state."""
        dcp.save({"train_state": _TrainState()}, checkpoint_id=str(tmp_path))

        assert not published_weight_version.saved_in(str(tmp_path))
