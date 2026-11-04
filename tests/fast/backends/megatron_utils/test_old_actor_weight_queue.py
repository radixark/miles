from types import ModuleType

import pytest
import torch
from tests.fast.backends.megatron_utils.test_actor import (
    _FakeWeightUpdater,
    _make_updating_actor,
    _update,
    actor_module,
)

from miles.utils.tensor_backper import TensorBackuper

__all__ = ["actor_module"]


class TestOldActorWeightQueue:
    @pytest.mark.parametrize(
        "keep_old,interval,old_value,rollout_value", [(False, 1, 1.0, 2.0), (True, 1, 2.0, 3.0), (True, 2, 3.0, 2.0)]
    )
    def test_weight_publication_rotates_real_tensor_snapshots_only_when_requested(
        self,
        actor_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        keep_old: bool,
        interval: int,
        old_value: float,
        rollout_value: float,
    ) -> None:
        """Old-actor retention controls the published-weight queue and its real saved tensors."""
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        tensor = torch.tensor([1.0])
        backups = TensorBackuper.create(source_getter=lambda: [("weight", tensor)])
        backups.backup("old_actor")
        tensor.fill_(2.0)
        backups.backup("rollout_actor")
        tensor.fill_(3.0)
        updater = _FakeWeightUpdater()
        actor = _make_updating_actor(actor_module=actor_module, monkeypatch=monkeypatch, updater=updater)
        actor.args.keep_old_actor = keep_old
        actor.args.update_weights_interval = interval
        actor.weights_backuper = backups

        output = _update(train_actor=actor, rollout_id=3)

        assert backups.get("old_actor")["weight"].item() == old_value
        assert backups.get("rollout_actor")["weight"].item() == rollout_value
        assert tensor.item() == 3.0
        assert updater.weight_versions == [7]
        assert output.weight_version == 7
