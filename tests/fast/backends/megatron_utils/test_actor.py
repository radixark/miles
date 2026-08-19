import inspect
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from miles.backends.megatron_utils import actor


class _FakeController:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_value = snapshot
        self.events: list[tuple[str, str | None]] = []
        self.retirement_completed = False
        self.snapshot_consumed = False
        self.freed_slots: set[str] = set()

    async def retire_adapters(self) -> None:
        self.retirement_completed = True
        self.events.append(("retire_adapters", None))

    async def snapshot(self) -> dict[str, Any]:
        self.snapshot_consumed = True
        self.events.append(("snapshot", None))
        return self.snapshot_value

    async def free_slot(self, name: str) -> None:
        self.freed_slots.add(name)
        self.events.append(("free_slot", name))


class TestReconcileAdapters:
    def test_the_independent_controller_coroutines_are_completed_during_reconciliation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First-replica reconciliation awaits retirement, snapshot, and cleanup-only slot release."""
        snapshot = {"active": {}, "pending": {}, "retiring": {}, "cleanup": ["orphan"]}
        controller = _FakeController(snapshot)
        train_actor = SimpleNamespace(
            args=SimpleNamespace(multi_lora=True),
            loaded_adapters={},
            model=object(),
            optimizer=object(),
            _multi_lora_pending_push=set(),
            weights_backuper=SimpleNamespace(backup=lambda _name: None),
        )
        monkeypatch.setattr(actor, "is_multi_lora_enabled", lambda _args: True)
        monkeypatch.setattr(actor, "is_first_replica_megatron_main_rank", lambda: True)
        monkeypatch.setattr(actor, "get_gloo_group", lambda: None)
        monkeypatch.setattr("miles.ray.multi_lora.controller.get_multi_lora_controller", lambda: controller)
        inspect.unwrap(actor.MegatronTrainRayActor.reconcile_adapters)(train_actor)

        assert controller.retirement_completed
        assert controller.snapshot_consumed
        assert controller.freed_slots == {"orphan"}


class TestSendCheckpoint:
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted
