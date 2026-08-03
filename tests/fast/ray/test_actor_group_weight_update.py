from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.ray.actor_group import RayTrainGroup

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


class _RemoteCall:
    def __init__(self, events: list[str], name: str, result: object = None) -> None:
        self._events = events
        self._name = name
        self._result = result

    async def remote(self, **kwargs: object) -> object:
        self._events.append(self._name)
        if self._name == "register":
            assert kwargs == {"rollout_engine_generation_ids": ["engine-generation-id"]}
        else:
            assert kwargs == {}
        return self._result


class _RolloutManager:
    def __init__(self, events: list[str]) -> None:
        self.recover_updatable_engines = _RemoteCall(events, "recover")
        self.get_updatable_engines_and_lock = _RemoteCall(
            events,
            "get_engines",
            result=SimpleNamespace(
                rollout_engines=["engine"],
                rollout_engine_generation_ids=["engine-generation-id"],
            ),
        )
        self.health_monitoring_pause = _RemoteCall(events, "pause")
        self.register_recovered_updatable_engines = _RemoteCall(events, "register")


@pytest.mark.asyncio
async def test_update_weights_registers_recovered_engines_after_transfer() -> None:
    events: list[str] = []
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        use_fault_tolerance=True,
    )
    group.rollout_manager = _RolloutManager(events)

    async def record_broadcast(method_name: str, *, info: object) -> None:
        assert method_name == "update_weights"
        assert info == SimpleNamespace(
            rollout_engines=["engine"],
            rollout_engine_generation_ids=["engine-generation-id"],
        )
        events.append("transfer")

    group._broadcast = AsyncMock(side_effect=record_broadcast)

    await group.update_weights()

    assert events == ["recover", "get_engines", "pause", "transfer", "register"]


@pytest.mark.asyncio
async def test_update_weights_does_not_register_when_transfer_fails() -> None:
    events: list[str] = []
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        use_fault_tolerance=True,
    )
    group.rollout_manager = _RolloutManager(events)
    group._broadcast = AsyncMock(side_effect=RuntimeError("transfer failed"))

    with pytest.raises(RuntimeError, match="transfer failed"):
        await group.update_weights()

    assert events == ["recover", "get_engines", "pause"]


@pytest.mark.asyncio
async def test_update_weights_publishes_pending_manual_start_without_fault_tolerance() -> None:
    events: list[str] = []
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        use_fault_tolerance=False,
    )
    group.rollout_manager = _RolloutManager(events)

    async def record_broadcast(method_name: str, *, info: object) -> None:
        assert method_name == "update_weights"
        assert info == SimpleNamespace(
            rollout_engines=["engine"],
            rollout_engine_generation_ids=["engine-generation-id"],
        )
        events.append("transfer")

    group._broadcast = AsyncMock(side_effect=record_broadcast)

    await group.update_weights()

    assert events == ["get_engines", "pause", "transfer", "register"]
