from argparse import Namespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.actor_group import RayTrainGroup


class _OrderRecordingInferenceController:
    def __init__(self, order: list[str]):
        self._order = order

    def __getattr__(self, name: str):
        recorder = self

        async def _method(*args, **kwargs):
            recorder._order.append(name)

        return _method


@pytest.mark.asyncio
async def test_v1_pauses_health_checks_before_snapshotting_the_engines():
    """The health monitor is paused before the engine set is snapshotted."""
    order: list[str] = []
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False, use_fault_tolerance=False)
    group._inference_controller = _OrderRecordingInferenceController(order)
    group._broadcast = AsyncMock()

    await group.update_weights()

    assert order[:2] == ["health_monitoring_pause", "get_updatable_engines_and_lock"]
    group._broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_pauses_health_checks_before_snapshotting_the_engines():
    """Same ordering requirement on the fault-tolerant trainer group."""
    from miles.ray.train.group import RayTrainGroup as FaultTolerantTrainGroup

    order: list[str] = []
    group = FaultTolerantTrainGroup.__new__(FaultTolerantTrainGroup)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False)
    group._inference_controller = _OrderRecordingInferenceController(order)
    group._execute_first_alive = AsyncMock()
    group._maybe_log_inference_engine_weight_checksums = AsyncMock()

    await group.update_weights()

    assert order[:2] == ["health_monitoring_pause", "get_updatable_engines_and_lock"]
