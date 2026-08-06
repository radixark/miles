from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

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

    assert order[:2] == ["health_monitoring_pause", "get_updatable_engines"]
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

    assert order[:2] == ["health_monitoring_pause", "get_updatable_engines"]


def test_fsdp_updater_flushes_only_after_every_engine_is_paused():
    """Every engine is paused before any engine is flushed."""
    from unittest.mock import patch

    from miles.backends.experimental.fsdp_utils.update_weight_utils import UpdateWeightFromTensor

    order: list[str] = []
    pause_modes: list[str] = []

    class _Client:
        def __init__(self, index: int):
            self._index = index

        async def pause_generation(self, *, mode: str = "retract"):
            order.append(f"pause-{self._index}")
            pause_modes.append(mode)

        async def flush_cache(self):
            order.append(f"flush-{self._index}")

        async def continue_generation(self):
            order.append(f"continue-{self._index}")

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(update_weight_buffer_size=1 << 30)
    updater.weight_version = 0
    updater.model = MagicMock()
    updater.model.state_dict.return_value = {}
    updater.rollout_engines = [_Client(0), _Client(1)]

    module = "miles.backends.experimental.fsdp_utils.update_weight_utils"
    with patch(f"{module}.dist") as dist_mock, patch(f"{module}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert set(order[:2]) == {"pause-0", "pause-1"}
    assert set(order[2:4]) == {"flush-0", "flush-1"}
    assert set(order[4:]) == {"continue-0", "continue-1"}
    assert pause_modes == ["retract", "retract"]
