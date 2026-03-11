from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
from miles.utils.ft.adapters.types import JobStatus


class FakeRmHandle:
    """Simulates RolloutManager Ray actor handle.

    Ray handle call pattern: handle.method.remote(args)
    .method returns a proxy with .remote() method.
    """

    def __init__(self) -> None:
        self.stop_cell = MagicMock()
        self.stop_cell.remote = AsyncMock()

        self.start_cell = MagicMock()
        self.start_cell.remote = AsyncMock(return_value=9)

        self.get_cell_status = MagicMock()
        self.get_cell_status.remote = AsyncMock(return_value=JobStatus.RUNNING)


@pytest.mark.anyio
class TestRayRolloutActuator:
    async def test_stop_calls_correct_cell(self) -> None:
        handle = FakeRmHandle()
        actuator = RayRolloutActuator(rm_handle=handle, cell_id="0")
        await actuator.stop()
        handle.stop_cell.remote.assert_awaited_once_with("0")

    async def test_start_returns_str(self) -> None:
        handle = FakeRmHandle()
        actuator = RayRolloutActuator(rm_handle=handle, cell_id="0")
        result = await actuator.start()
        handle.start_cell.remote.assert_awaited_once_with("0")
        assert result == "9"
        assert isinstance(result, str)

    async def test_get_status_running(self) -> None:
        handle = FakeRmHandle()
        actuator = RayRolloutActuator(rm_handle=handle, cell_id="0")
        status = await actuator.get_status()
        assert status == JobStatus.RUNNING
        handle.get_cell_status.remote.assert_awaited_once_with("0")

    async def test_get_status_failed(self) -> None:
        handle = FakeRmHandle()
        handle.get_cell_status.remote = AsyncMock(return_value=JobStatus.FAILED)
        actuator = RayRolloutActuator(rm_handle=handle, cell_id="1")
        status = await actuator.get_status()
        assert status == JobStatus.FAILED

    async def test_different_cell_ids_isolated(self) -> None:
        handle = FakeRmHandle()
        act_0 = RayRolloutActuator(rm_handle=handle, cell_id="0")
        act_1 = RayRolloutActuator(rm_handle=handle, cell_id="1")
        await act_0.stop()
        await act_1.stop()
        calls = handle.stop_cell.remote.await_args_list
        assert calls[0].args == ("0",)
        assert calls[1].args == ("1",)
