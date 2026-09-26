from types import SimpleNamespace

import pytest

from miles.utils.ft_utils.api_server import server
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.worker_provider.base import CellInfo


class FakeCellOperations:
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return {}

    async def suspend(self, *, cell_id: str) -> None:
        pass

    async def resume(self, *, cell_id: str) -> None:
        pass

    async def observe_fault_target(self, *, cell_id: str, rank: int) -> ObservedFaultHookTarget:
        raise NotImplementedError

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        raise NotImplementedError


class TestStartApiServer:
    def test_rollout_ft_requires_a_local_inference_controller(self) -> None:
        """Rollout fault tolerance fails before startup when no local inference controller exists."""

        with pytest.raises(
            AssertionError,
            match="rollout cells are suspended and resumed through the inference controller",
        ):
            server.start_api_server(
                args=SimpleNamespace(),
                trainer_models={},
                inference_controller=None,
                port=1234,
                ft_components=["rollout"],
                cell_operations=FakeCellOperations(),
            )
