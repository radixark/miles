from types import SimpleNamespace

import httpx
import pytest
from tests.fast.utils.ft_utils.api_server.conftest import PinnedCellOperations

from miles.utils.ft_utils.api_server import server
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class FakeCellOperations:
    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return {}

    async def suspend(self, *, cell_id: str) -> None:
        pass

    async def resume(self, *, cell_id: str) -> None:
        pass

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        pass


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


@pytest.mark.parametrize("replaced", [False, True])
async def test_observed_identity_is_forwarded_and_replacement_returns_precondition_failed(replaced: bool) -> None:
    """The HTTP round trip preserves every identity field and rejects a replaced worker."""
    operations = PinnedCellOperations()
    app = server._create_api_app(
        _CellRegistry(
            [
                _CellHandler(cell_type="actor", operations=operations, controllers=[], pool_ids=["actor"]),
            ]
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://control") as client:
        observed = await client.get("/api/v1/cells/actor-0/fault-target", params={"sub_index": 0})
        assert observed.status_code == 200
        assert observed.json() == operations.target.model_dump(mode="json")
        if replaced:
            operations.target = operations.target.model_copy(update={"boot_uuid": "boot-1"})
        result = await client.post(
            "/api/v1/cells/actor-0/inject-fault",
            json={
                "mode": "sigkill",
                "sub_index": 0,
                "expected_target": observed.json(),
            },
        )
    if replaced:
        assert result.status_code == 412
        assert result.json()["reason"] == "PreconditionFailed"
        assert operations.dispatched == []
    else:
        assert result.status_code == 200
        assert operations.dispatched == [operations.target]
        assert operations.modes == [FailureMode.SIGKILL]
