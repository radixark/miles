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
                "request_id": "request-1",
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
        assert operations.request_ids == ["request-1"]


@pytest.mark.parametrize("dispatch_failed", [False, True])
async def test_duplicate_submission_requires_evidence_and_never_dispatches_twice(dispatch_failed: bool) -> None:
    """An ambiguous first dispatch remains unconfirmed until a matching receipt arrives."""
    operations = PinnedCellOperations(dispatch_error=RuntimeError("Reply lost") if dispatch_failed else None)
    app = server._create_api_app(
        _CellRegistry([_CellHandler(cell_type="actor", operations=operations, controllers=[], pool_ids=["actor"])]),
        receipt_url="http://witness-host:18080",
    )
    body = {
        "mode": "sigkill",
        "sub_index": 0,
        "expected_target": operations.target.model_dump(mode="json"),
        "request_id": "request-1",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://control") as client:
        first = await client.post("/api/v1/cells/actor-0/inject-fault", json=body)
        assert first.status_code == (500 if dispatch_failed else 200)
        duplicate = await client.post("/api/v1/cells/actor-0/inject-fault", json=body)
        assert duplicate.status_code == 409
        assert (await client.get("/api/v1/fault-receipts/request-1")).json() is None
        published = await client.post("/api/v1/fault-receipts/request-1", json={"exited_pids": [42]})
        assert published.status_code == 200
        assert published.json() == {
            "request_id": "request-1",
            "target": body["expected_target"],
            "mode": "sigkill",
            "exited_pids": [42],
        }
        assert (await client.get("/api/v1/fault-receipts/request-1")).json() == published.json()
        confirmed = await client.post("/api/v1/cells/actor-0/inject-fault", json=body)
        assert confirmed.status_code == 200
        conflict = await client.post("/api/v1/fault-receipts/request-1", json={"exited_pids": [43]})
        assert conflict.status_code == 409
    assert operations.request_ids == ["request-1"]
    assert operations.receipt_urls == ["http://witness-host:18080"]
