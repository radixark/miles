from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests.fast.ray.rollout.conftest import make_args as make_rollout_args

from miles.ray.rollout.server_cell import compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server import server
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import MockHandler, MockInferenceController, MockWorkerManager, make_cell_summaries, make_mock_group


class TestGetHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetCells:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_cell_list(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [],
        }

    @pytest.mark.asyncio
    async def test_returns_all_cells_golden(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Golden test: full JSON response for GET /api/v1/cells with two cells."""
        actor_handler.add("actor-0", phase="Running")
        rollout_handler.add(
            "rollout-0",
            phase="Suspended",
            is_suspended=True,
            conditions=[
                {"type": "Allocated", "status": "False"},
                {"type": "Healthy", "status": "False"},
            ],
        )

        resp = await async_client.get("/api/v1/cells")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "CellList",
            "items": [
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "actor-0",
                        "labels": {"miles.io/cell-type": "actor", "miles.io/cell-id": "actor-0"},
                    },
                    "spec": {"suspend": False},
                    "status": {
                        "phase": "Running",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "True",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                    },
                },
                {
                    "apiVersion": "miles.io/v1",
                    "kind": "Cell",
                    "metadata": {
                        "name": "rollout-0",
                        "labels": {"miles.io/cell-type": "rollout", "miles.io/cell-id": "rollout-0"},
                    },
                    "spec": {"suspend": True},
                    "status": {
                        "phase": "Suspended",
                        "conditions": [
                            {
                                "type": "Allocated",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                            {
                                "type": "Healthy",
                                "status": "False",
                                "reason": None,
                                "message": None,
                                "lastTransitionTime": None,
                            },
                        ],
                    },
                },
            ],
        }


class TestGetCell:
    @pytest.mark.asyncio
    async def test_returns_single_cell_golden(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Golden test: full JSON response for GET /api/v1/cells/{name}."""
        actor_handler.add("actor-0", phase="Running")

        resp = await async_client.get("/api/v1/cells/actor-0")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {"miles.io/cell-type": "actor", "miles.io/cell-id": "actor-0"},
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_not_found_returns_k8s_status_golden(self, async_client: httpx.AsyncClient) -> None:
        """Golden test: K8s Status error response for 404."""
        resp = await async_client.get("/api/v1/cells/nonexistent")
        assert resp.status_code == 404
        assert resp.json() == {
            "apiVersion": "v1",
            "kind": "Status",
            "status": "Failure",
            "message": "Cell 'nonexistent' not found",
            "reason": "NotFound",
            "code": 404,
        }


class TestPatchCell:
    @pytest.mark.asyncio
    async def test_suspend_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1
        assert resp.json()["status"]["phase"] == "Suspended"
        assert resp.json()["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_resume_cell_via_patch(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})
        assert resp.status_code == 200
        assert cell.resume_calls == 1
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_with_no_spec_is_noop(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={})
        assert resp.status_code == 200
        assert cell.suspend_calls == 0
        assert cell.resume_calls == 0

    @pytest.mark.asyncio
    async def test_patch_with_empty_spec_does_not_suspend_or_resume(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A spec that omits suspend carries no instruction, so the cell must be left exactly as it was."""
        cell = actor_handler.add("actor-0", phase="Running")

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {}})

        assert resp.status_code == 200
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert resp.json()["spec"]["suspend"] is False
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.patch("/api/v1/cells/nonexistent", json={"spec": {"suspend": True}})
        assert resp.status_code == 404
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "NotFound"

    @pytest.mark.asyncio
    async def test_patch_suspend_idempotent(self, actor_handler: MockHandler, async_client: httpx.AsyncClient) -> None:
        cell = actor_handler.add("actor-0", phase="Suspended", is_suspended=True)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert cell.suspend_calls == 1

    @pytest.mark.asyncio
    async def test_patch_error_returns_500_k8s_status(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        actor_handler.add("actor-0", suspend_error=RuntimeError("engine crashed"))

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 500
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "InternalError"


class TestStartApiServerRegistration:
    def _start(
        self, monkeypatch: pytest.MonkeyPatch, *, ft_components: list[str], cell_ids: list[str]
    ) -> _CellRegistry:
        manager = MockWorkerManager(make_cell_summaries(*cell_ids))
        registries: list[_CellRegistry] = []

        monkeypatch.setattr(server, "RayWorkerManager", SimpleNamespace(get_handle=lambda: manager))
        monkeypatch.setattr(server, "_start_api_server_raw", lambda registry, port: registries.append(registry))

        server.start_api_server(
            args=make_rollout_args(),
            actor_model=make_mock_group([]),
            inference_controller=MockInferenceController(
                {cell_id: compute_pending_rollout_cell_status() for cell_id in cell_ids}
            ),
            port=18080,
            ft_components=ft_components,
        )

        (registry,) = registries
        return registry

    @pytest.mark.asyncio
    async def test_the_rollout_handler_enumerates_every_engine_cell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cell the handler never enumerates is invisible to the heal loop forever."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["inference-engine-0-0-1", "inference-engine-0-0-0"],
        )

        assert [cell.metadata.name for cell in await registry.list_cells()] == [
            "inference-engine-0-0-0",
            "inference-engine-0-0-1",
        ]

    @pytest.mark.asyncio
    async def test_no_rollout_handler_exists_when_rollout_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exposing suspend on engines nobody heals would let a request strand the pool."""
        registry = self._start(monkeypatch, ft_components=["train"], cell_ids=["inference-engine-0-0-0"])

        assert await registry.list_cells() == []

    @pytest.mark.asyncio
    async def test_both_handlers_coexist_under_mixed_ft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed ft heals trainer and rollout cells through the one endpoint."""
        registry = self._start(monkeypatch, ft_components=["train", "rollout"], cell_ids=["inference-engine-0-0-0"])

        assert [handler.cell_type for handler in registry._handlers] == ["actor", "rollout"]


class TestDynamicCells:
    @pytest.mark.asyncio
    async def test_a_cell_that_appears_after_startup_is_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Engine cells are reconciled in, so the server cannot snapshot them once at startup."""
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404

        rollout_handler.add("rollout-engine-0")

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

    @pytest.mark.asyncio
    async def test_a_cell_that_disappears_stops_being_served(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """A removed cell must 404 instead of reporting stale status."""
        rollout_handler.add("rollout-engine-0")
        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 200

        del rollout_handler.cells["rollout-engine-0"]

        assert (await async_client.get("/api/v1/cells/rollout-engine-0")).status_code == 404


class TestInjectFault:
    @pytest.mark.asyncio
    async def test_injection_reaches_the_handler_of_that_cell(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """CI fault injection targets one cell by name."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "sub_index": 1}
        )

        assert resp.status_code == 200
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.SIGKILL, 1)]

    @pytest.mark.asyncio
    async def test_a_handler_without_injection_support_answers_bad_request(
        self, actor_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Not every kind of cell can be crashed on demand."""
        actor_handler.add("actor-0")

        resp = await async_client.post("/api/v1/cells/actor-0/inject-fault", json={"mode": "sigkill"})

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_inject_fault_uses_zero_sub_index_by_default(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """The documented default targets worker zero, and a client omitting sub_index relies on it."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        resp = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "exit"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert rollout_handler.injected == [("rollout-engine-0", FailureMode.EXIT, 0)]

    @pytest.mark.asyncio
    async def test_inject_fault_rejects_missing_or_unknown_mode(
        self, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """An unrecognised failure mode must be refused by the schema rather than forwarded to the cell."""
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        missing = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={})
        unknown = await async_client.post("/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "nuke"})

        assert (missing.status_code, unknown.status_code) == (422, 422)
        assert rollout_handler.injected == []


class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_invalid_write_bodies_return_422_without_side_effects(
        self, actor_handler: MockHandler, rollout_handler: MockHandler, async_client: httpx.AsyncClient
    ) -> None:
        """Unknown fields must be refused outright, so a typo cannot silently half-apply a write."""
        cell = actor_handler.add("actor-0", phase="Running")
        rollout_handler.supports_inject_fault = True
        rollout_handler.add("rollout-engine-0")

        patch_resp = await async_client.patch(
            "/api/v1/cells/actor-0", json={"spec": {"suspend": True, "gracePeriod": 5}}
        )
        inject_resp = await async_client.post(
            "/api/v1/cells/rollout-engine-0/inject-fault", json={"mode": "sigkill", "subIndex": 1}
        )

        assert (patch_resp.status_code, inject_resp.status_code) == (422, 422)
        assert (cell.suspend_calls, cell.resume_calls) == (0, 0)
        assert rollout_handler.injected == []
