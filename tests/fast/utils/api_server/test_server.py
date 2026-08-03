from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from miles.utils.ft_utils.api_server import server
from miles.utils.ft_utils.api_server.registry import _CellRegistry

from .conftest import (
    MockHandle,
    MockInferenceController,
    MockRayTrainCell,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_group,
)


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
    async def test_returns_all_cells_golden(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        """Golden test: full JSON response for GET /api/v1/cells with two cells."""
        registry.register(MockHandle(cell_id="actor-0", cell_type="actor", phase="Running"))
        registry.register(
            MockHandle(
                cell_id="rollout-0",
                cell_type="rollout",
                phase="Suspended",
                is_suspended=True,
                conditions=[
                    {"type": "Allocated", "status": "False"},
                    {"type": "Healthy", "status": "False"},
                ],
            )
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
                        "labels": {"miles.io/cell-type": "actor", "miles.io/cell-index": "0"},
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
                        "labels": {"miles.io/cell-type": "rollout", "miles.io/cell-index": "0"},
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
    async def test_returns_single_cell_golden(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        """Golden test: full JSON response for GET /api/v1/cells/{name}."""
        registry.register(MockHandle(cell_id="actor-0", cell_type="actor", phase="Running"))

        resp = await async_client.get("/api/v1/cells/actor-0")
        assert resp.status_code == 200
        assert resp.json() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {"miles.io/cell-type": "actor", "miles.io/cell-index": "0"},
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
    async def test_suspend_cell_via_patch(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = MockHandle(cell_id="actor-0", cell_type="actor", phase="Running")
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert handle.suspend_calls == 1
        assert resp.json()["status"]["phase"] == "Suspended"
        assert resp.json()["spec"]["suspend"] is True

    @pytest.mark.asyncio
    async def test_resume_cell_via_patch(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = MockHandle(cell_id="actor-0", cell_type="actor", phase="Suspended", is_suspended=True)
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": False}})
        assert resp.status_code == 200
        assert handle.resume_calls == 1
        assert resp.json()["status"]["phase"] == "Running"

    @pytest.mark.asyncio
    async def test_patch_with_no_spec_is_noop(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = MockHandle(cell_id="actor-0", cell_type="actor", phase="Running")
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={})
        assert resp.status_code == 200
        assert handle.suspend_calls == 0
        assert handle.resume_calls == 0

    @pytest.mark.asyncio
    async def test_patch_not_found_returns_k8s_status(self, async_client: httpx.AsyncClient) -> None:
        resp = await async_client.patch("/api/v1/cells/nonexistent", json={"spec": {"suspend": True}})
        assert resp.status_code == 404
        assert resp.json()["kind"] == "Status"
        assert resp.json()["reason"] == "NotFound"

    @pytest.mark.asyncio
    async def test_patch_suspend_idempotent(self, registry: _CellRegistry, async_client: httpx.AsyncClient) -> None:
        handle = MockHandle(cell_id="actor-0", cell_type="actor", phase="Suspended", is_suspended=True)
        registry.register(handle)

        resp = await async_client.patch("/api/v1/cells/actor-0", json={"spec": {"suspend": True}})
        assert resp.status_code == 200
        assert handle.suspend_calls == 1

    @pytest.mark.asyncio
    async def test_patch_error_returns_500_k8s_status(
        self, registry: _CellRegistry, async_client: httpx.AsyncClient
    ) -> None:
        handle = MockHandle(cell_id="actor-0", cell_type="actor", suspend_error=RuntimeError("engine crashed"))
        registry.register(handle)

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
        monkeypatch.setattr(server, "compute_engine_pool_ids", lambda args: ["inference-engine-0-0"])
        monkeypatch.setattr(server, "ray", SimpleNamespace(get=lambda ref: ref.result()))
        monkeypatch.setattr(server, "_start_api_server_raw", lambda registry, port: registries.append(registry))

        server.start_api_server(
            args=SimpleNamespace(),
            actor_model=make_mock_group([MockRayTrainCell(), MockRayTrainCell()]),
            inference_controller=MockInferenceController(),
            port=0,
            ft_components=ft_components,
        )

        (registry,) = registries
        return registry

    @pytest.mark.asyncio
    async def test_every_engine_cell_gets_a_handle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cell the registry never learns about is invisible to the heal loop forever."""
        registry = self._start(
            monkeypatch,
            ft_components=["rollout"],
            cell_ids=["inference-engine-0-0-1", "inference-engine-0-0-0"],
        )

        assert [handle.cell_id for handle in registry.get_all()] == [
            "rollout-inference-engine-0-0-0",
            "rollout-inference-engine-0-0-1",
        ]

    @pytest.mark.asyncio
    async def test_rollout_cells_are_absent_when_rollout_ft_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exposing suspend on engines nobody heals would let a request strand the pool."""
        registry = self._start(monkeypatch, ft_components=["train"], cell_ids=["inference-engine-0-0-0"])

        assert all(handle.cell_type == "actor" for handle in registry.get_all())

    @pytest.mark.asyncio
    async def test_both_kinds_coexist_under_mixed_ft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed ft heals trainer and rollout cells through the one endpoint."""
        registry = self._start(monkeypatch, ft_components=["train", "rollout"], cell_ids=["inference-engine-0-0-0"])

        assert {handle.cell_type for handle in registry.get_all()} == {"actor", "rollout"}
