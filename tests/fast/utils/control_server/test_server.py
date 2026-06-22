from __future__ import annotations

import httpx
import pytest

from miles.utils.control_server.registry import _CellRegistry

from .conftest import MockHandle


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
        registry.register(MockHandle(cell_id="actor-0", cell_type="actor", cell_index=0, phase="Running"))
        registry.register(
            MockHandle(
                cell_id="rollout-0",
                cell_type="rollout",
                cell_index=0,
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
        registry.register(MockHandle(cell_id="actor-0", cell_type="actor", cell_index=0, phase="Running"))

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
