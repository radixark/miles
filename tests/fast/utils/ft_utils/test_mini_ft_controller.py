from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from miles.utils.ft_utils import mini_ft_controller
from miles.utils.ft_utils.mini_ft_controller import CellHealthStatus, _CellSnapshot, _MiniFTController


class TestControllerHealing:
    async def test_an_auto_resuming_cell_is_suspended_without_an_explicit_resume(self) -> None:
        """An auto-resuming cell completes healing without an explicit resume request."""
        get_cells = AsyncMock(return_value=[_CellSnapshot(name="cell-0", status=CellHealthStatus.UNHEALTHY)])
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()
        controller = _MiniFTController(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
            poll_interval=0.0,
            resume_delay=0.0,
            cells_auto_resume=True,
            clock=lambda: 10.0,
        )

        await controller._poll_and_heal()

        suspend_cell.assert_awaited_once_with("cell-0")
        resume_cell.assert_not_awaited()


class _Response:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _HttpClientFake:
    instances: list[_HttpClientFake] = []

    def __init__(self, **_: Any) -> None:
        self.patches: list[bool] = []
        self._poll_count = 0
        self.instances.append(self)

    async def get(self, _: str) -> _Response:
        self._poll_count += 1
        if self._poll_count > 1:
            raise asyncio.CancelledError
        return _Response(
            {
                "apiVersion": "miles.io/v1",
                "kind": "CellList",
                "items": [
                    {
                        "apiVersion": "miles.io/v1",
                        "kind": "Cell",
                        "metadata": {"name": "cell-0", "labels": {}},
                        "spec": {"suspend": False},
                        "status": {
                            "phase": "Running",
                            "conditions": [{"type": "Healthy", "status": "False"}],
                            "workers_hash": "hash-0",
                        },
                    }
                ],
            }
        )

    async def patch(self, _: str, *, content: str, headers: dict[str, str]) -> _Response:
        assert headers == {"Content-Type": "application/json"}
        self.patches.append(json.loads(content)["spec"]["suspend"])
        return _Response()

    async def aclose(self) -> None:
        return None


class _ThreadFake:
    def __init__(self, *, target: Any, daemon: bool) -> None:
        self._target = target

    def start(self) -> None:
        try:
            self._target()
        except asyncio.CancelledError:
            pass


class TestMaybeStartMiniFtController:
    @pytest.mark.parametrize(
        ("cluster_backend", "expected_patches"),
        [("ray", [True, False]), ("kubernetes", [True])],
    )
    def test_each_cluster_backend_emits_the_expected_healing_requests(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cluster_backend: str,
        expected_patches: list[bool],
    ) -> None:
        """Each cluster backend emits only the cell healing requests it owns."""
        _HttpClientFake.instances = []
        monkeypatch.setattr(mini_ft_controller.httpx, "AsyncClient", _HttpClientFake)
        monkeypatch.setattr(mini_ft_controller.threading, "Thread", _ThreadFake)

        mini_ft_controller.maybe_start_mini_ft_controller(
            argparse.Namespace(
                mini_ft_controller_enable=True,
                api_server_port=8080,
                mini_ft_controller_poll_interval=0.0,
                mini_ft_controller_resume_delay=0.0,
                cluster_backend=cluster_backend,
            )
        )

        assert len(_HttpClientFake.instances) == 1
        assert _HttpClientFake.instances[0].patches == expected_patches
