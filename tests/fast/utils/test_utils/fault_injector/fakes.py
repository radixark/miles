import asyncio
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from miles.utils.test_utils.fault_injector import controller
from miles.utils.test_utils.fault_injector.actions import process
from miles.utils.test_utils.fault_injector.actions.base import FaultHookResources
from miles.utils.test_utils.fault_injector.actions.cell import StopCellAction
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand, FaultHookOperation, _FaultHookController
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookRequest
from miles.utils.workers import process_utils


class _CellOperations:
    def __init__(self, *, reject_stop: bool = False, stop_gate: asyncio.Event | None = None) -> None:
        self.stopped: list[str] = []
        self.started: list[str] = []
        self.entered: list[str] = []
        self.reject_stop = reject_stop
        self.stop_gate = stop_gate

    async def suspend(self, *, cell_id: str) -> None:
        self.entered.append(cell_id)
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        if self.reject_stop:
            raise RuntimeError("worker manager rejected the stop")
        self.stopped.append(cell_id)

    async def resume(self, *, cell_id: str) -> None:
        self.started.append(cell_id)


class _Controller:
    def __init__(self, *, observed_after_reads: int = 0) -> None:
        self.reads = 0
        self.observed_after_reads = observed_after_reads

    @property
    def cell_ids(self) -> list[str]:
        self.reads += 1
        return [] if self.reads <= self.observed_after_reads else ["trainer-engine-actor-0"]


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Timer:
    def __init__(self, *, interval: float, function: Callable[..., None], kwargs: dict[str, Any]) -> None:
        self.interval = interval
        self.function = function
        self.kwargs = kwargs
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.function(**self.kwargs)


class _Effects:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.log: list[tuple[str, object]] = []
        monkeypatch.setattr(os, "kill", lambda pid, signum: self.log.append(("kill", (pid, signum))))
        monkeypatch.setattr(os, "_exit", lambda code: self.log.append(("exit", code)))
        monkeypatch.setattr(
            process_utils, "signal_process_tree", lambda proc, signum: self.log.append(("tree", (proc.pid, signum)))
        )
        monkeypatch.setattr(
            process,
            "ctypes",
            SimpleNamespace(
                CFUNCTYPE=lambda restype: lambda: lambda: self.log.append(("segfault", restype)),
                PyDLL=lambda name: SimpleNamespace(sleep=lambda seconds: self.log.append(("sleep", seconds))),
                c_uint=object(),
            ),
        )
        monkeypatch.setattr(
            process,
            "threading",
            SimpleNamespace(Event=lambda: SimpleNamespace(wait=lambda: self.log.append(("wait", None)))),
        )


class _MarkerOperations:
    def __init__(self, *, log: list[object], fail: bool) -> None:
        self.log = log
        self.fail = fail

    async def suspend(self, *, cell_id: str) -> None:
        self.log.append(("hook", cell_id))
        if self.fail:
            raise RuntimeError(f"fault hook {cell_id} failed")


def _arm_marker_hook(
    monkeypatch: pytest.MonkeyPatch,
    *,
    log: list[object],
    hook_name: FaultHookName,
    fail: bool = False,
    **filters: int,
) -> _FaultHookController:
    hooks = _FaultHookController()
    hooks.configure(resources=FaultHookResources(cell_operations=_MarkerOperations(log=log, fail=fail)))
    request = FaultHookRequest(
        request_id="marker", hook_name=hook_name, action=StopCellAction(cell_id=hook_name.value), **filters
    )
    hooks.apply(FaultHookCommand(operation=FaultHookOperation.SET, request=request))
    monkeypatch.setattr(controller, "fault_hook_controller", hooks)
    return hooks


class _ApiServer:
    def __init__(self, *, target: dict[str, object], get_status: int = 200, post_status: int = 200) -> None:
        self.target = target
        self.get_status = get_status
        self.post_status = post_status
        self.post_error: Exception | None = None
        self.requests: list[httpx.Request] = []
        self.client_timeouts: list[object] = []

    def client(self, *, timeout: float) -> httpx.AsyncClient:
        self.client_timeouts.append(timeout)
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), timeout=timeout)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            return httpx.Response(self.get_status, json=self.target)
        if self.post_error is not None:
            raise self.post_error
        return httpx.Response(self.post_status, json={})
