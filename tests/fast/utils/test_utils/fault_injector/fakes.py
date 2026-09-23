import asyncio
import os
from types import SimpleNamespace

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
    def __init__(
        self,
        *,
        observed_after_reads: int = 0,
        cell_ids: tuple[str, ...] = ("trainer-engine-actor-0",),
        initial_cell_ids: tuple[str, ...] = (),
    ) -> None:
        self.reads = 0
        self.observed_after_reads = observed_after_reads
        self._cell_ids = cell_ids
        self._initial_cell_ids = initial_cell_ids

    @property
    def cell_ids(self) -> list[str]:
        self.reads += 1
        return list(self._initial_cell_ids if self.reads <= self.observed_after_reads else self._cell_ids)


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
