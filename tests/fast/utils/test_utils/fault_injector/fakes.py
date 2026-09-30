import os
from types import SimpleNamespace

import pytest

from miles.utils.test_utils.fault_injector.actions import process


class _CellOperations:
    def __init__(self, *, reject_stop: bool = False) -> None:
        self.stopped: list[str] = []
        self.started: list[str] = []
        self.reject_stop = reject_stop

    async def suspend(self, *, cell_id: str) -> None:
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


class _Effects:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.log: list[tuple[str, object]] = []
        monkeypatch.setattr(os, "kill", lambda pid, signum: self.log.append(("kill", (pid, signum))))
        monkeypatch.setattr(os, "_exit", lambda code: self.log.append(("exit", code)))
        monkeypatch.setattr(
            process,
            "ctypes",
            SimpleNamespace(
                CFUNCTYPE=lambda restype: lambda: lambda: self.log.append(("segfault", restype)),
                PyDLL=lambda name: SimpleNamespace(sleep=lambda seconds: self.log.append(("sleep", seconds))),
                c_uint=object(),
            ),
        )
