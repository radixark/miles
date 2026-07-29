from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from pathlib import Path

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import rpc

_BLOCK_GUARD_SECONDS = 20.0


class Event(StrictBaseModel):
    tag: str
    phase: str
    thread_name: str
    at: float


WORKER_FACTORY_ERROR = "e2e worker factory refuses to build a worker"


class E2eWorker:
    def __init__(self, argv: list[str], state_dir: Path) -> None:
        self._argv = argv
        self._state_dir = state_dir
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._events: list[Event] = []
        self._sync_gates: dict[str, threading.Event] = {}
        self._async_gates: dict[str, asyncio.Event] = {}

    async def report_pid(self) -> int:
        return os.getpid()

    async def report_argv(self) -> list[str]:
        return self._argv

    async def report_counter(self, tag: str) -> int:
        with self._lock:
            return self._counters.get(tag, 0)

    async def report_events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def demo_sync(self, a: int, b: int) -> int:
        return a + b

    def demo_default_arg(self, name: str = "world") -> str:
        return f"hello {name}"

    def demo_optional_arg(self, name: str | None) -> str:
        return repr(name)

    async def demo_async(self, value: dict) -> dict:
        return value

    def demo_count_sync(self, tag: str) -> int:
        return self._bump(tag)

    async def demo_count_async(self, tag: str) -> int:
        return self._bump(tag)

    def demo_sleep_sync(self, tag: str, seconds: float) -> str:
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return tag

    async def demo_sleep_async(self, tag: str, seconds: float) -> str:
        self._mark(tag, "start")
        await asyncio.sleep(seconds)
        self._mark(tag, "end")
        return tag

    def demo_block_sync(self, tag: str) -> str:
        self._mark(tag, "start")
        self._sync_gate(tag).wait(timeout=_BLOCK_GUARD_SECONDS)
        self._mark(tag, "end")
        return threading.current_thread().name

    def demo_instant_sync(self, tag: str) -> str:
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    async def demo_block_async(self, tag: str) -> str:
        self._mark(tag, "start")
        await self._async_gate(tag).wait()
        self._mark(tag, "end")
        return threading.current_thread().name

    async def demo_instant_async(self, tag: str) -> str:
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_sleep_on_left(self, tag: str, seconds: float) -> str:
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_instant_on_left(self, tag: str) -> str:
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="right")
    def demo_sleep_on_right(self, tag: str, seconds: float) -> str:
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_block_on_left(self, tag: str) -> str:
        self._mark(tag, "start")
        self._sync_gate(tag).wait(timeout=_BLOCK_GUARD_SECONDS)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc()
    async def demo_async_on_left(self, tag: str) -> str:
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    async def release(self, tag: str) -> bool:
        self._sync_gate(tag).set()
        self._async_gate(tag).set()
        return True

    async def release_every_gate(self) -> int:
        with self._lock:
            gates = [*self._sync_gates.values(), *self._async_gates.values()]
        for gate in gates:
            gate.set()
        return len(gates)

    def _bump(self, tag: str) -> int:
        with self._lock:
            self._counters[tag] = self._counters.get(tag, 0) + 1
            return self._counters[tag]

    def _mark(self, tag: str, phase: str) -> None:
        event = Event(tag=tag, phase=phase, thread_name=threading.current_thread().name, at=time.monotonic())
        with self._lock:
            self._events.append(event)

    def _sync_gate(self, tag: str) -> threading.Event:
        with self._lock:
            return self._sync_gates.setdefault(tag, threading.Event())

    def _async_gate(self, tag: str) -> asyncio.Event:
        with self._lock:
            return self._async_gates.setdefault(tag, asyncio.Event())


def make_worker(argv: list[str]) -> E2eWorker:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    args, _ = parser.parse_known_args(argv)

    return E2eWorker(argv, Path(args.state_dir))


def make_raising_worker(argv: list[str]) -> E2eWorker:
    raise RuntimeError(WORKER_FACTORY_ERROR)
