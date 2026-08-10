from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import enum
import os
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import rpc
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"

_BLOCK_GUARD_SECONDS = 20.0


class Item(StrictBaseModel):
    name: str
    values: list[int]


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Nested(StrictBaseModel):
    item: Item
    lookup: dict[str, Item]
    tags: set[str]


@dataclasses.dataclass
class Point:
    x: int
    y: int


class Metric(StrictBaseModel):
    name: str
    value: float


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

    async def report_env(self, name: str) -> str | None:
        return os.environ.get(name)

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

    async def demo_model(self, item: Item) -> Item:
        return item

    async def demo_nested_model(self, payload: Nested) -> Nested:
        return payload

    async def report_nested_argument_types(self, payload: Nested) -> list[str]:
        return [
            type(payload).__name__,
            type(payload.item).__name__,
            type(payload.lookup["k"]).__name__,
            type(payload.tags).__name__,
            type(payload.item.values[0]).__name__,
        ]

    async def report_dataclass_argument_type(self, point: Point) -> str:
        return type(point).__name__

    async def report_enum_argument_is_member(self, colour: Colour) -> bool:
        return colour is Colour.BLUE

    async def report_scalar_argument_types(
        self, when: datetime.datetime, value: uuid.UUID, amount: Decimal, blob: bytes, pair: tuple[int, str]
    ) -> list[str]:
        return [
            type(when).__name__,
            type(value).__name__,
            type(amount).__name__,
            type(blob).__name__,
            type(pair).__name__,
        ]

    async def demo_enum(self, colour: Colour) -> Colour:
        return colour

    async def demo_dataclass(self, point: Point) -> Point:
        return Point(x=point.y, y=point.x)

    async def demo_datetime(self, when: datetime.datetime) -> datetime.datetime:
        return when

    async def demo_uuid(self, value: uuid.UUID) -> uuid.UUID:
        return value

    async def demo_decimal(self, value: Decimal) -> Decimal:
        return value

    async def demo_tuple(self, pair: tuple[int, str]) -> tuple[int, str]:
        return pair

    async def demo_optional(self, value: int | None) -> int | None:
        return value

    async def demo_union(self, value: int | str) -> int | str:
        return value

    async def demo_model_list(self, items: list[Item]) -> list[Item]:
        return items

    async def demo_bytes(self, blob: bytes) -> bytes:
        return blob

    async def demo_nan_result(self) -> float:
        return float("nan")

    async def demo_float(self, value: float) -> float:
        return value

    async def demo_optional_float(self, value: float | None) -> float | None:
        return value

    async def demo_float_metrics(self) -> dict:
        return {"loss": float("nan"), "grad_norm": float("inf"), "lr": -float("inf"), "step": 3.0}

    async def demo_float_list(self, values: list[float]) -> list[float]:
        return values

    async def demo_metric_model(self, metric: Metric) -> Metric:
        return metric

    async def report_float_repr(self, value: float) -> str:
        return repr(value)

    async def demo_bytes_list(self, blobs: list[bytes]) -> list[bytes]:
        return blobs

    async def report_union_argument_type(self, value: datetime.datetime | str) -> str:
        return type(value).__name__

    async def demo_none_result(self) -> None:
        return None

    async def demo_wrong_result_type(self) -> int:
        return "not-an-int"

    async def demo_unserializable_result(self) -> int:
        return object()

    def demo_sync_raises(self, message: str) -> None:
        raise ValueError(message)

    async def demo_async_raises(self, message: str) -> None:
        raise ValueError(message)

    def demo_system_exit(self) -> None:
        raise SystemExit(3)

    async def demo_system_exit_async(self) -> None:
        raise SystemExit(3)

    def demo_count_sync(self, tag: str) -> int:
        return self._bump(tag)

    async def demo_count_async(self, tag: str) -> int:
        return self._bump(tag)

    def demo_count_after_sleep(self, tag: str, seconds: float) -> int:
        time.sleep(seconds)
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

    async def demo_block_until_cancelled(self, tag: str) -> str:
        self._mark(tag, "start")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            (self._state_dir / f"cancelled_{tag}").touch()
            raise
        return tag

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

    def demo_marker_after_sleep(self, name: str, seconds: float) -> str:
        time.sleep(seconds)
        (self._state_dir / name).touch()
        return name

    async def demo_marker_after_sleep_async(self, name: str, seconds: float) -> str:
        await asyncio.sleep(seconds)
        (self._state_dir / name).touch()
        return name

    def demo_hang(self, tag: str) -> str:
        self._mark(tag, "start")
        threading.Event().wait(timeout=_BLOCK_GUARD_SECONDS)
        return tag

    def demo_large_upload(self, blob: str) -> int:
        return len(blob)

    def demo_large_download(self, size: int) -> list[int]:
        return list(range(size))

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


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [spec_of(worker_argv, env_var=lambda context: {"MILES_E2E_ARGV": ",".join(worker_argv)})]


def spec_of(worker_argv: list[str], *, env_var) -> ServeWorkerSpec:
    args = parse_run_args(worker_argv)
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=args.rpc_port)],
        env_var=env_var,
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.E2eWorker",
        ctor_kwargs=lambda context: dict(argv=worker_argv, state_dir=Path(args.state_dir)),
    )


def parse_run_args(worker_argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument(RPC_PORT_FLAG, type=int, required=True)
    args, _ = parser.parse_known_args(worker_argv)
    return args
