"""The monarch worker manager on real local procs: the same rpc conformance contract as the ray column."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("monarch")

from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.workers.conformance import (
    CHECK_IDS,
    CHECKS,
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    HandleCheck,
    compute_spec,
)
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_specs import FakeCommandSpec, FakeServeSpec
from tests.fast.utils.workers.monarch.monarch_context import in_monarch_context

from miles.utils.workers.monarch.manager import MonarchWorkerManager, create_monarch_backend_capability
from miles.utils.workers.monarch.runtime import GpuLayoutRequest
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import SchedulingSpec

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=["rpc-comm"])

CELL_ID = compute_cell_id(pool_id=POOL_ID, cell_index=0)
COMMAND_POOL_ID = "sleeper"
COMMAND_CELL_ID = compute_cell_id(pool_id=COMMAND_POOL_ID, cell_index=0)
LOST_CELL_TIMEOUT_SECONDS = 60.0
# A sleep no other process on the machine runs, so the test kills its own command and nothing else; computed here,
# in the test's process, and carried to the worker inside the spec, since the worker imports this module afresh
_SLEEPER_SECONDS = f"600.{os.getpid()}"

_NO_GPUS = GpuLayoutRequest(num_gpus=0, rollout_offset=0, use_critic=False, num_gpus_per_node=0)


@dataclass
class _LaunchedPool:
    manager: Any
    infos: list[WorkerInfo]
    handles: list[BaseWorkerHandle]


def _sleeper_spec(seconds: str) -> FakeCommandSpec:
    return FakeCommandSpec(
        name=COMMAND_POOL_ID,
        port_infos=[],
        fixed_scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0),
        command=lambda ctx: f"sleep {seconds}",
    )


@pytest.fixture(scope="module")
def launched_pool() -> Iterator[_LaunchedPool]:
    launched = in_monarch_context(_launch_pool)
    yield launched
    in_monarch_context(lambda: launched.manager.shutdown.call_one().get())


def _launch_pool() -> _LaunchedPool:
    manager = MonarchWorkerManager.launch(
        worker_manager_args(env_report_interval_seconds=0.0),
        [compute_spec(rpc_port=0), _sleeper_spec(_SLEEPER_SECONDS)],
        comm_backend=WorkerCommBackend.RPC,
        gpu_layout=_NO_GPUS,
    )
    provider = create_monarch_backend_capability(manager).dynamic_worker_provider(pool_ids=[POOL_ID])
    (infos,) = provider.get_worker_infos(cell_ids=[CELL_ID])
    handles = provider.get_handles_of_worker_infos(infos)
    return _LaunchedPool(manager=manager, infos=infos, handles=[handles[info.name] for info in infos])


@pytest.fixture
async def rpc_handle(launched_pool: _LaunchedPool) -> AsyncIterator[BaseWorkerHandle]:
    handle = launched_pool.handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestAMonarchLaunchedWorkerServedOverRpc:
    def test_the_launcher_answers_with_an_rpc_handle(self, launched_pool: _LaunchedPool):
        """Phase 1 serves every worker over rpc, so the driver is never handed a monarch actor to call."""
        assert isinstance(launched_pool.handles[0], RpcWorkerHandle)

    def test_the_worker_serves_on_the_port_the_launcher_allocated(self, launched_pool: _LaunchedPool):
        """Ports come from the backend-neutral allocator, probed inside the worker's own proc."""
        assert launched_pool.infos[0].self_addrs["rpc"].port > 0

    async def test_the_worker_is_reachable(self, rpc_handle: BaseWorkerHandle):
        """Monarch started the worker, http drives it."""
        assert await rpc_handle.add(a=2, b=5) == 7

    async def test_the_worker_runs_in_a_process_of_its_own(self, rpc_handle: BaseWorkerHandle):
        """A cell is a proc mesh: the worker must not share the test's process."""
        assert await rpc_handle.report_pid() != os.getpid()

    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_handle_contract_holds(self, rpc_handle: BaseWorkerHandle, check: HandleCheck):
        """The same contract as the ray column, now over a worker that monarch launched."""
        await check(rpc_handle)


class TestCellLifecycle:
    async def test_every_cell_starts_alive(self, launched_pool: _LaunchedPool):
        infos = await launched_pool.manager.get_cell_infos.call_one(pool_ids=[POOL_ID, COMMAND_POOL_ID], category=None)

        assert {cell_id: info.alive for cell_id, info in infos.items()} == {CELL_ID: True, COMMAND_CELL_ID: True}

    async def test_a_cell_whose_worker_dies_is_torn_down(self, launched_pool: _LaunchedPool):
        """Killing the command lets its CommandActor exit the proc; the liveness scan must tear the whole cell down."""
        infos = await launched_pool.manager.get_worker_infos.call_one(COMMAND_CELL_ID)
        assert len(infos) == 2
        os.kill(_pid_of_sleeper(), signal.SIGKILL)

        start = time.monotonic()
        while time.monotonic() < start + LOST_CELL_TIMEOUT_SECONDS:
            cells = await launched_pool.manager.get_cell_infos.call_one(pool_ids=[COMMAND_POOL_ID], category=None)
            if not cells[COMMAND_CELL_ID].alive:
                return
            await asyncio.sleep(0.2)
        raise AssertionError(
            f"cell {COMMAND_CELL_ID} is still alive {LOST_CELL_TIMEOUT_SECONDS}s after losing a worker"
        )

    async def test_a_stopped_cell_can_be_started_again(self, launched_pool: _LaunchedPool):
        manager = launched_pool.manager
        await manager.stop_cells.call_one([COMMAND_CELL_ID])
        stopped = await manager.get_cell_infos.call_one(pool_ids=[COMMAND_POOL_ID], category=None)
        assert not stopped[COMMAND_CELL_ID].alive

        await manager.start_cells.call_one([COMMAND_CELL_ID])

        started = await manager.get_cell_infos.call_one(pool_ids=[COMMAND_POOL_ID], category=None)
        assert started[COMMAND_CELL_ID].alive


def _pid_of_sleeper() -> int:
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                if f.read().split(b"\0")[:2] == [b"sleep", _SLEEPER_SECONDS.encode()]:
                    return int(entry)
        except OSError:
            continue
    raise AssertionError("no sleeper subprocess is running")


class TestACellThatCannotStart:
    def test_a_worker_whose_constructor_raises_fails_the_launch(self):
        """The failure has to reach the driver, and the half-started mesh must not outlive the attempt."""
        spec = FakeServeSpec(
            name="broken",
            port_infos=[],
            fixed_scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            worker_class=f"{__name__}._RaisingWorker",
        )

        with pytest.raises(Exception, match="cannot be built"):
            in_monarch_context(
                MonarchWorkerManager.launch,
                worker_manager_args(env_report_interval_seconds=0.0),
                [spec],
                comm_backend=WorkerCommBackend.RPC,
                gpu_layout=_NO_GPUS,
            )


class _RaisingWorker:
    def __init__(self) -> None:
        raise RuntimeError("this worker cannot be built")
