from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.workers.rpc.client.misc import RpcWorkerCallError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"

REPO_ROOT = Path(__file__).resolve().parents[4]

READY_TIMEOUT_SECONDS = 120.0


class ConformanceWorker:
    def __init__(self, *, tag: str) -> None:
        self._tag = tag

    def add(self, a: int, b: int) -> int:
        return a + b

    async def add_async(self, *, a: int, b: int) -> int:
        return a + b

    def report_tag(self) -> str:
        return self._tag

    def report_pid(self) -> int:
        return os.getpid()

    def report_ray_actor_id(self) -> str | None:
        import ray

        return ray.get_runtime_context().get_actor_id() if ray.is_initialized() else None

    def boom(self, *, message: str) -> None:
        raise RuntimeError(message)


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [compute_spec(rpc_port=_parse_rpc_port(worker_argv))]


def compute_spec(*, rpc_port: int) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=rpc_port, allow_dynamic=rpc_port == 0)],
        # prepend rather than replace: the inherited path is what carries sglang and megatron,
        # which the worker's own imports reach through miles.utils.arguments
        env_var=lambda _ctx: {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.ConformanceWorker",
        ctor_kwargs=lambda ctx: dict(tag=f"cell-{ctx.cell_index}-worker-{ctx.worker_in_cell_index}"),
    )


async def _a_call_returns_the_declared_type(handle: BaseWorkerHandle) -> None:
    result = await handle.add(a=3, b=4)
    assert result == 7 and isinstance(result, int)


async def _positional_arguments_reach_the_worker(handle: BaseWorkerHandle) -> None:
    assert await handle.add(3, 4) == 7


async def _an_async_method_answers_too(handle: BaseWorkerHandle) -> None:
    assert await handle.add_async(a=1, b=2) == 3


async def _the_worker_keeps_the_state_its_constructor_built(handle: BaseWorkerHandle) -> None:
    assert await handle.report_tag() == "cell-0-worker-0"


async def _the_call_runs_outside_the_caller(handle: BaseWorkerHandle) -> None:
    assert await handle.report_pid() != os.getpid()


async def _a_remote_failure_reaches_the_caller(handle: BaseWorkerHandle) -> None:
    with pytest.raises(RpcWorkerCallError, match="deliberate"):
        await handle.boom(message="deliberate")


async def _an_unknown_method_is_refused_before_the_wire(handle: BaseWorkerHandle) -> None:
    with pytest.raises(AttributeError):
        await handle.add_typo(a=1, b=2)


async def _a_wrong_argument_type_is_refused(handle: BaseWorkerHandle) -> None:
    with pytest.raises(ValidationError):
        await handle.add(a="three", b=4)


async def _readiness_is_answered(handle: BaseWorkerHandle) -> None:
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)


HandleCheck = Callable[[BaseWorkerHandle], Awaitable[None]]

SHARED_CHECKS: list[HandleCheck] = [
    _a_call_returns_the_declared_type,
    _positional_arguments_reach_the_worker,
    _an_async_method_answers_too,
    _the_worker_keeps_the_state_its_constructor_built,
    _the_call_runs_outside_the_caller,
    _an_unknown_method_is_refused_before_the_wire,
    _readiness_is_answered,
]

RPC_ONLY_CHECKS: list[HandleCheck] = [
    _a_remote_failure_reaches_the_caller,
    _a_wrong_argument_type_is_refused,
]

CHECKS: list[HandleCheck] = [*SHARED_CHECKS, *RPC_ONLY_CHECKS]

SHARED_CHECK_IDS = [check.__name__.lstrip("_") for check in SHARED_CHECKS]

CHECK_IDS = [check.__name__.lstrip("_") for check in CHECKS]


def _parse_rpc_port(worker_argv: list[str]) -> int:
    assert RPC_PORT_FLAG in worker_argv, f"{worker_argv} must name the port the conformance worker serves on"
    return int(worker_argv[worker_argv.index(RPC_PORT_FLAG) + 1])
