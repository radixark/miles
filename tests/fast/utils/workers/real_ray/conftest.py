from __future__ import annotations

import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import ray

from tests.fast.utils.workers.conftest import worker_manager_args

from miles.utils.workers.ray_worker_manager import _ACTOR_NAME, RayWorkerManager
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, PortInfo, SchedulingSpec

PLACEMENT_GROUP_READY_TIMEOUT = 120.0

_PROBE_SOURCE = """
import json, os, socket, sys, time

record_path, context_json, bind_port = sys.argv[1], sys.argv[2], int(sys.argv[3])
env_names = sys.argv[4:]

held_sockets = []
if bind_port:
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    held.bind(("", bind_port))
    held.listen(8)
    held_sockets.append(held)

record = {
    "pid": os.getpid(),
    "context": json.loads(context_json),
    "env": {name: os.environ.get(name) for name in env_names},
}
with open(record_path + ".tmp", "w") as f:
    json.dump(record, f)
os.rename(record_path + ".tmp", record_path)

time.sleep(600)
"""


@dataclass(kw_only=True)
class WorkerProbe:
    record_dir: str
    env_names: tuple[str, ...] = ()
    bind_primary: bool = False

    def launch_command(self, ctx: LaunchCommandContext) -> str:
        bind_port = ctx.self_addrs["primary"].port if self.bind_primary else 0
        return shlex.join(
            [
                sys.executable,
                "-c",
                _PROBE_SOURCE,
                str(self.record_path(cell_index=ctx.cell_index, worker_in_cell_index=ctx.worker_in_cell_index)),
                json.dumps(ctx.model_dump(mode="json")),
                str(bind_port),
                *self.env_names,
            ]
        )

    def record_path(self, *, cell_index: int, worker_in_cell_index: int) -> Path:
        return Path(self.record_dir) / f"{cell_index}-{worker_in_cell_index}.json"

    def wait_for_records(self, count: int, *, timeout: float = 120.0) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            records = self.read_records()
            if len(records) >= count:
                return records
            assert time.monotonic() < deadline, f"only {sorted(records)} of {count} records appeared"
            time.sleep(0.2)

    def read_records(self) -> dict[str, dict[str, Any]]:
        return {path.stem: json.loads(path.read_text()) for path in sorted(Path(self.record_dir).glob("*.json"))}

    def context_of(self, *, cell_index: int, worker_in_cell_index: int) -> dict[str, Any]:
        return self.read_records()[f"{cell_index}-{worker_in_cell_index}"]["context"]

    def wait_until_gone(self, pids: list[int], *, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            alive = [pid for pid in pids if is_process_running(pid)]
            if not alive:
                return
            assert time.monotonic() < deadline, f"processes {alive} are still running"
            time.sleep(0.2)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def make_command_spec(
    name: str,
    *,
    launch_command: Callable[[LaunchCommandContext], str],
    num_cells: int = 1,
    num_workers_per_cell: int = 1,
    port_infos: list[PortInfo] | None = None,
    env_var: dict[str, str] | None = None,
    num_gpus_per_worker: float = 0,
    num_gpu_slots_per_worker: int = 0,
    pg_name: str | None = None,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        port_infos=(
            port_infos if port_infos is not None else [PortInfo(name="primary", static_port=8000, allow_dynamic=True)]
        ),
        env_var=lambda _ctx: dict(env_var or {}),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=num_gpus_per_worker,
            num_gpu_slots_per_worker=num_gpu_slots_per_worker,
            pg_name=pg_name,
        ),
        launch_command=launch_command,
    )


def kill_quietly(handle: ray.actor.ActorHandle) -> None:
    try:
        ray.kill(handle, no_restart=True)
    except Exception:
        pass


def kill_named_worker_manager() -> None:
    try:
        handle = ray.get_actor(_ACTOR_NAME)
    except ValueError:
        return
    kill_quietly(handle)


def wait_until_named_manager_is_gone(*, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            ray.get_actor(_ACTOR_NAME)
        except ValueError:
            return
        assert time.monotonic() < deadline, f"the {_ACTOR_NAME} actor is still registered"
        time.sleep(0.2)


@pytest.fixture
def worker_probe_factory(tmp_path: Path) -> Callable[..., WorkerProbe]:
    """Builds probes whose worker command records its launch context, pid and environment, then lingers."""
    counter = iter(range(1000))

    def _make(*, env_names: tuple[str, ...] = (), bind_primary: bool = False) -> WorkerProbe:
        record_dir = tmp_path / f"probe-{next(counter)}"
        record_dir.mkdir()
        return WorkerProbe(record_dir=str(record_dir), env_names=env_names, bind_primary=bind_primary)

    return _make


@pytest.fixture(autouse=True)
def clean_named_worker_manager(ray_local_mode):
    """Keeps the fixed-name manager actor from leaking between tests of the shared cluster."""
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()
    yield
    kill_named_worker_manager()
    wait_until_named_manager_is_gone()


@pytest.fixture
def manager_factory(ray_local_mode) -> Callable[..., ray.actor.ActorHandle]:
    """Launches the manager the way production does: the fixed-name ray actor that owns its workers."""
    handles: list[ray.actor.ActorHandle] = []

    def _launch(specs: list[CommandWorkerSpec], pgs: dict[str, Any] | None = None) -> ray.actor.ActorHandle:
        handle = RayWorkerManager.launch(worker_manager_args(), specs, pgs if pgs is not None else {})
        handles.append(handle)
        return handle

    yield _launch

    for handle in handles:
        kill_quietly(handle)


class CellStoppableManager(RayWorkerManager):
    async def stop_cell(self, pool_id: str, cell_index: int) -> None:
        await self._pools[pool_id].cells[cell_index].stop()


@pytest.fixture
def placement_group_factory(ray_local_mode) -> Callable[..., Any]:
    """Creates a real placement group and describes it with a non-identity bundle and gpu mapping."""
    created: list[Any] = []

    def _make(*, num_bundles: int, first_gpu_id: int = 0):
        from miles.ray.placement_group import PlacementGroupInfo

        pg = ray.util.placement_group([{"CPU": 0.4, "GPU": 0.5} for _ in range(num_bundles)], strategy="PACK")
        # Bounded, because a cluster without the logical GPUs ray_local_mode declares leaves
        # this pending forever, and a hung shard is far harder to read than a named failure.
        try:
            ray.get(pg.ready(), timeout=PLACEMENT_GROUP_READY_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            raise AssertionError(
                f"placement group of {num_bundles} bundles never became ready; the cluster offers "
                f"{ray.cluster_resources()} and each bundle needs a 0.5 GPU slot"
            ) from None
        created.append(pg)
        return PlacementGroupInfo(
            pg=pg,
            pg_reordered_bundle_indices=list(reversed(range(num_bundles))),
            pg_reordered_gpu_ids=[first_gpu_id + index for index in range(num_bundles)],
        )

    yield _make

    for pg in created:
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            pass


@pytest.fixture
def cell_stoppable_manager_factory(ray_local_mode) -> Callable[..., ray.actor.ActorHandle]:
    """Launches the manager with a test-only entry point into its per-cell teardown."""
    handles: list[ray.actor.ActorHandle] = []

    def _launch(specs: list[CommandWorkerSpec], pgs: dict[str, Any] | None = None) -> ray.actor.ActorHandle:
        handle = ray.remote(CellStoppableManager).options(name=_ACTOR_NAME).remote()
        handles.append(handle)
        ray.get(handle.init.remote(worker_manager_args(), specs, pgs if pgs is not None else {}))
        return handle

    yield _launch

    for handle in handles:
        kill_quietly(handle)
