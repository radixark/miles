"""Monarch as the runtime of the worker manager: one Monarch ``ProcMesh`` per cell.

Monarch neither assigns gpus nor knows Miles' slots, so the layout here maps every logical gpu slot onto a
``(host, device)`` of the worker host mesh, packing hosts in mesh order like a Ray ``PACK`` placement group.
Every gpu worker sees its whole host, as under Ray, and selects the devices the manager placed it on.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from monarch.actor import HostMesh, ProcMesh, concurrent_endpoint, current_rank, default_bootstrap_cmd, endpoint

from miles.utils.misc import VISIBLE_DEVICES_ENV_VARS, get_current_node_ip, visible_devices
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.env_vars import WORKER_GPU_IDS_ENV_VAR
from miles.utils.workers.monarch.base import MilesActor
from miles.utils.workers.monarch.handles import (
    MonarchCellHandle,
    MonarchWorkerHandle,
    call_endpoint,
    probe_cell_is_dead,
)
from miles.utils.workers.rpc.common.metadata import canonicalize_method_arguments, collect_rpc_method_specs
from miles.utils.workers.rpc.server.executor import RpcCallExecutor
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunchRequest, WorkerProcess, WorkerRuntime

if TYPE_CHECKING:
    # manager.py imports this module, so the manager's class is imported for annotations only
    from miles.utils.workers.monarch.manager import MonarchWorkerManager

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_SECONDS = 60.0
_PROCS_DIM = "procs"


@dataclass(frozen=True, kw_only=True)
class GpuSlot:
    """One gpu a worker can be placed on."""

    # the host's position in the worker host mesh
    host_rank: int
    # the gpu's physical id on that host
    gpu_id: int


@dataclass(frozen=True, kw_only=True)
class GpuLayoutRequest:
    """What the driver asks of the gpu layout, as it asks Ray for placement groups."""

    # gpus the run's gpu workers take in total, trainers and rollout engines together
    num_gpus: int
    # the slot the rollout engines' slots start at: past the trainers' slots, or 0 when they share them (colocate)
    rollout_offset: int
    # whether the critic gets a layout too, which shares the actor's slots
    use_critic: bool
    # gpus to use on each host at most (--num-gpus-per-node), even where a host shows more
    num_gpus_per_node: int


@dataclass(frozen=True, kw_only=True)
class MonarchGpuLayout:
    """The same role as Ray's placement groups: logical slots of each named layout, in placement order."""

    slots_of_layout: dict[str, list[GpuSlot]]


@dataclass(frozen=True, kw_only=True)
class ProbedHost:
    """What a host reports about itself before the layout is computed."""

    # the address other hosts reach it by
    ip: str
    # physical ids, in the order the host's processes see them; what to_local_gpu_id maps a worker's gpu ids from
    gpu_ids: list[int]


def compute_monarch_gpu_layout(*, request: GpuLayoutRequest, hosts: list[ProbedHost]) -> MonarchGpuLayout:
    """Lay the run's gpu slots out on the hosts, filling each host before the next, as a Ray ``PACK`` group does.

    Every host offers the same number of gpus, the fewest any host shows, capped by ``num_gpus_per_node``. The
    "actor" layout is every slot, "rollout" the slots from ``rollout_offset`` on, and "critic", if used, the actor's.

    ``num_gpus`` is the run's total over all hosts, not a per-host count. Examples, as ``(host_rank, gpu_id)`` slots:

    - 2 hosts with 8 gpus each (num_gpus_per_node=8); the run needs 12 gpus in total, 8 for the trainers and 4 for
      rollout (num_gpus=12, rollout_offset=8). Slots fill host 0 first, then host 1:
      actor = (0,0) .. (0,7), (1,0) .. (1,3), of which the trainers take the first 8, i.e. all of host 0;
      rollout = (1,0) .. (1,3). Gpus 4..7 of host 1 stay unused.
    - the same 2 hosts with num_gpus_per_node=4: each host offers only gpus 0..3, 8 slots in total, so a run
      needing 12 fails. One needing 8, 4 for the trainers and 4 for rollout (num_gpus=8, rollout_offset=4), gets
      actor = (0,0) .. (0,3), (1,0) .. (1,3), the trainers taking host 0's 4; rollout = (1,0) .. (1,3).
    - 1 host whose processes see only gpus 2 and 3 (CUDA_VISIBLE_DEVICES=2,3); a colocated run of 2 gpus, the
      trainers and rollout sharing them (num_gpus=2, rollout_offset=0): actor = rollout = (0,2), (0,3).
    """
    num_gpus = request.num_gpus
    num_gpus_per_host = min([len(host.gpu_ids) for host in hosts] + [request.num_gpus_per_node])
    capacity = len(hosts) * num_gpus_per_host
    assert num_gpus <= capacity, (
        f"the run lays its gpu workers out on {num_gpus} gpus, but the {len(hosts)} monarch hosts offer "
        f"{num_gpus_per_host} each ({capacity} in total)"
    )
    slots = [
        GpuSlot(host_rank=host_rank, gpu_id=gpu_id)
        for host_rank, host in enumerate(hosts)
        for gpu_id in host.gpu_ids[:num_gpus_per_host]
    ][:num_gpus]
    slots_of_layout = {"actor": slots, "rollout": slots[request.rollout_offset :]}
    if request.use_critic:
        slots_of_layout["critic"] = slots_of_layout["actor"]
    return MonarchGpuLayout(slots_of_layout=slots_of_layout)


class MonarchNodeProbe(MilesActor):
    """A throwaway actor, one per worker host, that reports the host's address and gpus.

    Monarch has no view of a host's gpus, so the manager spawns one of these on every host before it lays the gpu
    slots out, and stops them once they answered.
    """

    @endpoint
    def probe(self) -> ProbedHost:
        # TODO: get_current_node_ip asks Ray's private lookup, which off a Ray node probes the default route;
        # Monarch could offer its own node address instead, but that lookup already works, so no need yet.
        return ProbedHost(ip=get_current_node_ip(), gpu_ids=visible_gpu_ids())


def visible_gpu_ids() -> list[int]:
    """The physical ids of the gpus processes on this host can use."""
    if (visible := visible_devices()) is not None:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        assert all(device.isdigit() for device in devices), f"monarch places gpus by index, not by uuid: {visible}"
        return [int(device) for device in devices]
    if any(name in os.environ for name in VISIBLE_DEVICES_ENV_VARS):
        # set, but to an empty list: every gpu is hidden
        return []
    devices = [name[len("nvidia") :] for name in os.listdir("/dev") if name.startswith("nvidia")]
    return sorted(int(index) for index in devices if index.isdigit())


async def probe_hosts(worker_hosts: HostMesh) -> list[ProbedHost]:
    """Every host's address and visible gpus, in mesh order."""
    probe_procs = worker_hosts.spawn_procs(per_host={_PROCS_DIM: 1}, name="miles-node-probe")
    try:
        probes = probe_procs.spawn("miles-node-probe", MonarchNodeProbe)
        answers = await probes.probe.call()
    finally:
        await probe_procs.stop()
    return [host for _, host in sorted(answers.items(), key=lambda item: item[0].rank)]


class MonarchWorkerActor(MilesActor):
    """Hosts one launched worker of a cell; the cell's proc mesh hosts one of these per worker.

    Every rank receives the targets of the whole cell and builds the one at its own rank, so a cell is spawned
    with a single ``ProcMesh.spawn``.
    """

    def __init__(self, launches: list[WorkerLaunchRequest], manager: MonarchWorkerManager) -> None:
        install_manager_handle(manager)
        target = launches[current_rank().rank]
        self._target = target.actor_class(**target.ctor_kwargs)
        # A worker served natively runs each call where its rpc specs put it, exactly as the rpc server would;
        # the launcher's own verbs (node probes, run, shutdown, ...) run on one thread of their own.
        self._specs = collect_rpc_method_specs(target.actor_class) if target.served_natively else {}
        self._rpc_executor: RpcCallExecutor | None = None
        self._launcher_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monarch-launcher")

    # Every call runs as a task of its own, so one still running blocks neither a call to another of the worker's
    # concurrency groups nor a liveness ping, as with the rpc server.
    @concurrent_endpoint
    async def invoke(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Call ``method`` of the worker this actor hosts, and answer what it returned."""
        return await self._dispatch(method, args=args, kwargs=kwargs)

    @concurrent_endpoint
    async def invoke_each(self, method: str, kwargs_per_worker: list[dict[str, Any]]) -> tuple[int, Any]:
        """One mesh call reaching the whole cell; every rank takes its own arguments and answers with its rank."""
        rank = current_rank().rank
        return rank, await self._dispatch(method, args=(), kwargs=kwargs_per_worker[rank])

    @concurrent_endpoint
    async def ping(self) -> bool:
        return True

    async def _dispatch(self, method: str, *, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run one call of the worker this actor hosts.

        A method the worker serves natively runs where its rpc specs put it (its concurrency group's thread, or the
        event loop), through the same executor the rpc server uses. Any other method, i.e. the launcher's own verbs,
        runs on this actor's event loop if it is a coroutine and on the launcher thread if not, so a blocking one
        never stalls the loop the liveness pings are answered on.
        """
        if (spec := self._specs.get(method)) is not None:
            if self._rpc_executor is None:
                self._rpc_executor = RpcCallExecutor(worker=self._target, specs=self._specs)
            canonical = canonicalize_method_arguments(spec=spec, args=args, kwargs=kwargs)
            return await self._rpc_executor.call(spec=spec, kwargs=canonical)

        fn = getattr(self._target, method)  # config-access-exempt: attribute selected at runtime from method
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(self._launcher_thread, partial(fn, *args, **kwargs))


# The worker manager of this run, as seen from this process: the driver installs it when it launches the manager,
# and every launched worker when it starts, since Monarch has no registry to look an actor up by name.
_MANAGER_HANDLE: MonarchWorkerManager | None = None


def install_manager_handle(handle: MonarchWorkerManager) -> None:
    global _MANAGER_HANDLE
    _MANAGER_HANDLE = handle


def installed_manager_handle() -> MonarchWorkerManager:
    assert _MANAGER_HANDLE is not None, "this process was not launched by a monarch worker manager"
    return _MANAGER_HANDLE


@dataclass(frozen=True, kw_only=True)
class _CellMesh:
    """The Monarch meshes one launch of a cell runs in."""

    # unique across the run's launches: the cell id and a counter, e.g. "trainer-engine-actor-00001-3"
    name: str
    # one process per worker, spawned on the hosts the cell is placed on
    proc_mesh: ProcMesh
    # one MonarchWorkerActor per process, each hosting its worker
    actor_mesh: MonarchWorkerActor
    # the cell's workers, one per rank of the mesh
    num_workers: int


class MonarchWorkerRuntime(WorkerRuntime):
    """Runs the worker manager's cells as Monarch proc meshes: one mesh per launch of a cell.

    A cell of gpu workers is spawned on the worker hosts its slots are laid out on; a cell of cpu-only workers on the
    head host, the driver's.
    """

    def __init__(
        self,
        *,
        head_host: HostMesh,
        worker_hosts: HostMesh,
        layout: MonarchGpuLayout,
        manager: MonarchWorkerManager,
        capability_factory: Callable[[], BackendCapability],
    ) -> None:
        self._head_host = head_host
        self._worker_hosts = worker_hosts
        self._layout = layout
        self._manager = manager
        self._capability_factory = capability_factory
        self._mesh_counter = itertools.count()

    async def launch_cell(self, *, cell_id: str, launches: list[WorkerLaunchRequest]) -> list[WorkerProcess]:
        """Spawn the cell's workers as one proc mesh, and answer once every one of them answers a ping.

        The mesh runs one process per worker on the hosts ``_place`` picks, each started with its own environment
        (its spec's variables and the gpus it may use), and one MonarchWorkerActor per process builds the worker.
        Worker ``i`` is the mesh's rank ``i``: process ``i % procs_per_host`` on the cell's ``i // procs_per_host``-th
        host.
        """
        host_mesh, host_ranks, procs_per_host = self._place(launches)
        mesh_name = f"{cell_id}-{next(self._mesh_counter)}"
        host_dim = _host_dim_of(host_mesh)
        host_slice = (
            host_mesh if host_dim is None else host_mesh.slice(**{host_dim: slice(host_ranks[0], host_ranks[-1] + 1)})
        )

        base_command = default_bootstrap_cmd()
        envs = [
            _compute_env(
                launch,
                gpu_ids=None if launch.gpu_slots is None else self.physical_gpu_ids(launch.gpu_slots),
            )
            for launch in launches
        ]
        proc_mesh = host_slice.spawn_procs(
            per_host={_PROCS_DIM: procs_per_host},
            name=mesh_name,
            bootstrap_command=lambda point: base_command.with_env(envs[point.rank]),
        )
        try:
            actor_mesh = proc_mesh.spawn(mesh_name, MonarchWorkerActor, launches, self._manager)
            await actor_mesh.ping.call()
        except BaseException:
            # The core never learns of processes it was not handed, so nobody else would stop this mesh. It is a
            # BaseException so a cancelled launch stops the mesh too; re-raising it unchanged leaves Monarch's own
            # handling of a BaseException in an endpoint (it fails the actor) as it is.
            await _stop_proc_mesh(proc_mesh, mesh_name=mesh_name)
            raise
        cell_mesh = _CellMesh(name=mesh_name, proc_mesh=proc_mesh, actor_mesh=actor_mesh, num_workers=len(launches))
        return [
            _MonarchWorkerProcess(
                cell_mesh=cell_mesh,
                rank=index,
                served_method_names=_served_method_names(launch),
                actor=actor_mesh.slice(
                    **({} if host_dim is None else {host_dim: index // procs_per_host}),
                    **{_PROCS_DIM: index % procs_per_host},
                ),
            )
            for index, launch in enumerate(launches)
        ]

    async def kill_worker(self, process: WorkerProcess) -> None:
        # A worker lives in its cell's proc mesh, which only stops as a whole in release_cell.
        pass

    async def release_cell(self, *, processes: list[WorkerProcess]) -> None:
        # every worker of a cell lives in the one mesh its launch_cell spawned
        [cell_mesh] = {process.cell_mesh for process in processes}
        await _stop_proc_mesh(cell_mesh.proc_mesh, mesh_name=cell_mesh.name)

    def physical_gpu_ids(self, slots: GpuSlotRef) -> list[int]:
        layout_slots = self._layout.slots_of_layout[slots.layout_name]
        return [layout_slots[slots.first_slot_index + i].gpu_id for i in range(slots.num_slots)]

    def worker_capability_factory(self) -> Callable[[], BackendCapability]:
        return self._capability_factory

    def _place(self, launches: list[WorkerLaunchRequest]) -> tuple[HostMesh, list[int], int]:
        """Pick the host mesh a cell is spawned on, the host rank of each of its workers, and the workers per host.

        A proc mesh puts the same number of processes on each of its hosts, so a cell of gpu workers must sit on a run
        of consecutive hosts with the same number of workers on each. Examples, with 8 gpus per host and one gpu
        per worker:

        - workers in slots 0..7: host ranks [0]*8, 8 per host.
        - workers in slots 8..23: host ranks [1]*8 + [2]*8, 8 per host.
        - workers in slots 4..11: host ranks [0]*4 + [1]*4, 4 per host.
        - workers in slots 4..15: host ranks [0]*4 + [1]*8, which fails: 4 on one host and 8 on the next.
        - n cpu-only workers (no slots): the head host, host ranks [0]*n, n per host.
        """
        gpu_slots = [launch.gpu_slots for launch in launches]
        if all(slots is None for slots in gpu_slots):
            return self._head_host, [0] * len(launches), len(launches)
        assert all(slots is not None for slots in gpu_slots), f"a cell mixes gpu and cpu-only workers: {launches}"

        host_ranks = [
            self._layout.slots_of_layout[slots.layout_name][slots.first_slot_index].host_rank for slots in gpu_slots
        ]
        procs_per_host = len(host_ranks) // (host_ranks[-1] - host_ranks[0] + 1)
        assert procs_per_host and host_ranks == [
            host_ranks[0] + i // procs_per_host for i in range(len(host_ranks))
        ], (
            f"monarch spawns a cell as one proc mesh, which needs the same number of workers on each of a run of "
            f"consecutive hosts, but the cell's workers sit on hosts {host_ranks}"
        )
        return self._worker_hosts, host_ranks, procs_per_host


def _compute_env(launch: WorkerLaunchRequest, *, gpu_ids: list[int] | None) -> dict[str, str | None]:
    """The environment a worker's process starts with: its spec's variables, and which gpus it may use.

    Examples:

    - a gpu worker on gpus [2, 3]: ``{**launch.env_vars, "MILES_WORKER_GPU_IDS": "2,3"}``
    - a cpu-only worker: ``{**launch.env_vars, "CUDA_VISIBLE_DEVICES": "", ...}``, every visible-devices variable
      set empty
    """
    if gpu_ids is None:
        # Ray hides every gpu from a worker it gives none
        gpus = dict.fromkeys(VISIBLE_DEVICES_ENV_VARS, "")
    else:
        # a gpu worker keeps the whole host it inherits, as under Ray, and learns which devices are its own
        gpus = {WORKER_GPU_IDS_ENV_VAR: ",".join(str(gpu_id) for gpu_id in gpu_ids)}
    return {**launch.env_vars, **gpus}


async def _stop_proc_mesh(proc_mesh: ProcMesh, *, mesh_name: str) -> None:
    """Stop a mesh's processes, waiting at most ``_STOP_TIMEOUT_SECONDS``; a failure is logged, not raised, so it
    never hides the error that made the caller stop the mesh."""
    try:
        await asyncio.wait_for(proc_mesh.stop(), timeout=_STOP_TIMEOUT_SECONDS)
        logger.info(f"Stopped proc mesh {mesh_name}")
    except Exception as e:
        logger.warning(f"Failed to stop proc mesh {mesh_name} ({e})")


def _host_dim_of(host_mesh: HostMesh) -> str | None:
    """The dimension a host mesh lays its hosts along; the driver's own host is a mesh with none."""
    labels = list(host_mesh.extent.keys())
    assert len(labels) <= 1, f"a worker host mesh is laid out along one dimension, not {labels}"
    return labels[0] if labels else None


def _served_method_names(launch: WorkerLaunchRequest) -> frozenset[str] | None:
    """The methods a natively served worker declares in its rpc specs, or None for a worker served over rpc."""
    if not launch.served_natively:
        return None
    return frozenset(collect_rpc_method_specs(launch.actor_class))


class _MonarchWorkerProcess(WorkerProcess):
    """One worker of a cell's mesh, as the worker manager core sees it: calls go to its MonarchWorkerActor."""

    def __init__(
        self,
        *,
        cell_mesh: _CellMesh,
        rank: int,
        actor: MonarchWorkerActor,
        served_method_names: frozenset[str] | None,
    ) -> None:
        self.cell_mesh = cell_mesh
        self._rank = rank
        self._actor = actor
        self._served_method_names = served_method_names
        self._unawaited_calls: set[asyncio.Task] = set()

    @property
    def served_natively(self) -> bool:
        return self._served_method_names is not None

    async def call(self, method: str, /, **kwargs: Any) -> Any:
        return await call_endpoint(self._actor.invoke.call_one(method, (), kwargs), describe=method)

    def call_without_result(self, method: str, /, **kwargs: Any) -> None:
        # Nobody awaits this call, so its failure is logged here, as ray logs the error of a call nobody awaits.
        task = asyncio.get_running_loop().create_task(_log_if_failed(self.call(method, **kwargs), describe=method))
        self._unawaited_calls.add(task)
        task.add_done_callback(self._unawaited_calls.discard)

    async def probe_is_dead(self) -> bool:
        return await probe_cell_is_dead(self.cell_mesh.actor_mesh)

    @property
    def native_handle(self) -> Any:
        """A MonarchWorkerHandle for a worker served natively; for one served over rpc, the bare actor, which only
        the launcher's own verbs are called on."""
        if self._served_method_names is None:
            return self._actor
        return MonarchWorkerHandle(
            actor=self._actor,
            cell=MonarchCellHandle(
                name=self.cell_mesh.name, actor_mesh=self.cell_mesh.actor_mesh, num_workers=self.cell_mesh.num_workers
            ),
            rank=self._rank,
            method_names=self._served_method_names,
        )


async def _log_if_failed(call: Awaitable[Any], *, describe: str) -> None:
    """Await a call nobody else awaits, and log it if it failed while its worker was still reachable."""
    try:
        await call
    except WorkerUnreachableError:
        # the worker is gone, which the liveness scan reports for its whole cell
        pass
    except Exception:
        logger.error(f"{describe!r}, which nobody awaits, failed on its worker", exc_info=True)
