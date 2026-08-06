import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.audit_utils.process_identity import InferenceControllerProcessIdentity
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import SimpleTicker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0


@enforce_lock_discipline
class InferenceController:
    @lock_exempt
    def __init__(
        self,
        args,
        *,
        engine_provider: BaseWorkerProvider,
        router_provider: BaseWorkerProvider,
    ) -> None:
        self.args = args
        self._engine_provider = engine_provider
        self._router_provider = router_provider
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self.eval_fleet: EvalFleet | None = None
        self._watcher_disposers: list[StopWatchFn] = []
        self._health_checker_activeness = ActivenessTracker(active=True)
        self._ticker: SimpleTicker | None = None

    @lock_exempt
    async def init(self) -> None:
        configure_logger(self.args, source=InferenceControllerProcessIdentity())

        if self.args.debug_train_only:
            return

        router_addrs = await resolve_router_addrs(self.args, provider=self._router_provider)
        self.servers = await create_rollout_servers(
            self.args,
            context_lock=self.context_lock,
            global_health_checker_activeness=self._health_checker_activeness.get,
            engine_provider=self._engine_provider,
            router_addrs=router_addrs,
        )
        self._watcher_disposers.append(await self._engine_provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)

        await asyncio.gather(*[srv.wait_expected_num_cells() for srv in self.servers.values()])

        if self.args.eval_num_gpus > 0:
            self.eval_fleet = await self._build_eval_fleet(srv=self.servers["eval"])

    @with_lock
    async def _build_eval_fleet(self, *, srv: RolloutServer) -> EvalFleet:
        return EvalFleet(
            self.args,
            api_clients=list(srv.api_clients),
            router_host=srv.router_ip,
            router_port=srv.router_port,
        )

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id: int) -> None:
        await self._health_monitoring_resume()
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def prepare_eval(self) -> None:
        await self._health_monitoring_resume()

    @with_lock
    async def dispose(self) -> None:
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await ticker.dispose()

        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

        for srv in self.servers.values():
            await srv.dispose()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None) -> None:
        await self._health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None) -> None:
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @acquires_lock
    async def start_update_weights(self) -> "UpdatableEngines":
        """Return engines eligible for weight updates."""
        await self._health_monitoring_pause()
        await self._ensure_cells_ready()

        srv = self._get_updatable_server()
        if not srv:
            return UpdatableEngines(
                rollout_engines=[],
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )

        return UpdatableEngines(
            rollout_engines=srv.api_clients,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell_id: cell.meta.workers_hash for cell_id, cell in srv.server_cells.items()},
        )

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        await asyncio.gather(
            *[
                cell.mark_weights_ready()
                for srv in self.servers.values()
                for cell_id, cell in srv.server_cells.items()
                if cell_id in snapshot_cell_id_to_hashes
                and snapshot_cell_id_to_hashes[cell_id] == cell.meta.workers_hash
                and cell.is_pending_weights
            ]
        )

    @requires_lock
    async def _ensure_cells_ready(self) -> None:
        deadline = time.monotonic() + CELLS_READY_TIMEOUT_SECONDS
        while True:
            cells = [cell for srv in self.servers.values() for cell in srv.server_cells.values()]
            if self.args.colocate:
                await asyncio.gather(*[cell.init() for cell in cells if cell.is_uninitialized])
            pending = [cell for cell in cells if not cell.is_pending_weights_or_serving]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {CELLS_READY_TIMEOUT_SECONDS}s waiting for "
                    f"{len(pending)}/{len(cells)} cells to become ready"
                )
            logger.info(f"Waiting for {len(pending)}/{len(cells)} cells to become ready...")
            async with self.context_lock.with_released():
                await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    @requires_lock
    def _get_updatable_server(self) -> RolloutServer | None:
        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Only one updatable server is supported."
                )

    # -------------------------- cell operations -----------------------------

    # TEMPORARY: exists only so a suspend can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        await RayWorkerManager.get_handle().stop_cells.remote([cell_id])

    # TEMPORARY: exists only so fault injection can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def inject_fault_between_weight_updates(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # TEMPORARY: colocate cannot kill rollout workers while trainer ranks own the shared GPUs
        if not self._health_checker_activeness.get().active:
            raise RuntimeError(f"Rollout cell {cell_id!r} is offloaded; refusing fault injection")
        await RayWorkerManager.get_handle().inject_fault.remote(
            cell_id, mode=mode.value, worker_in_cell_index=sub_index
        )

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {
            cell_id: cell.cell_status()
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ) -> list[Any]:
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server()
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

    @with_lock
    async def _tick_cells(self) -> None:
        cells = [cell for srv in list(self.servers.values()) for cell in list(srv.server_cells.values())]
        results = await asyncio.gather(
            *[asyncio.wait_for(cell.tick(), timeout=CELL_TICK_TIMEOUT_SECONDS) for cell in cells],
            return_exceptions=True,
        )
        for cell, result in zip(cells, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Ticking cell {cell.meta.cell_id} failed", exc_info=result)

    # -------------------------- reconcile -----------------------------

    @with_lock
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        async def _add(_cell_id: str, observed_info: CellInfo) -> None:
            observed_cell_meta = _compute_server_cell_meta_from_info(observed_info)
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)

        async def _remove(remove_cell_id: str) -> None:
            await actual_srv.remove_cell(remove_cell_id)

        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual_cell.meta.workers_hash if actual_cell is not None else None,
            add=_add,
            remove=_remove,
        )

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self) -> None:
        self._health_checker_activeness.bump_active(False)

    @requires_lock
    async def _health_monitoring_resume(self) -> None:
        self._health_checker_activeness.bump_active(True)


@dataclass(frozen=True)
class UpdatableEngines:
    rollout_engines: list[SGLangApiClient]
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]


# TODO may move and generalize later
def _compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )
