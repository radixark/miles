import asyncio
import logging
import time
from dataclasses import dataclass

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import wait_session_server_ready
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.specs.inference import compute_engine_pool_ids
from miles.utils.misc import SimpleTicker
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0


class InferenceController:
    def __init__(self, args):
        self.args = args
        self.servers: dict[str, RolloutServer] = {}
        self.rollout_id = -1
        self.eval_fleet: EvalFleet | None = None
        self._watcher_disposers: list[StopWatchFn] = []
        self._ticker: SimpleTicker | None = None

    async def init(self) -> None:
        if self.args.debug_train_only:
            return

        self.servers = await create_rollout_servers(self.args)
        if self.args.eval_num_gpus > 0:
            self.eval_fleet = EvalFleet(self.args, srv=self.servers["eval"])

        # TODO: may change to InferenceController.init(engine_provider, ...) later
        provider: BaseWorkerProvider = RayWorkerProvider.create(
            pool_ids=compute_engine_pool_ids(self.args)
        )  # TODO inject instance
        self._watcher_disposers.append(await provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)
        await wait_session_server_ready(self.args)

    # -------------------------- rollout lifecycle hooks -----------------------------

    async def prepare_rollout(self, rollout_id):
        self.rollout_id = rollout_id
        await self._health_monitoring_resume()
        if self.args.ci_test and self._rollout_ft_enabled and rollout_id >= 2:
            await self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)

    async def prepare_eval(self):
        await self._health_monitoring_resume()

    async def dispose(self):
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await ticker.dispose()

        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        await self._health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # -------------------------- engine management -----------------------------

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

    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]):
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
            await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    async def recover_updatable_engines(self) -> None:
        raise NotImplementedError("new ft to be implemented")

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

    # -------------------------- misc APIs -----------------------------

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server()
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

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

    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        observed_cell_meta: ServerCellMetadata | None = (
            _compute_server_cell_meta_from_info(observed) if observed is not None else None
        )

        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        if observed is not None and actual_srv is None:
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)
        elif observed is None and actual_srv is not None:
            await actual_srv.remove_cell(cell_id)
        elif (
            observed is not None
            and actual_srv is not None
            and observed_cell_meta.workers_hash != actual_cell.meta.workers_hash
        ):
            await actual_srv.remove_cell(cell_id)
            await actual_srv.add_cell(observed_cell_meta)

    # -------------------------- utils -----------------------------

    async def _health_monitoring_pause(self) -> None:
        self._assert_rollout_fault_tolerance_is_unsupported()

    async def _health_monitoring_resume(self) -> None:
        self._assert_rollout_fault_tolerance_is_unsupported()

    @property
    def _rollout_ft_enabled(self) -> bool:
        return self.args.use_fault_tolerance and "rollout" in self.args.ft_components

    def _assert_rollout_fault_tolerance_is_unsupported(self) -> None:
        if not self.args.debug_train_only and self._rollout_ft_enabled:
            raise NotImplementedError(
                "rollout fault tolerance is being rebuilt; health monitoring must pause before "
                "get_updatable_engines snapshots the engines"
            )

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    async def _try_ci_fault_injection(self):
        raise NotImplementedError("rollout fault injection is being rebuilt with rollout fault tolerance")


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
