import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import wait_router_ready
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock, enforce_lock_discipline, lock_exempt, requires_lock
from miles.utils.ft_utils.health_checker import ActiveAndEpoch
from miles.utils.retry_utils import retry_until_deadline

logger = logging.getLogger(__name__)

WAIT_CELLS_INITIAL_DELAY_SECONDS = 1.0
WAIT_CELLS_MAX_DELAY_SECONDS = 5.0


async def create_rollout_servers(
    args, context_lock: ContextLock, global_health_checker_activeness: Callable[[], ActiveAndEpoch]
) -> dict[str, "RolloutServer"]:
    """Create rollout servers: one per model, each with its own router."""
    assert args.sglang_router_ip is None, (
        "external router mode was removed: miles always starts its own routers "
        "(expected to return with the k8s-native mode)"
    )

    config = resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}

    for model_idx, model_cfg in enumerate(config.models):
        router_addr = await wait_router_ready(model_idx=model_idx)

        if model_idx == 0:
            args.sglang_router_ip = router_addr.host
            args.sglang_router_port = router_addr.port

        servers[model_cfg.name] = RolloutServer(
            server_cells={},
            args=args,
            context_lock=context_lock,
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
            global_health_checker_activeness=global_health_checker_activeness,
            expected_num_cells=model_cfg.num_server_cells,
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


@dataclasses.dataclass
@enforce_lock_discipline
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    context_lock: ContextLock
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    global_health_checker_activeness: Callable[[], ActiveAndEpoch] = lock_exempt(
        lambda: ActiveAndEpoch(active=True, epoch=0)
    )
    expected_num_cells: int = 0

    @property
    @requires_lock
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.meta.num_gpus_per_engine for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.meta.gpu_offset for cell in self._cells_by_gpu_offset()]

    @requires_lock
    def _cells_by_gpu_offset(self) -> list[ServerCell]:
        return sorted(self.server_cells.values(), key=lambda cell: cell.meta.gpu_offset)

    @lock_exempt
    async def probe_and_mark_dead(self):
        """Mark unreachable cells stopped so ``recover`` restarts them.

        For servers without a ``RolloutHealthMonitor``, which does the same job.
        """
        for cell in self.server_cells.values():
            await cell.probe_and_mark_dead()

    @requires_lock
    async def add_cell(self, cell_meta: ServerCellMetadata):
        cell_id = cell_meta.cell_id
        assert cell_id not in self.server_cells
        cell = ServerCell(
            args=self.args,
            router_api_client=self._router_api_client,
            meta=cell_meta,
            global_health_checker_activeness=self.global_health_checker_activeness,
        )
        self.server_cells[cell_id] = cell
        if not (self.args.colocate and cell_meta.needs_offload):
            await cell.init()

    @requires_lock
    async def remove_cell(self, cell_id: str):
        logger.info(f"Killing server {cell_id=}...")
        await self.server_cells[cell_id].dispose()
        del self.server_cells[cell_id]

    @requires_lock
    async def dispose(self) -> None:
        for cell_id in list(self.server_cells.keys()):
            await self.remove_cell(cell_id)

    @requires_lock
    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._addressable_cells()
            ]
        )

    @requires_lock
    def _addressable_cells(self) -> list[ServerCell]:
        return [cell for cell in self.server_cells.values() if cell.is_pending_weights_or_serving]

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = 3600):
        async def _check(remaining_seconds: float) -> None:
            count = self._count_startable_cells()
            if count < self.expected_num_cells:
                raise Exception(f"Only {count}/{self.expected_num_cells} cells of {self.model_name} are ready")

        await retry_until_deadline(
            _check,
            total_seconds=timeout,
            retry_on=Exception,
            initial_delay=WAIT_CELLS_INITIAL_DELAY_SECONDS,
            max_delay=WAIT_CELLS_MAX_DELAY_SECONDS,
            log_fields=dict(op="wait_expected_num_cells", model_name=self.model_name),
        )

    @lock_exempt
    def _count_startable_cells(self) -> int:
        return sum(
            1
            for cell in self.server_cells.values()
            if (self.args.colocate and cell.meta.needs_offload) or cell.is_pending_weights_or_serving
        )

    @property
    @requires_lock
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")
