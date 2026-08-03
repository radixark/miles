import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import wait_router_ready
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.retry_utils import retry_until_deadline

logger = logging.getLogger(__name__)

WAIT_CELLS_INITIAL_DELAY_SECONDS = 1.0
WAIT_CELLS_MAX_DELAY_SECONDS = 5.0


async def create_rollout_servers(args) -> dict[str, "RolloutServer"]:
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
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
            expected_num_cells=model_cfg.num_server_cells,
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    expected_num_cells: int = 0

    @property
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._cells_by_gpu_offset()]

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.meta.num_gpus_per_engine for cell in self._cells_by_gpu_offset()]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.meta.gpu_offset for cell in self._cells_by_gpu_offset()]

    def _cells_by_gpu_offset(self) -> list[ServerCell]:
        return sorted(self.server_cells.values(), key=lambda cell: cell.meta.gpu_offset)

    async def probe_and_mark_dead(self):
        """Mark unreachable cells stopped so ``recover`` restarts them.

        For servers without a ``RolloutHealthMonitor``, which does the same job.
        """
        for cell in self.server_cells.values():
            await cell.probe_and_mark_dead()

    async def add_cell(self, cell_meta: ServerCellMetadata):
        cell_id = cell_meta.cell_id
        assert cell_id not in self.server_cells
        cell = ServerCell(args=self.args, router_api_client=self._router_api_client, meta=cell_meta)
        if not self.args.colocate:
            await cell.init()
        self.server_cells[cell_id] = cell

    async def remove_cell(self, cell_id: str):
        logger.info(f"Killing server {cell_id=}...")
        await self.server_cells[cell_id].dispose()
        del self.server_cells[cell_id]

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

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

    def _addressable_cells(self) -> list[ServerCell]:
        return [cell for cell in self.server_cells.values() if cell.is_pending_weights_or_serving]

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

    def _count_startable_cells(self) -> int:
        if self.args.colocate:
            return len(self.server_cells)
        return sum(1 for cell in self.server_cells.values() if cell.is_pending_weights_or_serving)

    @property
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")
