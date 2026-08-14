import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import wait_router_ready
from miles.ray.rollout.server_cell import ServerCell, compute_nodes_per_engine

logger = logging.getLogger(__name__)


async def start_rollout_servers(args) -> dict[str, "RolloutServer"]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
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

        server_cells: dict[str, ServerCell] = {}

        for group_index, group_cfg in enumerate(model_cfg.server_groups):
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local
            nodes_per_engine = compute_nodes_per_engine(
                num_gpus_per_engine=gpus_per_engine, num_gpus_per_node=args.num_gpus_per_node
            )
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={group_cfg.gpu_offset}: "
                f"needs_offload={group_cfg.needs_offload}"
            )

            if group_cfg.worker_type != "placeholder":
                assert num_engines % nodes_per_engine == 0, (
                    f"group '{group_cfg.worker_type}' has {num_engines=} which is not a whole number of "
                    f"{nodes_per_engine=} engines; the trailing engine would have no node to run its remaining ranks"
                )
                for cell_start in range(0, num_engines, nodes_per_engine):
                    cell_id = format_cell_id(server_id=model_cfg.name, index=len(server_cells))
                    server_cells[cell_id] = ServerCell(
                        args=args,
                        worker_type=group_cfg.worker_type,
                        cell_id=cell_id,
                        num_gpus_per_engine=gpus_per_engine,
                        gpu_offset=group_cfg.gpu_offset + cell_start * num_gpu_per_engine_local,
                        sglang_overrides=group_cfg.overrides,
                        model_idx=model_idx,
                        group_index=group_index,
                        cell_index=cell_start // nodes_per_engine,
                        needs_offload=group_cfg.needs_offload,
                        model_path=group_cfg.model_path,
                        update_weights=model_cfg.update_weights,
                    )

        servers[model_cfg.name] = RolloutServer(
            server_cells=server_cells,
            args=args,
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    await asyncio.gather(*[srv.start_all_cells() for srv in servers.values()])

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool = False
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self.server_cells.values()]

    def clear_has_new_engines(self):
        self.has_new_engines = False

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.num_gpus_per_engine for cell in self.server_cells.values()]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.gpu_offset for cell in self.server_cells.values()]

    async def probe_and_mark_dead(self):
        """Mark unreachable cells stopped so ``recover`` restarts them.

        For servers without a ``RolloutHealthMonitor``, which does the same job.
        """
        for cell in self.server_cells:
            await cell.probe_and_mark_dead()

    async def start_all_cells(self):
        if self.args.debug_train_only:
            return

        cell_ids = [cell_id for cell_id, cell in self.server_cells.items() if not cell.is_allocated]
        await asyncio.gather(*[self.server_cells[cell_id].start(self._router_api_client) for cell_id in cell_ids])
        self.has_new_engines |= bool(cell_ids)

    async def stop_cells(self, cell_ids: list[str]):
        logger.info(f"Killing server {cell_ids=}...")
        for cell_id in sorted(set(cell_ids)):
            await self.server_cells[cell_id].stop(self._router_api_client)

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._allocated_cells_of()
            ]
        )

    async def wait_all_engines_alive(self, timeout: float = 600):
        # TODO: 600s default is hardcoded; make it configurable (e.g. via args) once we have a clearer
        # picture of init/recovery upper bounds across model sizes
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if all(cell.is_alive for cell in self.server_cells.values()):
                return
            await asyncio.sleep(sleep_time)
            logger.info("wait_all_engines_alive looping...")
        raise TimeoutError(f"Timed out after {timeout}s waiting for engines to become ready")

    def _allocated_cells_of(self, cell_ids: list[str] | None = None) -> list[ServerCell]:
        if cell_ids is None:
            cell_ids = list(self.server_cells)
        return [self.server_cells[cell_id] for cell_id in cell_ids if self.server_cells[cell_id].is_allocated]

    @property
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


def format_cell_id(*, server_id: str, index: int) -> str:
    return f"{server_id}-{index}"


def list_cell_ids(servers: dict[str, "RolloutServer"]) -> list[str]:
    return [cell_id for model_id in sorted(servers) for cell_id in servers[model_id].server_cells]
