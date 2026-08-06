import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.server_cell import SHUTDOWN_TIMEOUT, ServerCell
from miles.ray.rollout.server_engine import ServerEngine
from miles.utils import async_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    cells: list[ServerCell]
    num_gpus_per_engine: int
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    router_ip: str | None = None
    router_port: int | None = None

    def __post_init__(self):
        assert all(
            cell.rank_offset % self.nodes_per_engine == 0 for cell in self.cells
        ), f"every cell's rank_offset must be a multiple of {self.nodes_per_engine=}"

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self) -> list[ServerEngine]:
        """Node-0 engines only (for multi-node serving)."""
        return [cell.engines[0] for cell in self.cells]

    async def start_engines(
        self, port_allocator: PortAllocator, start_cell_indices: list[int] | None = None
    ) -> list[int]:
        """Create Ray actors, allocate ports, and run ``engine.init()`` on every new engine.

        Mutates ``port_allocator`` in place to advance past any newly assigned ports.
        Returns the indices of the cells that were just allocated. Actor creation,
        port allocation and state marking all happen before the first await point,
        so concurrent callers cannot double-start a slot.
        """
        if not self._precheck_engine_start():
            self.has_new_engines = False
            return []

        started_cell_indices: list[int] = []
        cell_starts = []
        for cell_index, cell in enumerate(self.cells):
            if (start_cell_indices is not None) and (cell_index not in start_cell_indices):
                continue

            if cell.is_allocated:
                continue

            started_cell_indices.append(cell_index)
            cell_starts.append(cell.start_engines(port_allocator))

        await asyncio.gather(*cell_starts)

        self.has_new_engines |= bool(started_cell_indices)
        return started_cell_indices

    def _precheck_engine_start(self) -> bool:
        return not (self.args.debug_train_only or self.worker_type == "placeholder")

    async def register_workers(self, cell_indices: list[int]) -> None:
        await asyncio.gather(
            *[cell.register(self._router_api_client) for cell in self._allocated_cells_of(cell_indices)]
        )

    async def unregister_workers(self, cell_indices: list[int]) -> None:
        await asyncio.gather(
            *[cell.unregister(self._router_api_client) for cell in self._allocated_cells_of(cell_indices)]
        )

    def _allocated_cells_of(self, cell_indices: list[int]) -> list[ServerCell]:
        return [self.cells[cell_index] for cell_index in cell_indices if self.cells[cell_index].is_allocated]

    @property
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")

    # Called from InferenceController.stop_cell (main thread, async): deliberately non-async here
    # to avoid introducing two states like "stopping (but not stopped)" vs "stopped", since
    # single-thread async code will not yield without an await point
    # it has the drawback of freezing the whole async thread, which may be avoided later by
    # moving `shutdown` mainly to local code
    def stop_engines(self, cell_indices: list[int]):
        logger.info(f"Killing server {cell_indices=}...")
        try:
            async_utils.run(asyncio.wait_for(self.unregister_workers(cell_indices), timeout=SHUTDOWN_TIMEOUT))
        except Exception as e:
            logger.warning(f"Unregistering {cell_indices=} from the router failed, tearing down anyway (e: {e})")
        for cell_index in sorted(set(cell_indices)):
            self.cells[cell_index].stop()

    async def recover(self, port_allocator: PortAllocator, filter_cell_indices: list[int] | None = None):
        if filter_cell_indices is None:
            filter_cell_indices = list(range(len(self.cells)))
        filter_cell_indices = [
            cell_index for cell_index in filter_cell_indices if not self.cells[cell_index].is_allocated
        ]

        if not self._precheck_engine_start():
            return

        await asyncio.gather(
            *[
                self.cells[cell_index].recover(port_allocator, self._router_api_client)
                for cell_index in filter_cell_indices
            ]
        )
        self.has_new_engines |= bool(filter_cell_indices)

        logger.info(f"Recovered {len(filter_cell_indices)} dead rollout cells (worker_type={self.worker_type})")

    def mark_alive(self, cell_indices: list[int]):
        for cell_index in cell_indices:
            self.cells[cell_index].mark_alive()

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self.cells if cell.is_allocated and cell.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self.cells if cell.is_allocated and cell.needs_offload]
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self.cells
                if cell.is_allocated
            ]
        )
