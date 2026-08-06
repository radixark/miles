import asyncio
import dataclasses
import functools
import logging
from typing import Any

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.server_cell import SHUTDOWN_TIMEOUT, ServerCell, flatten_cells, launch_sglang_ray_actor
from miles.ray.rollout.server_engine import AddrInfo, ServerEngine
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
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    cells: list[ServerCell]
    num_gpus_per_engine: int
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False
    model_path: str | None = None
    router_ip: str | None = None
    router_port: int | None = None
    update_weights: bool = True

    def __post_init__(self):
        assert (
            not self.cells or self.rank_offset % self.nodes_per_engine == 0
        ), f"{self.rank_offset=} must be a multiple of {self.nodes_per_engine=}"

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
        Returns the list of indices into the group's flat engine list that were just
        allocated. Actor creation, port allocation and state marking all happen before
        the first await point, so concurrent callers cannot double-start a slot.
        """
        assert not ({"host", "port"} & set(self.sglang_overrides)), (
            f"sglang_overrides must not override host/port ({self.sglang_overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )

        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.has_new_engines = False
            return []

        if self.args.rollout_external:
            raise NotImplementedError(
                "external rollout address allocation was removed and a new implementation is coming"
            )

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        new_engine_indices: list[int] = []
        init_handles: list = []
        for cell_index, cell in enumerate(self.cells):
            if (start_cell_indices is not None) and (cell_index not in start_cell_indices):
                continue

            if cell.is_allocated:
                continue

            new_entries: list[tuple[int, ServerEngine, Any]] = []
            for local_index, engine_slot in enumerate(cell.engines):
                flat_index = cell_index * self.nodes_per_engine + local_index
                global_rank = self.rank_offset + flat_index
                rollout_engine = launch_sglang_ray_actor(
                    args=self.args,
                    pg=self.pg,
                    global_rank=global_rank,
                    gpu_index=self.gpu_offset + flat_index * num_gpu_per_engine,
                    worker_type=self.worker_type,
                    sglang_overrides=self.sglang_overrides,
                    num_gpus_per_engine=self.num_gpus_per_engine,
                )

                new_entries.append((global_rank, engine_slot, rollout_engine))
                new_engine_indices.append(flat_index)
                engine_slot.mark_allocated_uninitialized(rollout_engine)

            self.has_new_engines = True

            addr_and_ports: dict[int, dict[str, Any]] = {}
            dist_init_addr = None
            for entry_index, (global_rank, _, actor) in enumerate(new_entries):
                node_ip, _ = ray.get(actor._get_current_node_ip_and_free_port.remote())
                alloc = functools.partial(port_allocator.alloc, engine=actor, node_ip=node_ip)

                if entry_index == 0:
                    dist_init_addr = f"{node_ip}:{alloc(consecutive=30 + self.args.sglang_dp_size)}"

                addr_and_ports[global_rank] = dict(
                    host=node_ip,
                    port=alloc(),
                    nccl_port=alloc(),
                    engine_info_bootstrap_port=alloc(),
                    dist_init_addr=dist_init_addr,
                )
                if self.worker_type == "prefill":
                    addr_and_ports[global_rank]["disaggregation_bootstrap_port"] = alloc()

            for global_rank, engine_slot, actor in new_entries:
                engine_addr_and_ports = addr_and_ports[global_rank]
                engine_slot.set_addressing(
                    AddrInfo(
                        server_url=build_server_url(
                            host=engine_addr_and_ports["host"], port=engine_addr_and_ports["port"]
                        ),
                        bootstrap_port=engine_addr_and_ports.get("disaggregation_bootstrap_port"),
                    )
                )
                init_handles.append(actor.init.remote(**addr_and_ports[global_rank]))

        await asyncio.gather(*init_handles)
        return new_engine_indices

    async def register_workers(self, engine_indices: list[int]) -> None:
        if self.args.rollout_external or not (self.router_ip and self.router_port):
            return
        await asyncio.gather(
            *[
                self._router_api_client.add_worker(
                    worker_url=engine.addr_info.server_url,
                    worker_type=self.worker_type,
                    use_legacy_api=use_legacy_router_api(self.args),
                    bootstrap_port=engine.addr_info.bootstrap_port,
                )
                for engine in self._primary_engines_of(engine_indices)
            ]
        )

    async def unregister_workers(self, engine_indices: list[int]) -> None:
        if self.args.rollout_external or not (self.router_ip and self.router_port):
            return
        await asyncio.gather(
            *[
                self._router_api_client.remove_worker(
                    worker_url=engine.addr_info.server_url,
                    use_legacy_api=use_legacy_router_api(self.args),
                )
                for engine in self._primary_engines_of(engine_indices)
            ]
        )

    def _engine_indices_of_cell(self, cell_index: int) -> range:
        return range(cell_index * self.nodes_per_engine, (cell_index + 1) * self.nodes_per_engine)

    def _primary_engines_of(self, engine_indices: list[int]) -> list[ServerEngine]:
        all_engines = flatten_cells(self.cells)
        return [
            all_engines[index]
            for index in engine_indices
            if index % self.nodes_per_engine == 0 and all_engines[index].is_allocated
        ]

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
        engine_indices = [i for cell_index in cell_indices for i in self._engine_indices_of_cell(cell_index)]
        try:
            async_utils.run(asyncio.wait_for(self.unregister_workers(engine_indices), timeout=SHUTDOWN_TIMEOUT))
        except Exception as e:
            logger.warning(f"Unregistering {engine_indices=} from the router failed, tearing down anyway (e: {e})")
        for cell_index in sorted(set(cell_indices)):
            self.cells[cell_index].stop()

    async def recover(self, port_allocator: PortAllocator, filter_cell_indices: list[int] | None = None):
        if filter_cell_indices is None:
            filter_cell_indices = [cell_index for cell_index, cell in enumerate(self.cells) if not cell.is_allocated]

        new_engine_indices = await self.start_engines(port_allocator, start_cell_indices=filter_cell_indices)

        all_engines = flatten_cells(self.cells)
        release_handles = []
        all_resume_engines = []
        logger.info(f"Recovered {len(new_engine_indices)} dead rollout engines (worker_type={self.worker_type})")
        if self.needs_offload and new_engine_indices:
            new_primary_engines = [all_engines[i] for i in new_engine_indices if i % self.nodes_per_engine == 0]
            release_handles.extend(engine.api_client.release_memory_occupation() for engine in new_primary_engines)
            if self.update_weights or self.model_path:
                all_resume_engines.extend(new_primary_engines)

        if release_handles:
            await asyncio.gather(*release_handles)
            if all_resume_engines:
                await asyncio.gather(
                    *[
                        engine.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

        self.mark_alive(engine_indices=new_engine_indices)
        await self.register_workers(new_engine_indices)

    def mark_alive(self, engine_indices: list[int]):
        all_engines = flatten_cells(self.cells)
        for engine_index in engine_indices:
            all_engines[engine_index].mark_alive()

    async def offload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return await asyncio.gather(*[cell.offload(tags=tags) for cell in self.cells if cell.is_allocated])

    async def onload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return await asyncio.gather(*[cell.onload(tags=tags) for cell in self.cells if cell.is_allocated])

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
