import asyncio
import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_engine import SGLangEngine, build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.cell_state import (
    AddrInfo,
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedUninitialized,
    StateStopped,
)
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    args: Any
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_nodes: int = 1
    pg: Any = None  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    num_gpus_per_engine: int = 1
    rank_offset: int = 0
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False
    model_path: str | None = None
    update_weights: bool = True
    _state: CellState = dataclasses.field(default_factory=StateStopped)

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, StateAllocatedAlive)

    @property
    def actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(self._state, StateAllocatedBase)
        return self._state.actor_handles

    @property
    def primary_actor_handle(self) -> ray.actor.ActorHandle:
        return self.actor_handles[0]

    @property
    def engine_gpu_ids(self) -> list[list[int]]:
        _, _, reordered_gpu_ids = self.pg
        gpus_on_node = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
        bases = [
            int(reordered_gpu_ids[self.gpu_offset + local_index * gpus_on_node])
            for local_index in range(self.num_nodes)
        ]
        return [list(range(base, base + gpus_on_node)) for base in bases]

    @property
    def addr_infos(self) -> list[AddrInfo]:
        assert isinstance(self._state, StateAllocatedBase)
        assert self._state.addr_infos is not None, f"{self._state=}"
        return self._state.addr_infos

    @property
    def addr_info(self) -> AddrInfo:
        return self.addr_infos[0]

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.addr_info.server_url)

    async def start_engines(self, port_allocator: PortAllocator) -> None:
        assert not ({"host", "port"} & set(self.sglang_overrides)), (
            f"sglang_overrides must not override host/port ({self.sglang_overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )
        assert not self.is_allocated, "the caller starts only stopped cells"

        if self.args.rollout_external:
            raise NotImplementedError(
                "external rollout address allocation was removed and a new implementation is coming"
            )

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        actor_handles = [
            launch_sglang_ray_actor(
                args=self.args,
                pg=self.pg,
                global_rank=self.rank_offset + local_index,
                gpu_index=self.gpu_offset + local_index * num_gpu_per_engine,
                worker_type=self.worker_type,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )
            for local_index in range(self.num_nodes)
        ]

        self._mark_allocated_uninitialized(actor_handles)

        global_ranks = [self.rank_offset + local_index for local_index in range(self.num_nodes)]

        node_ips = list(await asyncio.gather(*[actor._get_node_ip.remote() for actor in actor_handles]))

        addr_and_ports: dict[int, dict[str, Any]] = {}
        dist_init_addr = None
        for local_index, (rank, actor) in enumerate(zip(global_ranks, actor_handles, strict=True)):
            node_ip = node_ips[local_index]
            alloc = functools.partial(port_allocator.alloc, engine=actor, node_ip=node_ip)

            if local_index == 0:
                dist_init_addr = f"{node_ip}:{alloc(consecutive=30 + self.args.sglang_dp_size)}"

            addr_and_ports[rank] = dict(
                host=node_ip,
                port=alloc(),
                nccl_port=alloc(),
                engine_info_bootstrap_port=alloc(),
                dist_init_addr=dist_init_addr,
            )
            if self.worker_type == "prefill":
                addr_and_ports[rank]["disaggregation_bootstrap_port"] = alloc()

        self._mark_addressing(
            [
                AddrInfo(
                    server_url=build_server_url(
                        host=addr_and_ports[global_rank]["host"], port=addr_and_ports[global_rank]["port"]
                    ),
                    bootstrap_port=addr_and_ports[global_rank].get("disaggregation_bootstrap_port"),
                )
                for global_rank in global_ranks
            ]
        )

        await asyncio.gather(
            *[
                actor.init.remote(**addr_and_ports[global_rank])
                for global_rank, actor in zip(global_ranks, actor_handles, strict=True)
            ]
        )

    async def start(
        self, port_allocator: PortAllocator, router_api_client: SGLangRouterApiClient, recover: bool = False
    ) -> None:
        await self.start_engines(port_allocator)

        if recover and self.needs_offload:
            await self.api_client.release_memory_occupation()
            if self.update_weights or self.model_path:
                await self.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        self._mark_alive()

        await self.register(router_api_client)

    async def stop(self, router_api_client: SGLangRouterApiClient) -> None:
        if self.is_allocated:
            try:
                await asyncio.wait_for(self.unregister(router_api_client), timeout=SHUTDOWN_TIMEOUT)
            except Exception as e:
                logger.warning(f"Unregistering cell {self.cell_id} from the router failed, tearing down anyway ({e})")

            for local_index, actor_handle in enumerate(self.actor_handles):
                logger.info(f"Cell {self.cell_id}: shutting down and killing engine at cell-local index {local_index}")
                try:
                    ray.get(actor_handle.shutdown.remote(), timeout=SHUTDOWN_TIMEOUT)
                except Exception as e:
                    logger.warning(
                        f"Cell {self.cell_id}: graceful shutdown of engine at cell-local index {local_index} "
                        f"failed, killing anyway ({e})"
                    )
                try:
                    ray.kill(actor_handle)
                    logger.info(f"Cell {self.cell_id}: killed engine at cell-local index {local_index}")
                except Exception as e:
                    logger.warning(f"Cell {self.cell_id}: fail to kill engine at cell-local index {local_index} ({e})")
        else:
            logger.info(f"Cell {self.cell_id} is already stopped")
        self._mark_stopped()

    def _mark_allocated_uninitialized(self, actor_handles: list[ray.actor.ActorHandle]) -> None:
        self._change_state(
            "mark_allocated_uninitialized", StateStopped, StateAllocatedUninitialized(actor_handles=actor_handles)
        )

    def _mark_addressing(self, addr_infos: list[AddrInfo]) -> None:
        self._change_state(
            "mark_addressing",
            StateAllocatedUninitialized,
            StateAllocatedUninitialized(actor_handles=self.actor_handles, addr_infos=addr_infos),
        )

    def _mark_alive(self) -> None:
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(actor_handles=self.actor_handles, addr_infos=self.addr_infos),
        )

    def _mark_stopped(self) -> None:
        self._change_state("mark_stopped", (StateStopped, StateAllocatedBase), StateStopped())

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"Cell {self.cell_id} {debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"Cell {self.cell_id} {debug_name} end new={self._state}")

    async def probe_and_mark_dead(self) -> None:
        if not self.is_allocated:
            return
        try:
            await asyncio.wait_for(self.primary_actor_handle.get_weight_version.remote(), timeout=60)
        except Exception as e:
            logger.warning(f"Cell unreachable ({e!r}); marking stopped for recovery")
            for actor_handle in self.actor_handles:
                try:
                    ray.kill(actor_handle)
                except Exception:
                    pass
            self._mark_stopped()

    async def offload(self, tags: list[str] | None):
        return await self.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        return await self.api_client.resume_memory_occupation(tags=tags)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await self.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    async def register(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.add_worker(
            worker_url=self.addr_info.server_url,
            worker_type=self.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=self.addr_info.bootstrap_port,
        )

    async def unregister(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.remove_worker(
            worker_url=self.addr_info.server_url,
            use_legacy_api=use_legacy_router_api(self.args),
        )


def compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    return max(1, num_gpus_per_engine // num_gpus_per_node)


def launch_sglang_ray_actor(
    *,
    args: Any,
    pg: Any,
    global_rank: int,
    gpu_index: int,
    worker_type: str,
    sglang_overrides: dict,
    num_gpus_per_engine: int,
) -> ray.actor.ActorHandle:
    pg, reordered_bundle_indices, reordered_gpu_ids = pg

    num_gpus = 0.2
    num_cpus = num_gpus
    base_gpu_id = int(reordered_gpu_ids[gpu_index])

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=reordered_bundle_indices[gpu_index],
    )

    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        key: os.environ.get(key, default_val)
        for key, default_val in {
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            # TODO: this is hacky. Use env var SGLANG_DG_CACHE_DIR_PER_PROCESS=1
            # to enable this isolation.
            "SGLANG_DG_CACHE_DIR": f"/tmp/sglang_deep_gemm/{worker_type}_rank_{global_rank}",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                "0" if args.colocate and args.rollout_num_gpus_per_engine > 1 else "1"
            ),
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        }.items()
    }
    env_vars.update(dumper_utils.get_sglang_env(args))

    RolloutRayActor = ray.remote(SGLangEngine)
    return RolloutRayActor.options(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        scheduling_strategy=scheduling_strategy,
        runtime_env={
            "env_vars": env_vars,
        },
    ).remote(
        args,
        rank=global_rank,
        worker_type=worker_type,
        base_gpu_id=base_gpu_id,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
    )
