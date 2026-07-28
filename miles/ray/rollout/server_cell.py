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

from miles.backends.sglang_utils.sglang_engine import SGLangEngine, build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    args: Any
    worker_type: Literal["regular", "prefill", "decode"]
    engines: list[ServerEngine]
    pg: Any = None  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    num_gpus_per_engine: int = 1
    rank_offset: int = 0
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False
    model_path: str | None = None
    update_weights: bool = True

    @property
    def primary_engine(self) -> ServerEngine:
        return self.engines[0]

    @property
    def is_allocated(self) -> bool:
        states = {engine.is_allocated for engine in self.engines}
        assert len(states) == 1, f"a cell's engines are allocated and stopped together ({states=})"
        return states == {True}

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

        new_entries: list[tuple[int, ServerEngine, Any]] = []
        for local_index, engine_slot in enumerate(self.engines):
            global_rank = self.rank_offset + local_index
            rollout_engine = launch_sglang_ray_actor(
                args=self.args,
                pg=self.pg,
                global_rank=global_rank,
                gpu_index=self.gpu_offset + local_index * num_gpu_per_engine,
                worker_type=self.worker_type,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            new_entries.append((global_rank, engine_slot, rollout_engine))
            engine_slot.mark_allocated_uninitialized(rollout_engine)

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

        init_handles = []
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

    async def start(
        self, port_allocator: PortAllocator, router_api_client: SGLangRouterApiClient, recover: bool = False
    ) -> None:
        await self.start_engines(port_allocator)

        if recover and self.needs_offload:
            await self.primary_engine.api_client.release_memory_occupation()
            if self.update_weights or self.model_path:
                await self.primary_engine.api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        self.mark_alive()

        await self.register(router_api_client)

    def mark_alive(self):
        for engine in self.engines:
            engine.mark_alive()

    async def stop(self, router_api_client: SGLangRouterApiClient) -> None:
        if self.is_allocated:
            try:
                await asyncio.wait_for(self.unregister(router_api_client), timeout=SHUTDOWN_TIMEOUT)
            except Exception as e:
                logger.warning(f"Unregistering {self=} from the router failed, tearing down anyway (e: {e})")

        for local_index, engine in enumerate(self.engines):
            if engine.is_allocated:
                logger.info(f"Shutting down and killing engine at cell-local index {local_index}")
                try:
                    ray.get(engine.actor_handle.shutdown.remote(), timeout=SHUTDOWN_TIMEOUT)
                except Exception as e:
                    logger.warning(
                        f"Graceful shutdown of engine at cell-local index {local_index} failed, killing anyway (e: {e})"
                    )
                try:
                    ray.kill(engine.actor_handle)
                    logger.info(f"Successfully killed engine at cell-local index {local_index}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at cell-local index {local_index} (e: {e})")
            else:
                logger.info(f"Engine at cell-local index {local_index} is already None")
            self.engines[local_index].mark_stopped()

    async def offload(self, tags: list[str] | None):
        return await self.primary_engine.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        return await self.primary_engine.api_client.resume_memory_occupation(tags=tags)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await self.primary_engine.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    async def register(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.add_worker(
            worker_url=self.primary_engine.addr_info.server_url,
            worker_type=self.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=self.primary_engine.addr_info.bootstrap_port,
        )

    async def unregister(self, router_api_client: SGLangRouterApiClient) -> None:
        await router_api_client.remove_worker(
            worker_url=self.primary_engine.addr_info.server_url,
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
