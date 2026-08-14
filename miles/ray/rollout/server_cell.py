import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_server import RolloutServer

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    engines: list[ServerEngine]

    @property
    def primary_engine(self) -> ServerEngine:
        return self.engines[0]

    def stop(self):
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


def flatten_cells(cells: list[ServerCell]) -> list[ServerEngine]:
    return [engine for cell in cells for engine in cell.engines]


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


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    cell_index: int


def get_cell_indexer_of_id_map(servers: dict[str, "RolloutServer"]) -> list[CellIndexer]:
    """Flatten ``servers`` into a list whose position is the cell id.

    ``cell_index`` is the cell's position within its group. Order is sorted by
    ``srv_key``, so cell ids are stable across calls when the topology is
    unchanged.
    """
    result: list[CellIndexer] = []
    for srv_key in sorted(servers):
        srv = servers[srv_key]
        for group_index, group in enumerate(srv.server_groups):
            for cell_index in range(len(group.cells)):
                result.append(
                    CellIndexer(
                        srv_key=srv_key,
                        group_index=group_index,
                        cell_index=cell_index,
                    )
                )
    return result
