import asyncio
import dataclasses
import logging
import os
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.ray.rollout.addr_allocator import (
    PortCursors,
    allocate_rollout_engine_addr_and_ports_external,
    allocate_rollout_engine_addr_and_ports_normal,
)
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils

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
    all_engines: list[ServerEngine]
    num_gpus_per_engine: int
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (all_engines) later
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

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self) -> list[ServerEngine]:
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(
        self, port_cursors: PortCursors, start_indices: list[int] | None = None
    ) -> tuple[list, list[int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Mutates ``port_cursors`` in place to advance past any newly assigned ports.
        Returns ``(init_handles, new_engine_indices)`` where *init_handles* is a list
        of Ray ObjectRefs (one per newly created engine) and *new_engine_indices* is
        the list of indices into ``self.all_engines`` that were just allocated.
        """
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.has_new_engines = False
            return [], []

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        new_engines = []
        new_engine_indices = []
        for i in range(len(self.all_engines)):
            if (start_indices is not None) and (i not in start_indices):
                continue
            if self.all_engines[i].is_allocated:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    # TODO: this is hacky. Use env var SGLANG_DG_CACHE_DIR_PER_PROCESS=1
                    # to enable this isolation.
                    "SGLANG_DG_CACHE_DIR": f"/tmp/sglang_deep_gemm/{self.worker_type}_rank_{global_rank}",
                    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                        "0" if self.args.colocate and self.args.rollout_num_gpus_per_engine > 1 else "1"
                    ),
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }
            env_vars.update(dumper_utils.get_sglang_env(self.args))

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            new_engines.append((global_rank, rollout_engine))
            new_engine_indices.append(i)
            self.all_engines[i].mark_allocated_uninitialized(rollout_engine)

        curr_num_new_engines = len(new_engines)
        self.has_new_engines |= curr_num_new_engines > 0

        if curr_num_new_engines == 0:
            return [], []

        if self.args.rollout_external:
            addr_and_ports = allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=new_engines
            )
        else:
            base_port = port_cursors.next_base_port()
            addr_and_ports, next_port_cursors = allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=new_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )
            port_cursors.assign(next_port_cursors)

        init_handles = [
            engine.init.remote(
                **addr_and_ports[index],
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for index, engine in new_engines
        ]
        return init_handles, new_engine_indices

    # There are two callers, only one of them will exist in a running system
    # 1. For new callers (RolloutManager.stop_cell, main thread, async),
    #    deliberately make this function non-async here to avoid introducing two states
    #    like "stopping (but not stopped)" vs "stopped", since single-thread async code will not yield
    #    without an await point
    #    it has the drawback of freezing the whole async thread, which may be avoided later by
    #    moving `shutdown` mainly to local code
    # 2. For legacy callers (RolloutHealthMonitor, another thread, sync)
    #    it is still unsafe to be called in another thread
    #    because engine may be observed as non-stopped while being shutdown,
    #    but that is same as the original code
    def stop_engines(self, engine_indices: list[int]):
        logger.info(f"Killing server {engine_indices=}...")
        for i in engine_indices:
            engine = self.all_engines[i]
            if engine.is_allocated:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.actor_handle.shutdown.remote())
                    ray.kill(engine.actor_handle)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self.all_engines[i].mark_stopped()

    async def recover(self, port_cursors: PortCursors, filter_indices: list[int] | None = None):
        if filter_indices is None:
            filter_indices = [i for i, engine in enumerate(self.all_engines) if not engine.is_allocated]
        start_indices = [idx for idx in filter_indices if not self.all_engines[idx].is_allocated]

        handles, new_engine_indices = self.start_engines(port_cursors, start_indices=start_indices)
        await asyncio.gather(*handles)

        release_handles = []
        all_resume_engines = []
        logger.info(f"Recovered {len(new_engine_indices)} dead rollout engines (worker_type={self.worker_type})")
        assert len(new_engine_indices) == len(
            start_indices
        ), "curr_num_new_engines does not match start_indices length"
        if self.needs_offload and start_indices:
            new_engines = [self.all_engines[i] for i in start_indices]
            release_handles.extend(engine.actor_handle.release_memory_occupation.remote() for engine in new_engines)
            if self.update_weights or self.model_path:
                all_resume_engines.extend(new_engines)

        if release_handles:
            await asyncio.gather(*release_handles)
            if all_resume_engines:
                await asyncio.gather(
                    *[
                        engine.actor_handle.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

        self.mark_alive(engine_indices=new_engine_indices)

    def mark_alive(self, engine_indices: list[int]):
        for engine_index in engine_indices:
            self.all_engines[engine_index].mark_alive()

    def offload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return [
            engine.actor_handle.release_memory_occupation.remote(tags=tags)
            for engine in self.engines
            if engine.is_allocated
        ]

    def onload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return [
            engine.actor_handle.resume_memory_occupation.remote(tags=tags)
            for engine in self.engines
            if engine.is_allocated
        ]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups."""
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.actor_handle.update_weights_from_disk.remote(self.model_path)
            for engine in self.engines
            if engine.is_allocated
        ]

    async def check_weights(self, action: str):
        return await asyncio.gather(
            *[
                engine.actor_handle.check_weights.remote(action=action)
                for engine in self.engines
                if engine.is_allocated
            ]
        )
