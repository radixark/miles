import asyncio
import logging
from dataclasses import dataclass

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.utils import Lock
from miles.utils.health_monitor import RolloutHealthMonitor


logger = logging.getLogger(__name__)


class InferenceController:
    def __init__(self, args, pg):
        self.pg = pg
        self.args = args

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            self.servers = start_rollout_servers(args, pg)
            dashboard_hooks.register_router(args)
            start_session_server(args)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self.eval_fleet = EvalFleet(args, srv=self.servers["eval"]) if args.eval_num_gpus > 0 else None

        # TODO will be replaced by full ft, thus temporarily leave it without modifications
        self._health_monitors = []
        self._rollout_ft_enabled = self.args.use_fault_tolerance and "rollout" in self.args.ft_components
        self._ci_fault_injection_pending = False
        if not self.args.debug_train_only and self._rollout_ft_enabled:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test

    # -------------------------- rollout lifecycle hooks -----------------------------

    async def prepare_rollout(self, rollout_id):
        self.rollout_id = rollout_id
        self._health_monitoring_resume()
        if self.args.ci_test and self._rollout_ft_enabled and rollout_id >= 2:
            await self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)

    async def prepare_eval(self):
        self._health_monitoring_resume()

    async def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        await self.health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        if "weight" not in self.args.offload_rollout_level:
            return
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    async def offload_kv(self):
        tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
        if "kv_cache" in self.args.offload_rollout_level:
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
        await self.offload(tags=tags)

    async def offload_weights(self):
        if "weight" not in self.args.offload_rollout_level:
            return
        await self.offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    # -------------------------- engine management -----------------------------

    async def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        if not srv:
            return EnginesAndLock(
                rollout_engines=[],
                rollout_engine_lock=self.rollout_engine_lock,
                has_new_engines=False,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
            )

        await srv.wait_all_engines_alive()
        return EnginesAndLock(
            rollout_engines=[e.actor_handle for e in srv.engines],
            rollout_engine_lock=self.rollout_engine_lock,
            has_new_engines=srv.has_new_engines,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
        )

    async def clear_updatable_has_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear has_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.clear_has_new_engines()

    async def recover_updatable_engines(self) -> None:
        """Restart any dead rollout engines and update has_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        await self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        await srv.recover()

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

    # -------------------------- external start/stop -----------------------------

    async def start_cell(self, cell_id: int):
        port_cursors = PortCursors.empty()
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        await group.recover(port_cursors=port_cursors, filter_indices=idx.engine_indices)

    async def stop_cell(self, cell_id: int):
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        group.stop_engines(engine_indices=idx.engine_indices)

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

    # -------------------------- utils -----------------------------

    async def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def _health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    # TODO will be replaced by full ft, thus temporarily leave it without modifications
    async def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self._server
            and self._server.server_groups[0].all_engines
            and self._server.server_groups[0].all_engines[0].is_allocated
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self._server.server_groups[0].all_engines[0].actor_handle.simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")


@dataclass(frozen=True)
class EnginesAndLock:
    rollout_engines: list[ray.actor.ActorHandle]
    rollout_engine_lock: ray.actor.ActorHandle
    has_new_engines: bool
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
