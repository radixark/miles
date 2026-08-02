import logging
from dataclasses import dataclass

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, list_cell_ids, start_rollout_servers
from miles.ray.rollout.router_manager import wait_session_server_ready

logger = logging.getLogger(__name__)


class InferenceController:
    def __init__(self, args):
        self.args = args
        self.servers: dict[str, RolloutServer] = {}
        self.rollout_id = -1
        self.eval_fleet = EvalFleet(args, srv=self.servers["eval"]) if args.eval_num_gpus > 0 else None

    async def init(self) -> None:
        if self.args.debug_train_only:
            return

        self.servers = await start_rollout_servers(self.args)
        dashboard_hooks.register_router(self.args)
        await wait_session_server_ready(self.args)

    # -------------------------- rollout lifecycle hooks -----------------------------

    async def prepare_rollout(self, rollout_id):
        self.rollout_id = rollout_id
        await self.health_monitoring_resume()
        if self.args.ci_test and self._rollout_ft_enabled and rollout_id >= 2:
            await self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)

    async def prepare_eval(self):
        await self.health_monitoring_resume()

    async def dispose(self):
        pass

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
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # -------------------------- engine management -----------------------------

    async def get_updatable_engines(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        if not srv:
            return UpdatableEngines(
                rollout_engines=[],
                has_new_engines=False,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
            )

        await srv.wait_all_engines_alive()
        return UpdatableEngines(
            rollout_engines=srv.api_clients,
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

    async def start_cell(self, cell_id: str):
        await self._server_of(cell_id).recover(cell_ids=[cell_id])

    async def stop_cell(self, cell_id: str):
        await self._server_of(cell_id).stop_cells([cell_id])

    def list_cell_ids(self) -> list[str]:
        return list_cell_ids(self.servers)

    def _server_of(self, cell_id: str) -> RolloutServer:
        owners = [srv for srv in self.servers.values() if cell_id in srv.server_cells]
        assert len(owners) == 1, f"{cell_id=} must name exactly one cell, but {len(owners)} servers hold it"
        return owners[0]

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
        self._assert_rollout_fault_tolerance_is_unsupported()

    async def health_monitoring_resume(self) -> None:
        self._assert_rollout_fault_tolerance_is_unsupported()

    @property
    def _rollout_ft_enabled(self) -> bool:
        return self.args.use_fault_tolerance and "rollout" in self.args.ft_components

    def _assert_rollout_fault_tolerance_is_unsupported(self) -> None:
        if not self.args.debug_train_only and self._rollout_ft_enabled:
            raise NotImplementedError(
                "rollout fault tolerance is being rebuilt; health monitoring must pause before "
                "get_updatable_engines snapshots the engines"
            )

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    async def _try_ci_fault_injection(self):
        raise NotImplementedError("rollout fault injection is being rebuilt with rollout fault tolerance")


@dataclass(frozen=True)
class UpdatableEngines:
    rollout_engines: list[SGLangApiClient]
    has_new_engines: bool
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
