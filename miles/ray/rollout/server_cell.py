import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Literal

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, wait_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import CellState, StatePendingWeights, StateServing, StateUnknown
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


class ServerCellMetadata(FrozenStrictBaseModel):
    model_id: str
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_gpus_per_engine: int
    gpu_offset: int
    sglang_api_key: str | None
    worker_name: str
    needs_offload: bool
    update_weights: bool
    workers_hash: str


@dataclass
class ServerCell:
    args: Any
    meta: ServerCellMetadata
    router_api_client: SGLangRouterApiClient
    _state: CellState = dataclasses.field(default_factory=StateUnknown)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return isinstance(self._state, (StatePendingWeights, StateServing))

    @property
    def server_url(self) -> str:
        assert isinstance(self._state, (StatePendingWeights, StateServing))
        return self._state.server_url

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.server_url)

    async def add(self) -> None:
        if self.args.rollout_external:
            raise NotImplementedError(
                "external rollout address allocation was removed and a new implementation is coming"
            )

        provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
        master_addrs = await provider.get_addrs(worker_name=self.meta.worker_name)
        primary = master_addrs["primary"]
        server_url = build_server_url(host=primary.host, port=primary.port)
        bootstrap_port = x.port if (x := master_addrs.get("disaggregation_bootstrap")) else None

        await wait_server_healthy(
            server_url=server_url,
            api_key=self.meta.sglang_api_key,
        )

        if self.args.check_weight_update_equal and self.meta.update_weights:
            await self.check_weights(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)

        if self.meta.needs_offload:
            api_client = SGLangApiClient(server_url=server_url)
            await api_client.release_memory_occupation()
            await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        self._mark_pending_weights(server_url=server_url, bootstrap_port=bootstrap_port)

        await self.mark_weights_ready()

    async def mark_weights_ready(self):
        assert isinstance(self._state, StatePendingWeights), f"{self._state=}"
        bootstrap_port = self._state.bootstrap_port

        self._mark_serving()

        await self.router_api_client.add_worker(
            worker_url=self.server_url,
            worker_type=self.meta.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=bootstrap_port,
        )

    async def dispose(self) -> None:
        try:
            await asyncio.wait_for(
                self.router_api_client.remove_worker(
                    worker_url=self.server_url,
                    use_legacy_api=use_legacy_router_api(self.args),
                ),
                timeout=SHUTDOWN_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Unregistering cell {self.meta.cell_id} from the router failed, tearing down anyway ({e})")

    def _mark_pending_weights(self, server_url: str, bootstrap_port: int | None) -> None:
        self._change_state(
            "mark_pending_weights",
            StateUnknown,
            StatePendingWeights(server_url=server_url, bootstrap_port=bootstrap_port),
        )

    def _mark_serving(self) -> None:
        self._change_state("mark_serving", StatePendingWeights, StateServing(server_url=self.server_url))

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"Cell {self.meta.cell_id} {debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self._state=}"
        self._state = new_state
        logger.info(f"Cell {self.meta.cell_id} {debug_name} end new={self._state}")

    async def probe_and_mark_dead(self) -> None:
        if not self.is_allocated:
            return
        try:
            await asyncio.wait_for(self.api_client.get_weight_version(), timeout=60)
        except Exception as e:
            logger.warning(f"Cell unreachable ({e!r}); marking stopped for recovery")
            self._mark_stopped()

    async def offload(self, tags: list[str] | None):
        return await self.api_client.release_memory_occupation(tags=tags)

    async def onload(self, tags: list[str] | None):
        return await self.api_client.resume_memory_occupation(tags=tags)

    async def check_weights(self, action: str, allow_quant_error: bool, selector: str, skip_list: list[str] | None):
        return await self.api_client.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )


def compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    return max(1, num_gpus_per_engine // num_gpus_per_node)
