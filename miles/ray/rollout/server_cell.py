import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Literal

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, wait_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url, compute_api_key
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    AddrInfo,
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedUninitialized,
    StateStopped,
)
from miles.ray.specs.inference import compute_engine_pool
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ServerCell:
    args: Any
    worker_type: Literal["regular", "prefill", "decode"]
    cell_id: str
    num_gpus_per_engine: int = 1
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    model_idx: int = 0
    group_index: int = 0
    cell_index: int = 0
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
    def addr_info(self) -> AddrInfo:
        assert isinstance(self._state, StateAllocatedBase)
        assert self._state.addr_info is not None, f"{self._state=}"
        return self._state.addr_info

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.addr_info.server_url)

    @property
    def _pool_id(self) -> str:
        return compute_engine_pool(model_idx=self.model_idx, group_index=self.group_index)

    async def start_engines(self) -> None:
        assert not ({"host", "port"} & set(self.sglang_overrides)), (
            f"sglang_overrides must not override host/port ({self.sglang_overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )
        assert not self.is_allocated, "the caller starts only stopped cells"

        if self.args.rollout_external:
            raise NotImplementedError(
                "external rollout address allocation was removed and a new implementation is coming"
            )

        self._mark_allocated_uninitialized()

        provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
        worker_name = compute_worker_name(pool=self._pool_id, cell_index=self.cell_index)
        master_addrs = await provider.get_addrs(worker_name=worker_name)
        primary = master_addrs["primary"]
        disaggregation_bootstrap = master_addrs.get("disaggregation_bootstrap")
        self._mark_addressing(
            AddrInfo(
                server_url=build_server_url(host=primary.host, port=primary.port),
                bootstrap_port=disaggregation_bootstrap.port if disaggregation_bootstrap else None,
            )
        )

        await wait_server_healthy(
            server_url=self.addr_info.server_url,
            api_key=compute_api_key(self.args, sglang_overrides=self.sglang_overrides),
        )

    async def start(self, router_api_client: SGLangRouterApiClient, recover: bool = False) -> None:
        await self.start_engines()

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
        else:
            logger.info(f"Cell {self.cell_id} is already stopped")
        self._mark_stopped()

    def _mark_allocated_uninitialized(self) -> None:
        self._change_state("mark_allocated_uninitialized", StateStopped, StateAllocatedUninitialized())

    def _mark_addressing(self, addr_info: AddrInfo) -> None:
        self._change_state(
            "mark_addressing",
            StateAllocatedUninitialized,
            StateAllocatedUninitialized(addr_info=addr_info),
        )

    def _mark_alive(self) -> None:
        self._change_state(
            "mark_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(addr_info=self.addr_info),
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
