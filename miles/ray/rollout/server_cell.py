import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, probe_server_healthy
from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    CellState,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.engine_env_reporter import EngineEnvReporter
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    BaseHealthChecker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.launch_gate import GATE_PORT_NAME, activate_launch_gate
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30
INITIALIZING_TIMEOUT_SECONDS = 1800.0


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
    provider: BaseWorkerProvider
    global_health_checker_activeness: Callable[[], ActiveAndEpoch] = lambda: ActiveAndEpoch(active=True, epoch=0)
    _health_checker: BaseHealthChecker = dataclasses.field(init=False)
    _env_reporter: EngineEnvReporter = dataclasses.field(init=False)
    _state: CellState = dataclasses.field(default_factory=StateUninitialized)

    def __post_init__(self) -> None:
        self._env_reporter = EngineEnvReporter(interval_seconds=self.args.env_report_interval_seconds)
        self._health_checker = create_rollout_cell_health_checker(
            args=self.args,
            name=f"rollout-cell-{self.meta.cell_id}",
            get_api_client=lambda: self.api_client,
            get_activeness=self._get_health_checker_active_and_epoch,
        )
        self._health_checker.start()

    def _get_health_checker_active_and_epoch(self) -> ActiveAndEpoch:
        controller_active_and_epoch = self.global_health_checker_activeness()
        cell_active = isinstance(self._state, (StatePendingWeights, StateServing))
        return ActiveAndEpoch(
            active=cell_active and controller_active_and_epoch.active, epoch=controller_active_and_epoch.epoch
        )

    def __del__(self) -> None:
        assert isinstance(self._state, StateDisposed), (
            f"ServerCell {self.meta.cell_id} was garbage collected without dispose() ({self._state=}); "
            "every cell must be disposed so its health checker task is stopped"
        )

    def cell_status(self) -> CellStatus:
        match self._state:
            case StateUninitialized() | StateInitializing():
                return compute_pending_rollout_cell_status(
                    past_startup_deadline=self.is_initializing_past_deadline, workers_hash=self.meta.workers_hash
                )

            case StatePendingWeights() | StateServing():
                return CellStatus(
                    phase="Running",
                    conditions=[
                        CellCondition.allocated(TriState.TRUE),
                        CellCondition.from_health_checker_status(self._health_checker.status),
                        CellCondition.serving(TriState.TRUE if self.is_serving else TriState.FALSE),
                    ],
                    workers_hash=self.meta.workers_hash,
                )

            case StateDisposed():
                return CellStatus(
                    phase="Suspended",
                    conditions=[CellCondition.allocated(TriState.FALSE)],
                    workers_hash=self.meta.workers_hash,
                )

            case _:
                raise NotImplementedError(f"Unknown state: {self._state}")

    @property
    def is_uninitialized(self) -> bool:
        return isinstance(self._state, StateUninitialized)

    @property
    def is_initializing(self) -> bool:
        return isinstance(self._state, StateInitializing)

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return isinstance(self._state, (StatePendingWeights, StateServing))

    @property
    def is_pending_weights(self) -> bool:
        return isinstance(self._state, StatePendingWeights)

    @property
    def is_serving(self) -> bool:
        return isinstance(self._state, StateServing)

    @property
    def is_initializing_past_deadline(self) -> bool:
        return self.is_initializing and time.monotonic() - self._state.start_time >= INITIALIZING_TIMEOUT_SECONDS

    @property
    def addr_info(self) -> CellAddrInfo:
        assert isinstance(self._state, (StateInitializing, StatePendingWeights, StateServing))
        return self._state.addr_info

    @property
    def server_url(self) -> str:
        return self.addr_info.server_url

    @property
    def api_client(self) -> SGLangApiClient:
        return SGLangApiClient(server_url=self.server_url, api_key=self.meta.sglang_api_key)

    async def init(self) -> None:
        addr_info = await self._compute_addr_info()
        if (gate_url := addr_info.gate_url) is not None:
            await activate_launch_gate(gate_url=gate_url)
        self._change_state(
            "init", StateUninitialized, StateInitializing(addr_info=addr_info, start_time=time.monotonic())
        )

    async def tick(self) -> None:
        if isinstance(self._state, StateInitializing):
            await self._tick_when_initializing()
        await self._report_env_if_due()

    async def _report_env_if_due(self) -> None:
        if not self.is_pending_weights_or_serving:
            return
        await self._env_reporter.report_if_due(
            cell_id=self.meta.cell_id, server_url=self.server_url, api_client=self.api_client
        )

    async def _tick_when_initializing(self) -> None:
        addr_info = self._state.addr_info
        if not await probe_server_healthy(server_url=addr_info.server_url, api_key=self.meta.sglang_api_key):
            return

        if self.args.check_weight_update_equal and self.meta.update_weights:
            await self.check_weights(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)

        if self.meta.needs_offload:
            api_client = SGLangApiClient(server_url=addr_info.server_url)
            await api_client.release_memory_occupation()
            await api_client.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

        serve_without_weight_update: bool = not self.meta.update_weights or self.args.debug_rollout_only
        if not serve_without_weight_update and self.args.check_weight_update_equal:
            await self.check_weights(
                action="reset_tensors",
                allow_quant_error=False,
                selector="all",
                skip_list=self.args.check_weight_update_skip_list,
            )

        if serve_without_weight_update:
            await self._register_with_router(addr_info=addr_info)

        self._change_state("mark_pending_weights", StateInitializing, StatePendingWeights(addr_info=addr_info))

        if serve_without_weight_update:
            self._mark_serving()

    async def mark_weights_ready(self) -> None:
        assert isinstance(self._state, StatePendingWeights), f"{self._state=}"
        await self._register_with_router(addr_info=self._state.addr_info)
        self._mark_serving()

    async def _register_with_router(self, addr_info: CellAddrInfo) -> None:
        await self.router_api_client.add_worker(
            worker_url=addr_info.server_url,
            worker_type=self.meta.worker_type,
            use_legacy_api=use_legacy_router_api(self.args),
            bootstrap_port=addr_info.bootstrap_port,
        )

    async def dispose(self) -> None:
        self._health_checker.stop()

        match self._state:
            case StateServing():
                await self._unregister_from_router()
            case StateUninitialized() | StateInitializing() | StatePendingWeights() | StateDisposed():
                pass
            case _:
                raise ValueError(f"{self._state=}")

        self._change_state(
            "dispose",
            (StateUninitialized, StateInitializing, StatePendingWeights, StateServing, StateDisposed),
            StateDisposed(),
        )

    async def _unregister_from_router(self) -> None:
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

    async def _compute_addr_info(self) -> CellAddrInfo:
        master_addrs = await self.provider.get_addrs(worker_name=self.meta.worker_name)
        primary = master_addrs["primary"]
        gate = master_addrs.get(GATE_PORT_NAME)
        return CellAddrInfo(
            server_url=build_server_url(host=primary.host, port=primary.port),
            bootstrap_port=x.port if (x := master_addrs.get("disaggregation_bootstrap")) else None,
            gate_url=build_server_url(host=gate.host, port=gate.port) if gate else None,
        )

    def _mark_serving(self) -> None:
        self._change_state("mark_serving", StatePendingWeights, StateServing(addr_info=self.addr_info))

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


def create_rollout_cell_health_checker(
    *,
    args: Any,
    name: str,
    get_api_client: Callable[[], SGLangApiClient],
    get_activeness: Callable[[], ActiveAndEpoch],
) -> BaseHealthChecker:
    if "rollout" not in args.ft_components:
        return NoopHealthChecker()

    config = SimpleHealthCheckerConfig.from_args(args, prefix="rollout_health_check")

    async def _check() -> None:
        await get_api_client().health_generate(timeout=config.timeout)

    return SimpleHealthChecker(name=name, check_fn=_check, get_activeness=get_activeness, config=config)


def compute_pending_rollout_cell_status(*, workers_hash: str, past_startup_deadline: bool = False) -> CellStatus:
    return CellStatus(
        phase="Pending",
        conditions=[
            CellCondition.allocated(TriState.TRUE),
            *(
                [CellCondition.healthy(TriState.FALSE, reason="StartupDeadlineExceeded")]
                if past_startup_deadline
                else []
            ),
        ],
        workers_hash=workers_hash,
    )
