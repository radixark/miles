import asyncio
import logging
import time

from miles.ray.train.cell_monitor import compute_cell_status
from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import BaseHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.retry_utils import NonRetryableError
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import MASTER_PORT_NAME, HostAndPort

logger = logging.getLogger(__name__)

KILL_RPC_TIMEOUT_S = 10.0
CONFIRM_DEAD_TIMEOUT_S = 120.0


class TrainerCell:
    def __init__(
        self,
        *,
        args,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
        cell_id: str,
        cell_index: int,
        workers_hash: str,
        health_checker: BaseHealthChecker,
        provider: BaseWorkerProvider,
    ) -> None:
        self.args = args
        self.cell_id = cell_id
        self.cell_index = cell_index
        self.workers_hash = workers_hash
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher
        self.health_checker = health_checker

        (worker_infos,) = provider.get_worker_infos(cell_ids=[cell_id])
        self._master_addr: HostAndPort = worker_infos[0].self_addrs[MASTER_PORT_NAME]
        worker_handles = provider.get_handles_of_worker_infos(worker_infos)
        assert len(worker_handles) == len(worker_infos), f"cell {cell_id} holds workers that cannot be called"

        # NOTE: do *NOT* directly modify `self._state`, but instead use `self._change_state`
        self._state: CellState = StateAllocatedUninitialized(worker_handles=list(worker_handles.values()))

    # ------------------------ API ------------------------

    async def init(
        self,
        *,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        recv_ckpt_src_rank: int | None = None,
    ):
        await self.execute(
            "configure_master_addr_and_port",
            master_addr=self._master_addr.host.strip("[]"),
            master_port=self._master_addr.port,
        )
        results = await self.execute(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=indep_dp_info,
            indep_dp_store_addr=indep_dp_store_addr,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )
        self._mark_as_alive(indep_dp_info=indep_dp_info)
        self.health_checker.start()
        await asyncio.sleep(0)
        return results

    async def load_state(self) -> list:
        return await self.execute("load_state", kill_on_failure=False)

    async def train(
        self,
        *,
        rollout_id: int,
        rollout_data_ref,
        witness_info,
        attempt: int,
        external_data: list | None = None,
    ) -> list:
        if external_data is not None and len(external_data) != len(self._get_worker_handles()):
            raise NonRetryableError("external_data must contain one payload per train worker")

        return await self._execute_raw(
            "train",
            compute_kwargs=lambda i: dict(
                rollout_id=rollout_id,
                rollout_data_ref=rollout_data_ref,
                witness_info=witness_info,
                attempt=attempt,
                **({} if external_data is None else dict(external_data=external_data[i])),
            ),
        )

    # ------------------------ API :: cooperatively prepare ------------------------

    async def prepare_indep_dp_mode_alive(
        self,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        send_ckpt_dst_ranks: list[int],
    ):
        await self.execute(
            "reconfigure_indep_dp", indep_dp_info=indep_dp_info, indep_dp_store_addr=indep_dp_store_addr
        )
        self._update_indep_dp_info(indep_dp_info)

        for dst_rank in send_ckpt_dst_ranks:
            await self.execute("send_ckpt", dst_rank=dst_rank)

    async def prepare_indep_dp_mode_healing(
        self,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
        recv_ckpt_src_rank: int | None,
    ):
        await self.init(
            indep_dp_info=indep_dp_info,
            indep_dp_store_addr=indep_dp_store_addr,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )

    # ------------------------ state transition ------------------------

    async def _kill_workers_and_confirm_dead(self) -> None:
        handles = self._get_worker_handles() if self.is_allocated else []

        log_structured(
            logger.info, tag="ft", op="confirm_dead", phase="start", cell=self.cell_id, n_actors=len(handles)
        )
        start = time.monotonic()
        await asyncio.gather(*[_kill_worker(handle) for handle in handles])
        await asyncio.gather(*[handle.wait_dead(timeout=CONFIRM_DEAD_TIMEOUT_S) for handle in handles])
        log_structured(
            logger.info,
            tag="ft",
            op="confirm_dead",
            phase="end",
            cell=self.cell_id,
            elapsed_s=round(time.monotonic() - start, 1),
        )

    def _mark_as_alive(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_mark_as_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _update_indep_dp_info(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_update_indep_dp_info",
            StateAllocatedAlive,
            StateAllocatedAlive(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _mark_as_errored(self) -> None:
        assert isinstance(
            self._state, (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored)
        ), f"{self.cell_id=} {self._state=}"
        indep_dp_info = None if isinstance(self._state, StateAllocatedUninitialized) else self._state.indep_dp_info
        self._change_state(
            "_mark_as_errored",
            (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored),
            StateAllocatedErrored(worker_handles=self._state.worker_handles, indep_dp_info=indep_dp_info),
        )

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        log_structured(
            logger.info,
            tag="ft",
            op="state",
            phase="start",
            name=debug_name,
            cell=self.cell_id,
            from_state=type(self._state).__name__,
        )
        assert isinstance(self._state, old_state_cls), f"{self.cell_id=} {self._state=}"
        self._state = new_state
        log_structured(
            logger.info,
            tag="ft",
            op="state",
            phase="end",
            name=debug_name,
            cell=self.cell_id,
            to_state=type(self._state).__name__,
        )

    # ------------------------ API :: directly forward calls to actors ------------------------

    async def execute(self, fn_name: str, *, kill_on_failure: bool = True, **kwargs) -> list:
        return await self._execute_raw(
            fn_name,
            compute_kwargs=lambda _: kwargs,
            kill_on_failure=kill_on_failure,
        )

    async def _execute_raw(
        self,
        fn_name: str,
        compute_kwargs,
        kill_on_failure: bool = True,
    ) -> list:
        handles = self._get_worker_handles()
        log_structured(
            logger.info, tag="ft", op="execute", phase="start", cell=self.cell_id, fn=fn_name, n_actors=len(handles)
        )
        start = time.monotonic()
        try:
            result = await asyncio.gather(
                *[getattr(handle, fn_name)(**compute_kwargs(i)) for i, handle in enumerate(handles)]
            )
            log_structured(
                logger.info,
                tag="ft",
                op="execute",
                phase="end",
                cell=self.cell_id,
                fn=fn_name,
                ok=True,
                elapsed_s=round(time.monotonic() - start, 1),
            )
            return result
        except Exception:
            log_structured(
                logger.error,
                tag="ft",
                op="execute",
                phase="fail",
                cell=self.cell_id,
                fn=fn_name,
                elapsed_s=round(time.monotonic() - start, 1),
                exc_info=True,
            )
            if kill_on_failure:
                self._mark_as_errored()
                await self._kill_workers_and_confirm_dead()
            raise

    # ------------------------ state and misc queries ------------------------

    @property
    def is_allocated(self) -> bool:
        return isinstance(self._state, StateAllocatedBase)

    @property
    def is_alive(self) -> bool:
        return isinstance(self._state, StateAllocatedAlive)

    @property
    def is_errored(self) -> bool:
        return isinstance(self._state, StateAllocatedErrored)

    @property
    def is_uninitialized(self) -> bool:
        return isinstance(self._state, StateAllocatedUninitialized)

    @property
    def state_name(self) -> str:
        return type(self._state).__name__

    def cell_status(self) -> CellStatus:
        return compute_cell_status(self._state, self.health_checker.status)

    @property
    def indep_dp_info(self) -> IndepDPInfo | None:
        assert isinstance(self._state, (StateAllocatedAlive, StateAllocatedErrored))
        return self._state.indep_dp_info

    def _get_worker_handles(self) -> list[BaseWorkerHandle]:
        assert isinstance(
            self._state, StateAllocatedBase
        ), f"Cell {self.cell_id} is not allocated (state={type(self._state).__name__})"
        return self._state.worker_handles


async def _kill_worker(handle: BaseWorkerHandle) -> None:
    try:
        await asyncio.wait_for(handle.kill_self(), timeout=KILL_RPC_TIMEOUT_S)
    except WorkerUnreachableError:
        return
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("Timed out asking a worker to kill itself; falling back to the death confirmation probe")
