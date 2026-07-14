import asyncio
import logging
import time
from collections.abc import Callable

import ray

from miles.ray.train.cell_monitor import compute_cell_status
from miles.ray.train.cell_state import (
    CellState,
    StateAllocatedAlive,
    StateAllocatedBase,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
    StatePending,
    StateStopped,
)
from miles.utils.ft_utils.control_server.models import CellStatus
from miles.utils.ft_utils.health_checker import BaseHealthChecker
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)


ActorFactory = Callable[[], list[ray.actor.ActorHandle]]


class RayTrainCell:
    def __init__(
        self,
        *,
        args,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
        cell_index: int,
        actor_factory: ActorFactory,
        rollout_manager: object | None,
        health_checker: BaseHealthChecker,
    ) -> None:
        self.args = args
        self.cell_index = cell_index
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher
        self.rollout_manager = rollout_manager
        self.actor_factory = actor_factory
        self.health_checker = health_checker

        # NOTE: do *NOT* directly modify `self._state`, but instead use `self._change_state`
        self._state: CellState = StatePending()
        self.allocate_for_pending()

    # ------------------------ API ------------------------

    async def init(
        self,
        *,
        indep_dp_info: IndepDPInfo,
        recv_ckpt_src_rank: int | None = None,
    ):
        results = await self.execute(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            with_opd_teacher=self.with_opd_teacher,
            indep_dp_info=indep_dp_info,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )
        self._mark_as_alive(indep_dp_info=indep_dp_info)
        await self.health_checker.start()
        return results

    async def connect_actor_critic(self, critic_cell: "RayTrainCell") -> list:
        critic_handles = critic_cell._get_actor_handles()
        return await self._execute_raw(
            "connect_actor_critic",
            compute_args=lambda i: (critic_handles[i],),
            compute_kwargs=lambda _: {},
        )

    async def set_rollout_manager(self):
        if (m := self.rollout_manager) is not None:
            return await self.execute("set_rollout_manager", m)
        return []

    # ------------------------ API :: cooperatively prepare ------------------------

    async def prepare_indep_dp_mode_alive(
        self,
        indep_dp_info: IndepDPInfo,
        send_ckpt_dst_ranks: list[int],
    ):
        await self.execute("reconfigure_indep_dp", indep_dp_info=indep_dp_info)
        self._update_indep_dp_info(indep_dp_info)

        for dst_rank in send_ckpt_dst_ranks:
            await self.execute("send_ckpt", dst_rank=dst_rank)

    async def prepare_indep_dp_mode_healing(
        self,
        indep_dp_info: IndepDPInfo,
        recv_ckpt_src_rank: int | None,
    ):
        await self.init(
            indep_dp_info=indep_dp_info,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )

        await self.set_rollout_manager()

    # ------------------------ state transition ------------------------

    def stop(self) -> None:
        if self.is_stopped:
            log_structured(
                logger.info, op="state", name="stop", cell=self.cell_index, skipped=True, reason="already_stopped"
            )
            return

        if self.is_allocated:
            for actor in self._get_actor_handles():
                ray.kill(actor)

        self._change_state("stop", (StatePending, StateAllocatedBase), StateStopped())

    async def _stop_and_confirm_dead(self) -> None:
        if self.is_stopped:
            return

        handles = self._get_actor_handles() if self.is_allocated else []
        self.stop()

        log_structured(logger.info, op="confirm_dead", phase="start", cell=self.cell_index, n_actors=len(handles))
        start = time.monotonic()
        await asyncio.gather(*[_confirm_actor_dead(handle) for handle in handles])
        log_structured(
            logger.info,
            op="confirm_dead",
            phase="end",
            cell=self.cell_index,
            elapsed_s=round(time.monotonic() - start, 1),
        )

    def mark_as_pending(self) -> None:
        if self.is_pending or self.is_allocated:
            log_structured(
                logger.info,
                op="state",
                name="mark_as_pending",
                cell=self.cell_index,
                skipped=True,
                current=type(self._state).__name__,
            )
            return

        self._change_state("mark_as_pending", StateStopped, StatePending())

    def allocate_for_pending(self) -> None:
        actor_handles = self.actor_factory()
        self._change_state(
            "allocate_for_pending",
            StatePending,
            StateAllocatedUninitialized(actor_handles=actor_handles),
        )

    def _mark_as_alive(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_mark_as_alive",
            StateAllocatedUninitialized,
            StateAllocatedAlive(actor_handles=self._state.actor_handles, indep_dp_info=indep_dp_info),
        )

    def _update_indep_dp_info(self, indep_dp_info: IndepDPInfo) -> None:
        self._change_state(
            "_update_indep_dp_info",
            StateAllocatedAlive,
            StateAllocatedAlive(actor_handles=self._state.actor_handles, indep_dp_info=indep_dp_info),
        )

    def _mark_as_errored(self) -> None:
        assert isinstance(
            self._state, (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored)
        ), f"{self.cell_index=} {self._state=}"
        indep_dp_info = None if isinstance(self._state, StateAllocatedUninitialized) else self._state.indep_dp_info
        self._change_state(
            "_mark_as_errored",
            (StateAllocatedUninitialized, StateAllocatedAlive, StateAllocatedErrored),
            StateAllocatedErrored(actor_handles=self._state.actor_handles, indep_dp_info=indep_dp_info),
        )

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        log_structured(
            logger.info,
            op="state",
            phase="start",
            name=debug_name,
            cell=self.cell_index,
            from_state=type(self._state).__name__,
        )
        assert isinstance(self._state, old_state_cls), f"{self.cell_index=} {self._state=}"
        self._state = new_state
        log_structured(
            logger.info,
            op="state",
            phase="end",
            name=debug_name,
            cell=self.cell_index,
            to_state=type(self._state).__name__,
        )

    # ------------------------ API :: directly forward calls to actors ------------------------

    async def execute(self, fn_name: str, *args, kill_on_failure: bool = True, **kwargs) -> list:
        return await self._execute_raw(
            fn_name,
            compute_args=lambda _: args,
            compute_kwargs=lambda _: kwargs,
            kill_on_failure=kill_on_failure,
        )

    async def _execute_raw(
        self,
        fn_name: str,
        compute_args,
        compute_kwargs,
        kill_on_failure: bool = True,
    ) -> list:
        handles = self._get_actor_handles()
        log_structured(
            logger.info, op="execute", phase="start", cell=self.cell_index, fn=fn_name, n_actors=len(handles)
        )
        start = time.monotonic()
        try:
            result = await asyncio.gather(
                *[
                    getattr(actor, fn_name).remote(*compute_args(i), **compute_kwargs(i))
                    for i, actor in enumerate(handles)
                ]
            )
            log_structured(
                logger.info,
                op="execute",
                phase="end",
                cell=self.cell_index,
                fn=fn_name,
                ok=True,
                elapsed_s=round(time.monotonic() - start, 1),
            )
            return result
        except Exception:
            log_structured(
                logger.error,
                op="execute",
                phase="fail",
                cell=self.cell_index,
                fn=fn_name,
                elapsed_s=round(time.monotonic() - start, 1),
                exc_info=True,
            )
            if kill_on_failure:
                self._mark_as_errored()
                await self._stop_and_confirm_dead()
            raise

    # ------------------------ state and misc queries ------------------------

    @property
    def is_pending(self) -> bool:
        return isinstance(self._state, StatePending)

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
    def is_stopped(self) -> bool:
        return isinstance(self._state, StateStopped)

    @property
    def state_name(self) -> str:
        return type(self._state).__name__

    def cell_status(self) -> CellStatus:
        return compute_cell_status(self._state, self.health_checker.status)

    @property
    def indep_dp_info(self) -> IndepDPInfo | None:
        assert isinstance(self._state, (StateAllocatedAlive, StateAllocatedErrored))
        return self._state.indep_dp_info

    def _get_actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(
            self._state, StateAllocatedBase
        ), f"Cell {self.cell_index} is not allocated (state={type(self._state).__name__})"
        return self._state.actor_handles


async def _confirm_actor_dead(handle: ray.actor.ActorHandle) -> None:
    CONFIRM_DEAD_TIMEOUT_S = 120.0
    CONFIRM_DEAD_PROBE_INTERVAL_S = 1.0

    async def _probe() -> None:
        await handle.__ray_ready__.remote()

    deadline = time.monotonic() + CONFIRM_DEAD_TIMEOUT_S
    while True:
        try:
            await asyncio.wait_for(_probe(), timeout=CONFIRM_DEAD_PROBE_INTERVAL_S)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return
        except (TimeoutError, asyncio.TimeoutError):
            pass

        if time.monotonic() >= deadline:
            logger.error("Timed out after %.0fs confirming actor death; proceeding anyway", CONFIRM_DEAD_TIMEOUT_S)
            return

        await asyncio.sleep(CONFIRM_DEAD_PROBE_INTERVAL_S)
