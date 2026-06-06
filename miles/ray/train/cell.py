import asyncio
import logging
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
from miles.utils.control_server.models import CellStatus
from miles.utils.health_checker import BaseHealthChecker
from miles.utils.indep_dp import IndepDPInfo

logger = logging.getLogger(__name__)


ActorFactory = Callable[[], list[ray.actor.ActorHandle]]


class RayTrainCell:
    def __init__(
        self,
        *,
        args,
        role: str,
        with_ref: bool,
        cell_index: int,
        actor_factory: ActorFactory,
        rollout_manager: object | None,
        health_checker: BaseHealthChecker,
    ) -> None:
        self.args = args
        self.cell_index = cell_index
        self.role = role
        self.with_ref = with_ref
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
        self._mark_as_alive(indep_dp_info=indep_dp_info)
        results = await self.execute(
            "init",
            args=self.args,
            role=self.role,
            with_ref=self.with_ref,
            indep_dp_info=indep_dp_info,
            recv_ckpt_src_rank=recv_ckpt_src_rank,
        )
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
            logger.info(f"stop: cell {self.cell_index} already stopped, skipping")
            return

        if self.is_allocated:
            for actor in self._get_actor_handles():
                ray.kill(actor)

        self._change_state("stop", (StatePending, StateAllocatedBase), StateStopped())

    def mark_as_pending(self) -> None:
        if self.is_pending or self.is_allocated:
            logger.info(f"mark_as_pending: cell {self.cell_index} already {type(self._state).__name__}, skipping")
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
        # NOTE: do NOT kill actors here — external ft controller may need the actors
        # to be still there for stacktrace diagnostics before calling stop() to kill them
        # Validate state BEFORE building the new state, otherwise StateAllocatedUninitialized
        # has no `indep_dp_info` and we'd raise AttributeError instead of the expected AssertionError.
        assert isinstance(
            self._state, (StateAllocatedAlive, StateAllocatedErrored)
        ), f"{self.cell_index=} {self._state=}"
        self._change_state(
            "_mark_as_errored",
            (StateAllocatedAlive, StateAllocatedErrored),
            StateAllocatedErrored(actor_handles=self._state.actor_handles, indep_dp_info=self._state.indep_dp_info),
        )

    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[CellState] | tuple[type[CellState], ...],
        new_state: CellState,
    ) -> None:
        logger.info(f"{debug_name} start {self.cell_index=} old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self.cell_index=} {self._state=}"
        self._state = new_state
        logger.info(f"{debug_name} end {self.cell_index=} new={self._state}")

    # ------------------------ API :: directly forward calls to actors ------------------------

    async def execute(self, fn_name: str, *args, mark_errored_on_failure: bool = True, **kwargs) -> list:
        return await self._execute_raw(
            fn_name,
            compute_args=lambda _: args,
            compute_kwargs=lambda _: kwargs,
            mark_errored_on_failure=mark_errored_on_failure,
        )

    async def _execute_raw(
        self,
        fn_name: str,
        compute_args,
        compute_kwargs,
        mark_errored_on_failure: bool = True,
    ) -> list:
        handles = self._get_actor_handles()
        try:
            return await asyncio.gather(
                *[
                    getattr(actor, fn_name).remote(*compute_args(i), **compute_kwargs(i))
                    for i, actor in enumerate(handles)
                ]
            )
        except Exception:
            logger.error(f"Cell {self.cell_index} failed in {fn_name}", exc_info=True)
            if mark_errored_on_failure:
                self._mark_as_errored()
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

    def cell_status(self) -> CellStatus:
        return compute_cell_status(self._state, self.health_checker.status)

    @property
    def indep_dp_info(self) -> IndepDPInfo:
        assert isinstance(self._state, (StateAllocatedAlive, StateAllocatedErrored))
        return self._state.indep_dp_info

    def _get_actor_handles(self) -> list[ray.actor.ActorHandle]:
        assert isinstance(
            self._state, StateAllocatedBase
        ), f"Cell {self.cell_index} is not allocated (state={type(self._state).__name__})"
        return self._state.actor_handles
