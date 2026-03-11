from __future__ import annotations

import logging

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.controller.state_machines.controller.context import ControllerContext
from miles.utils.ft.controller.state_machines.controller.models import (
    ControllerState,
    NormalState,
    RestartingMainJobState,
)
from miles.utils.ft.controller.state_machines.main.models import RestartedMainJob, RestartingMainJob
from miles.utils.ft.utils.state_machine import StateHandler

logger = logging.getLogger(__name__)


class NormalStateHandler(StateHandler[NormalState, ControllerContext]):
    async def step(self, state: NormalState, context: ControllerContext) -> ControllerState | None:
        # Step 1: Step all sub-SMs (skip those already requesting restart)
        for name, entry in state.subsystems.items():
            if isinstance(entry.state_machine.state, RestartingMainJob):
                continue
            await entry.state_machine.step(None)

        # Step 2: Check if any sub-SM entered RestartingMainJob
        for name, entry in state.subsystems.items():
            if isinstance(entry.state_machine.state, RestartingMainJob):
                logger.info("sub-SM %r requested main job restart", name)
                await context.main_job.stop_job()
                await context.main_job.submit_job()
                return RestartingMainJobState(requestor_name=name)

        return None


class RestartingMainJobStateHandler(StateHandler[RestartingMainJobState, ControllerContext]):
    async def step(
        self, state: RestartingMainJobState, context: ControllerContext
    ) -> ControllerState | None:
        status = await context.main_job.get_job_status()

        if status == JobStatus.RUNNING:
            fresh = context.create_fresh_subsystems()
            if state.requestor_name in fresh:
                fresh[state.requestor_name].state_machine.force_state(RestartedMainJob())
            return NormalState(subsystems=fresh)

        if status == JobStatus.FAILED:
            logger.warning("main job restart failed, rebuilding subsystems for retry")
            fresh = context.create_fresh_subsystems()
            return NormalState(subsystems=fresh)

        # PENDING / STOPPED — keep waiting
        return None
