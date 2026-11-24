import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest, SoakTarget
from tests.utils.soak.core.views import SoakActionRecord, project_actions, trainer_step_ends
from tests.utils.soak.deploy.session import LauncherChain, start_launch
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails, ResizeStep

from miles.utils.test_utils.kubectl_reads import read_replicas

logger = logging.getLogger(__name__)

_RELAUNCH_TIMEOUT_SECONDS: float = 1800.0
_RELAUNCH_POLL_INTERVAL_SECONDS: float = 10.0


@dataclass(frozen=True, kw_only=True)
class ResizePoolForm(BaseSoakActionForm):
    namespace: str
    workload: str
    cell_type: str
    # TODO: generalize the rollout/phase schedule into a scheduler trigger any form can use
    schedule: tuple[ResizeStep, ...]
    relaunch_at_size: Callable[[int], Awaitable[None]]
    event_log: EventLog
    chain: LauncherChain

    @property
    def name(self) -> str:
        return f"resize:{self.cell_type}"

    @property
    def harms_target(self) -> bool:
        return False

    def maybe_create_request(
        self,
        *,
        target: SoakTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if not isinstance(target, PoolTarget) or target.identity != self.workload:
            return None

        index = sum(
            1
            for action in project_actions(events).values()
            if action.requested.request.target.identity == target.identity
            and isinstance(action.requested.request.details, ResizeDetails)
        )
        if index == len(self.schedule):
            return None
        step = self.schedule[index]
        state = _compute_step_state(events, step=step)
        assert state != "missed", (
            f"resize {index} was to fire during rollout {step.at_rollout}, and the run already trained it: the "
            f"scheduler missed the rollout"
        )
        if state != "due":
            return None

        return SoakActionRequest(
            target=target,
            form_name=self.name,
            details=ResizeDetails(step=step),
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, ResizeDetails), f"Request {request.request_id} names no pool size"
        replicas = request.details.step.replicas

        launcher = start_launch(
            self.relaunch_at_size(replicas), event_log=self.event_log, request_id=request.request_id, chain=self.chain
        )
        async with asyncio.timeout(_RELAUNCH_TIMEOUT_SECONDS):
            while (after := await self._read_replicas()) != replicas:
                assert not launcher.done(), (
                    f"the relaunch resizing {self.workload} to {replicas} ended while the pool still reads {after} "
                    f"replica(s): {'cancelled' if launcher.cancelled() else repr(launcher.exception())}"
                )
                await asyncio.sleep(_RELAUNCH_POLL_INTERVAL_SECONDS)

        logger.info(f"Resized {self.workload} {request.target.replicas} -> {after} ({request.details})")
        report_applied(PoolResizedEvidence())

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        if action.applied is None:
            return False
        details = action.requested.request.details
        assert isinstance(details, ResizeDetails)

        return any(
            observation.timestamp > action.applied.timestamp
            and isinstance(target, PoolTarget)
            and target.identity == self.workload
            and target.ready
            and target.replicas == details.step.replicas
            for observation in events
            if isinstance(observation, SoakObservationEvent)
            for target in observation.targets or []
        )

    async def _read_replicas(self) -> int:
        return await asyncio.to_thread(read_replicas, namespace=self.namespace, workload=self.workload)


def _compute_step_state(events: list[SoakEvent], *, step: ResizeStep) -> Literal["pending", "due", "missed"]:
    last_trained_rollout_id = max((event.rollout_id for event in trainer_step_ends(events)), default=-1)
    if last_trained_rollout_id >= step.at_rollout:
        return "missed"
    if last_trained_rollout_id >= step.at_rollout - 1:
        return "due"
    return "pending"
