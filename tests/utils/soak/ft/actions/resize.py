import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest, SoakTarget
from tests.utils.soak.core.views import SoakActionRecord, project_actions, trainer_step_ends
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails, ResizeStep

from miles.utils.test_utils.kubectl_reads import patch_replicas, read_replicas

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ResizePoolForm(BaseSoakActionForm):
    namespace: str
    workload: str
    cell_type: str
    # TODO: generalize the rollout/phase schedule into a scheduler trigger any form can use
    schedule: tuple[ResizeStep, ...] | None
    min_replicas: int
    max_replicas: int

    @property
    def name(self) -> str:
        name = f"resize:{self.cell_type}"
        if self.schedule is not None:
            name += ":scheduled"
        return name

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

        if self.schedule is None:
            details = self._random_details(target=target, rng=rng)
        else:
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
            details = ResizeDetails(replicas=step.replicas, at_rollout=step.at_rollout) if state == "due" else None
        if details is None:
            return None

        return SoakActionRequest(target=target, form_name=self.name, details=details)

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        assert isinstance(request.details, ResizeDetails), f"Request {request.request_id} names no pool size"
        assert request.target.identity == self.workload
        replicas = request.details.replicas

        before = await asyncio.to_thread(read_replicas, namespace=self.namespace, workload=self.workload)
        await asyncio.to_thread(patch_replicas, namespace=self.namespace, workload=self.workload, replicas=replicas)
        after = await asyncio.to_thread(read_replicas, namespace=self.namespace, workload=self.workload)
        assert after == replicas, f"{self.workload} reads {after} replica(s) right after being resized to {replicas}"

        logger.info(f"Resized {self.workload} {before} -> {after} ({request.details})")
        report_applied(PoolResizedEvidence(replicas_before=before, replicas_after=after))

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
            and target.replicas == details.replicas
            for observation in events
            if isinstance(observation, SoakObservationEvent)
            for target in observation.targets or []
        )

    def _random_details(self, *, target: PoolTarget, rng: random.Random) -> ResizeDetails | None:
        candidates = [
            replicas
            for replicas in (target.replicas - 1, target.replicas + 1)
            if self.min_replicas <= replicas <= self.max_replicas
        ]
        if not candidates:
            return None
        return ResizeDetails(replicas=rng.choice(candidates))


def _compute_step_state(events: list[SoakEvent], *, step: ResizeStep) -> Literal["pending", "due", "missed"]:
    last_trained_rollout_id = max((event.rollout_id for event in trainer_step_ends(events)), default=-1)
    if last_trained_rollout_id >= step.at_rollout:
        return "missed"
    if last_trained_rollout_id >= step.at_rollout - 1:
        return "due"
    return "pending"
