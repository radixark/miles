import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest, SoakTarget
from tests.utils.soak.core.views import SoakActionRecord, project_actions, sut_events
from tests.utils.soak.ft.types import Moment, PoolResizedEvidence, PoolTarget, ResizeDetails

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, MetricEvent
from miles.utils.test_utils.kubectl_reads import patch_replicas, read_replicas

logger = logging.getLogger(__name__)

LANDING_LAG_ROLLOUTS: int = 3
TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"

Standing = Literal["before", "at", "after"]


# ================================ the schedule ================================


@dataclass(frozen=True)
class ScalingStep:
    at_rollout: int
    replicas: int
    moment: Moment


def assert_schedule_leaves_room(
    schedule: tuple[ScalingStep, ...], *, initial_replicas: int, num_rollouts: int
) -> None:
    assert schedule, "an empty schedule resizes nothing, and this scenario is about a pool that changes size"
    replicas = initial_replicas
    previous_at = -LANDING_LAG_ROLLOUTS - 1
    for step in schedule:
        assert step.replicas != replicas, f"{step} keeps the pool at {replicas} replica(s), so it scales nothing"
        assert step.at_rollout > previous_at + LANDING_LAG_ROLLOUTS, (
            f"{step} fires while the step before it may still be landing (up to {LANDING_LAG_ROLLOUTS} rollouts "
            f"after rollout {previous_at}), so the two sizes could not be told apart"
        )
        replicas = step.replicas
        previous_at = step.at_rollout
    assert previous_at + LANDING_LAG_ROLLOUTS < num_rollouts - 1, (
        f"the last step fires at rollout {previous_at} and may land up to {LANDING_LAG_ROLLOUTS} rollouts later, "
        f"leaving no rollout of the {num_rollouts} to train at the final size"
    )


# ================================== the form ==================================


@dataclass(frozen=True, kw_only=True)
class ResizePoolForm(BaseSoakActionForm):
    namespace: str
    workload: str
    cell_type: str
    schedule: tuple[ScalingStep, ...] | None
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
            details = self._draw_details(target=target, rng=rng)
        else:
            details = self._scheduled_details(target=target, events=events, schedule=self.schedule)
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
            and len(target.cell_ids) == details.replicas
            for observation in events
            if isinstance(observation, SoakObservationEvent)
            for target in observation.targets or []
        )

    def _draw_details(self, *, target: PoolTarget, rng: random.Random) -> ResizeDetails | None:
        candidates = [
            replicas
            for replicas in (target.replicas - 1, target.replicas + 1)
            if self.min_replicas <= replicas <= self.max_replicas
        ]
        if not candidates:
            return None
        return ResizeDetails(replicas=rng.choice(candidates), at_rollout=None, moment=None)

    def _scheduled_details(
        self, *, target: PoolTarget, events: list[SoakEvent], schedule: tuple[ScalingStep, ...]
    ) -> ResizeDetails | None:
        index = sum(
            1
            for action in project_actions(events).values()
            if action.requested.request.target.identity == target.identity
            and isinstance(action.requested.request.details, ResizeDetails)
        )
        if index == len(schedule):
            return None
        step = schedule[index]

        progress = _compute_run_progress(events)
        standing = compute_standing(progress, step=step)
        assert standing != "after", (
            f"resize {index} was to fire while the run was {step.moment} rollout {step.at_rollout}, and the "
            f"run stands at {progress}: the soak missed the moment, so where the resize landed would be raced"
        )
        if standing != "at":
            return None
        return ResizeDetails(replicas=step.replicas, at_rollout=step.at_rollout, moment=step.moment)


# ========================== how far the run has come ==========================


@dataclass(frozen=True)
class RunProgress:
    last_generated_rollout_id: int | None
    last_trained_rollout_id: int | None
    last_weight_updated_rollout_id: int | None


def compute_standing(progress: RunProgress, *, step: ScalingStep) -> Standing:
    if step.moment == "training":
        opened, closed = progress.last_generated_rollout_id, progress.last_trained_rollout_id
        opens_at = step.at_rollout
    else:
        opened, closed = progress.last_weight_updated_rollout_id, progress.last_generated_rollout_id
        opens_at = step.at_rollout - 1

    if _reached(closed, step.at_rollout):
        return "after"
    return "at" if _reached(opened, opens_at) else "before"


def _compute_run_progress(events: list[SoakEvent]) -> RunProgress:
    observed = sut_events(events)
    metric_events = [event for event in observed if isinstance(event, MetricEvent) and event.rollout_id is not None]
    return RunProgress(
        last_generated_rollout_id=max(
            (event.rollout_id for event in metric_events if event.source.component == "rollout_executor"),
            default=None,
        ),
        last_trained_rollout_id=max(
            (event.rollout_id for event in metric_events if TRAIN_STEP_METRIC_KEY in event.metrics), default=None
        ),
        last_weight_updated_rollout_id=max(
            (event.rollout_id for event in observed if isinstance(event, InferenceEngineWeightChecksumEvent)),
            default=None,
        ),
    )


def _reached(rollout_id: int | None, threshold: int) -> bool:
    return rollout_id is not None and rollout_id >= threshold
