import logging
import random
import time
from dataclasses import dataclass, field
from typing import Literal

from tests.utils.soak.core.config import MomentTrigger, RunMoment, SoakRunnerConfig, SoakTargetConfig, TimerTrigger
from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionRequest, SoakForms, SoakTarget, find_form
from tests.utils.soak.core.views import (
    admission_closed,
    compute_num_injections,
    compute_successful_form_names,
    is_normal_step,
    latest_observation,
    project_actions,
    quiescent_polls_of_type,
    sut_events,
    trainer_step_ends,
)

from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, MetricEvent

logger = logging.getLogger(__name__)

_TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"

_Standing = Literal["before", "at", "after"]


@dataclass(frozen=True, kw_only=True)
class SoakActionScheduler:
    forms: SoakForms
    config: SoakRunnerConfig
    rng: random.Random = field(init=False)
    due_of_type: dict[str, float] = field(init=False)
    awaiting_applied: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng", random.Random(self.config.seed))
        now = time.monotonic()
        object.__setattr__(
            self,
            "due_of_type",
            {kind: self._draw_due_at(kind, now=now) for kind in sorted(self._timer_kinds())},
        )

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        self._redraw_applied(events=events, now=now)

        if admission_closed(events) is not None:
            return None
        if not self._is_started(events):
            return None
        if not self._all_recovered(events):
            return None

        observation = latest_observation(events)
        if observation is None or observation.errors:
            return None
        targets_of_type: dict[str, list[SoakTarget]] = {
            kind: [target for target in observation.targets or [] if target.kind == kind]
            for kind in self.config.target_configs
        }
        moment_kinds = self._moment_kinds()
        due_types = sorted(
            [
                *(kind for kind, due_at in self.due_of_type.items() if now >= due_at),
                *(kind for kind in moment_kinds if self._stands_at_next_moment(kind, events=events)),
            ]
        )
        if not due_types:
            return None

        polls_of_type = quiescent_polls_of_type(
            events,
            expected_count_of_kind={
                kind: policy.expected_count for kind, policy in self.config.target_configs.items()
            },
        )
        ready_types = [
            kind
            for kind in due_types
            if targets_of_type[kind]
            and (kind in moment_kinds or polls_of_type[kind] >= self.config.quiescent_polls_required)
        ]
        if not ready_types:
            logger.info(
                "Deferring injection: no due target kind is quiescent with a spare replica (due %s, "
                "quiescent polls %s, replicas %s)",
                due_types,
                {kind: polls_of_type[kind] for kind in due_types},
                {kind: len(targets_of_type[kind]) for kind in due_types},
            )
            return None

        kind = self.rng.choice(ready_types)
        form = _draw_form(self.forms[kind], events=events, kind=kind, rng=self.rng)
        targets = _eligible_targets(
            targets=targets_of_type[kind],
            policy=self.config.target_configs[kind],
            harms_target=form.harms_target,
        )
        if not targets:
            return None
        target = self.rng.choice(targets)
        request = form.maybe_create_request(target=target, observation=observation, events=events, rng=self.rng)
        if request is None:
            return None

        self.awaiting_applied[kind] = request.request_id
        return request

    def _redraw_applied(self, *, events: list[SoakEvent], now: float) -> None:
        applied = {event.request_id for event in events if isinstance(event, SoakActionAppliedEvent)}
        for kind, request_id in list(self.awaiting_applied.items()):
            if request_id in applied:
                if kind in self.due_of_type:
                    self.due_of_type[kind] = self._draw_due_at(kind, now=now)
                del self.awaiting_applied[kind]

    def _draw_due_at(self, kind: str, *, now: float) -> float:
        trigger = self.config.target_configs[kind].trigger
        assert isinstance(trigger, TimerTrigger), f"{kind} is not drawn by a timer"
        return now + self.rng.expovariate(1.0 / trigger.mean_interval_seconds)

    def _timer_kinds(self) -> set[str]:
        return {
            kind for kind, policy in self.config.target_configs.items() if isinstance(policy.trigger, TimerTrigger)
        }

    def _moment_kinds(self) -> set[str]:
        return {
            kind for kind, policy in self.config.target_configs.items() if isinstance(policy.trigger, MomentTrigger)
        }

    def _stands_at_next_moment(self, kind: str, *, events: list[SoakEvent]) -> bool:
        trigger = self.config.target_configs[kind].trigger
        assert isinstance(trigger, MomentTrigger), f"{kind} is not drawn at run moments"
        index = compute_num_injections(events, kind=kind)
        if index >= len(trigger.moments):
            return False
        moment = trigger.moments[index]

        progress = _compute_run_progress(events)
        standing = _compute_standing(progress, moment=moment)
        assert standing != "after", (
            f"{kind} action {index} was to fire while the run was {moment.phase} rollout {moment.at_rollout}, and "
            f"the run stands at {progress}: the soak missed the moment, so where the action landed would be raced"
        )
        return standing == "at"

    def _is_started(self, events: list[SoakEvent]) -> bool:
        if (start_after_rollout_id := self.config.start_after_rollout_id) is None:
            return True
        return any(
            step.rollout_id >= start_after_rollout_id and is_normal_step(step) for step in trainer_step_ends(events)
        )

    def _all_recovered(self, events: list[SoakEvent]) -> bool:
        for action in project_actions(events).values():
            request = action.requested.request
            form = find_form(self.forms, kind=request.target.kind, name=request.form_name)
            if (result := action.result) is not None and not result.returned:
                raise RuntimeError(f"Soak action failed: {request.request_id}: {result.error}")
            if not form.is_recovered(action=action, events=events):
                return False
        return True


def _eligible_targets(*, targets: list[SoakTarget], policy: SoakTargetConfig, harms_target: bool) -> list[SoakTarget]:
    ready = [target for target in targets if target.ready]
    if len(ready) != len(targets) or len(targets) != policy.expected_count:
        return []
    if harms_target and len(ready) < 2:
        return []
    return ready


def _draw_form(
    forms: list[BaseSoakActionForm], *, events: list[SoakEvent], kind: str, rng: random.Random
) -> BaseSoakActionForm:
    worked = compute_successful_form_names(events, kind=kind)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)


# ========================== how far the run has come ==========================


@dataclass(frozen=True)
class _RunProgress:
    last_generated_rollout_id: int | None
    last_trained_rollout_id: int | None
    last_weight_updated_rollout_id: int | None


def _compute_standing(progress: _RunProgress, *, moment: RunMoment) -> _Standing:
    if moment.phase == "training":
        opened, closed = progress.last_generated_rollout_id, progress.last_trained_rollout_id
        opens_at = moment.at_rollout
    else:
        opened, closed = progress.last_weight_updated_rollout_id, progress.last_generated_rollout_id
        opens_at = moment.at_rollout - 1

    if _reached(closed, moment.at_rollout):
        return "after"
    return "at" if _reached(opened, opens_at) else "before"


def _compute_run_progress(events: list[SoakEvent]) -> _RunProgress:
    observed = sut_events(events)
    metric_events = [event for event in observed if isinstance(event, MetricEvent) and event.rollout_id is not None]
    return _RunProgress(
        last_generated_rollout_id=max(
            (event.rollout_id for event in metric_events if event.source.component == "rollout_executor"),
            default=None,
        ),
        last_trained_rollout_id=max(
            (event.rollout_id for event in metric_events if _TRAIN_STEP_METRIC_KEY in event.metrics), default=None
        ),
        last_weight_updated_rollout_id=max(
            (event.rollout_id for event in observed if isinstance(event, InferenceEngineWeightChecksumEvent)),
            default=None,
        ),
    )


def _reached(rollout_id: int | None, threshold: int) -> bool:
    return rollout_id is not None and rollout_id >= threshold
