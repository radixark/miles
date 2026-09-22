import logging
import random
import time
from dataclasses import dataclass

from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetPolicy
from tests.utils.soak.core.events import SoakEvent, SoakScheduleEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionRequest, SoakForms, SoakTarget, find_form
from tests.utils.soak.core.views import (
    admission_closed,
    compute_successful_form_names,
    due_of_type,
    is_normal_step,
    latest_observation,
    project_actions,
    quiescent_polls_of_type,
    trainer_step_ends,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


@dataclass(frozen=True, kw_only=True)
class SoakActionScheduler:
    rng: random.Random
    mean_intervals: dict[str, float]
    forms: SoakForms
    config: SoakRunnerConfig
    quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED

    def __post_init__(self) -> None:
        if set(self.config.target_policies) != set(self.mean_intervals):
            raise ValueError("Target policies must name exactly the scheduled target kinds")

    def initial_schedule(self) -> SoakScheduleEvent:
        return SoakScheduleEvent(
            due_of_type={
                kind: time.monotonic() + self.rng.expovariate(1.0 / mean_interval_seconds)
                for kind, mean_interval_seconds in sorted(self.mean_intervals.items())
            }
        )

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
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
            for kind in self.mean_intervals
        }
        due_types = sorted(kind for kind, due_at in due_of_type(events).items() if now >= due_at)
        if not due_types:
            return None

        polls_of_type = quiescent_polls_of_type(
            events,
            expected_count_of_kind={
                kind: policy.expected_count for kind, policy in self.config.target_policies.items()
            },
        )
        ready_types = [
            kind
            for kind in due_types
            if targets_of_type[kind] and polls_of_type[kind] >= self.quiescent_polls_required
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
            policy=self.config.target_policies[kind],
            harms_target=form.harms_target,
        )
        if not targets:
            return None
        target = self.rng.choice(targets)
        request = form.maybe_create_request(target=target, observation=observation, events=events, rng=self.rng)
        if request is None:
            return None
        return request.model_copy(update={"next_due_at": now + self.rng.expovariate(1.0 / self.mean_intervals[kind])})

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


def _eligible_targets(*, targets: list[SoakTarget], policy: SoakTargetPolicy, harms_target: bool) -> list[SoakTarget]:
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
