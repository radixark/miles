# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import time

from tests.utils.soak.action import SoakActionForm
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy
from tests.utils.soak.fault_forms import CellFaultForms
from tests.utils.soak.policy import eligible_cells
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakAdmissionClosedEvent,
    SoakDeploymentTarget,
    SoakEvent,
    SoakObservation,
    SoakScheduleEvent,
    cell_is_alive,
    cell_type_of,
    target_type_of,
)
from tests.utils.soak.views import compute_successful_form_names, project_actions

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 2.0
QUIESCENT_POLLS_REQUIRED: int = 60


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


class SoakActionScheduler:
    def __init__(
        self,
        *,
        rng: random.Random,
        mean_intervals: dict[str, float],
        forms: CellFaultForms,
        quiescent_polls_required: int = QUIESCENT_POLLS_REQUIRED,
        policy: SoakPolicy | None = None,
    ) -> None:
        self._rng = rng
        self._mean_intervals = mean_intervals
        self._forms = forms
        self._quiescent_polls_required = quiescent_polls_required
        self.policy = policy if policy is not None else SoakPolicy()
        if set(self.policy.cell_policies) - set(mean_intervals):
            raise ValueError("Cell policies must name scheduled cell types")

    def initial_schedule(self) -> SoakScheduleEvent:
        return SoakScheduleEvent(
            due_of_type={
                cell_type: _compute_next_injection_time(self._rng, mean_interval_seconds)
                for cell_type, mean_interval_seconds in sorted(self._mean_intervals.items())
            },
            policy=self.policy,
        )

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        if any(isinstance(event, SoakAdmissionClosedEvent) for event in events):
            return None
        if self.policy.start_after_rollout_id is not None and not any(
            isinstance(step, TrainGroupStepEndEvent)
            and isinstance(step.source, TrainerControllerProcessIdentity)
            and step.source.trainer_id == "actor"
            and step.rollout_id >= self.policy.start_after_rollout_id
            and any(
                isinstance(outcomes, list) and TrainStepOutcome.NORMAL in outcomes
                for outcomes in step.cell_outcomes.values()
            )
            for observation in events
            if isinstance(observation, SoakObservation)
            for step in observation.training_events
        ):
            return None
        for action in project_actions(events).values():
            request = action.requested.request
            form = next(form for form in self._forms[target_type_of(request.target)] if form.name == request.form_name)
            if action.result is not None and not action.result.returned:
                raise RuntimeError(f"Soak action failed: {request.request_id}: {action.result.error}")
            if not form.is_recovered(action=action, events=events):
                return None
        due_of_type: dict[str, float] = {}
        # Quiescence is derived, not remembered: the largest replica count a kind ever showed, and
        # how many consecutive polls it has looked settled since its last injection attempt.
        max_num_cells_of_type: dict[str, int] = dict.fromkeys(self._mean_intervals, 0)
        quiescent_polls_of_type: dict[str, int] = dict.fromkeys(self._mean_intervals, 0)
        landed_request_ids = {event.request_id for event in events if isinstance(event, SoakActionAppliedEvent)}
        observation = None
        for event in events:
            if isinstance(event, SoakScheduleEvent):
                if event.policy is not None and event.policy != self.policy:
                    raise ValueError("Recorded scheduling policy differs from the scheduler configuration")
                due_of_type.update(event.due_of_type)
            elif isinstance(event, SoakActionRequestedEvent):
                # M38 moves the deadline only once an injection lands, and clears the streak on
                # every attempt, so a failed one leaves the kind due again on the next poll.
                if event.request.next_due_at is not None and event.request.request_id in landed_request_ids:
                    due_of_type[target_type_of(event.request.target)] = event.request.next_due_at
                quiescent_polls_of_type[target_type_of(event.request.target)] = 0
            elif isinstance(event, SoakObservation):
                observation = event
                if event.cells is not None:
                    polled_of_type: dict[str, list[dict]] = {
                        cell_type: [] for cell_type in self._mean_intervals if cell_type != "deployment"
                    }
                    for cell in event.cells:
                        if cell_type_of(cell) in polled_of_type:
                            polled_of_type[cell_type_of(cell)].append(cell)
                    for cell_type, kind_cells in sorted(polled_of_type.items()):
                        max_num_cells_of_type[cell_type] = max(max_num_cells_of_type[cell_type], len(kind_cells))
                        if _kind_is_quiescent(kind_cells, expected_num_cells=max_num_cells_of_type[cell_type]):
                            quiescent_polls_of_type[cell_type] += 1
                        else:
                            quiescent_polls_of_type[cell_type] = 0
                if "deployment" in quiescent_polls_of_type:
                    if event.deployments and not event.errors:
                        max_num_cells_of_type["deployment"] = max(
                            max_num_cells_of_type["deployment"], len(event.deployments)
                        )
                        if len(event.deployments) == max_num_cells_of_type["deployment"]:
                            quiescent_polls_of_type["deployment"] += 1
                        else:
                            quiescent_polls_of_type["deployment"] = 0
                    else:
                        quiescent_polls_of_type["deployment"] = 0
        if observation is None or observation.errors:
            return None
        cells_of_type: dict[str, list[dict | SoakDeploymentTarget]] = {
            cell_type: [] for cell_type in self._mean_intervals
        }
        for cell in observation.cells or []:
            if cell_type_of(cell) in cells_of_type:
                cells_of_type[cell_type_of(cell)].append(cell)
        if "deployment" in cells_of_type:
            cells_of_type["deployment"].extend(observation.deployments)
        due_types = sorted(kind for kind, due_at in due_of_type.items() if now >= due_at)
        if not due_types:
            return None

        # Inject only at a quiescent point: every replica of the kind present and alive for long
        # enough that the readings cannot all be stale. A due kind that is still recovering (or has
        # no spare replica to survive the kill) waits for a later poll.
        quiescent_types = {
            kind
            for kind in due_types
            if quiescent_polls_of_type[kind] >= self._quiescent_polls_required and cells_of_type[kind]
        }
        ready_types = [kind for kind in due_types if cells_of_type[kind] and kind in quiescent_types]
        if not ready_types:
            logger.info(
                "Deferring injection: no due cell kind is quiescent with a spare replica (due %s, "
                "quiescent polls %s, replicas %s)",
                due_types,
                {kind: quiescent_polls_of_type[kind] for kind in due_types},
                {kind: len(cells_of_type[kind]) for kind in due_types},
            )
            return None

        cell_type = self._rng.choice(ready_types)
        form = _draw_form(self._forms[cell_type], events=events, cell_type=cell_type, rng=self._rng)
        targets = cells_of_type[cell_type]
        if cell_type != "deployment":
            targets = eligible_cells(
                cells=targets,
                events=events,
                policy=self.policy.cell_policies.get(cell_type, SoakCellPolicy()),
                harms_cell=form.harms_cell,
            )
        if not targets:
            return None
        target = self._rng.choice(targets)
        if not form.is_eligible(events=events, target=target):
            return None
        request = form.prepare_request(target=target, observation=observation, events=events, rng=self._rng)
        if request is None:
            return None
        return request.model_copy(
            update={"next_due_at": now + self._rng.expovariate(1.0 / self._mean_intervals[cell_type])}
        )


def _kind_is_quiescent(kind_cells: list[dict], *, expected_num_cells: int) -> bool:
    if not kind_cells or len(kind_cells) < expected_num_cells:
        return False
    return all(cell_is_alive(cell) for cell in kind_cells)


def _draw_form(
    forms: list[SoakActionForm], *, events: list[SoakEvent], cell_type: str, rng: random.Random
) -> SoakActionForm:
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)
