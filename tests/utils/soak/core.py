# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import random
import time
from copy import deepcopy
from tests.utils.soak.action import SoakActionForm
from tests.utils.soak.batch import expand_fault_batches
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy
from tests.utils.soak.fault_forms import CellFaultForms, ExecSigkillFaultForm
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.policy import eligible_cells, pending_actions
from tests.utils.soak.sender_assignment import choose_sender_batch
from tests.utils.soak.state import (
    Event,
    ObservationsEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakAdmissionClosedEvent,
    SoakDeploymentTarget,
    SoakObservation,
    SoakScheduleEvent,
    cell_is_alive,
    cell_type_of,
    target_type_of,
)
from tests.utils.soak.views import compute_successful_form_names

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity

POLL_INTERVAL_SECONDS: float = 2.0


def _compute_next_injection_time(rng: random.Random, mean_interval_seconds: float) -> float:
    return time.monotonic() + rng.expovariate(1.0 / mean_interval_seconds)


class SoakActionScheduler:
    def __init__(
        self,
        *,
        rng: random.Random,
        mean_intervals: dict[str, float],
        forms: CellFaultForms,
        policy: SoakPolicy | None = None,
    ) -> None:
        self._rng = rng
        self._mean_intervals = mean_intervals
        self._forms = forms
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

    def choose(self, *, events: list[Event], now: float) -> SoakActionRequest | None:
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
        if len(pending_actions(events)) >= self.policy.max_concurrent_actions:
            return None
        due_of_type: dict[str, float] = {}
        observation = None
        for event in events:
            if isinstance(event, SoakScheduleEvent):
                if event.policy is not None and event.policy != self.policy:
                    raise ValueError("Recorded scheduling policy differs from the scheduler configuration")
                due_of_type.update(event.due_of_type)
            elif isinstance(event, SoakActionRequestedEvent) and event.request.next_due_at is not None:
                due_of_type[target_type_of(event.request.target)] = event.request.next_due_at
            elif isinstance(event, (ObservationsEvent, SoakObservation)):
                observation = event
        if observation is None:
            return None
        cells_of_type: dict[str, list[dict | SoakDeploymentTarget]] = {
            cell_type: [] for cell_type in self._mean_intervals
        }
        for cell in observation.cells or []:
            if cell_type_of(cell) in cells_of_type:
                cells_of_type[cell_type_of(cell)].append(cell)
        if isinstance(observation, SoakObservation) and "deployment" in cells_of_type:
            cells_of_type["deployment"].extend(observation.deployments)
        due_types = sorted(kind for kind, due_at in due_of_type.items() if now >= due_at)
        if not due_types:
            return None

        ready_types = [
            kind
            for kind in due_types
            if len(cells_of_type[kind])
            >= (1 if kind == "deployment" or isinstance(observation, SoakObservation) else 2)
        ]
        if not ready_types:
            return None

        cell_type = self._rng.choice(ready_types)
        form = _draw_form(self._forms[cell_type], events=events, cell_type=cell_type, rng=self._rng)
        reserved_triggers = {
            (action.hook_trigger.cell_id, action.hook_trigger.workers_hash)
            for action in pending_actions(events)
            if action.hook_trigger is not None
        }
        targets = cells_of_type[cell_type]
        if cell_type != "deployment" and isinstance(observation, SoakObservation):
            targets = eligible_cells(
                cells=targets,
                events=events,
                policy=self.policy.cell_policies.get(cell_type, SoakCellPolicy()),
                harms_cell=form.harms_cell,
            )
            if form.harms_cell:
                targets = [
                    cell
                    for cell in targets
                    if (cell["metadata"]["name"], cell["status"].get("workers_hash")) not in reserved_triggers
                ]
        if not targets:
            return None
        if isinstance(form, HookFaultForm) and form.all_targets:
            policy = self.policy.cell_policies.get(cell_type, SoakCellPolicy())
            if policy.min_survivors < 1 or policy.expected_cells is None:
                raise ValueError("All-sender-target hooks require surviving targets and an explicit fleet size")
            if not isinstance(observation, SoakObservation):
                return None
            targets = eligible_cells(
                cells=targets,
                events=events,
                policy=policy.model_copy(update={"require_ready_target": True}),
                harms_cell=True,
            )
            if len(targets) != policy.expected_cells or len(targets) < 2:
                return None
        target = self._rng.choice(targets)
        if not form.is_eligible(events=events, target=target):
            return None
        hook_trigger = None
        target_form = form
        if isinstance(form, HookFaultForm) and form.victim_form is not None:
            target_form = form.victim_form
            if not isinstance(observation, SoakObservation):
                return None
            reserved_victims = {
                (event.request.target["metadata"]["name"], event.request.target["status"].get("workers_hash"))
                for event in expand_fault_batches(events)
                if isinstance(event, SoakActionRequestedEvent)
                and event.request.harms_cell
                and isinstance(event.request.target, dict)
            }
            triggers = [
                identity
                for cell in observation.cells or []
                if cell_type_of(cell) == "actor" and cell_is_alive(cell)
                if (identity := observation.fault_targets.get(cell["metadata"]["name"])) is not None
                and identity.workers_hash == cell["status"].get("workers_hash")
                and (identity.cell_id, identity.workers_hash) not in reserved_triggers | reserved_victims
            ]
            if not triggers:
                return None
            hook_trigger = self._rng.choice(triggers)
        selected = [target]
        if isinstance(form, HookFaultForm) and form.all_targets:
            batch = choose_sender_batch(
                observation=observation,
                targets=targets,
                triggers=triggers,
                min_survivors=policy.min_survivors,
                rng=self._rng,
            )
            if batch is None:
                return None
            hook_trigger, selected = batch.trigger, batch.targets
        requests = [
            _build_observed_request(
                target=cell, form=target_form, harms_cell=form.harms_cell, observation=observation, rng=self._rng
            )
            for cell in selected
        ]
        if any(request is None for request in requests):
            return None
        request = requests[0]
        next_due_at = now + self._rng.expovariate(1.0 / self._mean_intervals[cell_type])
        return request.model_copy(
            update={
                "form_name": form.name,
                "next_due_at": next_due_at,
                "hook_trigger": hook_trigger,
                "hook_delay_ms": form.sample_delay(self._rng) if isinstance(form, HookFaultForm) else None,
                "additional_requests": requests[1:],
            }
        )


def _build_observed_request(
    *,
    target: dict | SoakDeploymentTarget,
    form: SoakActionForm,
    harms_cell: bool,
    observation: SoakObservation | ObservationsEvent,
    rng: random.Random,
) -> SoakActionRequest | None:
    fault_target = None
    candidates = None
    if isinstance(observation, SoakObservation) and form.name.startswith(("inject_fault:", "hook:")):
        assert isinstance(target, dict), "Fault injection requires a cell target"
        fault_target = observation.fault_targets.get(target["metadata"]["name"])
        if fault_target is None or fault_target.workers_hash != target["status"].get("workers_hash"):
            return None
    if isinstance(observation, SoakObservation) and form.name in {"delete_pod", "exec_sigkill", "exec_sigstop"}:
        assert isinstance(target, dict), "Pod faults require a cell target"
        candidates = observation.pods_of_cell.get(target["metadata"]["name"], [])
        if isinstance(form, ExecSigkillFaultForm):
            candidates = [
                pod
                for pod in candidates
                if all(
                    container in pod.process_targets and pod.process_targets[container].pattern == pattern
                    for container, pattern in form.process_patterns.items()
                )
            ]
        if not candidates:
            return None
    return SoakActionRequest(
        target=deepcopy(target),
        form_name=form.name,
        harms_cell=harms_cell,
        pod=rng.choice(candidates) if candidates is not None else None,
        fault_target=fault_target,
    )


def _draw_form(
    forms: list[SoakActionForm], *, events: list[Event], cell_type: str, rng: random.Random
) -> SoakActionForm:
    worked = compute_successful_form_names(events, cell_type=cell_type)
    unproven = [form for form in forms if form.name not in worked]
    return rng.choice(unproven or forms)
