from collections.abc import Sequence

from tests.utils.soak.state import (
    Event,
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakObservation,
    cell_is_alive,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import FaultHookEvent, TrainGroupStepEndEvent
from miles.utils.test_utils.fault_hooks import FaultHookRequest
from miles.utils.workers.naming import parse_cell_id


def assert_hook_effects(events: Sequence[Event], *, hook_events: Sequence[FaultHookEvent]) -> None:
    requests = {}
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            assert event.request.request_id not in requests, "Duplicate fault request"
            requests[event.request.request_id] = event.request

    applied = set()
    for event in events:
        if not isinstance(event, SoakActionAppliedEvent):
            continue
        request = requests[event.request_id]
        if not request.form_name.startswith("hook:"):
            continue
        assert event.request_id not in applied, "Duplicate hook effect"
        applied.add(event.request_id)
        hook_request = FaultHookRequest.model_validate(event.evidence["hook_request"])
        assert hook_request.request_id == request.request_id, "Hook evidence belongs to another request"
        assert request.form_name == (
            f"hook:{hook_request.hook}:{hook_request.mode}:{hook_request.delay_ms:g}ms"
        ), "Hook evidence names another form"
        matching = [event for event in hook_events if event.request_id == request.request_id]
        assert matching, "Applied hook has no worker-side evidence"
        assert all(
            event.instance_id == hook_request.instance_id
            and event.hook == hook_request.hook
            and event.mode == hook_request.mode
            for event in matching
        ), "Hook evidence mixes process incarnations or fault parameters"
        fired = [event for event in matching if event.status == "fired"]
        assert len(fired) == 1, "Applied hook must have exactly one dispatch"
        hit = fired[0]
        assert hit.weight_version is not None, "Weight-update hook lacks its exact version"
        assert hit.reached_at is not None and hit.due_at is not None, "Hook lacks target-local timing"
        assert abs(hit.due_at - hit.reached_at - hook_request.delay_ms / 1000) < 1e-6, "Wrong hook delay"
        assert hit.monotonic_time >= hit.due_at, "Hook fired before its target-local deadline"
        assert not any(
            event.status in {"cancelled", "expired", "failed"} for event in matching
        ), "Applied hook also claims cancellation, expiration or dispatch failure"

    assert applied, "No precise hook fault was confirmed"


def assert_hook_survivors(events: Sequence[Event], *, steps: Sequence[TrainGroupStepEndEvent]) -> None:
    observed: dict[str, str] = {}
    candidates: dict[str, dict[str, str]] = {}
    checked = 0
    for event in events:
        if isinstance(event, SoakObservation) and event.cells is not None:
            observed = {
                cell["metadata"]["name"]: cell["status"]["workers_hash"]
                for cell in event.cells
                if cell_is_alive(cell) and cell["status"].get("workers_hash")
            }
        elif isinstance(event, SoakActionRequestedEvent) and event.request.form_name.startswith("hook:"):
            assert event.request.fault_target is not None, "Hook lacks its original target"
            target = event.request.fault_target.cell_id
            pool = parse_cell_id(target).pool_id
            candidates[event.request.request_id] = {
                name: incarnation
                for name, incarnation in observed.items()
                if name != target and parse_cell_id(name).pool_id == pool
            }
        elif isinstance(event, SoakActionAppliedEvent) and event.request_id in candidates:
            survivors = candidates[event.request_id]
            assert survivors, "No original healthy peer was observed before the hook request"
            assert any(
                step.timestamp > event.timestamp
                and step.cell_incarnations.get(name) == incarnation
                and isinstance(outcomes := step.cell_outcomes.get(parse_cell_id(name).cell_index), list)
                and outcomes
                and all(outcome == TrainStepOutcome.NORMAL for outcome in outcomes)
                for name, incarnation in survivors.items()
                for step in steps
            ), "No original peer completed normal training after the hook effect"
            checked += 1

    assert checked, "No applied hook had survivor evidence"
