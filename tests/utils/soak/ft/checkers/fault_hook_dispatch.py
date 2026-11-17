from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import project_actions, weight_update_results
from tests.utils.soak.ft.types import InjectFaultDetails, ObservedCellFault

from miles.utils.audit_utils.event_logger.models import Event, FaultHookEvent
from miles.utils.test_utils.fault_injector.models import FaultHookStatus

UNFIRED_TERMINAL_STATUSES: frozenset[FaultHookStatus] = frozenset(
    {FaultHookStatus.CLEARED, FaultHookStatus.EXPIRED, FaultHookStatus.FAILED}
)


def assert_hook_dispatches(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    hook_events = [event for event in training_events if isinstance(event, FaultHookEvent)]
    checked = 0
    for action in project_actions(events).values():
        request = action.requested.request
        details = request.details
        if action.applied is None or not isinstance(details, InjectFaultDetails) or details.hook_name is None:
            continue
        evidence = action.applied.evidence
        assert (
            isinstance(evidence, ObservedCellFault)
            and evidence.request_id == request.request_id
            and evidence.target == details.fault_target
        ), "Hook effect belongs to another target"

        matching = [event for event in hook_events if event.record.request.request_id == request.request_id]
        assert matching, "Applied hook has no worker-side evidence"
        assert all(
            (event.record.request.hook_name, event.record.request.delay_ms) == (details.hook_name, details.delay_ms)
            for event in matching
        ), "Hook evidence mixes fault parameters"
        fired = [event for event in matching if event.record.status == FaultHookStatus.FIRED]
        assert len(fired) == 1, "Applied hook must have exactly one dispatch"
        _assert_fired_hook(fired[0])
        assert not any(
            event.record.status in UNFIRED_TERMINAL_STATUSES for event in matching
        ), "Applied hook also claims clearing, expiration or dispatch failure"
        checked += 1

    assert checked, "No hook-triggered fault was confirmed"


def assert_p2p_receiver_failures(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    results = {result.debug_weight_update_id: result for result in weight_update_results(training_events)}
    fired = {
        event.record.request.request_id: event
        for event in training_events
        if isinstance(event, FaultHookEvent) and event.record.status == FaultHookStatus.FIRED
    }
    checked = 0
    for action in project_actions(events).values():
        request = action.requested.request
        details = request.details
        if (
            action.applied is None
            or not isinstance(details, InjectFaultDetails)
            or details.hook_target == details.fault_target
        ):
            continue
        assert (
            context := fired[request.request_id].record.context
        ) is not None, "Receiver hook lacks its exact update"
        result = results[context.debug_weight_update_id]
        assert result.candidate_version == context.weight_version, "Triggered update changed its candidate version"
        assert (
            result.snapshot_cell_id_to_hashes.get(request.target.identity) == request.target.incarnation
        ), "Triggered update targeted another incarnation"
        assert request.target.identity in result.failed_cell_ids, "Receiver survived the update its hook fired in"
        checked += 1

    assert checked, "No receiver fault was checked against the update it fired in"


def _assert_fired_hook(event: FaultHookEvent) -> None:
    record = event.record
    assert (
        record.context is not None and record.context.debug_weight_update_id
    ), "Weight-update hook lacks its exact update"
    assert record.reached_at is not None and record.due_at is not None, "Hook lacks target-local timing"
    assert record.changed_at >= record.due_at, "Hook fired before its target-local deadline"
