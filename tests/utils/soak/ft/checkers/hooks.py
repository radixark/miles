from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import project_actions, weight_update_results
from tests.utils.soak.ft.actions.hook import UNFIRED_TERMINAL_STATUSES, assert_fired_hook
from tests.utils.soak.ft.types import (
    HookFaultDetails,
    HookFaultEvidence,
    RemoteHookFaultDetails,
    RemoteHookFaultEvidence,
)

from miles.utils.audit_utils.event_logger.models import Event, FaultHookEvent
from miles.utils.test_utils.fault_injector.actions.process import ObserveAction
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookStatus


def assert_hook_effects(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    hook_events = [event for event in training_events if isinstance(event, FaultHookEvent)]
    checked = 0
    for action in project_actions(events).values():
        if action.applied is None:
            continue
        request = action.requested.request
        details = request.details

        match action.applied.evidence:
            case HookFaultEvidence() as evidence:
                assert isinstance(details, HookFaultDetails), "Hook evidence answers another request"
                assert (
                    evidence.effect.request_id == request.request_id and evidence.effect.target == details.fault_target
                ), "Hook effect belongs to another target"
                _assert_single_dispatch(
                    evidence.hook_request_id,
                    action=details.action,
                    hook_name=details.hook_name,
                    delay_ms=details.delay_ms,
                    hook_events=hook_events,
                )
            case RemoteHookFaultEvidence() as evidence:
                assert isinstance(details, RemoteHookFaultDetails), "Remote evidence answers another request"
                hit = _assert_single_dispatch(
                    evidence.hit.record.request.request_id,
                    action=ObserveAction(),
                    hook_name=details.hook_name,
                    delay_ms=details.delay_ms,
                    hook_events=hook_events,
                )
                assert evidence.hit == hit, "Remote hit differs from worker evidence"
            case _:
                continue
        checked += 1

    assert checked, "No precise hook fault was confirmed"


def assert_remote_p2p_failures(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    results = {result.debug_weight_update_id: result for result in weight_update_results(training_events)}
    checked = 0
    for action in project_actions(events).values():
        request = action.requested.request
        if not isinstance(request.details, RemoteHookFaultDetails) or action.applied is None:
            continue
        assert isinstance(evidence := action.applied.evidence, RemoteHookFaultEvidence)
        assert (context := evidence.hit.record.context) is not None, "Remote P2P hit lacks its exact update"
        result = results[context.debug_weight_update_id]
        assert result.candidate_version == context.weight_version, "Triggered update changed its candidate version"
        assert (
            result.snapshot_cell_id_to_hashes.get(request.target.identity) == request.target.incarnation
        ), "Triggered update targeted another incarnation"
        assert request.target.identity in result.failed_cell_ids, "Remote P2P victim survived its triggered update"
        checked += 1

    assert checked, "No remote P2P fault was checked against its triggered update"


def _assert_single_dispatch(
    request_id: str,
    *,
    action: FaultAction,
    hook_name: FaultHookName,
    delay_ms: float,
    hook_events: list[FaultHookEvent],
) -> FaultHookEvent:
    matching = [event for event in hook_events if event.record.request.request_id == request_id]
    assert matching, "Applied hook has no worker-side evidence"
    assert all(
        (event.record.request.action, event.record.request.hook_name) == (action, hook_name) for event in matching
    ), "Hook evidence mixes fault parameters"

    fired = [event for event in matching if event.record.status == FaultHookStatus.FIRED]
    assert len(fired) == 1, "Applied hook must have exactly one dispatch"
    hit = fired[0]
    assert_fired_hook(hit)
    assert abs(hit.record.due_at - hit.record.reached_at - delay_ms / 1000) < 1e-6, "Wrong hook delay"
    assert not any(
        event.record.status in UNFIRED_TERMINAL_STATUSES for event in matching
    ), "Applied hook also claims clearing, expiration or dispatch failure"
    return hit
