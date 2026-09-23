from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import project_actions, weight_update_results
from tests.utils.soak.ft.actions.hook import OBSERVE_MODE, TERMINAL_WITHOUT_DISPATCH, assert_fired_hook
from tests.utils.soak.ft.actions.pod import DELETE_POD_FORM_NAME, EXEC_SIGKILL_FORM_NAME, EXEC_SIGSTOP_FORM_NAME
from tests.utils.soak.ft.types import (
    HookFaultDetails,
    HookFaultEvidence,
    InjectFaultDetails,
    ObservedCellFault,
    PodDetails,
    RemoteHookFaultDetails,
    RemoteHookFaultEvidence,
)
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence
from tests.utils.soak.k8s_utils.pod_processes import ProcessSignal, ProcessSignalReceipt

from miles.utils.audit_utils.event_logger.models import (
    Event,
    FaultHookAction,
    FaultHookEvent,
    FaultHookName,
    FaultHookStatus,
)
from miles.utils.test_utils.fault_injector import FailureMode

SIGNAL_OF_PROCESS_FORM: dict[str, ProcessSignal] = {
    EXEC_SIGKILL_FORM_NAME: ProcessSignal.KILL,
    EXEC_SIGSTOP_FORM_NAME: ProcessSignal.STOP,
}


def assert_hook_effects(events: list[SoakEvent], *, hook_events: list[FaultHookEvent]) -> None:
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
                    action=FaultHookAction.INJECT,
                    hook_name=details.hook_name,
                    mode=details.mode,
                    delay_ms=details.delay_ms,
                    hook_events=hook_events,
                )
            case RemoteHookFaultEvidence() as evidence:
                assert isinstance(details, RemoteHookFaultDetails), "Remote evidence answers another request"
                _assert_remote_victim_effect(request_id=request.request_id, details=details, evidence=evidence)
                hit = _assert_single_dispatch(
                    evidence.hit.record.request.request_id,
                    action=FaultHookAction.OBSERVE,
                    hook_name=details.hook_name,
                    mode=OBSERVE_MODE,
                    delay_ms=details.delay_ms,
                    hook_events=hook_events,
                )
                assert evidence.hit == hit, "Remote hit differs from worker evidence"
            case _:
                continue
        checked += 1

    assert checked, "No precise hook fault was confirmed"


def assert_remote_p2p_failures(events: list[SoakEvent], *, training_events: list[Event]) -> None:
    results = weight_update_results(training_events)
    expected_forms: set[str] = set()
    matched_forms: set[str] = set()
    for action in project_actions(events).values():
        request = action.requested.request
        if not isinstance(request.details, RemoteHookFaultDetails):
            continue
        expected_forms.add(request.form_name)
        if action.applied is None:
            continue

        assert isinstance(evidence := action.applied.evidence, RemoteHookFaultEvidence)
        assert (context := evidence.hit.record.context) is not None, "Remote P2P hit lacks its exact update"
        matching = [result for result in results if result.debug_weight_update_id == context.debug_weight_update_id]
        assert len(matching) == 1, "Remote P2P fault lacks one result from the triggered update"
        result = matching[0]
        assert result.candidate_version == context.weight_version, "Triggered update changed its candidate version"
        assert (
            result.snapshot_cell_id_to_hashes.get(request.target.identity) == request.target.incarnation
        ), "Triggered update targeted another incarnation"
        if request.target.identity in result.failed_cell_ids:
            matched_forms.add(request.form_name)

    assert matched_forms, "No remote P2P fault was checked against its triggered update"
    assert (
        matched_forms == expected_forms
    ), f"Remote P2P forms without a precise hit: {sorted(expected_forms - matched_forms)}"


def _assert_single_dispatch(
    request_id: str,
    *,
    action: FaultHookAction,
    hook_name: FaultHookName,
    mode: FailureMode,
    delay_ms: float,
    hook_events: list[FaultHookEvent],
) -> FaultHookEvent:
    matching = [event for event in hook_events if event.record.request.request_id == request_id]
    assert matching, "Applied hook has no worker-side evidence"
    assert all(
        (event.record.request.action, event.record.request.hook_name, event.record.request.mode)
        == (action, hook_name, mode)
        for event in matching
    ), "Hook evidence mixes fault parameters"

    fired = [event for event in matching if event.record.status == FaultHookStatus.FIRED]
    assert len(fired) == 1, "Applied hook must have exactly one dispatch"
    hit = fired[0]
    assert_fired_hook(hit)
    assert abs(hit.record.due_at - hit.record.reached_at - delay_ms / 1000) < 1e-6, "Wrong hook delay"
    assert not any(
        event.record.status in TERMINAL_WITHOUT_DISPATCH for event in matching
    ), "Applied hook also claims clearing, expiration or dispatch failure"
    return hit


def _assert_remote_victim_effect(
    *, request_id: str, details: RemoteHookFaultDetails, evidence: RemoteHookFaultEvidence
) -> None:
    victim = evidence.victim_evidence
    match victim:
        case ObservedCellFault():
            assert isinstance(details.victim, InjectFaultDetails), "Remote cell fault answers a pod request"
            assert (
                victim.request_id == request_id and victim.target == details.victim.fault_target
            ), "Remote cell fault belongs to another target"
            assert (
                details.victim_form == f"inject_fault:{victim.mode.value}"
            ), "Victim receipt names another fault mode"
        case ProcessSignalReceipt():
            assert isinstance(details.victim, PodDetails), "Remote process fault lacks its observed pod"
            assert victim.target in details.victim.pod.process_targets.values(), "Remote process identity changed"
            victim.validate_for(
                request_id=request_id, target=victim.target, operation=SIGNAL_OF_PROCESS_FORM[details.victim_form]
            )
        case PodDeletedEvidence():
            assert isinstance(details.victim, PodDetails), "Remote deletion lacks its observed pod"
            assert details.victim_form == DELETE_POD_FORM_NAME, "Victim receipt names another form"
            pod = details.victim.pod
            assert (victim.namespace, victim.pod_name, victim.pod_uid) == (
                pod.namespace,
                pod.name,
                pod.uid,
            ), "Remote deletion does not match the observed pod"
