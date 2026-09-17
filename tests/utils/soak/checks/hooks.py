from collections.abc import Sequence

from tests.utils.soak.fault_forms import ObservedCellFault
from tests.utils.soak.process_target import ProcessExitReceipt, ProcessStopReceipt
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakEvent,
    SoakObservation,
    cell_is_alive,
)

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import FaultHookEvent, TrainGroupStepEndEvent, WeightUpdateResultEvent
from miles.utils.test_utils.fault_hooks import FaultHookRecord, FaultHookRequest
from miles.utils.workers.cell_operations.base import FaultTarget
from miles.utils.workers.naming import parse_cell_id


def assert_hook_effects(events: Sequence[SoakEvent], *, hook_events: Sequence[FaultHookEvent]) -> None:
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
        if not request.form_name.startswith(("hook:", "remote_hook:")):
            continue
        assert event.request_id not in applied, "Duplicate hook effect"
        applied.add(event.request_id)
        hook_request = FaultHookRequest.model_validate(event.evidence["hook_request"])
        delay = f"{hook_request.delay_ms:g}ms"
        if request.hook_delay_ms is not None:
            assert hook_request.delay_ms == request.hook_delay_ms, "Hook delay differs from the recorded draw"
            delay = "random"
        remote = request.form_name.startswith("remote_hook:")
        if remote:
            assert hook_request.action == "observe", "Remote hook must leave its trigger unharmed"
            assert hook_request.request_id == f"{request.request_id}:trigger", "Wrong remote trigger request"
            assert request.form_name == (
                f"remote_hook:{hook_request.hook}:{event.evidence['victim_form']}:{delay}"
            ), "Remote hook evidence names another form"
            assert request.hook_trigger is not None
            assert FaultTarget.model_validate(event.evidence["hook_trigger"]) == request.hook_trigger
        else:
            assert hook_request.action == "inject", "Observation alone is not a local fault effect"
            assert hook_request.request_id == request.request_id, "Hook evidence belongs to another request"
            assert request.form_name == (
                f"hook:{hook_request.hook}:{hook_request.mode}:{delay}"
            ), "Hook evidence names another form"
        matching = [event for event in hook_events if event.request_id == hook_request.request_id]
        assert matching, "Applied hook has no worker-side evidence"
        assert all(
            event.instance_id == hook_request.instance_id
            and event.action == hook_request.action
            and event.hook == hook_request.hook
            and event.mode == hook_request.mode
            for event in matching
        ), "Hook evidence mixes process incarnations or fault parameters"
        fired = [event for event in matching if event.status == "fired"]
        assert len(fired) == 1, "Applied hook must have exactly one dispatch"
        hit = fired[0]
        assert hit.weight_version is not None, "Weight-update hook lacks its exact version"
        assert hit.update_id, "Weight-update hook lacks its exact update identity"
        assert hit.reached_at is not None and hit.due_at is not None, "Hook lacks target-local timing"
        assert abs(hit.due_at - hit.reached_at - hook_request.delay_ms / 1000) < 1e-6, "Wrong hook delay"
        assert hit.monotonic_time >= hit.due_at, "Hook fired before its target-local deadline"
        assert not any(
            event.status in {"cancelled", "expired", "failed"} for event in matching
        ), "Applied hook also claims cancellation, expiration or dispatch failure"
        if remote:
            recorded_hit = FaultHookRecord.model_validate(event.evidence["hook_hit"])
            assert recorded_hit.request == hook_request
            assert recorded_hit.status == "fired"
            assert (
                recorded_hit.weight_version,
                recorded_hit.update_id,
                recorded_hit.reached_at,
                recorded_hit.due_at,
                recorded_hit.changed_at,
            ) == (
                hit.weight_version,
                hit.update_id,
                hit.reached_at,
                hit.due_at,
                hit.monotonic_time,
            ), "Remote hit differs from worker evidence"
            assert recorded_hit.target_incarnations == hit.target_incarnations, "Remote sender assignment differs"
            _assert_remote_victim_effect(request=request, evidence=event.evidence)
    assert applied, "No precise hook fault was confirmed"


def assert_remote_p2p_failures(
    events: Sequence[SoakEvent],
    *,
    hook_events: Sequence[FaultHookEvent],
    update_events: Sequence[WeightUpdateResultEvent],
) -> set[str]:
    assert_hook_effects(events, hook_events=hook_events)
    requests = {
        event.request.request_id: event.request for event in events if isinstance(event, SoakActionRequestedEvent)
    }
    checked: set[str] = set()
    expected_forms = {
        request.form_name
        for request in requests.values()
        if request.form_name.startswith("remote_hook:trainer_before_weight_send:")
    }
    matched_forms: set[str] = set()
    for event in events:
        if not isinstance(event, SoakActionAppliedEvent):
            continue
        request = requests[event.request_id]
        if not request.form_name.startswith("remote_hook:trainer_before_weight_send:"):
            continue
        hit = FaultHookRecord.model_validate(event.evidence["hook_hit"])
        matching = [result for result in update_events if result.update_id == hit.update_id]
        assert len(matching) == 1, "Remote P2P fault lacks one result from the triggered update"
        result = matching[0]
        assert result.candidate_version == hit.weight_version, "Triggered update changed its candidate version"
        assert isinstance(request.target, dict)
        cell_id = request.target["metadata"]["name"]
        incarnation = request.target["status"]["workers_hash"]
        assert result.target_incarnations.get(cell_id) == incarnation, "Triggered update targeted another incarnation"
        updated, failed = set(result.updated_cell_ids), set(result.failed_cell_ids)
        assert not updated & failed, "P2P result reports both success and failure for one target"
        assert updated | failed == set(result.target_incarnations), "P2P result omits assigned targets"
        victims = {cell_id: incarnation}
        assert result.published_version == (
            result.candidate_version if updated else None
        ), "P2P result published a version inconsistent with its surviving targets"
        if not set(victims) <= failed:
            continue
        checked.add(request.request_id)
        matched_forms.add(request.form_name)

    assert checked, "No remote P2P fault was checked against its triggered update"
    assert (
        matched_forms == expected_forms
    ), f"Remote P2P forms without a precise hit: {sorted(expected_forms - matched_forms)}"
    return checked


def _assert_remote_victim_effect(*, request: SoakActionRequest, evidence: dict) -> None:
    victim_form = evidence["victim_form"]
    receipt = {
        key: value
        for key, value in evidence.items()
        if key not in {"hook_request", "hook_hit", "hook_trigger", "victim_form"}
    }
    if victim_form.startswith("inject_fault:"):
        effect = ObservedCellFault.model_validate(receipt)
        assert request.fault_target is not None
        assert effect.request_id == request.request_id and effect.target == request.fault_target.model_dump(
            mode="json"
        )
        assert victim_form == f"inject_fault:{effect.mode.value}", "Victim receipt names another fault mode"
    elif victim_form in {"exec_sigkill", "exec_sigstop"}:
        assert request.pod is not None, "Remote process fault lacks its observed pod"
        process_effect = (
            ProcessExitReceipt.model_validate(receipt)
            if victim_form == "exec_sigkill"
            else ProcessStopReceipt.model_validate(receipt)
        )
        assert process_effect.request_id == request.request_id
        assert process_effect.target in request.pod.process_targets.values(), "Remote process identity changed"
        process_effect.validate_for(request_id=request.request_id, target=process_effect.target)
    elif victim_form == "delete_pod":
        assert request.pod is not None
        assert receipt == {
            "kind": "pod_deleted",
            "namespace": request.pod.namespace,
            "pod_name": request.pod.name,
            "pod_uid": request.pod.uid,
        }, "Remote deletion does not match the observed pod"
    else:
        raise AssertionError(f"Unsupported remote hook effect: {victim_form}")


def assert_hook_survivors(events: Sequence[SoakEvent], *, steps: Sequence[TrainGroupStepEndEvent]) -> None:
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
