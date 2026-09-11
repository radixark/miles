from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.state import Event, SoakActionAppliedEvent, SoakActionRequest, SoakActionRequestedEvent

from miles.utils.audit_utils.event_logger.models import (
    EngineEnvReportEvent,
    EnvReportEvent,
    FaultHookEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity, TrainProcessIdentity
from miles.utils.test_utils.fault_hooks import FaultHookRecord, FaultHookRequest
from miles.utils.workers.cell_operations.base import FaultTarget


@dataclass
class HookEvidence:
    events: list[Event]
    hit: FaultHookEvent


@pytest.fixture
def deterministic_environment() -> list[EnvReportEvent | EngineEnvReportEvent]:
    timestamp = datetime(2026, 9, 11, tzinfo=timezone.utc)
    return [
        EnvReportEvent(
            timestamp=timestamp,
            source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0),
            report={
                "process": {
                    "hostname": "trainer",
                    "argv": [],
                    "args": {
                        "values": {
                            "deterministic_mode": True,
                            "debug_deterministic_collective": True,
                            "ft_components": ["rollout"],
                            "update_weight_transfer_mode": "p2p",
                            "sglang_router_policy": "round_robin",
                            "colocate": False,
                        },
                        "skipped_names": [],
                    },
                    "env_vars": {"NCCL_ALGO": "Ring"},
                    "launcher_env_report": None,
                },
                "key_versions": {},
                "editable_packages": [],
                "git_repos": [],
                "full_pip_list": [],
                "packages_probed": False,
            },
        ),
        EngineEnvReportEvent(
            timestamp=timestamp,
            source=TrainerControllerProcessIdentity(trainer_id="actor"),
            cell_id="rollout-0",
            workers_hash="generation-0",
            server_url="http://engine",
            server_info={
                "enable_deterministic_inference": True,
                "attention_backend": "flashinfer",
                "disable_radix_cache": True,
                "internal_states": [{"env_vars": {"SGLANG_ENABLE_JIT_DEEPGEMM": "false"}}],
            },
        ),
    ]


@pytest.fixture
def p2p_update_result(remote_hook_evidence: HookEvidence) -> WeightUpdateResultEvent:
    return WeightUpdateResultEvent(
        timestamp=remote_hook_evidence.hit.timestamp,
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        update_id=remote_hook_evidence.hit.update_id,
        rollout_id=24,
        candidate_version=23,
        published_version=23,
        target_incarnations={"rollout-1": "generation-0", "rollout-2": "generation-0"},
        updated_cell_ids=["rollout-2"],
        failed_cell_ids=["rollout-1"],
    )


@pytest.fixture
def remote_hook_evidence(hook_evidence: HookEvidence) -> HookEvidence:
    original_request = hook_evidence.events[0].request
    hook = FaultHookRequest(
        request_id=f"{original_request.request_id}:trigger",
        instance_id=hook_evidence.hit.instance_id,
        hook="trainer_before_weight_send",
        mode="sigkill",
        delay_ms=500,
        action="observe",
    )
    victim = FaultTarget(cell_id="rollout-1", sub_index=0, workers_hash="generation-0")
    request = original_request.model_copy(
        update={
            "form_name": "remote_hook:trainer_before_weight_send:inject_fault:sigkill:500ms",
            "target": typed_cell("rollout-1", "rollout"),
            "fault_target": victim,
            "hook_trigger": original_request.fault_target,
        }
    )
    hit = hook_evidence.hit.model_copy(update={"request_id": hook.request_id, "hook": hook.hook, "action": "observe"})
    record = FaultHookRecord(
        request=hook,
        status="fired",
        armed_at=10,
        changed_at=hit.monotonic_time,
        reached_at=hit.reached_at,
        due_at=hit.due_at,
        weight_version=hit.weight_version,
        update_id=hit.update_id,
    )
    return HookEvidence(
        events=[
            SoakActionRequestedEvent(request=request),
            SoakActionAppliedEvent(
                request_id=request.request_id,
                evidence={
                    "request_id": request.request_id,
                    "mode": "sigkill",
                    "target": victim.model_dump(mode="json"),
                    "exited_pids": [42],
                    "hook_request": hook.model_dump(mode="json"),
                    "hook_hit": record.model_dump(mode="json"),
                    "hook_trigger": original_request.fault_target.model_dump(mode="json"),
                    "victim_form": "inject_fault:sigkill",
                },
            ),
        ],
        hit=hit,
    )


@pytest.fixture
def batch_remote_hook_evidence(remote_hook_evidence: HookEvidence) -> HookEvidence:
    request = remote_hook_evidence.events[0].request
    child = request.model_copy(
        update={
            "request_id": "second-victim",
            "form_name": "inject_fault:sigkill",
            "target": typed_cell("rollout-2", "rollout"),
            "fault_target": FaultTarget(cell_id="rollout-2", sub_index=0, workers_hash="generation-0"),
        }
    )
    request = request.model_copy(update={"additional_requests": [child], "form_name": request.form_name + ":all"})
    assignments = {"rollout-1": "generation-0", "rollout-2": "generation-0"}
    evidence = remote_hook_evidence.events[-1].evidence
    evidence["hook_hit"]["target_incarnations"] = assignments
    evidence["batch_receipts"] = {
        child.request_id: {
            "request_id": child.request_id,
            "target": child.fault_target.model_dump(mode="json"),
            "mode": "sigkill",
            "exited_pids": [43],
        }
    }
    return HookEvidence(
        events=[SoakActionRequestedEvent(request=request), remote_hook_evidence.events[-1]],
        hit=remote_hook_evidence.hit.model_copy(update={"target_incarnations": assignments}),
    )


@pytest.fixture
def hook_evidence() -> HookEvidence:
    hook = FaultHookRequest(
        request_id="request",
        instance_id="instance",
        hook="trainer_before_all_gather",
        mode="sigkill",
        delay_ms=500,
    )
    target = FaultTarget(cell_id="actor-7", sub_index=0, workers_hash="generation-0")
    request = SoakActionRequest(
        request_id=hook.request_id,
        target=typed_cell("actor-7", "actor"),
        form_name="hook:trainer_before_all_gather:sigkill:500ms",
        harms_cell=True,
        fault_target=target,
    )
    return HookEvidence(
        events=[
            SoakActionRequestedEvent(request=request),
            SoakActionAppliedEvent(
                request_id=request.request_id,
                evidence={
                    "hook_request": hook.model_dump(mode="json"),
                    "request_id": request.request_id,
                    "target": target.model_dump(mode="json"),
                    "mode": "sigkill",
                    "exited_pids": [42],
                },
            ),
        ],
        hit=FaultHookEvent(
            timestamp=datetime(2026, 9, 11, tzinfo=timezone.utc),
            source=TrainProcessIdentity(component="actor", cell_index=7, rank_within_cell=0),
            request_id=hook.request_id,
            instance_id=hook.instance_id,
            hook=hook.hook,
            mode=hook.mode,
            status="fired",
            monotonic_time=11.6,
            reached_at=11.0,
            due_at=11.5,
            weight_version=23,
            update_id="update-23-first-attempt",
        ),
    )
