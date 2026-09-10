from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.state import Event, SoakActionAppliedEvent, SoakActionRequest, SoakActionRequestedEvent

from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.fault_hooks import FaultHookRequest
from miles.utils.workers.cell_operations.base import FaultTarget


@dataclass
class HookEvidence:
    events: list[Event]
    hit: FaultHookEvent


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
        ),
    )
