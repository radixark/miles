from datetime import datetime, timedelta, timezone

import pytest
from tests.fast.utils.soak.utils import cell
from tests.utils.soak.checks.tail import assert_tail_complete
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakObservation,
)

from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


@pytest.mark.parametrize("progress", ["normal", "early", "rollback", "critic", "discarded"])
def test_tail_requires_new_actor_progress_after_the_replacement_was_observed(progress: str) -> None:
    """Recovery alone, an old step, critic progress, or a discarded step cannot complete the tail."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    request = SoakActionRequest(
        target=cell("rollout-0", cell_type="rollout", healthy=True), form_name="kill", harms_cell=True
    )
    before = TrainGroupStepEndEvent(
        timestamp=start,
        source=TrainerControllerProcessIdentity(trainer_id="actor"),
        rollout_id=5,
        cell_outcomes={0: ["normal"]},
    )
    after = TrainGroupStepEndEvent(
        timestamp=start + timedelta(seconds=2 if progress == "early" else 4),
        source=TrainerControllerProcessIdentity(trainer_id="critic" if progress == "critic" else "actor"),
        rollout_id=4 if progress == "rollback" else 6,
        cell_outcomes={0: ["discarded_should_retry" if progress == "discarded" else "normal"]},
    )
    events = [
        SoakActionRequestedEvent(timestamp=start, request=request),
        SoakActionAppliedEvent(timestamp=start, request_id=request.request_id, evidence={"exited_pids": [1]}),
        SoakActionResultEvent(timestamp=start, request_id=request.request_id, returned=True),
        SoakObservation(timestamp=start, cells=[], training_events=[before]),
        SoakAdmissionClosedEvent(timestamp=start + timedelta(seconds=1)),
        SoakObservation(
            timestamp=start + timedelta(seconds=3),
            cells=[cell("rollout-0", cell_type="rollout", healthy=True, workers_hash="replacement")],
        ),
        SoakObservation(timestamp=start + timedelta(seconds=5), cells=[], training_events=[after]),
    ]
    if progress == "normal":
        assert_tail_complete(events)
    else:
        with pytest.raises(AssertionError, match="no successful training progress"):
            assert_tail_complete(events)


def test_an_unconfirmed_request_prevents_tail_success() -> None:
    """A failed command with unknown effect cannot disappear from the final verdict."""
    request = SoakActionRequest(target=cell("actor-0", healthy=True), form_name="kill", harms_cell=True)
    with pytest.raises(AssertionError, match="without confirmed effects"):
        assert_tail_complete(
            [
                SoakActionRequestedEvent(request=request),
                SoakActionResultEvent(request_id=request.request_id, returned=False, error="timeout"),
                SoakAdmissionClosedEvent(),
            ]
        )
