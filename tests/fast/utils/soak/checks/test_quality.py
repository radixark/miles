from datetime import datetime, timedelta, timezone

import pytest
from tests.utils.soak.checks.quality import assert_tail_quality
from tests.utils.soak.state import SoakActionAppliedEvent, SoakAdmissionClosedEvent, SoakObservation

from miles.utils.audit_utils.event_logger.models import MetricEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity


@pytest.mark.parametrize(
    "case", ["good", "degraded", "old_start", "old_rollout", "duplicate", "missing_start", "nan", "infinite", "bool"]
)
def test_quality_requires_two_distinct_fresh_evaluations_and_rejects_a_degraded_tail(case: str) -> None:
    """Historical peaks, stale asynchronous results, duplicate reads, and invalid scores cannot hide tail regression."""
    start = datetime(2026, 9, 11, tzinfo=timezone.utc)
    first = MetricEvent(
        timestamp=start + timedelta(seconds=5),
        evaluation_started_at=start + timedelta(seconds=4),
        source=SimpleProcessIdentity(component="rollout_executor"),
        rollout_id=11,
        metrics={"eval/gsm8k": 0.95},
    )
    second = MetricEvent(
        timestamp=start + timedelta(seconds=7),
        evaluation_started_at=(
            None if case == "missing_start" else start + timedelta(seconds=2 if case == "old_start" else 6)
        ),
        source=SimpleProcessIdentity(component="rollout_executor"),
        rollout_id=9 if case == "old_rollout" else 12,
        metrics={
            "eval/gsm8k": {"degraded": 0.4, "nan": float("nan"), "infinite": float("inf"), "bool": True}.get(
                case, 0.55
            )
        },
    )
    events = [
        SoakObservation(
            timestamp=start + timedelta(seconds=1),
            cells=[],
            training_events=[
                TrainGroupStepEndEvent(
                    timestamp=start,
                    source=TrainerControllerProcessIdentity(trainer_id="actor"),
                    rollout_id=10,
                    cell_outcomes={0: ["normal"]},
                )
            ],
        ),
        SoakAdmissionClosedEvent(timestamp=start + timedelta(seconds=2)),
        SoakActionAppliedEvent(timestamp=start + timedelta(seconds=3), request_id="last", evidence={}),
        SoakObservation(
            timestamp=start + timedelta(seconds=8),
            cells=[],
            training_events=[first, first if case == "duplicate" else second],
        ),
    ]
    if case == "good":
        assert_tail_quality(events, metric_key="eval/gsm8k", threshold=0.55)
    else:
        with pytest.raises(AssertionError):
            assert_tail_quality(events, metric_key="eval/gsm8k", threshold=0.55)
