import math

from tests.utils.soak.state import Event, SoakActionAppliedEvent, SoakAdmissionClosedEvent, SoakObservation

from miles.utils.audit_utils.event_logger.models import MetricEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


def assert_tail_quality(
    events: list[Event], *, metric_key: str, threshold: float, min_evaluations: int = 2, trainer_id: str = "actor"
) -> None:
    assert min_evaluations >= 2, "Tail quality requires repeated evaluations"
    assert math.isfinite(threshold), "Tail quality requires a finite threshold"
    closed = next((event for event in events if isinstance(event, SoakAdmissionClosedEvent)), None)
    assert closed is not None, "Quality evidence requires injection admission to close"
    cutoff = max(
        [closed.timestamp, *(event.timestamp for event in events if isinstance(event, SoakActionAppliedEvent))]
    )
    training_events = [
        training_event
        for event in events
        if isinstance(event, SoakObservation)
        for training_event in event.training_events
    ]
    floor = max(
        (
            event.rollout_id
            for event in training_events
            if isinstance(event, TrainGroupStepEndEvent)
            and isinstance(event.source, TrainerControllerProcessIdentity)
            and event.source.trainer_id == trainer_id
            and event.timestamp <= cutoff
        ),
        default=-1,
    )
    by_rollout: dict[int, MetricEvent] = {}
    for event in training_events:
        if (
            not isinstance(event, MetricEvent)
            or event.rollout_id is None
            or event.rollout_id <= floor
            or event.timestamp <= cutoff
            or event.evaluation_started_at is None
            or event.evaluation_started_at <= cutoff
            or event.evaluation_started_at > event.timestamp
            or metric_key not in event.metrics
        ):
            continue
        previous = by_rollout.get(event.rollout_id)
        if previous is None or previous.timestamp < event.timestamp:
            by_rollout[event.rollout_id] = event
    tail = [by_rollout[rollout_id] for rollout_id in sorted(by_rollout)[-min_evaluations:]]
    assert len(tail) == min_evaluations, (
        f"Expected {min_evaluations} distinct {metric_key} evaluations after the last fault and admission closure; "
        f"found {len(tail)} beyond rollout {floor}"
    )
    for event in tail:
        value = event.metrics[metric_key]
        assert (
            type(value) in (int, float) and math.isfinite(value) and value >= threshold
        ), f"Tail quality failed at rollout {event.rollout_id}: {metric_key}={value!r}, required >= {threshold}"
