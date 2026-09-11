from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import CurrentTrainerWitnessIssue
from miles.utils.audit_utils.event_logger.models import (
    Event,
    TrainerCpuWitnessEvent,
    TrainerWitnessCohortEvent,
    TrainingSampleCount,
)


def _latest_completed_cohort_snapshots(
    events: list[Event],
) -> tuple[list[TrainerCpuWitnessEvent], list[CurrentTrainerWitnessIssue]]:
    cohorts = [event for event in events if isinstance(event, TrainerWitnessCohortEvent)]
    if not cohorts:
        return [], [
            CurrentTrainerWitnessIssue(
                description="no completed trainer witness cohort was recorded",
                replicas=[],
            )
        ]

    cohort = max(enumerate(cohorts), key=lambda item: (item[1].timestamp, item[0]))[1]
    expected = set(cohort.replica_ids)
    if not expected:
        return [], [
            CurrentTrainerWitnessIssue(
                description="completed trainer witness cohort contains no replicas",
                replicas=[],
            )
        ]
    latest: dict[str, tuple[int, TrainerCpuWitnessEvent]] = {}
    for position, event in enumerate(events):
        if not isinstance(event, TrainerCpuWitnessEvent):
            continue
        if (
            event.cohort_id != cohort.cohort_id
            or event.rollout_id != cohort.rollout_id
            or event.replica_id not in expected
        ):
            continue
        current = latest.get(event.replica_id)
        if current is None or (event.timestamp, position) > (current[1].timestamp, current[0]):
            latest[event.replica_id] = (position, event)

    missing = sorted(expected - latest.keys())
    if missing:
        return [], [
            CurrentTrainerWitnessIssue(
                description="completed trainer witness cohort is missing replica snapshots",
                replicas=missing,
            )
        ]
    return [latest[replica_id][1] for replica_id in sorted(expected)], []


def _group_outputs_by_source_sample(rows: list[TrainingSampleCount]) -> dict[int, list[TrainingSampleCount]]:
    result: dict[int, list[TrainingSampleCount]] = {}
    for row in rows:
        result.setdefault(row.sample.source_sample_index, []).append(row)
    return result


def _describe_outputs(rows: list[TrainingSampleCount]) -> list[str]:
    return [f"row {row.sample.output_index}/{row.sample.output_count}: count {row.count}" for row in rows]


def _outputs_have_exactly_one_outcome(
    trained_rows: list[TrainingSampleCount],
    skipped_rows: list[TrainingSampleCount],
) -> bool:
    rows = [*trained_rows, *skipped_rows]
    if not rows:
        return False

    row_counts = {row.sample.output_count for row in rows}
    if len(row_counts) != 1 or (output_count := next(iter(row_counts))) <= 0:
        return False
    return (
        len(rows) == output_count
        and {row.sample.output_index for row in rows} == set(range(output_count))
        and all(row.count == 1 for row in rows)
    )
