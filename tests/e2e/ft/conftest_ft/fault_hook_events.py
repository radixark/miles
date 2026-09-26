from collections.abc import Iterable, Sequence
from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import FaultHookEvent, WeightUpdateResultEvent
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, FaultHookStatus


def assert_fault_hooks_fired(events_dir: Path, *, request_ids: Sequence[str]) -> None:
    records = _read_fault_hook_records(events_dir)
    status_of_request_id = {record.request.request_id: record.status for record in records}
    for request_id in request_ids:
        assert (status := status_of_request_id.get(request_id)) is FaultHookStatus.FIRED, (
            f"Declared fault hook {request_id} ended as {status}, so the run never took the fault it was built to "
            f"survive"
        )


def assert_weight_updates_published(events_dir: Path, *, rollout_ids: Iterable[int]) -> None:
    results = _read_weight_update_results(events_dir)
    for rollout_id in rollout_ids:
        rollout_results = [result for result in results if result.rollout_id == rollout_id]
        assert rollout_results, f"Rollout {rollout_id} recorded no weight update"
        for result in rollout_results:
            assert result.published_version is not None, (
                f"Rollout {rollout_id} published no weight version (failed cells {result.failed_cell_ids}), so the "
                f"engines generated the next rollout on stale weights"
            )


def assert_weight_update_failures(events_dir: Path, *, failed_cell_ids_of_rollout_id: dict[int, list[str]]) -> None:
    results = _read_weight_update_results(events_dir)
    seen_rollout_ids = {result.rollout_id for result in results}
    missing = sorted(set(failed_cell_ids_of_rollout_id) - seen_rollout_ids)
    assert not missing, f"Rollouts {missing} recorded no weight update"
    for result in results:
        if result.rollout_id is None:
            continue
        expected = failed_cell_ids_of_rollout_id.get(result.rollout_id, [])
        assert sorted(result.failed_cell_ids) == sorted(
            expected
        ), f"Rollout {result.rollout_id} failed to update cells {result.failed_cell_ids}, expected {expected}"


def _read_fault_hook_records(events_dir: Path) -> list[FaultHookRecord]:
    return [event.record for event in read_events(events_dir) if isinstance(event, FaultHookEvent)]


def _read_weight_update_results(events_dir: Path) -> list[WeightUpdateResultEvent]:
    return [event for event in read_events(events_dir) if isinstance(event, WeightUpdateResultEvent)]
