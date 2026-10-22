from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import (
    _applied,
    _at,
    _cell_target,
    _request,
    _requested,
    _weight_update_result,
    _write_sut_lines,
)
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent
from tests.utils.soak.deploy.checkers.publications import assert_publications_after_take_overs

from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineChecksumCollectionFailedEvent,
    InferenceEngineWeightChecksumEvent,
    WeightUpdateResultEvent,
)

_HASHES: dict[str, str] = {"rollout-0": "hash-a"}


def _published(update_id: str, *, second: float) -> WeightUpdateResultEvent:
    return _weight_update_result(update_id, at=_at(second), cell_hashes=_HASHES, updated=["rollout-0"])


def _unpublished(update_id: str, *, second: float) -> WeightUpdateResultEvent:
    return _weight_update_result(update_id, at=_at(second), cell_hashes=_HASHES, updated=[], failed=["rollout-0"])


def _soak(*, closed_at: float, applied_at: float | None = None) -> list[SoakEvent]:
    events: list[SoakEvent] = []
    if applied_at is not None:
        request = _request(_cell_target())
        events += [_requested(request, at=_at(applied_at - 1)), _applied(request, at=_at(applied_at))]
    return [*events, SoakAdmissionClosedEvent(timestamp=_at(closed_at))]


def _checksum(result: WeightUpdateResultEvent) -> InferenceEngineWeightChecksumEvent:
    return InferenceEngineWeightChecksumEvent(
        timestamp=result.timestamp,
        source=result.source,
        rollout_id=result.rollout_id,
        weight_version=result.published_version,
        debug_trainer_load_state_timestamp=result.debug_trainer_load_state_timestamp,
        debug_weight_update_id=result.debug_weight_update_id,
        engine_snapshots=[dict(cell_id="rollout-0", workers_hash="hash-a", tensor_checksums={"w": "value"})],
    )


def _source(tmp_path: Path, results: list[WeightUpdateResultEvent], *, records: list[Event] | None = None) -> Path:
    tmp_path.joinpath("events").mkdir()
    if records is None:
        records = [_checksum(result) for result in results if result.published_version is not None]
    _write_sut_lines(tmp_path / "events" / "controller.jsonl", [*results, *records])
    return tmp_path / "events"


class TestAssertPublicationsAfterTakeOvers:
    def test_two_collection_failures_do_not_prove_post_takeover_audit(self, tmp_path: Path) -> None:
        """The three-failure allowance must not let the entire recovery tail go unaudited."""
        results = [_published("u1", second=20), _published("u2", second=30)]
        failures = [
            InferenceEngineChecksumCollectionFailedEvent(
                timestamp=result.timestamp,
                source=result.source,
                rollout_id=result.rollout_id,
                weight_version=result.published_version,
                debug_trainer_load_state_timestamp=result.debug_trainer_load_state_timestamp,
                debug_weight_update_id=result.debug_weight_update_id,
                published_cell_id_to_hashes=_HASHES,
                reason="collection timed out",
            )
            for result in results
        ]
        source = _source(tmp_path, results, records=failures)

        with pytest.raises(AssertionError, match="No successfully audited"):
            assert_publications_after_take_overs(_soak(closed_at=10), source=source)

    @pytest.mark.parametrize("invalid", ["missing", "old_publication", "wrong_incarnation", "wrong_version"])
    def test_unrelated_or_invalid_checksums_do_not_prove_recovery(self, tmp_path: Path, invalid: str) -> None:
        """Recovery needs a successful checksum for the publication and receivers after takeover."""
        results = [_published("old", second=1), _published("u1", second=20), _published("u2", second=30)]
        checksum = _checksum(results[0] if invalid == "old_publication" else results[1])
        if invalid == "wrong_incarnation":
            checksum = checksum.model_copy(
                update={"engine_snapshots": [checksum.engine_snapshots[0].model_copy(update={"workers_hash": "old"})]}
            )
        elif invalid == "wrong_version":
            checksum = checksum.model_copy(update={"weight_version": 99})
        source = _source(tmp_path, results, records=[] if invalid == "missing" else [checksum])

        with pytest.raises(AssertionError, match="No successfully audited"):
            assert_publications_after_take_overs(_soak(closed_at=10), source=source)

    def test_one_valid_checksum_in_the_recovery_tail_is_enough(self, tmp_path: Path) -> None:
        """Recovery requires successful audit evidence without demanding every collection succeed."""
        results = [_published("u1", second=20), _published("u2", second=30)]
        source = _source(tmp_path, results, records=[_checksum(results[1])])

        assert_publications_after_take_overs(_soak(closed_at=10), source=source)

    def test_two_publications_after_the_tail_starts_pass(self, tmp_path: Path) -> None:
        """Two published updates after the last take-over prove the run serves new weights again."""
        source = _source(tmp_path, [_published("u1", second=20), _published("u2", second=30)])

        assert_publications_after_take_overs(_soak(closed_at=5, applied_at=10), source=source)

    @pytest.mark.parametrize("count", [0, 1])
    def test_fewer_than_two_publications_fail(self, tmp_path: Path, count: int) -> None:
        """One publication could be the in-flight update from before the take-over."""
        source = _source(tmp_path, [_published(f"u{index}", second=20 + index) for index in range(count)])

        with pytest.raises(AssertionError, match=f"Only {count} weight publications"):
            assert_publications_after_take_overs(_soak(closed_at=5), source=source)

    def test_unpublished_results_do_not_count(self, tmp_path: Path) -> None:
        """An update that reached no engine is a failure, not evidence of recovery."""
        source = _source(tmp_path, [_published("u1", second=20), _unpublished("u2", second=30)])

        with pytest.raises(AssertionError, match="Only 1 weight publications"):
            assert_publications_after_take_overs(_soak(closed_at=5), source=source)

    def test_publications_before_the_last_applied_take_over_do_not_count(self, tmp_path: Path) -> None:
        """The tail starts at the later of admission close and the last applied take-over."""
        source = _source(tmp_path, [_published("u1", second=8), _published("u2", second=30)])

        with pytest.raises(AssertionError, match="Only 1 weight publications"):
            assert_publications_after_take_overs(_soak(closed_at=5, applied_at=10), source=source)

    def test_publications_before_admission_closed_do_not_count(self, tmp_path: Path) -> None:
        """Publications while faults could still be drawn are not tail evidence."""
        source = _source(tmp_path, [_published("u1", second=4), _published("u2", second=30)])

        with pytest.raises(AssertionError, match="Only 1 weight publications"):
            assert_publications_after_take_overs(_soak(closed_at=5), source=source)

    def test_a_publication_at_the_tail_start_counts(self, tmp_path: Path) -> None:
        """The tail boundary is inclusive."""
        source = _source(tmp_path, [_published("u1", second=10), _published("u2", second=30)])

        assert_publications_after_take_overs(_soak(closed_at=5, applied_at=10), source=source)

    def test_a_soak_whose_admission_never_closed_is_rejected(self, tmp_path: Path) -> None:
        """Without a tail start there is no after to count publications in."""
        source = _source(tmp_path, [_published("u1", second=20), _published("u2", second=30)])

        with pytest.raises(AssertionError, match="admission never closed"):
            assert_publications_after_take_overs([], source=source)


class TestSuccessfulTailAuditIdentity:
    @pytest.mark.parametrize("mismatch", ["update_id", "lineage", "policy", "duplicate"])
    def test_a_checksum_must_identify_one_unambiguous_tail_publication(self, tmp_path: Path, mismatch: str) -> None:
        """Version equality alone cannot establish a successful recovery-tail audit."""
        results = [_published("u1", second=20), _published("u2", second=30)]
        checksum = _checksum(results[1])
        if mismatch == "update_id":
            checksum = checksum.model_copy(update={"debug_weight_update_id": "unrelated"})
        elif mismatch == "lineage":
            checksum = checksum.model_copy(update={"debug_trainer_load_state_timestamp": 99.0})
        elif mismatch == "policy":
            checksum = checksum.model_copy(update={"trainer_model_id": "unrelated"})
        records = [checksum, checksum] if mismatch == "duplicate" else [checksum]
        source = _source(tmp_path, results, records=records)

        with pytest.raises(AssertionError, match="No successfully audited"):
            assert_publications_after_take_overs(_soak(closed_at=10), source=source)

    def test_an_audit_before_the_last_applied_takeover_does_not_cover_the_tail(self, tmp_path: Path) -> None:
        """The final takeover invalidates audit evidence from earlier recovery windows."""
        results = [_published("before", second=12), _published("u1", second=21), _published("u2", second=30)]
        source = _source(tmp_path, results, records=[_checksum(results[0])])

        with pytest.raises(AssertionError, match="No successfully audited"):
            assert_publications_after_take_overs(_soak(closed_at=10, applied_at=20), source=source)
