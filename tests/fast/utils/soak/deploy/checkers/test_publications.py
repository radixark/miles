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

from miles.utils.audit_utils.event_logger.models import WeightUpdateResultEvent

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


def _source(tmp_path: Path, results: list[WeightUpdateResultEvent]) -> Path:
    tmp_path.joinpath("events").mkdir()
    _write_sut_lines(tmp_path / "events" / "controller.jsonl", results)
    return tmp_path / "events"


class TestAssertPublicationsAfterTakeOvers:
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
