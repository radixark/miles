from tests.fast.utils.event_analyzer.rules.weight_event_fakes import make_checksum, make_result, make_step_end

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_checksum_coverage import (
    check,
    settled_published_updates,
)
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, WeightUpdateResultEvent

_HASHES: dict[str, str] = {"a": "hash-a", "b": "hash-b"}
_TENSORS: dict[str, str] = {"rank0/w": "w1"}


def _published(
    update_id: str = "u1", *, second: float = 1.0, updated: list[str] | None = None
) -> WeightUpdateResultEvent:
    return make_result(
        second=second,
        update_id=update_id,
        published_version=1,
        cell_hashes=_HASHES,
        updated=["a", "b"] if updated is None else updated,
    )


def _record(
    update_id: str = "u1", *, snapshots: dict[str, tuple[str, dict[str, str]]] | None = None
) -> InferenceEngineWeightChecksumEvent:
    return make_checksum(
        second=1.5,
        update_id=update_id,
        weight_version=1,
        snapshots={"a": ("hash-a", _TENSORS), "b": ("hash-b", _TENSORS)} if snapshots is None else snapshots,
    )


class TestCheck:
    def test_a_settled_update_covered_by_exactly_its_updated_engines_passes(self) -> None:
        """One record naming every updated cell with its snapshot incarnation is the passing shape."""
        assert check([_published(), _record(), make_step_end(second=2.0)]) == []

    def test_a_settled_update_without_any_checksum_record_is_reported(self) -> None:
        """A publication whose receivers were never sampled must not pass silently."""
        [issue] = check([_published(), make_step_end(second=2.0)])

        assert issue.debug_weight_update_id == "u1"
        assert issue.weight_version == 1
        assert issue.description == "0 engine checksum records instead of one"

    def test_two_records_for_one_update_id_are_reported(self) -> None:
        """A duplicated record means two samplings raced, so neither can be trusted as the evidence."""
        [issue] = check([_published(), _record(), _record(), make_step_end(second=2.0)])

        assert issue.description == "2 engine checksum records instead of one"

    def test_a_record_of_another_update_id_does_not_cover_this_one(self) -> None:
        """Coverage is joined by update id, not by version or timestamp."""
        [issue] = check([_published("u1"), _record("u0"), make_step_end(second=2.0)])

        assert issue.description.startswith("0 engine checksum records")

    def test_a_record_missing_an_updated_engine_is_reported(self) -> None:
        """An updated cell whose weights nobody read could hold anything."""
        [issue] = check([_published(), _record(snapshots={"a": ("hash-a", _TENSORS)}), make_step_end(second=2.0)])

        assert issue.description == "checksum record covers ['a'] instead of ['a', 'b']"

    def test_a_record_covering_a_failed_engine_is_reported(self) -> None:
        """Sampling a cell outside the updated set means the record describes a different publication."""
        [issue] = check([_published(updated=["a"]), _record(), make_step_end(second=2.0)])

        assert issue.description == "checksum record covers ['a', 'b'] instead of ['a']"

    def test_a_record_from_a_replaced_incarnation_is_reported(self) -> None:
        """Same cell name with a new workers hash is a different process than the one that was updated."""
        issues = check(
            [
                _published(),
                _record(snapshots={"a": ("hash-a", _TENSORS), "b": ("hash-b-new", _TENSORS)}),
                make_step_end(second=2.0),
            ]
        )

        assert len(issues) == 1

    def test_an_unpublished_update_needs_no_record(self) -> None:
        """A candidate that never published has no serving weights to checksum."""
        unpublished = make_result(
            second=1.0, update_id="u1", published_version=None, cell_hashes=_HASHES, updated=[], failed=["a", "b"]
        )

        assert check([unpublished, make_step_end(second=2.0)]) == []

    def test_the_latest_update_is_skipped_until_a_later_event_settles_it(self) -> None:
        """The newest publication may still be sampling, so only a later step or result settles it."""
        assert check([_published()]) == []
        assert len(check([_published(), make_step_end(second=2.0)])) == 1

    def test_a_later_update_result_also_settles_the_previous_one(self) -> None:
        """A second result is evidence that the first publication finished sampling."""
        [issue] = check([_published("u1"), _published("u2", second=3.0)])

        assert issue.debug_weight_update_id == "u1"

    def test_include_latest_checks_the_newest_update_as_well(self) -> None:
        """Final analysis opts in to checking the tail publication."""
        [issue] = check([_published()], include_latest=True)

        assert issue.debug_weight_update_id == "u1"


class TestSettledPublishedUpdates:
    def test_a_step_at_the_same_instant_does_not_settle_the_update(self) -> None:
        """Settling is strictly-before, so a tie keeps the update pending."""
        assert settled_published_updates([_published(second=2.0), make_step_end(second=2.0)]) == []

    def test_no_results_yields_nothing(self) -> None:
        """Steps alone produce no publications to check."""
        assert settled_published_updates([make_step_end(second=1.0)]) == []
