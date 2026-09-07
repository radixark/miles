import pytest

from miles.backends.training_utils.weight_update.report import (
    WeightUpdateReport,
    build_lost_trainer_report,
    build_untouched_targets_report,
    build_weight_update_report,
    combine_trainer_reports,
    merge_rank_reports,
)


class TestWeightUpdateReportShape:
    """A malformed report would let the orchestration script publish or retire the wrong engines."""

    def test_a_cell_reported_as_both_updated_and_failed_is_rejected(self):
        """The script would both register it with the router and retire it, leaving the fleet inconsistent."""
        with pytest.raises(AssertionError, match="both updated and failed"):
            WeightUpdateReport(weight_version=1, updated_cell_ids=("cell-0",), failed_cell_ids=("cell-0",))

    def test_a_cell_named_twice_is_rejected(self):
        """A duplicated id hides a second, differently-named cell that was never accounted for."""
        with pytest.raises(AssertionError, match="names a cell twice"):
            WeightUpdateReport(weight_version=1, updated_cell_ids=("cell-0", "cell-0"), failed_cell_ids=())

    def test_a_trainer_that_published_nothing_can_still_report_failures(self):
        """A trainer that died mid-update published no version, yet its targets are all unusable."""
        report = build_lost_trainer_report(["cell-0", "cell-1"])

        assert report.weight_version is None
        assert report.updated_cell_ids == ()
        assert report.failed_cell_ids == ("cell-0", "cell-1")


class TestValidateAssignment:
    """The trainer must account for exactly the targets the controller handed it."""

    def test_a_cell_that_was_never_assigned_is_rejected(self):
        """Acting on it would retire or publish another trainer's engine."""
        report = WeightUpdateReport(weight_version=1, updated_cell_ids=("cell-9",), failed_cell_ids=())

        with pytest.raises(AssertionError, match="never assigned"):
            report.validate_assignment(["cell-0"])

    def test_an_assigned_cell_with_no_verdict_is_rejected(self):
        """An unmentioned target would silently keep the previous weights while counting as up to date."""
        report = WeightUpdateReport(weight_version=1, updated_cell_ids=("cell-0",), failed_cell_ids=())

        with pytest.raises(AssertionError, match="reported neither success nor failure"):
            report.validate_assignment(["cell-0", "cell-1"])

    def test_a_complete_disjoint_report_is_accepted(self):
        """The normal case: every assigned cell lands in exactly one of the two sets."""
        report = WeightUpdateReport(weight_version=1, updated_cell_ids=("cell-0",), failed_cell_ids=("cell-1",))

        report.validate_assignment(["cell-0", "cell-1"])


class TestBuildWeightUpdateReport:
    """The updater turns its health view into a report the controller can act on."""

    def test_the_failed_cells_are_taken_out_of_the_updated_set(self):
        """A cell that errored must never be counted as having received these weights."""
        report = build_weight_update_report(
            weight_version=4, assigned_cell_ids=["cell-0", "cell-1"], failed_cell_ids=["cell-1"]
        )

        assert (report.updated_cell_ids, report.failed_cell_ids) == (("cell-0",), ("cell-1",))

    def test_a_failure_outside_the_assignment_is_rejected(self):
        """A stale error from a previous connection would retire a cell this update never wrote to."""
        with pytest.raises(AssertionError, match="failed without being assigned"):
            build_weight_update_report(weight_version=4, assigned_cell_ids=["cell-0"], failed_cell_ids=["cell-9"])

    def test_an_untouched_report_publishes_no_version_and_fails_nobody(self):
        """--debug-skip-weight-update leaves every engine exactly as it was."""
        report = build_untouched_targets_report(["cell-0", "cell-1"])

        assert report.weight_version is None
        assert report.updated_cell_ids == ("cell-0", "cell-1")
        assert report.failed_cell_ids == ()


class TestMergeRankReports:
    """Every rank of a trainer cell synchronizes its failures, so their reports must be identical."""

    def test_identical_rank_reports_collapse_into_one(self):
        """The common case, where the cross-rank failure exchange already agreed."""
        report = WeightUpdateReport(weight_version=2, updated_cell_ids=("cell-0",), failed_cell_ids=())

        assert merge_rank_reports([report, report], debug_name="trainer-0") == report

    def test_ranks_that_disagree_are_rejected(self):
        """One rank silently keeping a cell healthy would let a half-written engine serve."""
        first = WeightUpdateReport(weight_version=2, updated_cell_ids=("cell-0",), failed_cell_ids=())
        second = WeightUpdateReport(weight_version=2, updated_cell_ids=(), failed_cell_ids=("cell-0",))

        with pytest.raises(AssertionError, match="disagree"):
            merge_rank_reports([first, second], debug_name="trainer-0")

    def test_an_empty_answer_is_rejected(self):
        """A trainer cell with no workers cannot have updated anything, so its silence must not read as success."""
        with pytest.raises(AssertionError, match="no report at all"):
            merge_rank_reports([], debug_name="trainer-0")


class TestCombineTrainerReports:
    """One update spans several trainers, and the caller acts on the union of their verdicts."""

    def test_the_verdicts_of_every_trainer_are_merged(self):
        """A failure only one trainer saw must still retire that engine."""
        first = WeightUpdateReport(weight_version=3, updated_cell_ids=("cell-0",), failed_cell_ids=("cell-1",))
        second = WeightUpdateReport(weight_version=3, updated_cell_ids=("cell-2",), failed_cell_ids=())

        combined = combine_trainer_reports([first, second])

        assert combined.updated_cell_ids == ("cell-0", "cell-2")
        assert combined.failed_cell_ids == ("cell-1",)
        assert combined.weight_version == 3

    def test_a_lost_trainer_does_not_hide_the_version_the_others_published(self):
        """The surviving engines really do serve the new weights, and the executor must stamp them with it."""
        lost = build_lost_trainer_report(["cell-0"])
        published = WeightUpdateReport(weight_version=3, updated_cell_ids=("cell-1",), failed_cell_ids=())

        combined = combine_trainer_reports([lost, published])

        assert combined.weight_version == 3
        assert combined.failed_cell_ids == ("cell-0",)

    def test_trainers_that_published_different_versions_are_rejected(self):
        """Half the fleet serving another version would silently mislabel every sample it produces."""
        first = WeightUpdateReport(weight_version=3, updated_cell_ids=("cell-0",), failed_cell_ids=())
        second = WeightUpdateReport(weight_version=4, updated_cell_ids=("cell-1",), failed_cell_ids=())

        with pytest.raises(AssertionError, match="different weight versions"):
            combine_trainer_reports([first, second])

    def test_an_update_where_every_trainer_was_lost_publishes_no_version(self):
        """Nothing serves the new weights, so the executor must not be told a version exists."""
        combined = combine_trainer_reports(
            [build_lost_trainer_report(["cell-0"]), build_lost_trainer_report(["cell-1"])]
        )

        assert combined.weight_version is None
        assert combined.failed_cell_ids == ("cell-0", "cell-1")
