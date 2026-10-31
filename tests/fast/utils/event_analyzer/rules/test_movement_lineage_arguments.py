from tests.fast.utils.event_analyzer.rules.test_inference_engine_weight_movement import _version
from tests.fast.utils.event_analyzer.rules.weight_event_fakes import make_step_end, make_trainer_args

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_movement import check


class TestLineageArgumentBoundary:
    def test_future_arguments_do_not_retroactively_disable_an_existing_lineage(self) -> None:
        """An environment report after the first checksum cannot change that lineage's applicability."""
        events = [
            make_trainer_args(second=0),
            _version(1, {"w": "same"}, second=1),
            make_trainer_args(second=2, lora_rank=8),
            _version(2, {"w": "same"}, second=3),
            make_step_end(second=4),
        ]

        [issue] = check(events)

        assert (issue.weight_version_before, issue.weight_version_after) == (1, 2)

    def test_an_argument_report_at_the_first_checksum_is_effective(self) -> None:
        """The applicability lookup includes a report timestamped exactly at the lineage boundary."""
        events = [
            make_trainer_args(second=0),
            make_trainer_args(second=1, update_weights_interval=2),
            _version(1, {"w": "same"}, second=1),
            _version(2, {"w": "same"}, second=2),
            make_step_end(second=3),
        ]

        assert check(events) == []

    def test_future_reports_alone_cannot_authorize_checks_of_older_weights(self) -> None:
        """A later process configuration does not establish how an earlier publication was produced."""
        events = [
            _version(1, {"w": "same"}, second=1),
            _version(2, {"w": "same"}, second=2),
            make_trainer_args(second=3),
            make_step_end(second=4),
        ]

        assert check(events) == []
