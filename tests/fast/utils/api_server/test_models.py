import pytest
from pydantic import ValidationError

from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState


class TestCellConditionFromHealthCheckerStatus:
    def test_a_failed_probe_carries_a_reason(self):
        """The reason is what tells an operator why a cell was recycled."""
        condition = CellCondition.from_health_checker_status(TriState.FALSE)

        assert (condition.type, condition.status, condition.reason) == ("Healthy", TriState.FALSE, "HealthCheckFailed")

    def test_no_verdict_yet_is_distinguishable_from_a_failure(self):
        """A checker that has not probed yet must not be read as a dead cell."""
        condition = CellCondition.from_health_checker_status(TriState.UNKNOWN)

        assert (condition.status, condition.reason) == (TriState.UNKNOWN, "HealthCheckUnknown")

    def test_a_passing_probe_needs_no_reason(self):
        """A healthy cell has nothing to explain."""
        condition = CellCondition.from_health_checker_status(TriState.TRUE)

        assert (condition.status, condition.reason) == (TriState.TRUE, None)


class TestCellStatusSerialization:
    def test_the_worker_generation_stamp_reaches_the_response(self):
        """A client comparing two polls needs the generation a status describes, so it is part of the published document."""
        status = CellStatus(
            phase="Running",
            conditions=[CellCondition.allocated(TriState.TRUE)],
            workers_hash="pseudo-hash-0",
        )

        assert status.model_dump()["workers_hash"] == "pseudo-hash-0"


class TestCellStatusRequiresItsGeneration:
    def test_a_status_cannot_be_built_without_the_generation_it_describes(self):
        """An optional stamp would let an unstamped status be published as the current generation's verdict."""
        with pytest.raises(ValidationError):
            CellStatus(phase="Running", conditions=[CellCondition.allocated(TriState.TRUE)])

    def test_a_published_status_survives_a_round_trip_with_its_generation(self):
        """A client polling the api server compares the stamp across polls, so it must reparse into the same value."""
        status = CellStatus(
            phase="Running",
            conditions=[CellCondition.allocated(TriState.TRUE)],
            workers_hash="pseudo-hash-7",
        )

        assert CellStatus.model_validate(status.model_dump()) == status
