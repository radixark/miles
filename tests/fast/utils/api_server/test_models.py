from miles.utils.ft_utils.api_server.models import CellCondition, TriState


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
