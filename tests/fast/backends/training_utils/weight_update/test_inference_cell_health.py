from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth

_MODULE = "miles.backends.training_utils.weight_update.inference_cell_health"


class TestErrorBookkeeping:
    """The set of cells this rank still believes it can update."""

    def test_a_fresh_assignment_declares_every_cell_healthy(self) -> None:
        """A rank that reported nothing must not silently keep every cell out of the update."""
        health = InferenceCellHealth(["cell-0", "cell-1"])

        assert health.cell_ids == ("cell-0", "cell-1")
        assert health.healthy_cell_ids == ["cell-0", "cell-1"]
        assert health.errored_cell_ids == []

    def test_an_errored_cell_leaves_the_healthy_set(self) -> None:
        """Every later stage picks its targets from the healthy set, so the failure has to land there."""
        health = InferenceCellHealth(["cell-0", "cell-1"])

        health.mark_errored("cell-0", RuntimeError("boom"))

        assert health.is_errored("cell-0") is True
        assert health.healthy_cell_ids == ["cell-1"]
        assert health.errored_cell_ids == ["cell-0"]

    def test_the_first_error_of_a_cell_survives_the_later_ones(self) -> None:
        """The first failure names the real cause; the later ones are its consequences."""
        health = InferenceCellHealth(["cell-0"])
        first = RuntimeError("the connect handshake timed out")

        health.mark_errored("cell-0", first)
        health.mark_errored("cell-0", RuntimeError("the write failed too"))

        assert health.error_of("cell-0") is first

    def test_a_cell_outside_the_assignment_cannot_be_marked(self) -> None:
        """A failure reported against an unknown id means the ids drifted, and would be lost silently."""
        health = InferenceCellHealth(["cell-0"])

        with pytest.raises(AssertionError, match="not part of this weight update"):
            health.mark_errored("cell-9", RuntimeError("boom"))

    def test_a_verdict_cannot_be_erased_from_an_assignment(self) -> None:
        """A connection's verdicts are final: the next connection is a new object, not a cleared one."""
        health = InferenceCellHealth(["cell-0"])
        health.mark_errored("cell-0", RuntimeError("boom"))

        assert not hasattr(health, "reset")
        assert InferenceCellHealth(["cell-0", "cell-1"]).errored_cell_ids == []
        assert health.errored_cell_ids == ["cell-0"]

    def test_a_repeated_cell_id_in_an_assignment_is_rejected(self) -> None:
        """Two engines answering to one id would let one of them hide the other's failure."""
        with pytest.raises(AssertionError, match="must appear once"):
            InferenceCellHealth(["cell-0", "cell-0"])


def _patched_dist(reports: list[list[str]]):
    def all_gather_object(gathered, obj, group=None):
        for index, report in enumerate(reports):
            gathered[index] = list(report)

    dist_mock = MagicMock()
    dist_mock.get_world_size.return_value = len(reports)
    dist_mock.all_gather_object.side_effect = all_gather_object
    return patch(f"{_MODULE}.dist", dist_mock)


class TestCrossRankAggregation:
    """One rank failing to write a cell makes that cell errored on every rank of the trainer cell."""

    def test_a_cell_only_another_rank_lost_becomes_errored_here_too(self) -> None:
        """This rank would happily resume a cell that is missing another rank's shard of the model."""
        health = InferenceCellHealth(["cell-0", "cell-1"])

        with _patched_dist([[], ["cell-0"]]):
            health.synchronize(group=object())

        assert health.errored_cell_ids == ["cell-0"]
        assert "trainer rank 1" in str(health.error_of("cell-0"))

    def test_a_cell_this_rank_lost_keeps_its_own_error(self) -> None:
        """The local exception is the real evidence; the aggregate would replace it with a rank number."""
        health = InferenceCellHealth(["cell-0"])
        local = ConnectionError("cell-0 is unreachable")
        health.mark_errored("cell-0", local)

        with _patched_dist([["cell-0"], ["cell-0"]]):
            health.synchronize(group=object())

        assert health.error_of("cell-0") is local

    def test_a_rank_that_lost_nothing_reports_an_empty_set(self) -> None:
        """Every rank has to enter the collective, including the ones with nothing to report."""
        health = InferenceCellHealth(["cell-0"])
        reported: list[list[str]] = []

        def all_gather_object(gathered, obj, group=None):
            reported.append(list(obj))
            gathered[0] = list(obj)
            gathered[1] = []

        with patch(f"{_MODULE}.dist") as dist_mock:
            dist_mock.get_world_size.return_value = 2
            dist_mock.all_gather_object.side_effect = all_gather_object
            health.synchronize(group=object())

        assert reported == [[]]
        assert health.errored_cell_ids == []

    def test_a_rank_without_any_target_of_a_cell_still_adopts_the_verdict(self) -> None:
        """A rank that sends nothing to a cell still has to skip it when it resumes the engines."""
        health = InferenceCellHealth(["cell-0", "cell-1"])

        with _patched_dist([[], ["cell-0", "cell-1"]]):
            health.synchronize(group=object())

        assert health.healthy_cell_ids == []
