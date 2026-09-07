import pytest

from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth


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
