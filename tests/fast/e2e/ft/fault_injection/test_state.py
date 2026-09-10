from tests.e2e.ft.conftest_ft.fault_injection import state
from tests.fast.e2e.ft.fault_injection.utils import RUNNING_NOT_SERVING, SERVING, cell, staged


def test_cell_is_alive_true_only_when_healthy_condition_is_true() -> None:
    """cell_is_alive reflects the Healthy condition status."""
    assert state.cell_is_alive(cell("c", healthy=True))
    assert not state.cell_is_alive(cell("c", healthy=False))


def test_cell_is_alive_false_when_no_healthy_condition_present() -> None:
    """A cell with no Healthy condition is not considered alive."""
    assert not state.cell_is_alive({"metadata": {"name": "c"}, "status": {"conditions": []}})


def test_a_running_cell_that_is_not_in_the_router_is_not_serving() -> None:
    """The api server renders PendingWeights and Serving alike, so the Serving condition must split them."""
    assert state.compute_observed_cell_state(staged("c", RUNNING_NOT_SERVING)) is RUNNING_NOT_SERVING
    assert state.compute_observed_cell_state(staged("c", SERVING)) is SERVING


class TestObservedGeneration:
    def test_the_api_cell_schema_carries_generation_in_status(self) -> None:
        """The injector reads the real API generation field rather than a test-only metadata label."""
        from miles.utils.ft_utils.api_server.models import Cell, CellMetadata, CellSpec, CellStatus

        observed = Cell(
            metadata=CellMetadata(name="actor-0", labels={"miles.io/cell-type": "actor"}),
            spec=CellSpec(),
            status=CellStatus(phase="Running", conditions=[], workers_hash="replacement"),
        )
        log = state.EventLog()

        log.observe([observed.model_dump(mode="json")])

        assert log.events[0].cell_infos["actor-0"].workers_hash == "replacement"
