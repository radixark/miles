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


def test_virtual_cells_can_be_observed_without_claiming_a_generation() -> None:
    """Non-destructive orchestration faults use virtual cells with no worker incarnation."""
    virtual = cell("script", healthy=True, cell_type="script")
    del virtual["status"]["workers_hash"]
    log = state.EventLog()

    log.observe([virtual])

    assert log.events[0].cell_infos["script"].workers_hash is None
