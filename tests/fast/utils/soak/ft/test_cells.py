import pytest
from tests.fast.utils.soak.soak_fakes import _cell
from tests.utils.soak.ft.cells import cell_is_alive, cell_is_ready, cell_type_of

from miles.utils.ft_utils.api_server.models import TriState


class TestCellIsAlive:
    @pytest.mark.parametrize(
        ("healthy", "alive"), [(TriState.TRUE, True), (TriState.FALSE, False), (TriState.UNKNOWN, True)]
    )
    def test_only_a_failed_health_check_makes_a_running_cell_dead(self, healthy: TriState, alive: bool) -> None:
        """A health checker paused for a weight update reads unknown, which is not a failure."""
        assert cell_is_alive(_cell("actor-0", cell_type="actor", healthy=healthy)) is alive

    @pytest.mark.parametrize("phase", ["Pending", "Suspended"])
    def test_a_cell_that_is_not_running_is_not_alive(self, phase: str) -> None:
        """A cell being healed or not yet started is not alive whatever its last health reading."""
        assert not cell_is_alive(_cell("actor-0", cell_type="actor", phase=phase, healthy=TriState.UNKNOWN))


class TestCellIsReady:
    def test_a_running_healthy_trainer_is_ready_without_serving(self) -> None:
        """Trainer cells never serve, so Serving is not part of their readiness."""
        assert cell_is_ready(_cell("actor-0", cell_type="actor", serving=TriState.FALSE))

    @pytest.mark.parametrize(
        ("serving", "ready"), [(TriState.TRUE, True), (TriState.FALSE, False), (TriState.UNKNOWN, False)]
    )
    def test_a_rollout_cell_is_ready_only_while_serving(self, serving: TriState, ready: bool) -> None:
        """A healthy engine that is not yet serving has not recovered."""
        assert cell_is_ready(_cell("rollout-0", cell_type="rollout", serving=serving)) is ready

    @pytest.mark.parametrize("phase", ["Pending", "Suspended"])
    def test_a_cell_that_is_not_running_is_not_ready(self, phase: str) -> None:
        """Pending or suspended cells are not ready even when healthy and serving."""
        assert not cell_is_ready(_cell("rollout-0", cell_type="rollout", phase=phase))

    @pytest.mark.parametrize("healthy", [TriState.FALSE, TriState.UNKNOWN])
    def test_a_running_cell_without_a_passing_health_check_is_not_ready(self, healthy: TriState) -> None:
        """Readiness needs a passing health check, which a paused or failed checker cannot give."""
        assert not cell_is_ready(_cell("rollout-0", cell_type="rollout", healthy=healthy))


class TestCellTypeOf:
    def test_the_type_comes_from_the_cell_type_label(self) -> None:
        """Observation filters cells by the chart's cell-type label."""
        assert cell_type_of(_cell("x-0", cell_type="rollout")) == "rollout"
