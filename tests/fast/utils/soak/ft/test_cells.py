import pytest
from tests.fast.utils.soak.soak_fakes import _cell
from tests.utils.soak.ft.cells import cell_is_alive, cell_is_ready, cell_type_of

from miles.utils.ft_utils.api_server.models import TriState


class TestCellIsAlive:
    @pytest.mark.parametrize(
        ("healthy", "alive"), [(TriState.TRUE, True), (TriState.FALSE, False), (TriState.UNKNOWN, False)]
    )
    def test_only_a_true_healthy_condition_is_alive(self, healthy: TriState, alive: bool) -> None:
        """An unknown health reading must not count as a live cell."""
        assert cell_is_alive(_cell("actor-0", cell_type="actor", healthy=healthy)) is alive


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

    def test_a_running_but_unhealthy_cell_is_not_ready(self) -> None:
        """Readiness requires liveness first."""
        assert not cell_is_ready(_cell("rollout-0", cell_type="rollout", healthy=TriState.FALSE))


class TestCellTypeOf:
    def test_the_type_comes_from_the_cell_type_label(self) -> None:
        """Observation filters cells by the chart's cell-type label."""
        assert cell_type_of(_cell("x-0", cell_type="rollout")) == "rollout"
