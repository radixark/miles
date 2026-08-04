from types import SimpleNamespace

import pytest

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    CellState,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata, compute_pending_rollout_cell_status
from miles.utils.ft_utils.api_server.models import CellStatus, TriState

_ADDR_INFO = CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:13000")


def _make_cell(state: CellState, health: TriState = TriState.TRUE, workers_hash: str = "pseudo-hash-0") -> ServerCell:
    cell = ServerCell(
        args=make_args(),
        meta=ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id="inference-engine-0-0-0",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            worker_name="inference-engine-0-0-0-0",
            needs_offload=False,
            update_weights=True,
            workers_hash=workers_hash,
        ),
        router_api_client=SimpleNamespace(),
    )
    cell._state = state
    cell._health_checker = SimpleNamespace(status=health)
    return cell


def _conditions(status: CellStatus) -> list[tuple[str, TriState]]:
    return [(c.type, c.status) for c in status.conditions]


class TestServerCellStatus:
    def test_a_gated_cell_is_pending_without_a_health_verdict(self):
        """A colocated engine waits gated for the next window; a Healthy=False here would heal-loop."""
        status = _make_cell(StateUninitialized()).cell_status()

        assert status.phase == "Pending"
        assert _conditions(status) == [("Allocated", TriState.TRUE)]

    def test_a_booting_cell_is_pending_without_a_health_verdict(self):
        """Its port is not listening yet, so no probe result exists to report."""
        status = _make_cell(StateInitializing(addr_info=_ADDR_INFO)).cell_status()

        assert status.phase == "Pending"
        assert _conditions(status) == [("Allocated", TriState.TRUE)]

    def test_a_cell_holding_stale_weights_is_already_running(self):
        """It answers requests with stale weights, so a crash there is a real failure."""
        status = _make_cell(StatePendingWeights(addr_info=_ADDR_INFO)).cell_status()

        assert status.phase == "Running"
        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
            ("Serving", TriState.FALSE),
        ]

    def test_a_cell_holding_stale_weights_is_not_reported_as_serving(self):
        """Running alone cannot witness a recovery: an engine never re-registered in the router reads Running too."""
        status = _make_cell(StatePendingWeights(addr_info=_ADDR_INFO)).cell_status()

        assert ("Serving", TriState.FALSE) in _conditions(status)

    def test_a_serving_cell_reports_its_probe_verdict(self):
        """This is the signal the mini ft controller heals on."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.FALSE).cell_status()

        assert status.phase == "Running"
        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
            ("Serving", TriState.TRUE),
        ]

    def test_a_cell_registered_in_the_router_is_reported_as_serving(self):
        """This is the only externally visible proof that a replaced engine really came back."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO)).cell_status()

        assert ("Serving", TriState.TRUE) in _conditions(status)

    def test_a_serving_cell_with_no_verdict_yet_is_unknown(self):
        """A checker that has not completed a probe must not be read as healthy."""
        status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.UNKNOWN).cell_status()

        assert _conditions(status) == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.UNKNOWN),
            ("Serving", TriState.TRUE),
        ]

    def test_a_disposed_cell_is_suspended(self):
        """Nothing is left to probe once the cell has been torn down."""
        status = _make_cell(StateDisposed()).cell_status()

        assert status.phase == "Suspended"
        assert _conditions(status) == [("Allocated", TriState.FALSE)]


class TestServerCellStatusGeneration:
    @pytest.mark.parametrize(
        "state",
        [
            StateUninitialized(),
            StateInitializing(addr_info=_ADDR_INFO, start_time=time.monotonic()),
            StatePendingWeights(addr_info=_ADDR_INFO),
            StateServing(addr_info=_ADDR_INFO),
            StateDisposed(),
        ],
    )
    def test_a_status_is_stamped_with_the_generation_of_the_cell_that_computed_it(self, state: CellState):
        """No state may publish a verdict without saying which process it is about, whatever phase it reports."""
        status = _make_cell(state, workers_hash="hash-7").cell_status()

        assert status.workers_hash == "hash-7"

    def test_a_cell_stuck_booting_past_its_deadline_still_names_its_generation(self):
        """The deadline branch builds its own condition list, so it must not drop the stamp with it."""
        started_long_ago = time.monotonic() - INITIALIZING_TIMEOUT_SECONDS - 1.0
        cell = _make_cell(StateInitializing(addr_info=_ADDR_INFO, start_time=started_long_ago), workers_hash="hash-7")

        assert cell.cell_status().workers_hash == "hash-7"

    def test_two_cells_of_different_generations_report_different_stamps(self):
        """A stamp shared by every cell could not distinguish a replaced engine from its predecessor."""
        old = _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-1").cell_status()
        new = _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-2").cell_status()

        assert (old.workers_hash, new.workers_hash) == ("hash-1", "hash-2")


class TestComputePendingRolloutCellStatus:
    @pytest.mark.parametrize("past_startup_deadline", [False, True])
    def test_a_pending_status_carries_the_generation_it_was_asked_about(self, past_startup_deadline: bool):
        """The api server builds this itself for a cell the controller has not observed yet, from the live listing."""
        status = compute_pending_rollout_cell_status(
            workers_hash="hash-7", past_startup_deadline=past_startup_deadline
        )

        assert status.workers_hash == "hash-7"


class TestGetCellStatuses:
    def _controller(self, servers: dict[str, SimpleNamespace]) -> SimpleNamespace:
        return SimpleNamespace(servers=servers)

    def test_every_cell_of_every_server_is_reported(self):
        """The api server renders one row per cell, so a missing model means missing rows."""
        cell_a = _make_cell(StateServing(addr_info=_ADDR_INFO))
        cell_b = _make_cell(StateUninitialized())
        controller = self._controller(
            {
                "actor": SimpleNamespace(server_cells={"engine-0": cell_a}),
                "critic": SimpleNamespace(server_cells={"engine-1": cell_b}),
            }
        )

        statuses = InferenceController.get_cell_statuses(controller)

        assert {cell_id: status.phase for cell_id, status in statuses.items()} == {
            "engine-0": "Running",
            "engine-1": "Pending",
        }

    def test_each_status_carries_the_generation_it_describes(self):
        """A status without its generation is indistinguishable from one about the process it replaced."""
        controller = self._controller(
            {
                "actor": SimpleNamespace(
                    server_cells={"engine-0": _make_cell(StateServing(addr_info=_ADDR_INFO), workers_hash="hash-7")}
                )
            }
        )

        statuses = InferenceController.get_cell_statuses(controller)

        assert statuses["engine-0"].workers_hash == "hash-7"

    def test_a_controller_without_servers_reports_nothing(self):
        """debug-train-only runs have no rollout cells, and must not fabricate any."""
        assert InferenceController.get_cell_statuses(self._controller({})) == {}

    def test_each_pending_cell_gets_its_own_status_object(self):
        """A shared status instance would let one cell's mutation rewrite every other pending cell."""
        controller = self._controller(
            {
                "actor": SimpleNamespace(
                    server_cells={
                        "engine-0": _make_cell(StateUninitialized()),
                        "engine-1": _make_cell(StateUninitialized()),
                    }
                )
            }
        )

        statuses = InferenceController.get_cell_statuses(controller)

        assert statuses["engine-0"] is not statuses["engine-1"]
