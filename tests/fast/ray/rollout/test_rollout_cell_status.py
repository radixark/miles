from types import SimpleNamespace

from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.ft_utils.api_server.models import TriState

_ADDR_INFO = CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:13000")


def _make_cell(state, health: TriState = TriState.TRUE) -> ServerCell:
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
            workers_hash="pseudo-hash-0",
        ),
        router_api_client=SimpleNamespace(),
    )
    cell._state = state
    cell._health_checker = SimpleNamespace(status=health)
    return cell


def _conditions(status):
    return [(c.type, c.status) for c in status.conditions]


def test_a_gated_cell_is_pending_without_a_health_verdict():
    """A colocated engine waits gated for the next window; a Healthy=False here would heal-loop."""
    status = _make_cell(StateUninitialized()).cell_status()

    assert status.phase == "Pending"
    assert _conditions(status) == [("Allocated", TriState.TRUE)]


def test_a_booting_cell_is_pending_without_a_health_verdict():
    """Its port is not listening yet, so no probe result exists to report."""
    status = _make_cell(StateInitializing(addr_info=_ADDR_INFO)).cell_status()

    assert status.phase == "Pending"
    assert _conditions(status) == [("Allocated", TriState.TRUE)]


def test_a_cell_holding_stale_weights_is_already_running():
    """It answers requests with stale weights, so a crash there is a real failure."""
    status = _make_cell(StatePendingWeights(addr_info=_ADDR_INFO)).cell_status()

    assert status.phase == "Running"
    assert _conditions(status) == [("Allocated", TriState.TRUE), ("Healthy", TriState.TRUE)]


def test_a_serving_cell_reports_its_probe_verdict():
    """This is the signal the mini ft controller heals on."""
    status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.FALSE).cell_status()

    assert status.phase == "Running"
    assert _conditions(status) == [("Allocated", TriState.TRUE), ("Healthy", TriState.FALSE)]


def test_a_serving_cell_with_no_verdict_yet_is_unknown():
    """A checker that has not completed a probe must not be read as healthy."""
    status = _make_cell(StateServing(addr_info=_ADDR_INFO), health=TriState.UNKNOWN).cell_status()

    assert _conditions(status) == [("Allocated", TriState.TRUE), ("Healthy", TriState.UNKNOWN)]


def test_a_disposed_cell_is_suspended():
    """Nothing is left to probe once the cell has been torn down."""
    status = _make_cell(StateDisposed()).cell_status()

    assert status.phase == "Suspended"
    assert _conditions(status) == [("Allocated", TriState.FALSE)]
