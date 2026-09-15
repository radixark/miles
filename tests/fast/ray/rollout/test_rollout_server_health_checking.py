from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, NoopHealthChecker, SimpleHealthChecker
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ADDRS: NamedHostAndPorts = {
    "primary": HostAndPort(host="10.0.0.1", port=30000),
    GATE_PORT_NAME: HostAndPort(host="10.0.0.1", port=13000),
}


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return _ADDRS


def _make_meta(cell_id: str = "cell-0", **overrides) -> ServerCellMetadata:
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": cell_id,
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": f"{cell_id}-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


def _make_server(*, ft_components=("rollout",), **overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=make_args(colocate=True, ft_components=list(ft_components)),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


def _make_cell(*, global_activeness=None, router=None, **meta_overrides) -> ServerCell:
    return track_server_cell(
        ServerCell(
            args=make_args(ft_components=["rollout"]),
            meta=_make_meta(**meta_overrides),
            router_api_client=router or SimpleNamespace(),
            provider=_StubProvider(),
            global_health_checker_activeness=global_activeness or (lambda: ActiveAndEpoch(active=True, epoch=0)),
        )
    )


def _stub_network(monkeypatch, *, ready: bool = True) -> None:
    async def _activate(gate_url: str) -> None:
        pass

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        return ready

    monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)


async def _noop_add_worker(**kwargs) -> None:
    pass


class TestHealthCheckerActiveness:
    async def test_a_gated_cell_is_not_probed(self):
        """A colocated cell waits gated for the next window; its port is not even listening."""
        cell = _make_cell()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_booting_cell_is_not_probed(self, monkeypatch):
        """Its engine has not answered yet, so a probe would count a false failure."""
        _stub_network(monkeypatch, ready=False)
        cell = _make_cell()

        await cell.init()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_cell_holding_stale_weights_is_probed(self, monkeypatch):
        """It answers requests with stale weights, so a crash there is a real failure."""
        _stub_network(monkeypatch)
        cell = _make_cell()

        await cell.init()
        await cell.tick()

        assert cell._get_health_checker_active_and_epoch().active

    async def test_a_cell_that_skips_pending_weights_is_probed(self, monkeypatch):
        """A frozen model goes straight to serving, and an unwatched serving cell is a hole in FT."""
        _stub_network(monkeypatch)
        cell = _make_cell(router=SimpleNamespace(add_worker=_noop_add_worker), update_weights=False)

        await cell.init()
        await cell.tick()

        assert cell._get_health_checker_active_and_epoch().active

    async def test_a_disposed_cell_is_not_probed(self, monkeypatch):
        """Nothing is left to answer once the cell has been torn down."""
        _stub_network(monkeypatch)
        cell = _make_cell()

        await cell.init()
        await cell.tick()
        await cell.dispose()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_forced_pause_wins_over_the_cell_state(self, monkeypatch):
        """Engines are unusable while offloaded or mid weight update, whatever state they are in."""
        _stub_network(monkeypatch)
        active = {"value": True}
        cell = _make_cell(global_activeness=lambda: ActiveAndEpoch(active=active["value"], epoch=0))

        await cell.init()
        await cell.tick()
        assert cell._get_health_checker_active_and_epoch().active

        active["value"] = False

        assert not cell._get_health_checker_active_and_epoch().active


class TestAddCellHealthChecker:
    async def test_a_new_cell_starts_its_checker(self):
        """Activeness is pulled per loop, so starting unconditionally is safe and needs no replay."""
        srv = _make_server()

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            checker = srv.server_cells["cell-0"]._health_checker
            assert isinstance(checker, SimpleHealthChecker)
            assert checker._task is not None
            await srv.dispose()

    async def test_a_cell_added_mid_window_does_not_probe(self):
        """The window releases the lock, so reconcile can add a cell while probing is paused."""
        srv = _make_server(global_health_checker_activeness=lambda: ActiveAndEpoch(active=False, epoch=0))

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            assert not srv.server_cells["cell-0"]._get_health_checker_active_and_epoch().active
            await srv.dispose()

    async def test_no_checker_is_created_without_rollout_fault_tolerance(self):
        """Without rollout FT nothing consumes the health status, so nothing probes."""
        srv = _make_server(ft_components=())

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            assert isinstance(srv.server_cells["cell-0"]._health_checker, NoopHealthChecker)
            await srv.dispose()


class TestDispose:
    async def test_dispose_stops_the_checker_task(self):
        """An inactive checker still polls its predicate forever unless the task is cancelled."""
        cell = _make_cell()
        assert cell._health_checker._task is not None

        await cell.dispose()

        assert cell._health_checker._task is None
