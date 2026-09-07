from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StateErrored, StateServing
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ADDR_INFO = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url="http://10.0.0.1:13000",
)


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        return dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            gate=HostAndPort(host="10.0.0.1", port=13000),
        )


class _RecordingRouterApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.remove_worker_error: Exception | None = None

    async def add_worker(self, **kwargs) -> None:
        self.calls.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs) -> None:
        self.calls.append(("remove_worker", kwargs))
        if self.remove_worker_error is not None:
            raise self.remove_worker_error


class _RecordingApiClient:
    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        self.server_url = server_url
        self.api_key = api_key

    async def release_memory_occupation(self, tags=None) -> None:
        return None

    async def resume_memory_occupation(self, tags=None) -> None:
        return None

    async def check_weights(self, **kwargs) -> None:
        return None


class _RecordingHealthChecker:
    def __init__(self) -> None:
        self.stop_count = 0

    @property
    def status(self) -> TriState:
        return TriState.TRUE

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_count += 1


def _make_meta(**overrides) -> ServerCellMetadata:
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": "inference-engine-0-0-0",
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": "inference-engine-0-0-0-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


@pytest.fixture
def cell_env(monkeypatch):
    """Stub out everything the cell dials, so only its own state machine is under test."""
    health: dict[str, bool] = {"ready": True}

    async def _activate(gate_url: str) -> None:
        return None

    async def _compute_addr_info(self) -> CellAddrInfo:
        return _ADDR_INFO

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        return health["ready"]

    monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
    monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)

    return dict(health=health)


def _make_cell(*, router: _RecordingRouterApiClient | None = None, **meta_overrides) -> ServerCell:
    cell = track_server_cell(
        ServerCell(
            args=make_args(),
            meta=_make_meta(**meta_overrides),
            router_api_client=router or _RecordingRouterApiClient(),
            provider=_StubProvider(),
        )
    )
    cell._health_checker = _RecordingHealthChecker()
    return cell


async def _make_serving_cell(router: _RecordingRouterApiClient) -> ServerCell:
    cell = _make_cell(router=router)
    await cell.init()
    await cell.tick()
    await cell.mark_weights_ready()
    return cell


async def _make_pending_weights_cell(router: _RecordingRouterApiClient) -> ServerCell:
    cell = _make_cell(router=router)
    await cell.init()
    await cell.tick()
    return cell


class TestMarkErrored:
    async def test_a_serving_cell_is_withdrawn_from_the_router(self, cell_env):
        """A cell whose weight update failed must stop taking traffic it would answer with stale weights."""
        router = _RecordingRouterApiClient()
        cell = await _make_serving_cell(router)

        await cell.mark_errored()

        assert cell.is_errored
        assert [name for name, _kwargs in router.calls] == ["add_worker", "remove_worker"]

    async def test_a_serving_cell_is_no_longer_serving_before_the_router_call_is_awaited(self, cell_env):
        """Anything reading the cell while the unregister is in flight must not treat it as a live engine."""
        router = _RecordingRouterApiClient()
        cell = await _make_serving_cell(router)
        seen: list[tuple[bool, bool]] = []

        async def _remove_worker(**kwargs) -> None:
            seen.append((cell.is_serving, cell.is_errored))

        router.remove_worker = _remove_worker

        await cell.mark_errored()

        assert seen == [(False, True)]

    async def test_a_cell_awaiting_weights_is_never_registered(self, cell_env):
        """It never reached the router, so erroring it must not publish an engine holding stale weights."""
        router = _RecordingRouterApiClient()
        cell = await _make_pending_weights_cell(router)

        await cell.mark_errored()

        assert cell.is_errored
        assert router.calls == []

    async def test_a_booting_cell_can_be_marked_errored(self, cell_env):
        """A cell whose engine died during startup is errored the same way, with no router call to undo."""
        cell_env["health"]["ready"] = False
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()

        await cell.mark_errored()

        assert cell.is_errored
        assert router.calls == []

    async def test_marking_errored_twice_unregisters_only_once(self, cell_env):
        """Several reporters may name the same failed cell, and a second removal races a replacement."""
        router = _RecordingRouterApiClient()
        cell = await _make_serving_cell(router)

        await cell.mark_errored()
        await cell.mark_errored()

        assert cell.is_errored
        assert [name for name, _kwargs in router.calls].count("remove_worker") == 1

    async def test_a_disposed_cell_is_not_revived(self, cell_env):
        """Reconcile may have torn the cell down before the failure report arrives."""
        router = _RecordingRouterApiClient()
        cell = await _make_serving_cell(router)
        await cell.dispose()

        await cell.mark_errored()

        assert isinstance(cell._state, StateDisposed)
        assert not cell.is_errored

    async def test_an_uninitialized_cell_cannot_be_marked_errored(self, cell_env):
        """It has no address and was never a weight-update target, so a report about it is a bug."""
        cell = _make_cell()

        with pytest.raises(ValueError):
            await cell.mark_errored()

    async def test_the_health_checker_is_stopped(self, cell_env):
        """Probing an engine that is being torn down only produces noise and heal loops."""
        cell = await _make_serving_cell(_RecordingRouterApiClient())

        await cell.mark_errored()

        assert cell._health_checker.stop_count == 1

    async def test_a_router_that_rejects_the_removal_still_leaves_the_cell_errored(self, cell_env):
        """The cell must not fall back to serving just because the router could not be reached."""
        router = _RecordingRouterApiClient()
        router.remove_worker_error = RuntimeError("router rejected the removal")
        cell = await _make_serving_cell(router)

        await cell.mark_errored()

        assert cell.is_errored

    async def test_the_address_its_cleanup_needs_is_kept(self, cell_env):
        """Killing the engine later needs the address of the very incarnation that failed."""
        cell = await _make_serving_cell(_RecordingRouterApiClient())

        await cell.mark_errored()

        assert cell._state == StateErrored(addr_info=_ADDR_INFO)
        assert cell.addr_info == _ADDR_INFO


class TestAnErroredCellIsInert:
    async def test_the_sweep_does_not_move_it_back_to_serving(self, cell_env):
        """A cell erroring while it was still booting would otherwise be published by the next tick."""
        cell_env["health"]["ready"] = False
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()
        await cell.mark_errored()

        cell_env["health"]["ready"] = True
        await cell.tick()

        assert cell.is_errored
        assert router.calls == []

    async def test_the_sweep_does_not_ask_it_for_its_env(self, cell_env, monkeypatch):
        """Its engine is being reclaimed, so the report would only time out every tick."""
        cell = await _make_serving_cell(_RecordingRouterApiClient())
        asked: list[str] = []

        async def _report_if_due(*, cell_id: str, server_url: str, api_client) -> None:
            asked.append(cell_id)

        monkeypatch.setattr(cell._env_reporter, "report_if_due", _report_if_due)
        await cell.mark_errored()

        await cell.tick()

        assert asked == []

    async def test_it_is_not_counted_as_pending_weights_or_serving(self, cell_env):
        """Every fan-out in the rollout server keys off this, so an errored cell must fall out of all of them."""
        cell = await _make_serving_cell(_RecordingRouterApiClient())

        await cell.mark_errored()

        assert not cell.is_pending_weights_or_serving
        assert not cell.is_pending_weights
        assert not cell.is_serving

    async def test_it_cannot_be_marked_weights_ready(self, cell_env):
        """The failed update must never publish the target it failed to reach."""
        cell = await _make_pending_weights_cell(_RecordingRouterApiClient())
        await cell.mark_errored()

        with pytest.raises(AssertionError):
            await cell.mark_weights_ready()

        assert cell.is_errored

    async def test_its_health_checker_stays_inactive(self, cell_env):
        """An errored cell answers no probe, so the checker must not publish verdicts about it."""
        cell = _make_cell()
        await cell.init()
        await cell.tick()
        await cell.mark_errored()

        assert not cell._get_health_checker_active_and_epoch().active


class TestDisposeAfterErrored:
    async def test_an_errored_cell_is_disposed_without_a_second_unregister(self, cell_env):
        """The router already dropped it, and dialling it again would only delay the teardown."""
        router = _RecordingRouterApiClient()
        cell = await _make_serving_cell(router)
        await cell.mark_errored()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)
        assert [name for name, _kwargs in router.calls].count("remove_worker") == 1

    async def test_a_cell_errored_before_it_ever_served_is_disposed_without_touching_the_router(self, cell_env):
        """It was never registered, so removing it would name a worker the router does not know."""
        router = _RecordingRouterApiClient()
        cell = await _make_pending_weights_cell(router)
        await cell.mark_errored()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)
        assert router.calls == []

    async def test_erroring_a_disposed_cell_keeps_the_destructor_contract(self, cell_env):
        """The destructor asserts on the disposed state, so a late report may not move the cell out of it."""
        cell = await _make_serving_cell(_RecordingRouterApiClient())
        await cell.dispose()

        await cell.mark_errored()
        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)


class TestErroredCellIsDistinctFromServing:
    async def test_a_healthy_cell_keeps_serving_and_stays_registered(self, cell_env):
        """Erroring one cell must not disturb the cells that did receive their weights."""
        router = _RecordingRouterApiClient()
        healthy = await _make_serving_cell(router)
        failed = await _make_serving_cell(router)

        await failed.mark_errored()

        assert isinstance(healthy._state, StateServing)
        assert healthy.is_serving
        assert [kwargs for name, kwargs in router.calls if name == "remove_worker"] == [
            dict(worker_url="http://10.0.0.1:30000", use_legacy_api=False)
        ]
