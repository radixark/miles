from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["eval"])

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from tests.fast.fixtures.args_fixtures import sglang_router_args
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import eval_pool as eval_pool_module
from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StateServing
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.rollout.eval_fleet import wait_router_generate_ready
from miles.ray.rollout.eval_pool import EvalPoolCarveOut, InferenceControllerEvalPool
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.workers.worker_spec import HostAndPort

EVAL_ROUTER_URL = "http://10.0.0.9:31000"


class _RecordingRegistry:
    def __init__(self, log: list, router_url: str, *, fail_register: bool = False):
        self._log = log
        self._router_url = router_url
        self._fail_register = fail_register

    async def register(self, *, worker_url, worker_type, bootstrap_port):
        if self._fail_register:
            raise RuntimeError("eval router unreachable")
        self._log.append(("register", self._router_url, worker_url))

    async def unregister(self, *, worker_url):
        self._log.append(("unregister", self._router_url, worker_url))


@pytest.fixture
def router_log(monkeypatch: pytest.MonkeyPatch) -> list:
    log: list = []
    monkeypatch.setattr(server_cell_module, "_make_registry", lambda url: _RecordingRegistry(log, url))
    return log


def _training_router_client(log: list) -> MagicMock:
    client = MagicMock()

    async def add_worker(*, worker_url, worker_type, bootstrap_port=None):
        log.append(("register", "training", worker_url))

    async def remove_worker(*, worker_url):
        log.append(("unregister", "training", worker_url))

    client.add_worker = AsyncMock(side_effect=add_worker)
    client.remove_worker = AsyncMock(side_effect=remove_worker)
    return client


class _StubProvider:
    async def get_addrs(self, worker_name: str):
        raise AssertionError("these tests never address a cell through the provider")


@pytest.fixture
def make_serving_cell(router_log):
    def _make(server_url: str = "http://10.0.0.2:30000") -> ServerCell:
        cell = ServerCell(
            args=make_args(num_gpus_per_node=8),
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
                workers_hash="hash-0",
            ),
            router_api_client=_training_router_client(router_log),
            provider=_StubProvider(),
        )
        cell._state = StateServing(addr_info=CellAddrInfo(server_url=server_url, bootstrap_port=None, gate_url=None))
        return cell

    # every test disposes its cell, as ServerCell.__del__ requires
    return _make


class TestServerCellEvalPin:
    @pytest.mark.asyncio
    async def test_pinning_moves_the_cell_from_the_training_router_to_the_eval_router(
        self, make_serving_cell, router_log
    ):
        """A pinned engine still on the training router would take training turns at the frozen version."""
        cell = make_serving_cell()

        await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)

        assert router_log == [
            ("unregister", "training", "http://10.0.0.2:30000"),
            ("register", EVAL_ROUTER_URL, "http://10.0.0.2:30000"),
        ]
        assert cell.is_eval_pinned and not cell.is_serving
        assert cell.server_url == "http://10.0.0.2:30000"
        await cell.dispose()

    @pytest.mark.asyncio
    async def test_a_failed_eval_registration_leaves_the_cell_pending_weights(
        self, make_serving_cell, router_log, monkeypatch
    ):
        """Out of both routers, it must come back through the next weight update rather than stay lost."""
        cell = make_serving_cell()
        monkeypatch.setattr(
            server_cell_module, "_make_registry", lambda url: _RecordingRegistry(router_log, url, fail_register=True)
        )

        with pytest.raises(RuntimeError, match="eval router unreachable"):
            await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)

        assert cell.is_pending_weights
        assert router_log == [("unregister", "training", "http://10.0.0.2:30000")]
        await cell.dispose()

    @pytest.mark.asyncio
    async def test_unpinning_rejoins_training_only_after_the_weights_are_ready(self, make_serving_cell, router_log):
        """Back on the training router at the stale version, it would generate training samples with old weights."""
        cell = make_serving_cell()
        await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)
        router_log.clear()

        await cell.unpin_from_eval()
        assert cell.is_pending_weights
        assert router_log == [("unregister", EVAL_ROUTER_URL, "http://10.0.0.2:30000")]

        await cell.mark_weights_ready()
        assert cell.is_serving
        assert router_log[-1] == ("register", "training", "http://10.0.0.2:30000")
        await cell.dispose()

    @pytest.mark.asyncio
    async def test_an_errored_pinned_cell_leaves_the_eval_router(self, make_serving_cell, router_log):
        """The eval router would keep dialing a dead engine and fail the rest of the eval."""
        cell = make_serving_cell()
        await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)
        router_log.clear()

        await cell.mark_errored()

        assert cell.is_errored
        assert router_log == [("unregister", EVAL_ROUTER_URL, "http://10.0.0.2:30000")]
        await cell.dispose()

    @pytest.mark.asyncio
    async def test_a_pinned_cell_reports_itself_allocated_and_serving(self, make_serving_cell):
        """Fault tolerance reads this status; an unmatched state would raise in the status endpoint."""
        cell = make_serving_cell()
        await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)

        status = cell.cell_status()

        assert status.phase == "Running"
        conditions = {condition.type: condition.status for condition in status.conditions}
        assert conditions["Allocated"] == TriState.TRUE
        assert conditions["Serving"] == TriState.TRUE
        await cell.dispose()


def _fake_cell(*, cell_id: str, gpu_offset: int, version: str = "7", serving: bool = True) -> SimpleNamespace:
    cell = SimpleNamespace(
        meta=SimpleNamespace(cell_id=cell_id, gpu_offset=gpu_offset, num_gpus_per_engine=1),
        api_client=SimpleNamespace(get_weight_version=AsyncMock(return_value=version)),
        is_serving=serving,
        is_errored=False,
        is_eval_pinned=False,
        pin_calls=0,
    )

    async def pin_for_eval(*, eval_router_url):
        cell.pin_calls += 1
        cell.is_serving = False
        cell.is_eval_pinned = True

    async def unpin_from_eval():
        cell.is_eval_pinned = False

    cell.pin_for_eval = pin_for_eval
    cell.unpin_from_eval = unpin_from_eval
    return cell


class _FakeServer:
    def __init__(self, cells: list[SimpleNamespace]):
        self.all_server_cells = {cell.meta.cell_id: cell for cell in cells}
        self.context_lock = ContextLock("test")

    @property
    def eval_pinned_cells(self) -> dict:
        return {cell_id: cell for cell_id, cell in self.all_server_cells.items() if cell.is_eval_pinned}


@pytest.fixture
def router_always_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_pool_module, "wait_router_generate_ready", AsyncMock())


def _pool(
    cells: list[SimpleNamespace], num_engines: int = 2, router_policy: str | None = None
) -> tuple[InferenceControllerEvalPool, _FakeServer]:
    srv = _FakeServer(cells)
    args = SimpleNamespace(
        eval_pool_num_engines=num_engines,
        rollout_num_gpus_per_engine=1,
        sglang=SimpleNamespace(models=[SimpleNamespace(has_pd_disaggregation=False)]),
        **sglang_router_args(router_policy),
    )
    return InferenceControllerEvalPool(args, srv=srv, router=HostAndPort(host="10.0.0.9", port=31000)), srv


@pytest.mark.usefixtures("router_always_ready")
class TestCarveOut:
    @pytest.mark.asyncio
    async def test_the_last_serving_engines_are_pinned_at_their_shared_version(self):
        cells = [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(4)]
        pool, srv = _pool(cells)

        carve_out = await pool.carve_out()

        assert carve_out == EvalPoolCarveOut(weight_version="7", skip_reason=None)
        assert sorted(srv.eval_pinned_cells) == ["engine-2", "engine-3"]

    @pytest.mark.asyncio
    async def test_training_always_keeps_an_engine(self):
        cells = [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(2)]
        pool, srv = _pool(cells, num_engines=2)

        carve_out = await pool.carve_out()

        assert carve_out.skip_reason == "insufficient_engines"
        assert srv.eval_pinned_cells == {}

    @pytest.mark.asyncio
    async def test_engines_at_different_versions_are_never_pinned(self):
        """An eval spread over two versions measures neither; nothing may move before the check."""
        cells = [
            _fake_cell(cell_id="engine-0", gpu_offset=0),
            _fake_cell(cell_id="engine-1", gpu_offset=1, version="7"),
            _fake_cell(cell_id="engine-2", gpu_offset=2, version="6"),
        ]
        pool, srv = _pool(cells)

        carve_out = await pool.carve_out()

        assert carve_out.skip_reason == "pin_violation"
        assert all(cell.pin_calls == 0 for cell in cells)

    @pytest.mark.asyncio
    async def test_a_failed_pin_gives_back_the_engines_already_pinned(self):
        cells = [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(4)]

        async def broken_pin(*, eval_router_url):
            raise RuntimeError("router down")

        cells[3].pin_for_eval = broken_pin
        pool, srv = _pool(cells)

        carve_out = await pool.carve_out()

        assert carve_out.skip_reason == "unhealthy"
        assert srv.eval_pinned_cells == {}

    @pytest.mark.asyncio
    async def test_a_second_carve_out_while_engines_are_pinned_is_refused(self):
        cells = [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(4)]
        pool, _ = _pool(cells)
        await pool.carve_out()

        with pytest.raises(AssertionError, match="still pinned"):
            await pool.carve_out()

    @pytest.mark.asyncio
    async def test_the_version_check_rejects_a_pinned_engine_that_moved(self):
        cells = [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(4)]
        pool, _ = _pool(cells)
        await pool.carve_out()

        assert (await pool.check_version("7")).skip_reason is None
        cells[3].api_client.get_weight_version = AsyncMock(return_value="8")
        assert (await pool.check_version("7")).skip_reason == "pin_violation"

    @pytest.mark.asyncio
    async def test_info_sizes_the_eval_state_by_the_carved_engines(self):
        pool, _ = _pool([_fake_cell(cell_id="engine-0", gpu_offset=0)], num_engines=3)

        info = await pool.info()

        assert info.engine_gpu_counts == [1, 1, 1]
        assert info.router == HostAndPort(host="10.0.0.9", port=31000)


class TestRouterProbe:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("router_policy", "headers"),
        [("manual", {"X-SMG-Routing-Key": "router-generate-probe"}), ("cache_aware", {})],
        ids=["key-routed", "not-key-routed"],
    )
    async def test_the_probe_carries_a_routing_key_only_for_a_key_routed_router(
        self, monkeypatch: pytest.MonkeyPatch, router_policy, headers
    ):
        """A session-server run routes by key (--policy manual); the router 400s a probe without one."""
        probe = AsyncMock()
        monkeypatch.setattr(eval_pool_module, "wait_router_generate_ready", probe)
        pool, _ = _pool(
            [_fake_cell(cell_id=f"engine-{i}", gpu_offset=i) for i in range(4)], router_policy=router_policy
        )

        await pool.carve_out()
        await pool.check_version("7")

        assert [call.kwargs["headers"] for call in probe.await_args_list] == [headers, headers]

    @pytest.mark.asyncio
    async def test_a_key_routed_router_answers_the_probe_once_it_carries_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def sgl_router_manual_policy(request: httpx.Request) -> httpx.Response:
            if "X-SMG-Routing-Key" not in request.headers:
                return httpx.Response(400, json={"error": {"code": "missing_routing_key"}})
            return httpx.Response(200, json={"text": ""})

        transport, client_class = httpx.MockTransport(sgl_router_manual_policy), httpx.AsyncClient
        monkeypatch.setattr("miles.utils.http_utils.httpx.AsyncClient", lambda: client_class(transport=transport))

        await wait_router_generate_ready(EVAL_ROUTER_URL, headers={"X-SMG-Routing-Key": "probe"}, timeout=0)
        with pytest.raises(TimeoutError, match="HTTP 400"):
            await wait_router_generate_ready(EVAL_ROUTER_URL, headers={}, timeout=0)


class TestWeightReceivers:
    @pytest.mark.asyncio
    async def test_pinned_cells_receive_no_weight_update(self):
        """A pinned engine in the update would leave its eval at a different version than the one it started at."""
        from miles.ray.rollout.rollout_server import RolloutServer

        cells = {
            "engine-0": SimpleNamespace(
                meta=SimpleNamespace(cell_id="engine-0", gpu_offset=0, num_gpus_per_engine=1),
                api_client="client-0",
                is_errored=False,
                is_eval_pinned=False,
            ),
            "engine-1": SimpleNamespace(
                meta=SimpleNamespace(cell_id="engine-1", gpu_offset=1, num_gpus_per_engine=1),
                api_client="client-1",
                is_errored=False,
                is_eval_pinned=True,
            ),
        }
        srv = RolloutServer(
            all_server_cells=cells,
            args=SimpleNamespace(),
            context_lock=ContextLock("test"),
            engine_provider=None,
        )

        async with srv.context_lock:
            assert list(srv.weight_receiver_cells) == ["engine-0"]
            assert srv.engine_cell_ids == ["engine-0"]
            assert srv.api_clients == ["client-0"]
            assert list(srv.eval_pinned_cells) == ["engine-1"]


class TestGiveBackPinnedCells:
    """Pins whose eval never gave them back, e.g. because the orchestration script died during the eval."""

    @staticmethod
    def _server(cells: dict) -> RolloutServer:
        return RolloutServer(
            all_server_cells=cells,
            args=make_args(colocate=False),
            context_lock=ContextLock("InferenceController"),
            engine_provider=_StubProvider(),
            init_expected_num_cells=len(cells),
        )

    @pytest.mark.asyncio
    async def test_cells_left_pinned_rejoin_the_receivers_and_the_take_over_gate(self, make_serving_cell, monkeypatch):
        """Left pinned, they are short of the fleet gate of the next script and never receive another update."""
        monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_INITIAL_DELAY_SECONDS", 0.001)
        monkeypatch.setattr(rollout_server_module, "WAIT_CELLS_MAX_DELAY_SECONDS", 0.001)
        cells = {"a": make_serving_cell("http://10.0.0.2:30000"), "b": make_serving_cell("http://10.0.0.3:30000")}
        await cells["b"].pin_for_eval(eval_router_url=EVAL_ROUTER_URL)
        srv = self._server(cells)

        async with srv.context_lock:
            await srv.give_back_eval_pinned_cells()
            assert srv.eval_pinned_cells == {}
            assert sorted(srv.weight_receiver_cells) == ["a", "b"]
        await srv.wait_init_expected_num_cells(timeout=1)

        for cell in cells.values():
            await cell.dispose()

    @pytest.mark.asyncio
    async def test_a_cell_that_cannot_leave_the_eval_router_does_not_hold_back_the_others(
        self, make_serving_cell, router_log, monkeypatch
    ):
        cells = {"a": make_serving_cell("http://10.0.0.2:30000"), "b": make_serving_cell("http://10.0.0.3:30000")}
        for cell in cells.values():
            await cell.pin_for_eval(eval_router_url=EVAL_ROUTER_URL)

        class _RefusingRegistry(_RecordingRegistry):
            async def unregister(self, *, worker_url):
                if worker_url == "http://10.0.0.2:30000":
                    raise RuntimeError("eval router unreachable")
                await super().unregister(worker_url=worker_url)

        monkeypatch.setattr(server_cell_module, "_make_registry", lambda url: _RefusingRegistry(router_log, url))
        monkeypatch.setattr(server_cell_module, "UNREGISTER_FROM_ROUTER_TIMEOUT", 0.01)
        srv = self._server(cells)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="eval router unreachable"):
                await srv.give_back_eval_pinned_cells()
            assert list(srv.eval_pinned_cells) == ["a"]
        assert cells["b"].is_pending_weights

        for cell in cells.values():
            await cell.dispose()


class _FakeController:
    def __init__(self, log: list, carve_out: EvalPoolCarveOut):
        self._log = log
        self._carve_out = carve_out

    async def carve_out_eval_pool(self) -> EvalPoolCarveOut:
        self._log.append("carve_out")
        return self._carve_out

    async def give_back_eval_pool(self) -> None:
        self._log.append("give_back")


class _FakeRolloutManager:
    def __init__(self, log: list, *, eval_error: BaseException | None = None, eval_gate: asyncio.Event | None = None):
        self._log = log
        self._eval_error = eval_error
        self._eval_gate = eval_gate

    async def eval(self, rollout_id, *, weight_version=None):
        self._log.append(("eval", rollout_id, weight_version))
        if self._eval_gate is not None:
            await self._eval_gate.wait()
        if self._eval_error is not None:
            raise self._eval_error

    async def report_eval_skip(self, rollout_id, reason):
        self._log.append(("skip", rollout_id, reason))


def _dispatcher(log: list, *, carve_out=None, **manager_kwargs) -> EvalDispatcher:
    args = make_args(eval_pool_num_engines=2, eval_max_in_flight=1, eval_overflow_policy="backpressure")
    return EvalDispatcher(
        args,
        actor_model=None,
        rollout_manager=_FakeRolloutManager(log, **manager_kwargs),
        inference_controller=_FakeController(log, carve_out or EvalPoolCarveOut(weight_version="7", skip_reason=None)),
    )


class TestDispatchOnCarvedPool:
    @pytest.mark.asyncio
    async def test_the_eval_runs_at_the_carved_version_without_blocking_and_gives_the_engines_back(self):
        log: list = []
        gate = asyncio.Event()
        dispatcher = _dispatcher(log, eval_gate=gate)

        await dispatcher.dispatch(10)
        await asyncio.sleep(0)

        assert log == ["carve_out", ("eval", 10, "7")]
        gate.set()
        await dispatcher.drain()
        assert log == ["carve_out", ("eval", 10, "7"), "give_back"]

    @pytest.mark.asyncio
    async def test_a_crashed_eval_still_gives_the_engines_back(self):
        """Engines left pinned would serve no training and refuse the next carve-out."""
        log: list = []
        dispatcher = _dispatcher(log, eval_error=RuntimeError("engine gone"))

        await dispatcher.dispatch(10)
        await dispatcher.drain()

        assert "give_back" in log
        assert log[-1] == ("skip", 10, "crashed")

    @pytest.mark.asyncio
    async def test_a_skipped_carve_out_is_reported_and_runs_no_eval(self):
        log: list = []
        dispatcher = _dispatcher(log, carve_out=EvalPoolCarveOut(weight_version=None, skip_reason="pin_violation"))

        await dispatcher.dispatch(10)
        await dispatcher.drain()

        assert log == ["carve_out", ("skip", 10, "pin_violation")]

    @pytest.mark.asyncio
    async def test_the_next_point_waits_for_the_previous_eval_to_give_its_engines_back(self):
        log: list = []
        dispatcher = _dispatcher(log)

        await dispatcher.dispatch(10)
        await dispatcher.dispatch(20)
        await dispatcher.drain()

        assert log == ["carve_out", ("eval", 10, "7"), "give_back", "carve_out", ("eval", 20, "7"), "give_back"]

    @pytest.mark.asyncio
    async def test_the_previous_eval_gives_its_engines_back_before_the_weight_update_of_the_next_point(self):
        """Settled only at dispatch, after the update, its engines would still be pending weights at the carve-out."""
        log: list = []
        gate = asyncio.Event()
        dispatcher = _dispatcher(log, eval_gate=gate)
        await dispatcher.dispatch(10)
        await asyncio.sleep(0)

        settling = asyncio.ensure_future(dispatcher.make_room_before_weight_update(20, force=False))
        await asyncio.sleep(0)
        assert not settling.done()
        gate.set()
        await settling
        log.append("update_weights")
        await dispatcher.dispatch(20)
        await dispatcher.drain()

        assert log == [
            "carve_out",
            ("eval", 10, "7"),
            "give_back",
            "update_weights",
            "carve_out",
            ("eval", 20, "7"),
            "give_back",
        ]

    @pytest.mark.asyncio
    async def test_a_point_skipped_before_the_update_is_not_carved_after_it(self):
        log: list = []
        gate = asyncio.Event()
        args = make_args(eval_pool_num_engines=2, eval_max_in_flight=1, eval_overflow_policy="skip")
        dispatcher = EvalDispatcher(
            args,
            actor_model=None,
            rollout_manager=_FakeRolloutManager(log, eval_gate=gate),
            inference_controller=_FakeController(log, EvalPoolCarveOut(weight_version="7", skip_reason=None)),
        )
        await dispatcher.dispatch(10)
        await asyncio.sleep(0)

        await dispatcher.make_room_before_weight_update(20, force=False)
        await dispatcher.dispatch(20)
        gate.set()
        await dispatcher.drain()

        assert log == ["carve_out", ("eval", 10, "7"), ("skip", 20, "busy"), "give_back"]
