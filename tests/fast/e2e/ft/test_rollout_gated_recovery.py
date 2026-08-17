from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest
from tests.e2e.ft.conftest_ft import fault_injection as fi
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout import rollout_server as rollout_server_module
from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.ft_utils.mini_ft_controller import _compute_cell_snapshot, _MiniFTController
from miles.utils.workers.cell_operations.ray import RayCellOperations
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts

_POOL_ID = "inference-engine-0"
_CELL_IDS = ["inference-engine-0-0-0", "inference-engine-0-0-1"]


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"this module addresses cells through a patched _compute_addr_info ({worker_name=})")


class _FakeRouter:
    def __init__(self) -> None:
        self.worker_urls: list[str] = []

    async def add_worker(
        self, *, worker_url: str, worker_type: str, use_legacy_api: bool, bootstrap_port: int | None
    ) -> None:
        self.worker_urls.append(worker_url)

    async def remove_worker(self, *, worker_url: str, use_legacy_api: bool) -> None:
        self.worker_urls.remove(worker_url)


class _FakeHealthChecker:
    def __init__(self) -> None:
        self.status: TriState = TriState.TRUE
        self.started: bool = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


class _FakeEngineApiClient:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def release_memory_occupation(self, tags: list[str] | None = None) -> None:
        return None

    async def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        return None

    async def get_server_info(self) -> dict[str, Any]:
        return {"model_path": "/fake/model", "internal_states": [{"env_vars": {"RANK": "0"}}]}


class _RemoteMethod:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class _FakeWorkerManager:
    def __init__(self, *, cell_ids: list[str], reconcile: Callable[[str, CellInfo | None], Any]) -> None:
        self._alive: dict[str, bool] = {cell_id: True for cell_id in cell_ids}
        self._generation: dict[str, int] = {cell_id: 0 for cell_id in cell_ids}
        self._reconcile = reconcile
        self.get_cell_infos = _RemoteMethod(self._get_cell_infos)
        self.stop_cells = _RemoteMethod(self._stop_cells)
        self.start_cells = _RemoteMethod(self._start_cells)

    def cell_info(self, cell_id: str) -> CellInfo:
        index = _CELL_IDS.index(cell_id)
        return CellInfo(
            cell_id=cell_id,
            pool_id=_POOL_ID,
            alive=self._alive[cell_id],
            worker_names=[f"{cell_id}-0"],
            workers_hash=f"hash-{cell_id}-{self._generation[cell_id]}",
            meta={
                "model_id": "default",
                "worker_type": "regular",
                "num_gpus_per_engine": 1,
                "gpu_offset": index,
                "sglang_api_key": None,
                "needs_offload": True,
                "update_weights": True,
            },
        )

    async def arrive_all(self) -> None:
        for cell_id in _CELL_IDS:
            await self._reconcile(cell_id, self.cell_info(cell_id))

    async def _get_cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return {cell_id: self.cell_info(cell_id) for cell_id in _CELL_IDS}

    async def _stop_cells(self, cell_ids: list[str]) -> None:
        for cell_id in cell_ids:
            self._alive[cell_id] = False
            await self._reconcile(cell_id, None)

    async def _start_cells(self, cell_ids: list[str]) -> None:
        for cell_id in cell_ids:
            self._alive[cell_id] = True
            self._generation[cell_id] += 1
            await self._reconcile(cell_id, self.cell_info(cell_id))


class _Harness:
    def __init__(self, *, monkeypatch: pytest.MonkeyPatch) -> None:
        self.router = _FakeRouter()
        self.health_checkers: dict[str, _FakeHealthChecker] = {}
        self.activated_gate_urls: list[str] = []

        async def _activate(gate_url: str) -> None:
            self.activated_gate_urls.append(gate_url)

        async def _probe(server_url: str, api_key: str | None, timeout: float = 5.0) -> bool:
            return True

        async def _compute_addr_info(cell: ServerCell) -> CellAddrInfo:
            index = _CELL_IDS.index(cell.meta.cell_id)
            return CellAddrInfo(
                server_url=f"http://10.0.0.{index}:30000",
                bootstrap_port=None,
                gate_url=f"http://10.0.0.{index}:13000",
            )

        def _create_health_checker(
            *, args: Any, name: str, get_api_client: Any, get_activeness: Any
        ) -> _FakeHealthChecker:
            checker = _FakeHealthChecker()
            self.health_checkers[name.removeprefix("rollout-cell-")] = checker
            return checker

        monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
        monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _FakeEngineApiClient)
        monkeypatch.setattr(server_cell_module, "create_rollout_cell_health_checker", _create_health_checker)
        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)
        monkeypatch.setattr(rollout_server_module, "SGLangRouterApiClient", lambda router_url: self.router)
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_POLL_INTERVAL_SECONDS", 0.0)

        self.args = make_args(colocate=True, use_fault_tolerance=True, ft_components=["rollout"])
        self.controller = InferenceController.__new__(InferenceController)
        self.controller.args = self.args
        self.controller.context_lock = ContextLock("InferenceController")
        self.controller._watcher_disposers = []
        self.controller._ticker = None
        self.controller._health_checker_activeness = ActivenessTracker(active=True)
        self.controller.servers = {
            "default": RolloutServer(
                server_cells={},
                args=self.args,
                context_lock=self.controller.context_lock,
                engine_provider=_StubProvider(),
                router_ip="10.0.0.9",
                router_port=20000,
                model_name="default",
                update_weights=True,
                expected_num_cells=len(_CELL_IDS),
            )
        }
        self.worker_manager = _FakeWorkerManager(cell_ids=_CELL_IDS, reconcile=self.controller._reconcile)
        self.handler = _CellHandler(
            cell_type="rollout",
            operations=RayCellOperations(worker_manager_handle=self.worker_manager),
            controllers=[self.controller],
            pool_ids=[_POOL_ID],
        )
        self.ft_controller = _MiniFTController(
            get_cells=self._get_cell_snapshots,
            suspend_cell=self.handler.suspend,
            resume_cell=self.handler.resume,
            poll_interval=0.0,
            resume_delay=0.0,
        )

    async def observe(self) -> dict[str, dict]:
        cells = await self.handler.list_cells()
        return {cell.metadata.name: cell.model_dump(mode="json") for cell in cells}

    async def open_weight_update_window(self, *, mark_weights_ready: bool = True) -> None:
        tick_task = asyncio.create_task(self._tick_forever())
        try:
            engines = await asyncio.wait_for(self.controller.start_update_weights(), timeout=5)
            await self.controller.end_update_weights(engines.snapshot_cell_id_to_hashes if mark_weights_ready else {})
        finally:
            tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tick_task

    def crash(self, cell_id: str) -> None:
        self.health_checkers[cell_id].status = TriState.FALSE

    async def run_ft_controller_once(self) -> None:
        await self.ft_controller._poll_and_heal()

    async def _get_cell_snapshots(self) -> list[Any]:
        return [_compute_cell_snapshot(cell) for cell in await self.handler.list_cells()]

    async def _tick_forever(self) -> None:
        while True:
            await self.controller._tick_cells()
            await asyncio.sleep(0)


@pytest.fixture
async def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """A colocated 2-engine rollout deployment wired to a fake worker manager, up and serving."""
    harness = _Harness(monkeypatch=monkeypatch)
    await harness.worker_manager.arrive_all()
    await harness.open_weight_update_window()
    return harness


def _serving_condition(cell: dict) -> str | None:
    return next((cond["status"] for cond in cell["status"]["conditions"] if cond["type"] == "Serving"), None)


async def test_a_healthy_deployment_reports_every_engine_as_serving(harness: _Harness) -> None:
    """The soak's terminal criterion is Serving, so a fault-free deployment must report exactly that."""
    cells = await harness.observe()

    assert [cell["status"]["phase"] for cell in cells.values()] == ["Running", "Running"]
    assert [_serving_condition(cell) for cell in cells.values()] == ["True", "True"]
    assert sorted(harness.router.worker_urls) == ["http://10.0.0.0:30000", "http://10.0.0.1:30000"]


async def test_a_crashed_engine_is_suspended_and_withdrawn_from_the_router(harness: _Harness) -> None:
    """Healing starts by tearing the engine down; leaving it registered would keep routing at a dead engine."""
    harness.crash(_CELL_IDS[0])

    await harness.handler.suspend(_CELL_IDS[0])
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Suspended"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_mini_ft_controller_heals_exactly_the_crashed_engine(harness: _Harness) -> None:
    """A heal that also cycled the healthy engines would take the whole deployment down at once."""
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Pending"
    assert cells[_CELL_IDS[1]]["status"]["phase"] == "Running"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_replacement_engine_stays_gated_until_the_next_window(harness: _Harness) -> None:
    """Under colocation the trainer owns the gpus mid-step, so the relaunched engine must not claim memory yet."""
    harness.crash(_CELL_IDS[0])
    gates_before = list(harness.activated_gate_urls)

    await harness.run_ft_controller_once()
    cells = await harness.observe()

    assert harness.controller.servers["default"].server_cells[_CELL_IDS[0]].is_uninitialized
    assert harness.activated_gate_urls == gates_before
    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Pending"


async def test_the_replacement_engine_is_not_serving_before_it_gets_weights(harness: _Harness) -> None:
    """Regression: a relaunched engine holding stale weights reads Running, which alone proves no recovery."""
    harness.crash(_CELL_IDS[0])
    await harness.run_ft_controller_once()

    await harness.open_weight_update_window(mark_weights_ready=False)
    cells = await harness.observe()

    assert cells[_CELL_IDS[0]]["status"]["phase"] == "Running"
    assert _serving_condition(cells[_CELL_IDS[0]]) == "False"
    assert harness.router.worker_urls == ["http://10.0.0.1:30000"]


async def test_the_next_window_puts_the_replacement_engine_back_in_the_router(harness: _Harness) -> None:
    """This is the recovery the soak asserts: the replaced engine ends up serving again."""
    harness.crash(_CELL_IDS[0])
    await harness.run_ft_controller_once()

    await harness.open_weight_update_window()
    cells = await harness.observe()

    assert _serving_condition(cells[_CELL_IDS[0]]) == "True"
    assert sorted(harness.router.worker_urls) == ["http://10.0.0.0:30000", "http://10.0.0.1:30000"]


async def test_the_observed_sequence_satisfies_the_soak_recovery_witness(harness: _Harness) -> None:
    """The fast-layer stand-in is only worth anything if the e2e witness accepts the sequence it produces."""
    log = fi.EventLog()
    log.observe(list((await harness.observe()).values()))
    log.note_injected(_CELL_IDS[0])
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    log.observe(list((await harness.observe()).values()))
    await harness.open_weight_update_window()
    log.observe(list((await harness.observe()).values()))

    assert fi.compute_num_completed_recoveries(log.events, cell_type="rollout") == 1
    assert fi.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {}


async def test_the_witness_rejects_a_replacement_that_never_reaches_the_router(harness: _Harness) -> None:
    """A weight update that silently skips the replaced cell leaves it Running forever, and must fail the soak."""
    log = fi.EventLog()
    log.observe(list((await harness.observe()).values()))
    log.note_injected(_CELL_IDS[0])
    harness.crash(_CELL_IDS[0])

    await harness.run_ft_controller_once()
    log.observe(list((await harness.observe()).values()))
    await harness.open_weight_update_window(mark_weights_ready=False)
    log.observe(list((await harness.observe()).values()))

    assert fi.compute_num_completed_recoveries(log.events, cell_type="rollout") == 0
    assert fi.compute_cells_with_unfinished_recovery(log.events, cell_type="rollout") == {_CELL_IDS[0]: 1}
