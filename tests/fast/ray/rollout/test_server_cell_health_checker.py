from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StatePendingWeights, StateServing, StateUninitialized
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import (
    ActiveAndEpoch,
    ActivenessTracker,
    NoopHealthChecker,
    SimpleHealthChecker,
    SimpleHealthCheckerConfig,
)
from miles.utils.test_utils.clock import FakeClock

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ENDPOINT_CALLS: list[tuple[str, str]] = []


def _make_meta() -> ServerCellMetadata:
    return ServerCellMetadata(
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
    )


def _make_cell(*, ft_components: list[str], global_activeness: bool = True) -> ServerCell:
    return track_server_cell(
        ServerCell(
            args=make_args(ft_components=ft_components),
            meta=_make_meta(),
            router_api_client=MagicMock(),
            global_health_checker_activeness=lambda: ActiveAndEpoch(active=global_activeness, epoch=0),
        )
    )


def _addr_info() -> CellAddrInfo:
    return CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url="http://10.0.0.1:31000")


class _RecordingApiClient:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def health_generate(self, timeout: float = 5.0) -> bool:
        _ENDPOINT_CALLS.append(("health_generate", self.server_url))
        return True


class _NoopRouterApiClient:
    async def add_worker(self, **kwargs: Any) -> None:
        pass

    async def remove_worker(self, **kwargs: Any) -> None:
        pass


class TestRolloutCellHealthCheckerGating:
    async def test_a_cell_gets_no_checker_when_rollout_ft_is_off(self):
        """Probing engines nobody will heal only produces noise and load."""
        cell = _make_cell(ft_components=["train"])
        assert isinstance(cell._health_checker, NoopHealthChecker)

    async def test_a_cell_gets_a_real_checker_when_rollout_ft_is_on(self):
        """Rollout healing needs liveness, so the checker must actually be wired up."""
        cell = _make_cell(ft_components=["rollout"])
        assert isinstance(cell._health_checker, SimpleHealthChecker)

    async def test_the_checker_never_waits_a_grace_period(self):
        """Activeness flips every weight update window, so a grace period would restart forever."""
        cell = _make_cell(ft_components=["rollout"])
        assert cell._health_checker._config.first_wait == 0.0


class TestRolloutCellHealthCheckerActiveness:
    @pytest.mark.parametrize(
        "state, expected",
        [
            (StateUninitialized(), False),
            (StatePendingWeights(addr_info=_addr_info()), True),
            (StateServing(addr_info=_addr_info()), True),
        ],
    )
    async def test_only_a_started_engine_is_probed(self, state, expected):
        """An engine whose process is not up yet would fail every probe and look unhealthy."""
        cell = _make_cell(ft_components=["rollout"])
        cell._state = state
        assert cell._health_checker._get_activeness().active is expected

    async def test_the_global_flag_can_silence_a_serving_cell(self):
        """During a weight update the engine is offloaded, so probing it would kill a healthy cell."""
        cell = _make_cell(ft_components=["rollout"], global_activeness=False)
        cell._state = StateServing(addr_info=_addr_info())
        assert cell._health_checker._get_activeness().active is False


class TestRolloutCellHealthCheckerProbeEndpoint:
    async def test_the_probe_hits_health_generate_on_the_cells_own_engine(self, monkeypatch):
        """A probe aimed at another endpoint or another engine would never notice this engine dying."""
        _ENDPOINT_CALLS.clear()
        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
        cell = _make_cell(ft_components=["rollout"])
        cell._state = StateServing(addr_info=_addr_info())

        await cell._health_checker._check_fn()

        assert _ENDPOINT_CALLS == [("health_generate", "http://10.0.0.1:30000")]


class TestRolloutCellHealthCheckerPauseBarrier:
    async def test_pausing_the_controller_makes_the_predicate_false_for_every_cell(self):
        """The whole point of the pause is that no cell is probed inside the protected window."""
        controller, cell = await _make_controller_with_serving_cell()
        assert cell._health_checker._get_activeness().active is True

        await controller.offload()

        assert cell._health_checker._get_activeness().active is False

    async def test_pausing_discards_a_probe_that_was_already_in_flight(self):
        """A probe launched before the pause must not publish a failure inside the protected window."""
        results: list[bool] = []
        probe_started = asyncio.Event()

        async def _hanging_check() -> None:
            probe_started.set()
            await asyncio.sleep(3600)

        controller, cell = await _make_controller_with_serving_cell()
        checker = _restart_checker_with(cell, check_fn=_hanging_check, on_result=results.append)
        await asyncio.wait_for(probe_started.wait(), timeout=1)

        await controller.offload()

        assert checker._probe_task is None
        assert results == []

    async def test_the_pause_returns_only_after_the_in_flight_probe_is_really_gone(self):
        """Returning while the probe is still running is exactly the race the barrier has to close."""
        probe_started = asyncio.Event()
        probe_cancelled = asyncio.Event()

        async def _hanging_check() -> None:
            probe_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                probe_cancelled.set()
                raise

        controller, cell = await _make_controller_with_serving_cell()
        _restart_checker_with(cell, check_fn=_hanging_check, on_result=None)
        await asyncio.wait_for(probe_started.wait(), timeout=1)

        await controller.offload()

        assert probe_cancelled.is_set()


class TestRolloutCellActiveAndEpoch:
    async def test_reading_the_active_and_epoch_twice_returns_the_very_same_value(self):
        """A pull that samples into a tracker of its own degrades the epoch back into a boolean read."""
        _, cell = await _make_controller_with_serving_cell()

        first = cell._get_health_checker_active_and_epoch()
        second = cell._get_health_checker_active_and_epoch()

        assert first == second == ActiveAndEpoch(active=True, epoch=0)

    async def test_a_pause_resume_window_completed_between_two_polls_resets_the_failure_counter(self):
        """The controller opens and closes the whole window while the loop sleeps, so only the epoch reveals it."""
        controller, cell = await _make_controller_with_serving_cell()
        checker, clock = _make_fake_clock_checker(cell)

        checker.start()
        await _settle(clock)
        await clock.elapse(100.0)
        await clock.elapse(10.0)
        assert checker._consecutive_failures == 2

        await controller.offload()
        await controller.prepare_eval()
        await clock.elapse(10.0)

        assert checker._consecutive_failures == 0
        checker.stop()


class TestRolloutCellHealthCheckerDisposal:
    async def test_disposing_a_cell_stops_its_checker(self):
        """A removed cell whose loop keeps polling leaks the task and the whole cell it closes over."""
        cell = _make_cell(ft_components=["rollout"])
        assert cell._health_checker._task is not None

        await cell.dispose()

        assert cell._health_checker._task is None

    async def test_a_cell_dropped_without_dispose_complains_on_collection(self):
        """Silently dropping a cell leaks its checker task forever, so the mistake must be loud."""
        cell = _make_cell(ft_components=["rollout"])

        with pytest.raises(AssertionError, match="without dispose"):
            cell.__del__()

    async def test_a_disposed_cell_is_collected_quietly(self):
        """The collection guard must not fire on the normal teardown path."""
        cell = _make_cell(ft_components=["rollout"])
        await cell.dispose()

        cell.__del__()


async def _make_controller_with_serving_cell() -> tuple[InferenceController, ServerCell]:
    args: Any = make_args(ft_components=["rollout"], colocate=True)
    controller = InferenceController.__new__(InferenceController)
    controller.args = args
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = ActivenessTracker(active=True)

    srv = RolloutServer(
        server_cells={},
        args=args,
        context_lock=controller.context_lock,
        global_health_checker_activeness=controller._health_checker_activeness.get,
    )
    controller.servers = {"default": srv}

    async with controller.context_lock:
        await srv.add_cell(_make_meta())

    cell: ServerCell = track_server_cell(srv.server_cells["inference-engine-0-0-0"])
    cell.router_api_client = _NoopRouterApiClient()
    cell._state = StateServing(addr_info=_addr_info())
    return controller, cell


def _make_fake_clock_checker(cell: ServerCell) -> tuple[SimpleHealthChecker, FakeClock]:
    cell._health_checker.stop()

    async def _failing_check() -> None:
        raise RuntimeError("engine down")

    clock = FakeClock()
    checker = SimpleHealthChecker(
        name=f"rollout-cell-{cell.meta.cell_id}",
        check_fn=_failing_check,
        get_activeness=cell._get_health_checker_active_and_epoch,
        config=SimpleHealthCheckerConfig(interval=10.0, timeout=5.0, first_wait=100.0, failure_threshold=3),
        clock=clock,
    )
    return checker, clock


async def _settle(clock: FakeClock) -> None:
    for _ in range(1000):
        if clock.pending_count >= 1:
            return
        await asyncio.sleep(0)


def _restart_checker_with(cell: ServerCell, *, check_fn: Any, on_result: Any) -> SimpleHealthChecker:
    checker: SimpleHealthChecker = cell._health_checker
    checker.stop()
    checker._check_fn = check_fn
    checker._on_result = on_result
    checker.start()
    return checker
