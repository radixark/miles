from __future__ import annotations

import argparse
import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from miles.utils.arguments import (
    _DEFAULT_FT_API_SERVER_PORT,
    _resolve_api_server_port,
    _resolve_mini_ft_controller_enable,
)
from miles.utils.ft_utils import mini_ft_controller
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.ft_utils.mini_ft_controller import (
    CellHealthStatus,
    _CellSnapshot,
    _compute_cell_snapshot,
    _MiniFTController,
    _MiniFTControllerRunner,
)

# ------------------------ helpers ------------------------


def _make_cell(
    *,
    name: str = "cell-0",
    status: CellHealthStatus = CellHealthStatus.HEALTHY,
) -> _CellSnapshot:
    return _CellSnapshot(name=name, status=status)


HEALTHY = CellHealthStatus.HEALTHY
UNHEALTHY = CellHealthStatus.UNHEALTHY
NOT_APPLICABLE = CellHealthStatus.NOT_APPLICABLE


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_controller(
    *,
    get_cells: AsyncMock | None = None,
    suspend_cell: AsyncMock | None = None,
    resume_cell: AsyncMock | None = None,
    poll_interval: float = 0.01,
    resume_delay: float = 0.0,
    clock: _FakeClock | None = None,
) -> _MiniFTController:
    return _MiniFTController(
        get_cells=get_cells or AsyncMock(return_value=[]),
        suspend_cell=suspend_cell or AsyncMock(),
        resume_cell=resume_cell or AsyncMock(),
        poll_interval=poll_interval,
        resume_delay=resume_delay,
        clock=clock or _FakeClock(),
    )


def _run_controller_for_n_polls(
    controller: _MiniFTController,
    get_cells_mock: AsyncMock,
    responses: list[list[_CellSnapshot]],
) -> None:
    """Configure get_cells to return `responses` then stop the controller."""
    poll_count = 0

    async def _side_effect() -> list[_CellSnapshot]:
        nonlocal poll_count
        result = responses[poll_count] if poll_count < len(responses) else []
        poll_count += 1
        if poll_count >= len(responses):
            controller.request_stop()
        return result

    get_cells_mock.side_effect = _side_effect


def _build_cell_json(
    *,
    name: str = "actor-0",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
    workers_hash: str = "hash-0",
) -> dict[str, Any]:
    return {
        "apiVersion": "miles.io/v1",
        "kind": "Cell",
        "metadata": {
            "name": name,
            "labels": {"miles.io/cell-type": "actor", "miles.io/cell-id": name},
        },
        "spec": {"suspend": False},
        "status": {
            "phase": "Running",
            "workers_hash": workers_hash,
            "conditions": [
                {"type": "Allocated", "status": "True"},
                {"type": "Healthy", "status": healthy_status, "reason": healthy_reason},
            ],
        },
    }


def _build_cell_list_json(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"apiVersion": "miles.io/v1", "kind": "CellList", "items": cells}


def _create_runner() -> _MiniFTControllerRunner:
    return _MiniFTControllerRunner(
        api_server_url="http://127.0.0.1:8080",
        poll_interval=10.0,
        resume_delay=5.0,
    )


def _mock_response(*, status_code: int = 200, json_data: Any = None) -> httpx.Response:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
    return response


# ------------------------ snapshot tests ------------------------


def _make_cell_object(
    *, name: str = "actor-0", healthy: list[CellCondition] | None = None, phase: str = "Running"
) -> Cell:
    conditions: list[CellCondition] = [CellCondition.allocated(TriState.TRUE)]
    if healthy is not None:
        conditions.extend(healthy)
    return Cell(
        metadata=CellMetadata(name=name, labels={}),
        spec=CellSpec(),
        status=CellStatus(phase=phase, conditions=conditions, workers_hash="pseudo-hash-0"),
    )


class TestComputeCellSnapshot:
    def test_healthy_true_maps_to_healthy(self):
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.TRUE)])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=HEALTHY)

    def test_healthy_false_maps_to_unhealthy(self):
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.FALSE, reason="HealthCheckFailed")])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=UNHEALTHY)

    def test_healthy_unknown_does_not_trigger_heal(self):
        """Regression: UNKNOWN is a transient state (e.g. health checker paused
        during healing); controller must not treat it as unhealthy and reconcile."""
        cell = _make_cell_object(healthy=[CellCondition.healthy(TriState.UNKNOWN, reason="HealthCheckUnknown")])

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot.status != UNHEALTHY

    def test_missing_healthy_condition_maps_to_not_applicable(self):
        cell = _make_cell_object(healthy=None)

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    def test_suspended_cell_maps_to_unhealthy_so_controller_resumes(self):
        """A Suspended cell was shrunk out by the train side (cell.stop()) and must be resumed;
        it carries no Healthy condition, so the controller must heal it by phase, not be misled
        into NOT_APPLICABLE (which would lose the cell forever)."""
        cell = _make_cell_object(healthy=None, phase="Suspended")

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=UNHEALTHY)

    def test_pending_cell_maps_to_not_applicable_not_healed(self):
        """A Pending cell is mid-heal (being re-allocated); it also has no Healthy condition but
        must NOT be treated as needing heal, otherwise the controller fights the ongoing heal."""
        cell = _make_cell_object(healthy=None, phase="Pending")

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    def test_any_false_among_multiple_conditions_wins(self):
        """If any Healthy condition is FALSE, the cell is unhealthy regardless of
        other TRUE/UNKNOWN conditions — fail-loud over fail-quiet."""
        cell = _make_cell_object(
            healthy=[
                CellCondition.healthy(TriState.TRUE),
                CellCondition.healthy(TriState.FALSE),
            ]
        )

        snapshot = _compute_cell_snapshot(cell)

        assert snapshot.status == UNHEALTHY


# ------------------------ controller tests ------------------------


class TestControllerHealing:
    @pytest.mark.asyncio
    async def test_heal_on_unhealthy_cell(self) -> None:
        """Unhealthy cell triggers suspend → resume."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_called_once_with("cell-0")
        resume_cell.assert_called_once_with("cell-0")

    @pytest.mark.asyncio
    async def test_skip_healthy_cell(self) -> None:
        """Healthy cell does not trigger heal."""
        healthy_cell = _make_cell(name="cell-0", status=HEALTHY)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[healthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_heal_multiple_unhealthy_cells(self) -> None:
        """Multiple unhealthy cells are each healed."""
        cells = [
            _make_cell(name="cell-0", status=UNHEALTHY),
            _make_cell(name="cell-1", status=UNHEALTHY),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert suspend_cell.call_count == 2
        assert resume_cell.call_count == 2
        suspend_cell.assert_any_call("cell-0")
        suspend_cell.assert_any_call("cell-1")
        resume_cell.assert_any_call("cell-0")
        resume_cell.assert_any_call("cell-1")

    @pytest.mark.asyncio
    async def test_no_action_when_all_healthy(self) -> None:
        """All healthy → no suspend/resume calls."""
        cells = [
            _make_cell(name="cell-0", status=HEALTHY),
            _make_cell(name="cell-1", status=HEALTHY),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()


class TestControllerBackoff:
    @pytest.mark.asyncio
    async def test_backoff_on_heal_failure(self) -> None:
        """Suspend raises → consecutive_failures increments, next_attempt_at increases."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=100.0)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("connection failed"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1
        assert backoff.next_attempt_at == 110.0  # 100 + 5*2^1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Verify backoff delays: 5*2^1=10, 5*2^2=20, ..., capped at 300."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        expected_delays = [10, 20, 40, 80, 160, 300, 300]
        for expected_delay in expected_delays:
            # Advance clock past backoff window so the heal attempt is made
            backoff = controller._cell_backoffs.get("cell-0")
            if backoff:
                clock.now = backoff.next_attempt_at

            _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
            await asyncio.wait_for(controller.run(), timeout=5.0)

            backoff = controller._cell_backoffs["cell-0"]
            assert backoff.next_attempt_at == clock.now + expected_delay

    @pytest.mark.asyncio
    async def test_skips_heal_when_within_backoff_window(self) -> None:
        """Cell is not healed again until clock passes next_attempt_at."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: First poll → heal attempt fails, sets next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1

        # Step 2: Poll again without advancing clock → should skip heal
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 1  # no new call

        # Step 3: Advance clock past backoff → should attempt heal again
        backoff = controller._cell_backoffs["cell-0"]
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert suspend_cell.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_heal_resets_backoff(self) -> None:
        """Successful heal resets consecutive_failures to 0."""
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        clock = _FakeClock(start=0.0)

        call_count = 0

        async def failing_then_succeeding_suspend(name: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")

        get_cells = AsyncMock()
        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=AsyncMock(side_effect=failing_then_succeeding_suspend),
            resume_cell=AsyncMock(),
            clock=clock,
        )

        # Step 1: First attempt fails
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        backoff = controller._cell_backoffs["cell-0"]
        assert backoff.consecutive_failures == 1

        # Step 2: Advance clock past backoff, second attempt succeeds
        clock.now = backoff.next_attempt_at
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert backoff.consecutive_failures == 0
        assert backoff.next_attempt_at == clock.now


class TestControllerNotApplicable:
    @pytest.mark.asyncio
    async def test_not_applicable_cell_skipped(self) -> None:
        """NOT_APPLICABLE cell does not trigger heal."""
        na_cell = _make_cell(name="cell-0", status=NOT_APPLICABLE)
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [[na_cell]])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_not_called()
        resume_cell.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_statuses_only_heals_unhealthy(self) -> None:
        """HEALTHY and NOT_APPLICABLE skipped, only UNHEALTHY healed."""
        cells = [
            _make_cell(name="cell-0", status=HEALTHY),
            _make_cell(name="cell-1", status=UNHEALTHY),
            _make_cell(name="cell-2", status=NOT_APPLICABLE),
        ]
        get_cells = AsyncMock()
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
        )
        _run_controller_for_n_polls(controller, get_cells, [cells])

        await asyncio.wait_for(controller.run(), timeout=5.0)

        suspend_cell.assert_called_once_with("cell-1")
        resume_cell.assert_called_once_with("cell-1")

    @pytest.mark.asyncio
    async def test_backoff_cleared_when_cell_leaves_unhealthy(self) -> None:
        """Cell goes UNHEALTHY (heal fails) → NOT_APPLICABLE → backoff is cleaned up."""
        clock = _FakeClock(start=0.0)
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        na_cell = _make_cell(name="cell-0", status=NOT_APPLICABLE)

        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: Unhealthy → heal fails → backoff entry created
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" in controller._cell_backoffs

        # Step 2: Cell becomes NOT_APPLICABLE → backoff entry cleaned
        _run_controller_for_n_polls(controller, get_cells, [[na_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" not in controller._cell_backoffs

    @pytest.mark.asyncio
    async def test_backoff_cleared_when_cell_becomes_healthy(self) -> None:
        """Cell goes UNHEALTHY (heal fails) → HEALTHY → backoff is cleaned up."""
        clock = _FakeClock(start=0.0)
        unhealthy_cell = _make_cell(name="cell-0", status=UNHEALTHY)
        healthy_cell = _make_cell(name="cell-0", status=HEALTHY)

        get_cells = AsyncMock()
        suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

        controller = _make_controller(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            clock=clock,
        )

        # Step 1: Unhealthy → heal fails → backoff entry created
        _run_controller_for_n_polls(controller, get_cells, [[unhealthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" in controller._cell_backoffs

        # Step 2: Cell becomes HEALTHY → backoff entry cleaned
        _run_controller_for_n_polls(controller, get_cells, [[healthy_cell]])
        await asyncio.wait_for(controller.run(), timeout=5.0)
        assert "cell-0" not in controller._cell_backoffs


class TestControllerLifecycle:
    @pytest.mark.asyncio
    async def test_poll_continues_after_get_cells_failure(self) -> None:
        """get_cells raises → controller does not exit, continues polling."""
        call_count = 0

        async def failing_then_stopping(controller_ref: list[_MiniFTController]) -> list[_CellSnapshot]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("network error")
            controller_ref[0].request_stop()
            return []

        controller = _make_controller()
        controller_ref = [controller]

        async def _side_effect() -> list[_CellSnapshot]:
            return await failing_then_stopping(controller_ref)

        controller._get_cells = AsyncMock(side_effect=_side_effect)

        await asyncio.wait_for(controller.run(), timeout=5.0)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_request_stop_exits_loop(self) -> None:
        """call request_stop → run() returns."""
        get_cells = AsyncMock()
        controller = _make_controller(get_cells=get_cells)
        _run_controller_for_n_polls(controller, get_cells, [[]])

        await asyncio.wait_for(controller.run(), timeout=5.0)


# ------------------------ runner tests ------------------------


class TestRunnerGetCells:
    @pytest.mark.asyncio
    async def test_parses_healthy_and_unhealthy(self) -> None:
        """Parse CellList JSON into _CellSnapshot with correct healthy bool."""
        runner = _create_runner()

        cells_json = _build_cell_list_json(
            [
                _build_cell_json(name="actor-0", healthy_status="True"),
                _build_cell_json(name="actor-1", healthy_status="False"),
            ]
        )

        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 2
        assert result[0] == _CellSnapshot(name="actor-0", status=HEALTHY)
        assert result[1] == _CellSnapshot(name="actor-1", status=UNHEALTHY)

    @pytest.mark.asyncio
    async def test_missing_healthy_condition_treated_as_not_applicable(self) -> None:
        """Cell with no Healthy condition → status=NOT_APPLICABLE (not healed)."""
        runner = _create_runner()

        cell_json = _build_cell_json(name="actor-0")
        # Remove the Healthy condition, keep only Allocated
        cell_json["status"]["conditions"] = [{"type": "Allocated", "status": "True"}]

        cells_json = _build_cell_list_json([cell_json])
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(json_data=cells_json))

        result = await runner._get_cells()

        assert len(result) == 1
        assert result[0] == _CellSnapshot(name="actor-0", status=NOT_APPLICABLE)

    @pytest.mark.asyncio
    async def test_http_error_raises(self) -> None:
        """Verify HTTP 4xx/5xx propagated."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.get = AsyncMock(return_value=_mock_response(status_code=500))

        with pytest.raises(httpx.HTTPStatusError):
            await runner._get_cells()


class TestRunnerPatchCell:
    @pytest.mark.asyncio
    async def test_suspend_sends_correct_patch(self) -> None:
        """Verify PATCH body for suspend uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._suspend_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": True}}

    @pytest.mark.asyncio
    async def test_resume_sends_correct_patch(self) -> None:
        """Verify PATCH body for resume uses CellPatch model."""
        runner = _create_runner()
        runner._client = AsyncMock()
        runner._client.patch = AsyncMock(return_value=_mock_response())

        await runner._resume_cell("actor-0")

        runner._client.patch.assert_called_once()
        call_args = runner._client.patch.call_args
        assert call_args[0][0] == "/api/v1/cells/actor-0"
        body = json.loads(call_args[1]["content"])
        assert body == {"spec": {"suspend": False}}


class TestFtControllerDefaults:
    @staticmethod
    def _resolve(**overrides) -> tuple[int, bool]:
        args = argparse.Namespace(
            **{
                "api_server_port": None,
                "mini_ft_controller_enable": None,
                "ft_components": [],
                **overrides,
            }
        )
        args.api_server_port = _resolve_api_server_port(args)
        return args.api_server_port, _resolve_mini_ft_controller_enable(args)

    def test_asking_for_fault_tolerance_opens_the_port_and_starts_the_healing_loop(self) -> None:
        """The health checkers only publish a status; without both of these a run watches an
        engine die and leaves it routed."""
        assert self._resolve(ft_components=["rollout"]) == (_DEFAULT_FT_API_SERVER_PORT, True)

    def test_a_run_without_fault_tolerance_opens_no_port_and_starts_no_loop(self) -> None:
        """There is nothing to heal, so the port would be surface area for nobody."""
        assert self._resolve(ft_components=[]) == (0, False)

    def test_an_explicitly_disabled_port_also_disables_the_healing_loop(self) -> None:
        """The loop drives cells over that port, so leaving it on would fail every poll."""
        assert self._resolve(ft_components=["rollout"], api_server_port=0) == (0, False)

    def test_an_explicit_port_is_kept_as_given(self) -> None:
        """A run that pins the port has an external controller expecting to find it there."""
        assert self._resolve(ft_components=["rollout"], api_server_port=9999) == (9999, True)

    def test_an_explicit_healing_choice_wins_over_the_default(self) -> None:
        """The flag is how a run opts out of healing while keeping the health reporting."""
        assert self._resolve(ft_components=["rollout"], mini_ft_controller_enable=False)[1] is False


class _FakeRunner:
    def __init__(self, *, api_server_url: str, poll_interval: float, resume_delay: float) -> None:
        self.api_server_url = api_server_url
        self.poll_interval = poll_interval
        self.resume_delay = resume_delay
        self.ran = threading.Event()
        self.thread_was_daemon: bool | None = None
        self.thread_was_main_thread: bool | None = None

    async def run(self) -> None:
        current_thread = threading.current_thread()
        self.thread_was_daemon = current_thread.daemon
        self.thread_was_main_thread = current_thread is threading.main_thread()
        self.ran.set()


class TestMaybeStartMiniFtController:
    def test_enabled_controller_uses_api_server_port_and_starts_a_daemon_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Enabling the mini FT controller aims the runner at the local API server port and runs it on a daemon thread."""
        created_runners: list[_FakeRunner] = []

        def _create_runner_spy(**kwargs: Any) -> _FakeRunner:
            runner = _FakeRunner(**kwargs)
            created_runners.append(runner)
            return runner

        monkeypatch.setattr(mini_ft_controller, "_MiniFTControllerRunner", _create_runner_spy)

        mini_ft_controller.maybe_start_mini_ft_controller(
            argparse.Namespace(
                mini_ft_controller_enable=True,
                api_server_port=18231,
                mini_ft_controller_poll_interval=1.5,
                mini_ft_controller_resume_delay=2.5,
            )
        )

        assert len(created_runners) == 1
        runner = created_runners[0]
        assert runner.api_server_url == "http://127.0.0.1:18231"
        assert runner.poll_interval == 1.5
        assert runner.resume_delay == 2.5
        assert runner.ran.wait(timeout=5.0)
        assert runner.thread_was_daemon is True
        assert runner.thread_was_main_thread is False
