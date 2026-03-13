from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.agents.core.rollout.health_checker import (
    CellEntry,
    RolloutHealthChecker,
    _probe_cell,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockEngine:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    async def health_check(self) -> None:
        if not self.alive:
            raise ConnectionError("engine dead")


async def _engine_health_fn(engine: object) -> None:
    await engine.health_check()  # type: ignore[attr-defined]


class _ReportCollector:
    """Collects (cell_id, is_healthy) reports for assertions."""

    def __init__(self) -> None:
        self.reports: list[tuple[str, bool]] = []

    def __call__(self, *, cell_id: str, is_healthy: bool) -> None:
        self.reports.append((cell_id, is_healthy))

    def latest(self, cell_id: str) -> bool | None:
        for cid, healthy in reversed(self.reports):
            if cid == cell_id:
                return healthy
        return None


def _make_checker(
    engines_by_cell: dict[str, list[_MockEngine]],
    *,
    check_interval: float = 0.05,
) -> tuple[RolloutHealthChecker, _ReportCollector]:
    collector = _ReportCollector()
    checker = RolloutHealthChecker(
        cells=[
            CellEntry(cell_id=cid, get_engines=lambda es=es: es)
            for cid, es in engines_by_cell.items()
        ],
        engine_health_fn=_engine_health_fn,
        report_fn=collector,
        check_interval=check_interval,
    )
    return checker, collector


# ===========================================================================
# _probe_cell
# ===========================================================================


class _SlowEngine:
    async def health_check(self) -> None:
        await asyncio.sleep(999)


class _BrokenEngine:
    async def health_check(self) -> None:
        raise ConnectionError("engine crashed")


class TestProbeCell:
    @pytest.mark.anyio
    async def test_alive_engine_returns_true(self) -> None:
        result = await _probe_cell(
            engines=[_MockEngine(True)],
            engine_health_fn=_engine_health_fn,
            cell_id="a0",
            timeout=10.0,
        )

        assert result is True

    @pytest.mark.anyio
    async def test_timeout_returns_false(self) -> None:
        result = await _probe_cell(
            engines=[_SlowEngine()],
            engine_health_fn=_engine_health_fn,
            cell_id="a0",
            timeout=0.01,
        )

        assert result is False

    @pytest.mark.anyio
    async def test_exception_returns_false(self) -> None:
        result = await _probe_cell(
            engines=[_BrokenEngine()],
            engine_health_fn=_engine_health_fn,
            cell_id="a0",
            timeout=10.0,
        )

        assert result is False

    @pytest.mark.anyio
    async def test_probes_only_engine0(self) -> None:
        """Only engines[0] is probed; if it's dead the whole cell is dead."""
        result = await _probe_cell(
            engines=[_BrokenEngine(), _MockEngine(True)],
            engine_health_fn=_engine_health_fn,
            cell_id="c1",
            timeout=10.0,
        )

        assert result is False

    @pytest.mark.anyio
    async def test_empty_engines_raises(self) -> None:
        with pytest.raises(ValueError, match="engines must not be empty"):
            await _probe_cell(
                engines=[],
                engine_health_fn=_engine_health_fn,
                cell_id="empty",
                timeout=10.0,
            )

    @pytest.mark.anyio
    async def test_lead_engine_none_returns_false(self) -> None:
        """engines[0] is None (killed engine) → probe returns False without calling health_fn."""
        called = False

        async def _spy_health_fn(engine: object) -> None:
            nonlocal called
            called = True

        result = await _probe_cell(
            engines=[None],  # type: ignore[list-item]
            engine_health_fn=_spy_health_fn,
            cell_id="killed",
            timeout=10.0,
        )

        assert result is False
        assert not called, "health_fn should not be called when lead engine is None"

    @pytest.mark.anyio
    async def test_lead_engine_none_with_healthy_followers_returns_false(self) -> None:
        """engines[0] is None but followers are alive → still returns False."""
        result = await _probe_cell(
            engines=[None, _MockEngine(True), _MockEngine(True)],  # type: ignore[list-item]
            engine_health_fn=_engine_health_fn,
            cell_id="killed-lead",
            timeout=10.0,
        )

        assert result is False

    # P2 item 27: additional probe edge cases
    @pytest.mark.anyio
    async def test_engine_raises_timeout_error_returns_false(self) -> None:
        """Engine health_fn that raises asyncio.TimeoutError → probe returns False."""

        async def _timeout_fn(engine: object) -> None:
            raise asyncio.TimeoutError("health check timed out")

        result = await _probe_cell(
            engines=[_MockEngine(True)],
            engine_health_fn=_timeout_fn,
            cell_id="timeout-cell",
            timeout=10.0,
        )
        assert result is False

    @pytest.mark.anyio
    async def test_engine_raises_connection_refused_returns_false(self) -> None:
        """Engine health_fn that raises ConnectionRefusedError → probe returns False."""

        async def _refused_fn(engine: object) -> None:
            raise ConnectionRefusedError("connection refused")

        result = await _probe_cell(
            engines=[_MockEngine(True)],
            engine_health_fn=_refused_fn,
            cell_id="refused-cell",
            timeout=10.0,
        )
        assert result is False


# ===========================================================================
# RolloutHealthChecker (aggregate, with loop)
# ===========================================================================


class TestLoopReportsHealthy:
    @pytest.mark.anyio
    async def test_healthy_engines_reported(self) -> None:
        checker, collector = _make_checker(
            {"default": [_MockEngine(True), _MockEngine(True)]},
        )

        try:
            await asyncio.sleep(0.15)
            assert collector.latest("default") is True
        finally:
            await checker.shutdown()


class TestLoopReportsUnhealthy:
    @pytest.mark.anyio
    async def test_dead_engine0_reported(self) -> None:
        checker, collector = _make_checker(
            {"default": [_MockEngine(False), _MockEngine(True)]},
        )

        try:
            await asyncio.sleep(0.15)
            assert collector.latest("default") is False
        finally:
            await checker.shutdown()


class TestPause:
    @pytest.mark.anyio
    async def test_pause_stops_reporting(self) -> None:
        engines = [_MockEngine(True)]
        checker, collector = _make_checker({"default": engines})

        try:
            # Step 1: wait for initial healthy report
            await asyncio.sleep(0.15)
            assert collector.latest("default") is True

            # Step 2: pause, then kill engine
            checker.pause()
            engines[0].alive = False
            count_before = len(collector.reports)

            # Step 3: wait — no new reports should arrive
            await asyncio.sleep(0.15)
            assert len(collector.reports) == count_before
        finally:
            await checker.shutdown()


class TestResume:
    @pytest.mark.anyio
    async def test_resume_restarts_reporting(self) -> None:
        engines = [_MockEngine(True)]
        checker, collector = _make_checker({"default": engines})

        try:
            await asyncio.sleep(0.15)

            # Step 1: pause and kill engine
            checker.pause()
            engines[0].alive = False
            await asyncio.sleep(0.15)

            # Step 2: resume — should pick up the dead engine
            checker.resume()
            await asyncio.sleep(0.15)
            assert collector.latest("default") is False
        finally:
            await checker.shutdown()


class TestLoopSurvivesException:
    @pytest.mark.anyio
    async def test_broken_cell_reports_unhealthy(self) -> None:
        """H-2: If get_engines() raises, report_fn must still be called with
        is_healthy=False so the metric transitions to 0 (previously the
        exception silently swallowed the report, leaving the metric stale)."""

        def _exploding_engines() -> list[object]:
            raise RuntimeError("simulated failure")

        collector = _ReportCollector()
        checker = RolloutHealthChecker(
            cells=[
                CellEntry(cell_id="healthy", get_engines=lambda: [_MockEngine(True)]),
                CellEntry(cell_id="broken", get_engines=_exploding_engines),
            ],
            engine_health_fn=_engine_health_fn,
            report_fn=collector,
            check_interval=0.05,
        )

        try:
            await asyncio.sleep(0.15)

            # Step 1: healthy cell still reported
            assert collector.latest("healthy") is True

            # Step 2: broken cell reported as unhealthy (not silently skipped)
            assert collector.latest("broken") is False

            # Step 3: loop still running
            assert not checker._task.done()
        finally:
            await checker.shutdown()


class TestCheckOneCellExceptionReportsUnhealthy:
    @pytest.mark.anyio
    async def test_exception_in_check_still_reports_unhealthy(self) -> None:
        """_check_one_cell must always call report_fn even when an
        exception occurs. Previously, the exception path could skip
        the report_fn call, leaving the metric at a stale 'healthy' value."""

        async def _crashing_health_fn(engine: object) -> None:
            raise RuntimeError("unexpected health check failure")

        collector = _ReportCollector()
        checker = RolloutHealthChecker(
            cells=[
                CellEntry(cell_id="crash", get_engines=lambda: [_MockEngine(True)]),
            ],
            engine_health_fn=_crashing_health_fn,
            report_fn=collector,
            check_interval=0.05,
        )

        try:
            await asyncio.sleep(0.15)
            assert collector.latest("crash") is False
        finally:
            await checker.shutdown()


class TestNoneEngineIntegration:
    """RolloutHealthChecker correctly reports unhealthy when engines become None."""

    @pytest.mark.anyio
    async def test_none_engine_reported_unhealthy(self) -> None:
        """get_engines() returns [None] (killed engine) → reported as unhealthy."""
        collector = _ReportCollector()
        checker = RolloutHealthChecker(
            cells=[
                CellEntry(cell_id="killed", get_engines=lambda: [None]),  # type: ignore[list-item]
            ],
            engine_health_fn=_engine_health_fn,
            report_fn=collector,
            check_interval=0.05,
        )

        try:
            await asyncio.sleep(0.15)
            assert collector.latest("killed") is False
        finally:
            await checker.shutdown()

    @pytest.mark.anyio
    async def test_engine_becomes_none_after_initial_healthy(self) -> None:
        """Engine starts healthy, then gets replaced with None → transitions to unhealthy."""
        engines: list[object | None] = [_MockEngine(True)]
        collector = _ReportCollector()
        checker = RolloutHealthChecker(
            cells=[
                CellEntry(cell_id="dynamic", get_engines=lambda: engines),  # type: ignore[return-value]
            ],
            engine_health_fn=_engine_health_fn,
            report_fn=collector,
            check_interval=0.05,
        )

        try:
            # Step 1: initially healthy
            await asyncio.sleep(0.15)
            assert collector.latest("dynamic") is True

            # Step 2: engine gets killed (set to None)
            engines[0] = None
            await asyncio.sleep(0.15)
            assert collector.latest("dynamic") is False
        finally:
            await checker.shutdown()


class TestMultiCell:
    @pytest.mark.anyio
    async def test_independent_cell_results(self) -> None:
        checker, collector = _make_checker({
            "a0": [_MockEngine(True)],
            "a1": [_MockEngine(False)],
        })

        try:
            await asyncio.sleep(0.15)
            assert collector.latest("a0") is True
            assert collector.latest("a1") is False
        finally:
            await checker.shutdown()


class TestShutdown:
    @pytest.mark.anyio
    async def test_shutdown_immediately_is_safe(self) -> None:
        checker, _ = _make_checker(
            {"default": [_MockEngine(True)]},
            check_interval=10.0,
        )

        await checker.shutdown()
        assert checker._task.done()

    @pytest.mark.anyio
    async def test_cell_ids(self) -> None:
        checker, _ = _make_checker(
            {"a": [_MockEngine()], "b": [_MockEngine()]},
        )

        try:
            assert checker.cell_ids == ["a", "b"]
        finally:
            await checker.shutdown()
