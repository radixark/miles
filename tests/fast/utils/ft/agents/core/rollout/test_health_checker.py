from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.agents.core.rollout.health_checker import (
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
        cells={cid: (lambda es=es: es) for cid, es in engines_by_cell.items()},
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
    async def test_broken_cell_does_not_crash_loop(self) -> None:
        """If one cell's engine list callback raises, the loop continues."""

        def _exploding_engines() -> list[object]:
            raise RuntimeError("simulated failure")

        collector = _ReportCollector()
        checker = RolloutHealthChecker(
            cells={
                "healthy": lambda: [_MockEngine(True)],
                "broken": _exploding_engines,
            },
            engine_health_fn=_engine_health_fn,
            report_fn=collector,
            check_interval=0.05,
        )

        try:
            await asyncio.sleep(0.15)

            # Step 1: healthy cell still reported
            assert collector.latest("healthy") is True

            # Step 2: broken cell never reported (exception caught)
            assert collector.latest("broken") is None

            # Step 3: loop still running
            assert not checker._task.done()
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
