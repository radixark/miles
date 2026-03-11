from __future__ import annotations

import asyncio

import pytest

from miles.utils.ft.agents.core.rollout.health_checker import RolloutCellHealthChecker


async def _engine_method_health_checker(engine: object) -> None:
    await engine.health_check()  # type: ignore[attr-defined]


class _AliveEngine:
    async def health_check(self) -> None:
        pass


class _SlowEngine:
    async def health_check(self) -> None:
        await asyncio.sleep(999)


class _BrokenEngine:
    async def health_check(self) -> None:
        raise ConnectionError("engine crashed")


# ---------------------------------------------------------------------------
# _check_single_engine (migrated from test_rollout_cell_agent.py)
# ---------------------------------------------------------------------------


class TestCheckSingleEngine:
    """Tests the real _check_single_engine."""

    @pytest.mark.anyio
    async def test_alive_engine_returns_true(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker._check_single_engine(engine=_AliveEngine(), index=0)

        assert result is True

    @pytest.mark.anyio
    async def test_timeout_returns_false(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
            timeout=0.01,
        )

        result = await checker._check_single_engine(engine=_SlowEngine(), index=0)

        assert result is False

    @pytest.mark.anyio
    async def test_exception_returns_false(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker._check_single_engine(engine=_BrokenEngine(), index=0)

        assert result is False


# ---------------------------------------------------------------------------
# check_health (aggregation)
# ---------------------------------------------------------------------------


class TestCheckHealth:
    @pytest.mark.anyio
    async def test_all_alive_returns_healthy(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="c0",
            engine_health_fn=_engine_method_health_checker,
        )
        engines: list[object] = [_AliveEngine(), _AliveEngine()]

        result = await checker.check_health(engines=engines)

        assert result.is_healthy is True
        assert result.alive_engines == 2
        assert result.total_engines == 2
        assert result.dead_engine_indices == ()
        assert result.cell_id == "c0"

    @pytest.mark.anyio
    async def test_dead_engine0_returns_all_dead(self) -> None:
        """Only engines[0] is probed; if it's dead the whole cell is dead."""
        checker = RolloutCellHealthChecker(
            cell_id="c1",
            engine_health_fn=_engine_method_health_checker,
        )
        engines: list[object] = [_BrokenEngine(), _AliveEngine(), _AliveEngine()]

        result = await checker.check_health(engines=engines)

        assert result.is_healthy is False
        assert result.alive_engines == 0
        assert result.dead_engine_indices == (0, 1, 2)

    @pytest.mark.anyio
    async def test_caches_last_result(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="c0",
            engine_health_fn=_engine_method_health_checker,
        )
        assert checker.last_result is None

        result = await checker.check_health(engines=[_AliveEngine()])

        assert checker.last_result is result
        assert checker.is_healthy() is True

    @pytest.mark.anyio
    async def test_empty_engines_raises(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="empty",
            engine_health_fn=_engine_method_health_checker,
        )

        with pytest.raises(ValueError, match="engines must not be empty"):
            await checker.check_health(engines=[])


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


class TestInvalidate:
    @pytest.mark.anyio
    async def test_invalidate_clears_cached_result(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="c0",
            engine_health_fn=_engine_method_health_checker,
        )
        await checker.check_health(engines=[_AliveEngine()])
        assert checker.is_healthy() is True

        checker.invalidate()

        assert checker.is_healthy() is False
        assert checker.last_result is None
