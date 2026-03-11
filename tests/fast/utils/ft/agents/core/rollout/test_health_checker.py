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
# _probe_engine
# ---------------------------------------------------------------------------


class TestProbeEngine:
    @pytest.mark.anyio
    async def test_alive_engine_returns_true(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker._probe_engine(engine=_AliveEngine())

        assert result is True

    @pytest.mark.anyio
    async def test_timeout_returns_false(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
            timeout=0.01,
        )

        result = await checker._probe_engine(engine=_SlowEngine())

        assert result is False

    @pytest.mark.anyio
    async def test_exception_returns_false(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="a0",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker._probe_engine(engine=_BrokenEngine())

        assert result is False


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    @pytest.mark.anyio
    async def test_all_alive_returns_true(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="c0",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker.check_health(engines=[_AliveEngine(), _AliveEngine()])

        assert result is True

    @pytest.mark.anyio
    async def test_dead_engine0_returns_false(self) -> None:
        """Only engines[0] is probed; if it's dead the whole cell is dead."""
        checker = RolloutCellHealthChecker(
            cell_id="c1",
            engine_health_fn=_engine_method_health_checker,
        )

        result = await checker.check_health(engines=[_BrokenEngine(), _AliveEngine()])

        assert result is False

    @pytest.mark.anyio
    async def test_caches_result_for_is_healthy(self) -> None:
        checker = RolloutCellHealthChecker(
            cell_id="c0",
            engine_health_fn=_engine_method_health_checker,
        )
        assert checker.is_healthy() is False

        await checker.check_health(engines=[_AliveEngine()])

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
