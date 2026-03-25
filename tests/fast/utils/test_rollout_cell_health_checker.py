from __future__ import annotations

import asyncio
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.rollout_cell_health_checker import CellEntry, RolloutCellHealthChecker


def _make_engine(*, healthy: bool = True, delay: float = 0.0) -> MagicMock:
    engine = MagicMock()
    if healthy:

        async def _remote() -> bool:
            if delay:
                await asyncio.sleep(delay)
            return True

        engine.health_generate.remote = _remote
    else:

        async def _remote_fail() -> bool:
            raise RuntimeError("engine dead")

        engine.health_generate.remote = _remote_fail
    return engine


def _make_cell(cell_id: str, engines: list[MagicMock]) -> CellEntry:
    return CellEntry(cell_id=cell_id, get_engines=lambda: engines)


class _MockHandle:
    def __init__(self) -> None:
        self.set_gauge_calls: list[dict] = []

    class _RemoteProxy:
        def __init__(self, owner: _MockHandle) -> None:
            self._owner = owner

        def remote(self, name: str, value: float, extra_labels: dict[str, str] | None = None) -> str:
            self._owner.set_gauge_calls.append({"name": name, "value": value, "extra_labels": extra_labels})
            return "ref"

    @property
    def set_gauge(self) -> _RemoteProxy:
        return self._RemoteProxy(self)


@pytest.fixture()
def mock_prom() -> _MockHandle:
    handle = _MockHandle()
    with patch("miles.utils.rollout_cell_health.get_prometheus", return_value=handle):
        yield handle


def _find_call(handle: _MockHandle, cell_id: str) -> dict | None:
    for c in handle.set_gauge_calls:
        if c["extra_labels"] and c["extra_labels"].get("cell_id") == cell_id:
            return c
    return None


class TestCheckOneCell:
    @pytest.mark.asyncio()
    async def test_healthy_cell_reports_alive(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 1.0
        assert c["name"] == "miles_rollout_cell_alive"

    @pytest.mark.asyncio()
    async def test_unhealthy_engine_reports_dead(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=False)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_none_engine_reports_dead(self, mock_prom: _MockHandle) -> None:
        cell = _make_cell("cell-0", [None])  # type: ignore[list-item]

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_empty_engines_list_reports_dead(self, mock_prom: _MockHandle) -> None:
        cell = CellEntry(cell_id="cell-0", get_engines=lambda: [])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_timeout_reports_dead(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True, delay=5.0)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(
            cells=[cell],
            session_id="sess-1",
            check_interval=100.0,
            timeout=0.01,
        )
        checker.start()
        await asyncio.sleep(0.1)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_multiple_cells_checked_independently(self, mock_prom: _MockHandle) -> None:
        engine_a = _make_engine(healthy=True)
        engine_b = _make_engine(healthy=False)
        cell_a = _make_cell("cell-a", [engine_a])
        cell_b = _make_cell("cell-b", [engine_b])

        checker = RolloutCellHealthChecker(
            cells=[cell_a, cell_b],
            session_id="sess-1",
            check_interval=100.0,
        )
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        ca = _find_call(mock_prom, "cell-a")
        cb = _find_call(mock_prom, "cell-b")
        assert ca is not None and ca["value"] == 1.0
        assert cb is not None and cb["value"] == 0.0

    @pytest.mark.asyncio()
    async def test_report_noop_when_prometheus_none(self) -> None:
        """When get_prometheus() returns None, _report() is a no-op (no error raised)."""
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        with patch("miles.utils.rollout_cell_health.get_prometheus", return_value=None):
            checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
            checker.start()
            await asyncio.sleep(0.05)
            await checker.shutdown()

    @pytest.mark.asyncio()
    async def test_extra_labels_contain_session_id(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="my-sess", check_interval=100.0)
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["extra_labels"]["session_id"] == "my-sess"


class TestPauseResume:
    @pytest.mark.asyncio()
    async def test_pause_reports_minus_one(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell_a = _make_cell("cell-a", [engine])
        cell_b = _make_cell("cell-b", [engine])

        checker = RolloutCellHealthChecker(cells=[cell_a, cell_b], session_id="sess-1", check_interval=0.01)
        checker.pause()
        checker.start()

        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0
        for call in mock_prom.set_gauge_calls:
            assert call["value"] == -1.0

        ca = _find_call(mock_prom, "cell-a")
        cb = _find_call(mock_prom, "cell-b")
        assert ca is not None
        assert cb is not None

    @pytest.mark.asyncio()
    async def test_resume_re_enables_check(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker.pause()
        checker.start()
        await asyncio.sleep(0.03)

        mock_prom.set_gauge_calls.clear()
        checker.resume()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0
        c = _find_call(mock_prom, "cell-0")
        assert c is not None
        assert c["value"] == 1.0


class TestLifecycle:
    @pytest.mark.asyncio()
    async def test_shutdown_clears_task(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker.start()
        await checker.shutdown()

        assert checker._task is None

    @pytest.mark.asyncio()
    async def test_double_start_is_idempotent(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker.start()
        first_task = checker._task

        checker.start()
        assert checker._task is first_task

        await checker.shutdown()

    @pytest.mark.asyncio()
    async def test_restart_after_shutdown(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=100.0)
        checker.start()
        await checker.shutdown()

        mock_prom.set_gauge_calls.clear()
        checker.start()
        await asyncio.sleep(0.05)
        await checker.shutdown()

        assert len(mock_prom.set_gauge_calls) > 0

    @pytest.mark.asyncio()
    async def test_check_interval_respected(self, mock_prom: _MockHandle) -> None:
        engine = _make_engine(healthy=True)
        cell = _make_cell("cell-0", [engine])

        checker_fast = RolloutCellHealthChecker(cells=[cell], session_id="sess-1", check_interval=0.01)
        checker_fast.start()
        await asyncio.sleep(0.1)
        await checker_fast.shutdown()
        fast_count = len(mock_prom.set_gauge_calls)

        mock_prom.set_gauge_calls.clear()

        checker_slow = RolloutCellHealthChecker(cells=[cell], session_id="sess-2", check_interval=0.05)
        checker_slow.start()
        await asyncio.sleep(0.1)
        await checker_slow.shutdown()
        slow_count = len(mock_prom.set_gauge_calls)

        assert fast_count > slow_count


class _MockServerGroup:
    def __init__(self, worker_type: str = "regular") -> None:
        self.worker_type = worker_type
        self.all_engines = [MagicMock()]

    @property
    def engines(self) -> list[MagicMock]:
        return self.all_engines


class _MockServer:
    def __init__(self, groups: list[_MockServerGroup]) -> None:
        self.server_groups = groups


class TestMaybeCreate:
    def test_maybe_create_returns_none_when_prometheus_disabled(self) -> None:
        args = Namespace(
            use_prometheus=False,
            session_id="sess",
            rollout_health_check_interval=30.0,
            rollout_health_check_timeout=30.0,
        )
        result = RolloutCellHealthChecker.maybe_create(servers={"srv0": _MockServer([_MockServerGroup()])}, args=args)
        assert result is None

    def test_maybe_create_returns_none_when_no_cells(self) -> None:
        args = Namespace(
            use_prometheus=True,
            session_id="sess",
            rollout_health_check_interval=30.0,
            rollout_health_check_timeout=30.0,
        )
        result = RolloutCellHealthChecker.maybe_create(servers={}, args=args)
        assert result is None

    @pytest.mark.asyncio()
    async def test_maybe_create_returns_checker_when_enabled(self, mock_prom: _MockHandle) -> None:
        args = Namespace(
            use_prometheus=True,
            session_id="sess",
            rollout_health_check_interval=30.0,
            rollout_health_check_timeout=30.0,
        )
        servers = {"srv0": _MockServer([_MockServerGroup()])}

        checker = RolloutCellHealthChecker.maybe_create(servers=servers, args=args)
        assert isinstance(checker, RolloutCellHealthChecker)
        assert checker._task is not None

        await checker.shutdown()
