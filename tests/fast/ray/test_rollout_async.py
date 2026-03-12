"""Tests for async behavior of ServerCell, RolloutServer, and RolloutManager recovery.

Verifies that:
- ServerCell.start()/stop() use native ``await`` (not ``ray.get``)
- Per-cell parallel recovery: multiple dead cells recover concurrently
- generate (to_thread) and start_cell (async) can overlap on the event loop
- Edge cases: no-op on healthy cells, FAILED status on exception

No GPU, no Ray cluster, no pytest-asyncio required — pure asyncio + mock.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from miles.ray.rollout import JobStatus, ServerCell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_future(value=None):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    fut.set_result(value)
    return fut


def _delayed_future(delay: float, value=None):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    async def _resolve():
        await asyncio.sleep(delay)
        if not fut.done():
            fut.set_result(value)

    asyncio.ensure_future(_resolve())
    return fut


def _mock_engine():
    engine = MagicMock()
    engine.shutdown.remote.return_value = _resolved_future()
    engine.release_memory_occupation.remote.return_value = _resolved_future()
    engine.resume_memory_occupation.remote.return_value = _resolved_future()
    return engine


def _mock_server_group(num_engines, dead_indices=None, needs_offload=False, start_delay=0.0):
    dead_indices = set(dead_indices or [])
    engines = [_mock_engine() if i not in dead_indices else None for i in range(num_engines)]

    group = MagicMock()
    group.all_engines = engines
    group.needs_offload = needs_offload

    def fake_start_engines(port_cursors=None, target_indices=None):
        target = set(target_indices) if target_indices is not None else {
            i for i, e in enumerate(engines) if e is None
        }
        handles = []
        for i in target:
            if group.all_engines[i] is None:
                group.all_engines[i] = _mock_engine()
                handles.append(_delayed_future(start_delay) if start_delay else _resolved_future())
        group.num_new_engines = len(handles)
        return handles, port_cursors or {}

    group.start_engines = fake_start_engines
    return group


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServerCellStart:
    def test_start_recovers_dead_engines(self):
        async def _run():
            group = _mock_server_group(2, dead_indices=[0, 1])
            cell = ServerCell("c0", group, [0, 1])
            cell._status = JobStatus.FAILED

            num = await cell.start()
            assert num == 2
            assert cell.status == JobStatus.RUNNING
            assert all(group.all_engines[i] is not None for i in [0, 1])

        asyncio.run(_run())

    def test_start_is_noop_when_healthy(self):
        async def _run():
            group = _mock_server_group(2, dead_indices=[])
            cell = ServerCell("c0", group, [0, 1])
            cell._status = JobStatus.RUNNING

            num = await cell.start()
            assert num == 0

        asyncio.run(_run())

    def test_start_handles_offload(self):
        async def _run():
            group = _mock_server_group(2, dead_indices=[0], needs_offload=True)
            cell = ServerCell("c0", group, [0])
            cell._status = JobStatus.FAILED

            num = await cell.start()
            assert num == 1
            new_engine = group.all_engines[0]
            new_engine.release_memory_occupation.remote.assert_called_once()
            new_engine.resume_memory_occupation.remote.assert_called_once()

        asyncio.run(_run())

    def test_start_sets_failed_on_exception(self):
        async def _run():
            group = MagicMock()
            group.all_engines = [None]
            group.needs_offload = False
            group.start_engines = MagicMock(side_effect=RuntimeError("boom"))

            cell = ServerCell("bad", group, [0])
            cell._status = JobStatus.FAILED

            with pytest.raises(RuntimeError, match="boom"):
                await cell.start()
            assert cell.status == JobStatus.FAILED

        asyncio.run(_run())


class TestServerCellStop:
    def test_stop_kills_engines(self):
        async def _run():
            group = _mock_server_group(2, dead_indices=[])
            cell = ServerCell("c0", group, [0, 1])
            cell._status = JobStatus.RUNNING

            await cell.stop()
            assert cell.status == JobStatus.STOPPED
            assert all(e is None for e in group.all_engines)

        asyncio.run(_run())

    def test_stop_is_noop_when_already_stopped(self):
        async def _run():
            group = _mock_server_group(1, dead_indices=[])
            cell = ServerCell("c0", group, [0])
            cell._status = JobStatus.STOPPED

            await cell.stop()
            assert cell.status == JobStatus.STOPPED
            group.all_engines[0].shutdown.remote.assert_not_called()

        asyncio.run(_run())


class TestPerCellParallelRecovery:
    def test_multiple_cells_recover_in_parallel(self):
        """If 3 cells each take ~0.1s, parallel recovery should take ~0.1s total."""

        async def _run():
            DELAY = 0.1
            N = 3
            group = _mock_server_group(N, dead_indices=list(range(N)), start_delay=DELAY)

            cells = []
            for i in range(N):
                c = ServerCell(f"c{i}", group, [i])
                c._status = JobStatus.FAILED
                cells.append(c)

            start = time.monotonic()
            results = await asyncio.gather(*[c.start() for c in cells])
            elapsed = time.monotonic() - start

            assert sum(results) == N
            assert all(c.status == JobStatus.RUNNING for c in cells)
            assert elapsed < DELAY * 2, (
                f"Cells recovered sequentially ({elapsed:.2f}s), expected parallel (~{DELAY}s)"
            )

        asyncio.run(_run())


class TestConcurrentGenerateAndStartCell:
    def test_generate_and_start_cell_overlap(self):
        """generate (to_thread) and start_cell (async) should run concurrently."""

        async def _run():
            DELAY = 0.1
            group = _mock_server_group(2, dead_indices=[1], start_delay=DELAY)
            cell = ServerCell("c1", group, [1])
            cell._status = JobStatus.FAILED

            async def fake_generate():
                await asyncio.to_thread(time.sleep, DELAY)
                return "gen_done"

            start = time.monotonic()
            gen_result, start_result = await asyncio.gather(fake_generate(), cell.start())
            elapsed = time.monotonic() - start

            assert gen_result == "gen_done"
            assert start_result == 1
            assert elapsed < DELAY * 1.8, (
                f"Not concurrent ({elapsed:.2f}s), expected overlap (~{DELAY}s)"
            )

        asyncio.run(_run())
