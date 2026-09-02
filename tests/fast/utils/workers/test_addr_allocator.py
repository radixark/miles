from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tests.fast.ray.rollout.conftest import fake_engine

from miles.utils.workers.addr_allocator import PortAllocator


def _make_probe_fail(engine: MagicMock) -> None:
    async def _fail(start_port: int = 15000, count: int = 1):
        raise RuntimeError("free port probe failed")

    engine._get_free_port_block.remote.side_effect = _fail


def _make_probe_slow(engine: MagicMock, *, delay_seconds: float) -> None:
    async def _slow(start_port: int = 15000, count: int = 1):
        await asyncio.sleep(delay_seconds)
        port = max(engine._port_cursor, start_port)
        engine._port_cursor = port + count
        return port

    engine._get_free_port_block.remote.side_effect = _slow


class TestPortAllocator:
    def test_a_fresh_allocator_has_no_cursors(self):
        """A brand new allocator starts with no per-node cursors."""
        c = PortAllocator()
        assert c._next_port_of_ip == {}

    async def test_alloc_advances_the_cursor_of_its_node(self):
        """Two allocations on the same node must hand out non-overlapping ports."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = await cursors.alloc(engine, node_ip="10.0.0.1")
        second = await cursors.alloc(engine, node_ip="10.0.0.1")
        assert second > first
        assert cursors._next_port_of_ip["10.0.0.1"] == second + 1

    async def test_alloc_starts_from_the_base_port_on_an_unseen_node(self):
        """A node with no cursor yet starts at the base port, away from ray's range."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        assert await cursors.alloc(engine, node_ip="10.0.0.1") == 20000

    async def test_alloc_consecutive_reserves_a_whole_block(self):
        """A consecutive=N allocation must move this node's cursor past the entire block."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        first = await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=5)
        assert cursors._next_port_of_ip["10.0.0.1"] == first + 5

    async def test_a_skipped_candidate_range_advances_from_the_returned_block(self):
        """When the actor skips occupied candidates, the cursor must advance from the returned port plus the block size."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=25000)
        first = await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4)
        assert first == 25000
        assert cursors._next_port_of_ip["10.0.0.1"] == 25004

    async def test_a_failed_probe_does_not_advance_the_node_cursor(self):
        """A probe that fails on result retrieval propagates its error and leaves this node's cursor untouched."""
        cursors = PortAllocator()
        cursors._next_port_of_ip["10.0.0.1"] = 20005
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=3)
        assert cursors._next_port_of_ip == {"10.0.0.1": 20005}

    async def test_a_failed_probe_does_not_create_a_cursor_for_an_unseen_node(self):
        """A probe that fails on result retrieval leaves a never-seen node without any cursor."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=3)
        assert cursors._next_port_of_ip == {}

    async def test_a_failed_probe_releases_the_lock_for_the_next_allocation(self):
        """A failed probe must not leave the allocator locked, so a later allocation still goes through."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_fail(engine)
        with pytest.raises(RuntimeError, match="free port probe failed"):
            await cursors.alloc(engine, node_ip="10.0.0.1")

        healthy = fake_engine(host="10.0.0.1", port_seed=0)
        assert await asyncio.wait_for(cursors.alloc(healthy, node_ip="10.0.0.1"), timeout=1.0) == 20000

    async def test_alloc_tracks_nodes_independently(self):
        """Each node ip owns its own cursor."""
        cursors = PortAllocator()
        engine_a = fake_engine(host="10.0.0.1", port_seed=0)
        engine_b = fake_engine(host="10.0.0.2", port_seed=0)
        await cursors.alloc(engine_a, node_ip="10.0.0.1")
        await cursors.alloc(engine_b, node_ip="10.0.0.2")
        assert set(cursors._next_port_of_ip.keys()) == {"10.0.0.1", "10.0.0.2"}

    async def test_a_cursor_past_the_last_port_restarts_at_the_base_port(self):
        """Ports are never reclaimed, so a long fault-tolerance run walks the cursor off the end."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65535

        assert await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 20000

    async def test_a_cursor_that_still_fits_is_left_alone(self):
        """Resetting early would hand out ports that are still in use by live cells."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        cursors._next_port_of_ip["10.0.0.1"] = 65530

        assert await cursors.alloc(engine, node_ip="10.0.0.1", consecutive=4) == 65530

    async def test_concurrent_allocations_on_one_node_never_overlap(self):
        """Allocations racing on the same node while a probe is in flight must still get disjoint blocks."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_slow(engine, delay_seconds=0.01)

        ports = await asyncio.gather(*[cursors.alloc(engine, node_ip="10.0.0.1", consecutive=2) for _ in range(5)])

        assert sorted(ports) == [20000, 20002, 20004, 20006, 20008]
        assert cursors._next_port_of_ip["10.0.0.1"] == 20010

    async def test_a_probe_in_flight_does_not_block_the_event_loop(self):
        """The probe is awaited, so unrelated coroutines keep running while it is pending."""
        cursors = PortAllocator()
        engine = fake_engine(host="10.0.0.1", port_seed=0)
        _make_probe_slow(engine, delay_seconds=0.05)
        ticks: list[int] = []

        async def _tick() -> None:
            for i in range(3):
                ticks.append(i)
                await asyncio.sleep(0.01)

        await asyncio.gather(cursors.alloc(engine, node_ip="10.0.0.1"), _tick())

        assert ticks == [0, 1, 2]
