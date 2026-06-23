import asyncio

import pytest

from miles.utils.clock import FakeClock, RealClock


class TestRealClock:
    def test_time_returns_current_time(self):
        clock = RealClock()
        import time

        before = time.time()
        result = clock.time()
        after = time.time()
        assert before <= result <= after

    async def test_sleep_actually_waits(self):
        clock = RealClock()
        import time

        start = time.time()
        await clock.sleep(0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.04


class TestFakeClockTime:
    def test_initial_time(self):
        clock = FakeClock(start=100.0)
        assert clock.time() == 100.0

    def test_default_start_is_zero(self):
        clock = FakeClock()
        assert clock.time() == 0.0

    async def test_elapse_advances_time(self):
        clock = FakeClock(start=10.0)
        await clock.elapse(5.0)
        assert clock.time() == 15.0

    async def test_multiple_elapse_accumulate(self):
        clock = FakeClock()
        await clock.elapse(3.0)
        await clock.elapse(7.0)
        assert clock.time() == 10.0

    async def test_elapse_zero_is_allowed(self):
        clock = FakeClock(start=5.0)
        await clock.elapse(0.0)
        assert clock.time() == 5.0

    async def test_elapse_negative_raises(self):
        clock = FakeClock()
        with pytest.raises(AssertionError, match="negative"):
            await clock.elapse(-1.0)


class TestFakeClockSleep:
    async def test_sleep_zero_returns_immediately(self):
        clock = FakeClock()
        completed = False

        async def task() -> None:
            nonlocal completed
            await clock.sleep(0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert completed

    async def test_sleep_blocks_until_elapse(self):
        clock = FakeClock()
        completed = False

        async def task() -> None:
            nonlocal completed
            await clock.sleep(10.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert not completed

        await clock.elapse(10.0)
        assert completed

    async def test_sleep_does_not_resolve_before_target(self):
        clock = FakeClock()
        completed = False

        async def task() -> None:
            nonlocal completed
            await clock.sleep(10.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(9.9)
        assert not completed

        await clock.elapse(0.1)
        assert completed

    async def test_multiple_sleeps_resolve_in_order(self):
        clock = FakeClock()
        order: list[str] = []

        async def task_a() -> None:
            await clock.sleep(5.0)
            order.append("a")

        async def task_b() -> None:
            await clock.sleep(10.0)
            order.append("b")

        async def task_c() -> None:
            await clock.sleep(3.0)
            order.append("c")

        asyncio.create_task(task_a())
        asyncio.create_task(task_b())
        asyncio.create_task(task_c())
        await asyncio.sleep(0)

        await clock.elapse(5.0)
        assert order == ["c", "a"]

        await clock.elapse(5.0)
        assert order == ["c", "a", "b"]

    async def test_same_target_time_all_resolve(self):
        clock = FakeClock()
        count = 0

        async def task() -> None:
            nonlocal count
            await clock.sleep(5.0)
            count += 1

        for _ in range(3):
            asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(5.0)
        assert count == 3

    async def test_overshoot_resolves_all_pending(self):
        clock = FakeClock()
        order: list[int] = []

        for delay in [1, 5, 10]:
            d = delay

            async def task(d: int = d) -> None:
                await clock.sleep(d)
                order.append(d)

            asyncio.create_task(task())
        await asyncio.sleep(0)

        await clock.elapse(100.0)
        assert sorted(order) == [1, 5, 10]

    async def test_sleep_negative_returns_immediately(self):
        clock = FakeClock()
        completed = False

        async def task() -> None:
            nonlocal completed
            await clock.sleep(-1.0)
            completed = True

        asyncio.create_task(task())
        await asyncio.sleep(0)
        assert completed


class TestFakeClockPendingCount:
    async def test_no_pending_initially(self):
        clock = FakeClock()
        assert clock.pending_count == 0

    async def test_pending_count_tracks_sleeps(self):
        clock = FakeClock()

        asyncio.create_task(clock.sleep(5.0))
        asyncio.create_task(clock.sleep(10.0))
        await asyncio.sleep(0)
        assert clock.pending_count == 2

        await clock.elapse(5.0)
        assert clock.pending_count == 1

        await clock.elapse(5.0)
        assert clock.pending_count == 0


class TestFakeClockChainedSleeps:
    async def test_sequential_sleeps_in_coroutine(self):
        """A coroutine that does multiple sleeps in sequence."""
        clock = FakeClock()
        checkpoints: list[float] = []

        async def task() -> None:
            checkpoints.append(clock.time())
            await clock.sleep(5.0)
            checkpoints.append(clock.time())
            await clock.sleep(3.0)
            checkpoints.append(clock.time())

        asyncio.create_task(task())
        await asyncio.sleep(0)

        assert checkpoints == [0.0]

        await clock.elapse(5.0)
        assert checkpoints == [0.0, 5.0]

        await clock.elapse(3.0)
        assert checkpoints == [0.0, 5.0, 8.0]

    async def test_periodic_loop(self):
        """Simulates a periodic loop like SimpleHealthChecker._loop."""
        clock = FakeClock()
        ticks: list[float] = []

        async def loop() -> None:
            await clock.sleep(10.0)
            while len(ticks) < 3:
                ticks.append(clock.time())
                await clock.sleep(5.0)

        task = asyncio.create_task(loop())
        await asyncio.sleep(0)

        # Step 1: first_wait=10
        assert ticks == []
        await clock.elapse(10.0)
        assert ticks == [10.0]

        # Step 2: interval=5
        await clock.elapse(5.0)
        assert ticks == [10.0, 15.0]

        # Step 3: interval=5
        await clock.elapse(5.0)
        assert ticks == [10.0, 15.0, 20.0]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
