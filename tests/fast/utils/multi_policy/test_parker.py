import asyncio

import pytest

from miles.utils.multi_policy import parker as parker_module
from miles.utils.multi_policy.parker import Parker


class TestWithAllParked:
    async def test_the_body_runs_only_once_every_follower_has_parked(self):
        """The body writes the run's checkpoint, so a follower still training would be caught mid round."""
        parker = Parker(num_followers=2)
        entered = False

        async def follower(delay: float) -> None:
            await asyncio.sleep(delay)
            await parker.maybe_park_follower()

        tasks = [asyncio.create_task(follower(0.05)), asyncio.create_task(follower(0.1))]

        async with parker.with_all_parked():
            entered = True
            assert all(not task.done() for task in tasks)

        assert entered
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    async def test_a_parked_follower_stays_parked_until_the_body_returns(self):
        """A follower that resumes early would train past the round the checkpoint claims to hold."""
        parker = Parker(num_followers=1)
        resumed = False

        async def follower() -> None:
            nonlocal resumed
            await parker.maybe_park_follower()
            resumed = True

        task = asyncio.create_task(follower())

        async with parker.with_all_parked():
            await asyncio.sleep(0.05)
            assert not resumed

        await asyncio.wait_for(task, timeout=5)
        assert resumed

    async def test_a_second_round_waits_for_the_first_one_to_disperse(self):
        """Entering while a follower is still leaving would count its stale arrival as this round's."""
        parker = Parker(num_followers=1)
        laps = 0

        async def follower() -> None:
            nonlocal laps
            while True:
                await parker.maybe_park_follower()
                laps += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(follower())
        try:
            async with parker.with_all_parked():
                laps_of_the_first_round = laps
            async with parker.with_all_parked():
                laps_of_the_second_round = laps
        finally:
            task.cancel()

        assert laps_of_the_second_round > laps_of_the_first_round

    async def test_waiting_for_a_follower_that_never_parks_fails_loudly(self, monkeypatch):
        """A silent hang here stalls the whole run, so the wait has to end as a failure instead."""
        monkeypatch.setattr(parker_module, "PARK_TIMEOUT_SECONDS", 0.0)
        parker = Parker(num_followers=1)

        with pytest.raises(AssertionError, match="ready 0, want 1"):
            async with parker.with_all_parked():
                pass


class TestMaybeParkFollower:
    async def test_a_follower_is_not_slowed_down_while_no_checkpoint_is_running(self):
        """Every round of every follower passes through here, so the common case must not yield."""
        parker = Parker(num_followers=1)
        finished = False

        async def follower() -> None:
            nonlocal finished
            await parker.maybe_park_follower()
            finished = True

        task = asyncio.create_task(follower())
        await asyncio.wait_for(task, timeout=5)

        assert finished


class TestTheFollowerAlwaysGivesTheLoopATurn:
    async def test_an_open_gate_still_suspends_the_follower(self):
        """An open gate that returns without suspending lets the follower's unbounded loop starve the loop."""
        parker = Parker(num_followers=1)
        other_task_ran = False

        async def other_task() -> None:
            nonlocal other_task_ran
            other_task_ran = True

        asyncio.create_task(other_task())

        await parker.maybe_park_follower()

        assert other_task_ran
