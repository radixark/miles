from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import logging
from argparse import Namespace

import pytest

from miles.rollout.submission_scheduler import (
    GroupLevelSubmission,
    SampleBackfillSubmission,
    make_submission_scheduler,
)
from miles.utils.types import Sample

GROUP_SIZE = 2


def _make_group(group_index: int) -> list[Sample]:
    return [
        Sample(group_index=group_index, index=group_index * 10 + i, prompt="prompt", response="ok")
        for i in range(GROUP_SIZE)
    ]


def _parked_task() -> asyncio.Task:
    return asyncio.create_task(asyncio.Event().wait())


async def _settle(times: int = 20) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class TestWaitForProgress:
    async def test_the_groups_that_are_still_running_stay_pending(self) -> None:
        """The driver resubmits from what it gets back, so a still-running group must not be reported done."""
        scheduler = GroupLevelSubmission()
        released = asyncio.Event()
        finishing = asyncio.create_task(released.wait())
        parked = _parked_task()

        waiting = asyncio.create_task(scheduler.wait_for_progress({finishing, parked}))
        await _settle()
        assert not waiting.done()

        released.set()
        done, pending = await waiting

        assert done == {finishing}
        assert pending == {parked}
        parked.cancel()


class TestSampleBackfillLeak:
    def test_a_leaked_in_flight_count_is_reported_before_it_is_reset(self, caplog) -> None:
        """Silently resetting the count would hide a generate function that never reports its samples."""
        scheduler = SampleBackfillSubmission(GROUP_SIZE)
        scheduler.on_submit([_make_group(1)])

        with caplog.at_level(logging.WARNING, logger="miles.rollout.submission_scheduler"):
            assert scheduler.has_capacity(pending_groups=0, group_budget=2)

        assert scheduler.samples_in_flight == 0
        assert "resetting to 0" in caplog.text


class TestMakeSubmissionScheduler:
    def test_the_sample_scheduler_is_built_with_the_configured_group_size(self) -> None:
        """A wrong group size drifts the sample budget by a whole group in one direction or the other."""
        args = Namespace(rollout_submission_granularity="sample", n_samples_per_prompt=5)

        assert make_submission_scheduler(args, default="group").group_size == 5

    def test_an_unknown_granularity_is_refused(self) -> None:
        """Falling back silently would pace the run in a way the user never asked for."""
        args = Namespace(rollout_submission_granularity="batch", n_samples_per_prompt=GROUP_SIZE)

        with pytest.raises(AssertionError, match="unknown submission granularity"):
            make_submission_scheduler(args, default="sample")
