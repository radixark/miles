from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_train as train
from miles.rollout.inference_rollout.inference_rollout_common import SubmissionScheduler
from miles.utils.types import Sample

GROUP_SIZE = 4


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=GROUP_SIZE,
        over_sampling_batch_size=1,
        rollout_sample_completion_backfill=False,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_group(group_index: int) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=Sample.Status.COMPLETED,
        )
        for i in range(GROUP_SIZE)
    ]


class FakeGenerateState:
    def __init__(self, args: Namespace):
        self.args = args
        self.sampling_params = {}
        self.aborted = False
        self.reset_count = 0

    def reset(self) -> None:
        self.aborted = False
        self.reset_count += 1


class Harness:
    """Drives generate_rollout_async with hand-controlled group and sample completions."""

    def __init__(self, monkeypatch, args: Namespace):
        self.args = args
        self.state = FakeGenerateState(args)
        self.submitted_group_indices: list[int] = []
        self.callbacks: list = []
        self._next_group_index = 0
        self._blockers: list[asyncio.Event] = []

        def fake_submit_generate_tasks(_state, samples, sample_done_callback=None):
            tasks = []
            for group in samples:
                self.submitted_group_indices.append(group[0].group_index)
                blocker = asyncio.Event()
                self._blockers.append(blocker)
                self.callbacks.append(sample_done_callback)
                tasks.append(asyncio.create_task(self._run_group(group, blocker)))
            return tasks

        async def fake_abort(_state, pendings, _rollout_id):
            _state.aborted = True
            for task in pendings:
                task.cancel()
            await asyncio.gather(*pendings, return_exceptions=True)
            return []

        async def noop_configure_sglang(_args):
            return None

        async def noop_recompute(*_args, **_kwargs):
            return None

        monkeypatch.setattr(train, "submit_generate_tasks", fake_submit_generate_tasks)
        monkeypatch.setattr(train, "abort", fake_abort)
        monkeypatch.setattr(train, "load_function", lambda _path: None)
        monkeypatch.setattr(train.dumper_utils, "configure_sglang", noop_configure_sglang)
        monkeypatch.setattr(train, "recompute_samples_rollout_logprobs_via_prefill", noop_recompute)

    async def _run_group(self, group, blocker):
        await blocker.wait()
        return group

    def data_source(self, num_groups: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_groups):
            self._next_group_index += 1
            groups.append(make_group(self._next_group_index))
        return groups

    def run(self):
        return asyncio.create_task(train.generate_rollout_async(self.state, 0, self.data_source))

    def finish_group(self, submission_index: int) -> None:
        self._blockers[submission_index].set()

    def finish_samples(self, submission_index: int, count: int) -> None:
        """Report `count` individual sample completions for a still-pending group."""
        callback = self.callbacks[submission_index]
        assert callback is not None, "backfill is disabled, no sample callback was wired"
        for _ in range(count):
            callback()


async def test_group_level_scheduling_is_the_default(monkeypatch):
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2))
    task = harness.run()
    await asyncio.sleep(0)

    # Two groups fill the batch; no sample callback is wired at all.
    assert harness.submitted_group_indices == [1, 2]
    assert harness.callbacks == [None, None]

    harness.finish_group(0)
    harness.finish_group(1)
    output, _ = await task
    assert [group[0].group_index for group in output.samples] == [1, 2]


async def test_backfill_submits_replacement_before_the_group_returns(monkeypatch):
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2, rollout_sample_completion_backfill=True))
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2]

    # A whole group's worth of samples finishes, spread across two still-pending groups.
    harness.finish_samples(0, GROUP_SIZE - 1)
    harness.finish_samples(1, 1)
    await asyncio.sleep(0.01)

    # A replacement group is in flight even though no group task has returned.
    assert harness.submitted_group_indices == [1, 2, 3]

    for i in range(3):
        harness.finish_group(i)
    output, _ = await task
    assert len(output.samples) == 2


async def test_backfill_does_not_oversubmit_below_one_group(monkeypatch):
    harness = Harness(monkeypatch, make_args(rollout_batch_size=2, rollout_sample_completion_backfill=True))
    task = harness.run()
    await asyncio.sleep(0)
    assert harness.submitted_group_indices == [1, 2]

    # One sample short of a full group: no replacement yet.
    harness.finish_samples(0, GROUP_SIZE - 1)
    await asyncio.sleep(0.01)
    assert harness.submitted_group_indices == [1, 2]

    harness.finish_group(0)
    harness.finish_group(1)
    output, _ = await task
    assert len(output.samples) == 2


class TestSubmissionScheduler:
    def test_disabled_counts_groups(self):
        scheduler = SubmissionScheduler(make_args())
        assert scheduler.sample_done_callback is None
        assert scheduler.has_capacity(pending_groups=1, group_budget=2)
        assert not scheduler.has_capacity(pending_groups=2, group_budget=2)

    def test_enabled_counts_samples(self):
        scheduler = SubmissionScheduler(make_args(rollout_sample_completion_backfill=True))
        assert scheduler.sample_done_callback is not None

        scheduler.on_submit([make_group(1), make_group(2)])
        assert scheduler.samples_in_flight == 2 * GROUP_SIZE
        # Full: pending_groups is irrelevant once samples are accounted for.
        assert not scheduler.has_capacity(pending_groups=0, group_budget=2)

        for _ in range(GROUP_SIZE - 1):
            scheduler.sample_done_callback()
        assert not scheduler.has_capacity(pending_groups=0, group_budget=2)

        scheduler.sample_done_callback()
        assert scheduler.has_capacity(pending_groups=0, group_budget=2)

    @pytest.mark.parametrize("enabled", [False, True])
    async def test_wait_for_progress_returns_on_group_completion(self, enabled):
        scheduler = SubmissionScheduler(make_args(rollout_sample_completion_backfill=enabled))
        blocker = asyncio.Event()

        async def group():
            await blocker.wait()

        task = asyncio.create_task(group())
        waiter = asyncio.create_task(scheduler.wait_for_progress({task}))
        await asyncio.sleep(0)
        assert not waiter.done()

        blocker.set()
        done, pending = await waiter
        assert done == {task}
        assert pending == set()

    async def test_wait_for_progress_returns_on_sample_completion(self):
        scheduler = SubmissionScheduler(make_args(rollout_sample_completion_backfill=True))
        never = asyncio.create_task(asyncio.Event().wait())

        scheduler.arm()
        waiter = asyncio.create_task(scheduler.wait_for_progress({never}))
        await asyncio.sleep(0)
        assert not waiter.done()

        scheduler.sample_done_callback()
        done, pending = await waiter
        assert done == set()  # the sample waiter is filtered out of the result
        assert pending == {never}

        never.cancel()
