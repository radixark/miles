from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from tests.fast.ray.rollout.test_rollout_executor import make_executor

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import argparse
import asyncio
import copy
from argparse import Namespace
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import RolloutHoldingsSnapshotEvent, SampleOwner
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

N_SAMPLES_PER_PROMPT = 2


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
    reward: float = 1,
) -> list[Sample]:
    versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
        for version in weight_versions or []
    ]
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=reward,
            status=status,
            weight_versions=list(versions),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        reward_key=None,
        async_max_concurrent_samples=None,
        async_data_buffer_capacity_factor=1000.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_request_timeout_secs=14400,
        eval_num_gpus=0,
        save=None,
        load=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        for sample in group:
            sample.status = Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    return fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_eval_without_fleet_pauses_producer(monkeypatch):
    """Shared-engine eval: producer submissions pause during eval and resume after."""
    release = asyncio.Event()
    entered = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        entered.set()
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch, make_args(rollout_batch_size=2, eval_num_gpus=0), data_source, generate=blocking_generate
    )

    eval_started = asyncio.Event()
    eval_release = asyncio.Event()
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}

    async def fake_run_eval_datasets(state, cache):
        assert state is fn.state  # shared-engine eval uses the train state
        eval_started.set()
        await eval_release.wait()
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    # Start the producer via a train call, then run eval concurrently.
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await entered.wait()
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await drain
    assert data_source.num_get_calls == submitted_before_eval

    eval_release.set()
    output = await eval_task
    assert output.data == eval_results

    # Producer resumes and the train drain completes.
    assert (await drain).samples


async def test_eval_runs_on_dedicated_fleet(monkeypatch):
    """RolloutManager (not the fn) decides fleet-vs-shared and builds the fleet's
    GenerateState; it hands it in via RolloutFnEvalInput.generate_state. The fn must
    use that state as-is (not self.state) and must not touch the producer/data_source.
    Building/caching the fleet state itself is RolloutExecutorEvalFleet's job, covered in
    tests/fast/rollout/test_checkpoint_eval.py.
    """
    args = make_args(eval_num_gpus=1, eval_num_gpus_per_engine=1)
    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, args, data_source)

    fleet_state = FakeGenerateState(args)
    eval_results = {"fake_ds": {"rewards": [1.0], "truncated": [False], "samples": []}}
    seen_states = []

    async def fake_run_eval_datasets(state, cache):
        seen_states.append(state)
        return eval_results

    monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

    output = await fn(RolloutFnEvalInput(rollout_id=0, generate_state=fleet_state, weight_version="0"))

    assert output.data == eval_results
    assert seen_states == [fleet_state]  # used the fleet's state, not fn.state
    # Eval must not start the producer or consume training prompts.
    assert fn._worker is None
    assert data_source.num_get_calls == 0


def abort_once_generate(aborted_group_index: int):
    seen: set[int] = set()

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
        first_time = group[0].group_index == aborted_group_index and group[0].group_index not in seen
        seen.add(group[0].group_index)
        for sample in group:
            sample.status = Sample.Status.ABORTED if first_time else Sample.Status.COMPLETED
        return group

    return generate


class TestRetryBuffer:
    async def test_an_aborted_group_is_resubmitted_from_the_retry_buffer(self, monkeypatch):
        """The rollout function owns its retry buffer, and the read-only data source refuses add_samples."""
        aborted = make_group(1)
        data_source = FakeDataSource(scripted=[aborted])
        args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
        fn = make_fn(monkeypatch, args, data_source, generate=abort_once_generate(1))

        output = await fn(RolloutFnTrainInput(rollout_id=0))

        assert output.samples[0][0].group_index == 1
        assert data_source.num_get_calls == 1
        assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1

    async def test_a_recycled_group_is_reset_before_it_goes_back_into_the_buffer(self, monkeypatch):
        """Generated tokens are written into the prompt samples in place, so a resubmission must clear them."""
        args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        group = make_group(1, weight_versions=["5"])

        fn._recycle(group, reason=data_buffer.UnusedReason.ABORTED, trainer_model_id=None)

        [pending] = fn._retry_buffer
        assert [sample.index for sample in pending.samples] == [sample.index for sample in group]
        assert all(sample.response == "" and sample.weight_versions == [] for sample in pending.samples)
        assert all(sample.response == "ok" and sample.weight_versions for sample in group)

    async def test_the_next_submission_prefers_the_retry_buffer_over_the_data_source(self, monkeypatch):
        """A recycled prompt that queued behind the whole dataset would come back an epoch later."""
        data_source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)
        recycled = make_group(7)
        fn._retry_buffer.append(fully_async._PendingPrompt(samples=recycled))

        fn._submit_one_group()

        assert data_source.num_get_calls == 0
        assert [pending.samples for pending in fn._in_flight.values()] == [recycled]
        assert not fn._retry_buffer


async def test_stale_group_recycled(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = [
                        WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
                        for version in data_source_fresh_versions
                    ]
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    args = make_args(rollout_batch_size=1, max_weight_staleness=2, async_unused_samples_handler="retry")

    async def regenerate_with_fresh_weights(state, group, **kwargs):
        for sample in group:
            sample.status = Sample.Status.COMPLETED
            if not sample.weight_versions:
                sample.weight_versions = make_group(0, weight_versions=["10"])[0].weight_versions
        return group

    fn = make_fn(monkeypatch, args, data_source, generate=regenerate_with_fresh_weights)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert all(sample.response == "ok" and sample.oldest_weight_version == 5 for sample in stale)
    assert [sample.index for sample in output.samples[0]] == [sample.index for sample in stale]
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 0


async def test_stale_group_dropped_by_default(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert not fn._retry_buffer
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        entered.set()
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await entered.wait()
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_worker_failure_beats_queued_groups(monkeypatch):
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        raise RuntimeError("generation exploded")

    fn._output = make_buffer()[0]
    group = make_group(1)
    await fn._output.put(data_buffer.DataBufferInput(prompt_group=group, group=group))
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            for sample in group:
                sample.status = Sample.Status.COMPLETED
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample, status=Sample.Status.COMPLETED)])
        return expanded

    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert all(isinstance(sample, Sample) for sample in submitted[1])
    assert len(submitted) > 1
    assert len(output.samples) == 1


def reject_group_1(args, group, **kwargs):
    keep = group[0].group_index != 1
    return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    args = make_args(
        rollout_batch_size=1,
        dynamic_sampling_filter_path=f"{__name__}.reject_group_1",
        async_unused_samples_handler="retry",
    )
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Dropped even with handler="retry": filter rejections bypass the unused handler.
    assert not fn._retry_buffer
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_staleness_filter_off_before_the_first_weight_update(monkeypatch):
    """weight_version is None until the trainer pushes weights; staleness is unknown, not zero."""
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=0), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert not fn._retry_buffer
    assert output.samples[0][0].group_index == 1
    assert "rollout/fully_async/max_staleness" not in output.metrics


# ── DataBuffer: staleness-bounded buffering ─────────────────────────


def make_buffer(max_groups=None, max_staleness=None):
    unused = []
    args = make_args(
        rollout_batch_size=1,  # capacity is factor * batch size; batch size 1 makes it count groups
        async_data_buffer_capacity_factor=max_groups or 1000.0,
        max_weight_staleness=max_staleness,
    )
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=args, unused_handler_fn=lambda group, *, reason, trainer_model_id: unused.append(group)
        )
    )
    return buffer, unused


async def put_group(buffer, group):
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(prompt_group=group, group=group))


async def test_buffer_reports_unfiltered_raw_reward_across_kept_and_dropped():
    """The accepted-only raw_reward is conditioned by the filter, so this mean must still see dropped groups."""
    args = make_args(rollout_batch_size=1, dynamic_sampling_filter_path=f"{__name__}.reject_group_1")
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=args, unused_handler_fn=lambda group, *, reason, trainer_model_id: None
        )
    )

    await put_group(buffer, make_group(1, reward=0))
    await put_group(buffer, make_group(2, reward=1))

    metrics = buffer.get_metrics()
    assert metrics["rollout/raw_reward_unfiltered"] == 0.5
    assert metrics["rollout/dynamic_filter/drop_rejected"] == 1
    assert "rollout/raw_reward_unfiltered" not in buffer.get_metrics()


async def test_buffer_blocks_producer_when_full():
    buffer, _ = make_buffer(max_groups=2)
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    blocked = asyncio.create_task(put_group(buffer, make_group(3)))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    assert (await buffer.get(num_groups=1))[0].group[0].group_index == 1
    await blocked
    assert (await buffer.get(num_groups=1))[0].group[0].group_index == 2
    assert (await buffer.get(num_groups=1))[0].group[0].group_index == 3


async def test_buffer_get_ignores_unknown_context_keys():
    """get(**context) lets the driver add keys without breaking existing buffers."""
    buffer, _ = make_buffer()
    await put_group(buffer, make_group(1))

    assert (await buffer.get(num_groups=1, current_version=1, some_future_key=2))[0].group[0].group_index == 1


async def test_buffer_get_skips_groups_stale_at_consumption_time():
    """Both groups were fresh when buffered; only the version passed to get() decides."""
    buffer, unused = make_buffer(max_staleness=2)
    stale = make_group(1, weight_versions=["5"])
    await put_group(buffer, stale)
    await put_group(buffer, make_group(2, weight_versions=["9"]))

    assert (await buffer.get(num_groups=1, current_version=10))[0].group[0].group_index == 2
    assert unused == [stale]
    assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1


async def test_buffer_staleness_metrics():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    assert "rollout/fully_async/buffer_avg_staleness" not in buffer.get_metrics()  # engine version never seen

    await put_group(buffer, make_group(2, weight_versions=["6"]))
    await put_group(buffer, make_group(3, weight_versions=["8"]))
    await buffer.get(num_groups=1, current_version=10)  # pops group 1 and tracks the engine version clock
    metrics = buffer.get_metrics()
    assert metrics["rollout/fully_async/avg_staleness"] == 6.0  # consumed group 1: 10 - 4
    assert metrics["rollout/fully_async/buffer_avg_staleness"] == 3.0  # buffered groups 2, 3: (4 + 2) / 2
    assert metrics["rollout/fully_async/buffer_max_staleness"] == 4


def make_multi_policy_group(group_index: int, *trainer_model_ids: str) -> list[Sample]:
    """One prompt group whose samples train different policy models, as a multi policy generate fn returns."""
    group = make_group(group_index)
    for sample, trainer_model_id in zip(group, trainer_model_ids, strict=True):
        sample.trainer_model_id = trainer_model_id
    return group


def make_multi_buffer(*model_ids: str, max_staleness=None, paths_per_model=None):
    unused = []
    args = make_args(
        rollout_batch_size=1,
        async_data_buffer_capacity_factor=1000.0,
        max_weight_staleness=max_staleness,
        megatron_config=encode_megatron_config(*model_ids),
        custom_async_data_buffer_path_per_model=paths_per_model,
    )
    buffer = data_buffer.DefaultMultiDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=args, unused_handler_fn=lambda group, *, reason, trainer_model_id: unused.append(group)
        )
    )
    return buffer, unused


class TestPerPolicyQueues:
    async def test_a_group_of_two_policies_lands_in_a_queue_of_each(self):
        """One generate call feeds both policies, and a shared queue would hand them each other's samples."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 1
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_a_policy_only_ever_drains_its_own_samples(self):
        """Training a policy on another policy's responses is the failure this queue split exists to stop."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        [entry] = await buffer.get(num_groups=1, trainer_model_id="verifier")

        assert [sample.trainer_model_id for sample in data_buffer.iter_samples(entry.group)] == ["verifier"]

    async def test_a_policy_waits_for_its_own_queue_instead_of_taking_from_another(self):
        """A policy that consumed a queue it does not own would starve the policy that does."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "solver"))

        waiting = asyncio.create_task(buffer.get(num_groups=1, trainer_model_id="verifier"))
        await asyncio.sleep(0.01)

        assert not waiting.done()
        waiting.cancel()

    async def test_an_untagged_sample_is_refused_at_the_split(self):
        """Every sample of a multi policy run is stamped by the generate function, so an unstamped one is a bug."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="must stamp every sample"):
            await put_group(buffer, make_group(1))

    async def test_a_sample_of_an_unknown_policy_is_refused(self):
        """Its groups would queue up in a buffer no trainer ever drains, and the run would simply stall."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await put_group(buffer, make_multi_policy_group(1, "solver", "reviewer"))

    async def test_the_prompt_group_of_a_split_group_stays_whole(self):
        """Recycling a rejected group resubmits prompts, which are not owned by either policy."""
        buffer, unused = make_multi_buffer("solver", "verifier", max_staleness=0)
        group = make_group(1, weight_versions=["1"])
        group[0].trainer_model_id, group[1].trainer_model_id = "solver", "verifier"

        await put_group(buffer, group)
        drained = asyncio.create_task(buffer.get(num_groups=1, current_version=9, trainer_model_id="solver"))
        await asyncio.sleep(0.01)

        assert unused == [group]
        drained.cancel()

    async def test_getting_for_a_policy_this_run_does_not_train_is_refused(self):
        """A typo in the trainer's model id would wait forever on a queue that is never fed."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await buffer.get(num_groups=1, trainer_model_id="reviewer")

    async def test_every_policy_of_the_config_gets_a_queue_of_its_own(self):
        """The queues are built from --megatron-config, so a policy missing one has nowhere to put its groups."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert buffer.get_metrics("solver")["rollout/fully_async/queue_size"] == 0
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 0

    async def test_a_policy_reads_and_resets_only_its_own_metric_window(self):
        """Draining one policy used to read and clear every policy's counters, moving them onto the wrong curve."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "verifier"))

        solver_metrics = buffer.get_metrics("solver")

        assert set(solver_metrics) == {key for key in solver_metrics if not key.startswith(("solver/", "verifier/"))}
        assert buffer.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1

    async def test_an_inner_buffer_is_told_which_policy_asks_for_a_group(self):
        """A custom per policy buffer cannot filter or account by policy if the composite eats that context."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        seen: list[dict] = []

        class _RecordingInner:
            async def get(self, **context):
                seen.append(context)
                return ["entry"]

        buffer._inners["solver"] = _RecordingInner()

        assert await buffer.get(num_groups=1, current_version=4, trainer_model_id="solver") == ["entry"]
        assert seen == [{"num_groups": 1, "current_version": 4, "trainer_model_id": "solver"}]


def make_tagged_sample(index: int, trainer_model_id: str | None) -> Sample:
    sample = make_group(index)[0]
    sample.trainer_model_id = trainer_model_id
    return sample


def split(group: data_buffer.Group, *, prompt_group=None) -> dict:
    return data_buffer._split_by_trainer_model_id(
        data_buffer.DataBufferInput(prompt_group=prompt_group if prompt_group is not None else [], group=group)
    )


class TestSplitByTrainerModelId:
    def test_a_group_of_one_policy_lands_whole_in_that_policy_alone(self):
        """The common group is a group of one policy, and every sample of it must reach that policy."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, second])

        assert list(ans) == ["solver"]
        assert ans["solver"].group == [first, second]

    def test_a_mixed_group_becomes_one_input_per_policy(self):
        """Each policy trains on its own samples only, so the group has to be cut along the tags."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([solver, verifier])

        assert list(ans) == ["solver", "verifier"]
        assert ans["solver"].group == [solver]
        assert ans["verifier"].group == [verifier]

    def test_the_samples_of_a_policy_keep_the_order_they_arrived_in(self):
        """Order carries the trajectory, and a reordered group trains on a reshuffled conversation."""
        first, second = make_tagged_sample(1, "solver"), make_tagged_sample(2, "solver")

        ans = split([first, make_tagged_sample(3, "verifier"), second])

        assert ans["solver"].group == [first, second]

    def test_a_sub_group_of_a_multi_sample_trajectory_is_filtered_per_policy(self):
        """A generate function may return several samples per trajectory, and they need not share a policy."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver, verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_a_sub_group_no_sample_of_which_survives_is_dropped(self):
        """An empty sub-group is a trajectory with no samples, which the consumers cannot make sense of."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        ans = split([[solver], [verifier]])

        assert ans["solver"].group == [[solver]]
        assert ans["verifier"].group == [[verifier]]

    def test_every_trajectory_of_the_group_is_split_on_its_own(self):
        """One finished group carries several trajectories, and each of them may mix policies differently."""
        first, second, third = (
            make_tagged_sample(1, "solver"),
            make_tagged_sample(2, "verifier"),
            make_tagged_sample(3, "solver"),
        )

        ans = split([[first, second], [third]])

        assert ans["solver"].group == [[first], [third]]
        assert ans["verifier"].group == [[second]]

    def test_the_prompt_group_travels_whole_into_every_split(self):
        """A rejected group is recycled by resubmitting its prompts, which belong to no policy in particular."""
        prompt_group = make_group(7)

        ans = split([make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")], prompt_group=prompt_group)

        assert ans["solver"].prompt_group is prompt_group
        assert ans["verifier"].prompt_group is prompt_group

    def test_an_untagged_sample_is_refused_before_anything_is_routed(self):
        """Nothing downstream can guess where an unstamped sample belongs, so the split is where it must stop."""
        with pytest.raises(AssertionError, match="must stamp every sample"):
            split([make_tagged_sample(1, "solver"), make_tagged_sample(2, None)])

    def test_a_group_with_no_sample_left_reaches_no_policy(self):
        """A group every filter emptied belongs to nobody, and inventing a key for it would assert on None."""
        assert split([]) == {}


class TestFilterGroup:
    def test_it_keeps_only_the_samples_of_the_policy_asked_for(self):
        """This is what stops a policy from training on another policy's responses."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer.filter_group([solver, verifier], trainer_model_id="solver") == [solver]

    def test_it_keeps_a_sub_group_that_still_has_samples(self):
        """A trajectory whose samples are split across policies survives on both sides, one sample each."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer.filter_group([[solver, verifier]], trainer_model_id="solver") == [[solver]]

    def test_it_drops_a_sub_group_that_lost_every_sample(self):
        """An empty list left in place would be a trajectory that consumers must special-case forever."""
        assert data_buffer.filter_group([[make_tagged_sample(1, "verifier")]], trainer_model_id="solver") == []

    def test_it_leaves_the_group_it_was_given_untouched(self):
        """It runs once per policy over the same group, so a mutating filter would eat the later policies' samples."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")
        group = [solver, [verifier]]

        data_buffer.filter_group(group, trainer_model_id="solver")

        assert group == [solver, [verifier]]


class RecordingBuffer(data_buffer.DefaultDataBuffer):
    constructed_with = None

    def __init__(self, input):
        super().__init__(input)
        RecordingBuffer.constructed_with = input


class TestPerPolicyBufferClass:
    def test_every_policy_keeps_the_built_in_buffer_by_default(self):
        """The flag is opt-in, so a run that does not pass it must compose exactly what it composed before."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        assert [type(inner) for inner in buffer._inners.values()] == [
            data_buffer.DefaultDataBuffer,
            data_buffer.DefaultDataBuffer,
        ]

    def test_a_named_policy_gets_the_class_the_flag_names(self):
        """Two policies can need different dataflow, which is the whole point of one buffer per policy."""
        buffer, _ = make_multi_buffer("solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"])

        assert type(buffer._inners["solver"]) is RecordingBuffer
        assert type(buffer._inners["verifier"]) is data_buffer.DefaultDataBuffer

    def test_the_custom_class_is_built_with_the_same_constructor_input(self):
        """A custom buffer owns staleness and recycling, so it needs the handler the built-in one gets."""
        buffer, unused = make_multi_buffer(
            "solver", "verifier", paths_per_model=[f"solver={__name__}.RecordingBuffer"]
        )

        assert RecordingBuffer.constructed_with.args is buffer._inners["verifier"]._args

    def test_a_policy_this_run_does_not_train_is_refused(self):
        """A typo would silently leave the policy it meant to configure on the built-in buffer."""
        with pytest.raises(AssertionError, match="train no policy of this run"):
            make_multi_buffer("solver", "verifier", paths_per_model=[f"reviewer={__name__}.RecordingBuffer"])


class TestParseDataBufferPaths:
    def test_it_maps_every_model_id_to_its_class_path(self):
        """This is the mapping the composite buffer is built from."""
        assert data_buffer._parse_data_buffer_paths(["solver=pkg.A", "verifier=pkg.B"]) == {
            "solver": "pkg.A",
            "verifier": "pkg.B",
        }

    def test_an_unset_flag_is_an_empty_mapping(self):
        """Default is every policy on the built-in buffer, which is the empty mapping."""
        assert data_buffer._parse_data_buffer_paths(None) == {}

    @pytest.mark.parametrize("entry", ["solver", "=pkg.A", "solver=", "solver =  "])
    def test_a_malformed_entry_is_refused(self, entry):
        """Silently ignoring it would run the policy on a buffer the user did not ask for."""
        with pytest.raises(ValueError, match="expected MODEL_ID=PATH"):
            data_buffer._parse_data_buffer_paths([entry])

    def test_the_whitespace_around_an_entry_is_not_part_of_the_names(self):
        """A shell-quoted entry keeps its spaces, and an import path with them resolves to nothing."""
        assert data_buffer._parse_data_buffer_paths([" solver = pkg.A "]) == {"solver": "pkg.A"}

    def test_a_model_id_named_twice_is_refused(self):
        """One of the two class paths would win silently, and which one is not something to guess."""
        with pytest.raises(ValueError, match="Duplicate model id"):
            data_buffer._parse_data_buffer_paths(["solver=pkg.A", "solver=pkg.B"])


class TestDataBufferArgumentRegistration:
    def test_the_per_model_flag_is_declared_by_the_rollout_function_that_uses_it(self):
        """The framework asks the selected rollout function for its flags, so this hook must declare it."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        parsed = parser.parse_args(["--custom-async-data-buffer-path-per-model", "solver=pkg.A", "verifier=pkg.B"])

        assert parsed.custom_async_data_buffer_path_per_model == ["solver=pkg.A", "verifier=pkg.B"]

    def test_a_run_that_never_passes_the_flag_leaves_every_policy_on_the_built_in_buffer(self):
        """The default has to be None, which _parse_data_buffer_paths reads as the empty mapping."""
        parser = argparse.ArgumentParser()
        fully_async.FullyAsyncRolloutFn.add_arguments(parser)

        assert parser.parse_args([]).custom_async_data_buffer_path_per_model is None


async def test_custom_data_buffer_path_replaces_default(monkeypatch):
    path = f"{__name__}.RecordingBuffer"
    args = make_args(custom_async_data_buffer_path=path, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert type(fn._output) is RecordingBuffer
    assert RecordingBuffer.constructed_with.unused_handler_fn == fn._recycle
    assert len(output.samples) == 2


class MultiPolicyDataSource(FakeDataSource):
    """Stamps every sample of a group with one policy, alternating, as a multi policy generate function does."""

    def get_samples(self, num_samples):
        [group] = super().get_samples(num_samples)
        for sample in group:
            sample.trainer_model_id = "a" if self.num_get_calls % 2 == 1 else "b"
        return [group]


class WedgedBuffer(data_buffer.DataBuffer):
    def __init__(self, input: data_buffer.DataBufferConstructorInput):
        self._never = asyncio.Event()
        self.entered = asyncio.Event()

    async def put(self, input: data_buffer.DataBufferInput) -> data_buffer.PutOutcomes:
        self.entered.set()
        await self._never.wait()
        raise AssertionError("the wedged buffer never accepts a group")

    async def get(self, *, num_groups: int, **context) -> list[data_buffer.DataBufferInput]:
        self.entered.set()
        await self._never.wait()
        raise AssertionError("the wedged buffer never hands out a group")

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        return {}

    def snapshot(self) -> data_buffer.DataBufferState:
        return {}

    def restore(self, state: data_buffer.DataBufferState) -> None:
        raise NotImplementedError


class TestDisposal:
    async def test_disposing_ends_the_producer_before_it_returns(self, monkeypatch):
        """Teardown has to be done when it returns, or the rest of it races the producer's unwinding."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0)
        await fn._output.entered.wait()
        assert not step.done()

        await asyncio.wait_for(fn.dispose(), timeout=5)

        assert fn._worker.cancelled()

    async def test_disposing_fails_a_step_that_is_waiting_for_groups(self, monkeypatch):
        """A run whose producer is parked when it ends must not leave the waiting step there forever."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0)
        await fn._output.entered.wait()

        await fn.dispose()

        with pytest.raises(RuntimeError, match="disposed while a step waited for groups"):
            await asyncio.wait_for(step, timeout=5)

    async def test_disposing_a_run_that_never_started_is_quiet(self, monkeypatch):
        """dispose runs on every teardown path, including one that failed before the first step."""
        fn = make_fn(monkeypatch, make_args(), FakeDataSource())

        await fn.dispose()

        assert fn._worker is None


class RecordingMultiBuffer(data_buffer.DefaultMultiDataBuffer):
    get_calls: list[dict] = []

    async def get(self, **context):
        RecordingMultiBuffer.get_calls.append(context)
        return await super().get(**context)


class TestBufferSelection:
    async def test_a_single_policy_run_keeps_the_plain_buffer(self, monkeypatch):
        """Every existing run goes through this line, and a per-policy buffer would key it under a model id."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert type(fn._output) is data_buffer.DefaultDataBuffer

    async def test_a_multi_policy_run_defaults_to_the_per_policy_buffer(self, monkeypatch):
        """One shared queue would hand a policy the groups another policy generated."""
        args = make_args(rollout_batch_size=1, megatron_config=encode_megatron_config("a", "b"))
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        output = await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is data_buffer.DefaultMultiDataBuffer
        assert [sample.trainer_model_id for group in output.samples for sample in group] == ["a", "a"]

    async def test_a_custom_buffer_still_wins_in_a_multi_policy_run(self, monkeypatch):
        """--custom-async-data-buffer-path is how a run replaces the queue, whatever the default would have been."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="a"))

        assert type(fn._output) is RecordingMultiBuffer

    async def test_the_consumer_asks_the_buffer_for_the_policy_that_called_it(self, monkeypatch):
        """The queue is keyed by policy, so a consumer that forgets to name itself drains whoever answers first."""
        RecordingMultiBuffer.get_calls = []
        args = make_args(
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            custom_async_data_buffer_path=f"{__name__}.RecordingMultiBuffer",
        )
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=4, trainer_model_id="a"))

        assert RecordingMultiBuffer.get_calls == [dict(num_groups=1, current_version=4, trainer_model_id="a")]


async def test_worker_defaults_to_sample_granularity(monkeypatch):
    """Unset --rollout-submission-granularity: this driver backfills on sample completion."""
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    # Report every sample of the still-pending group as finished.
    for _ in range(N_SAMPLES_PER_PROMPT):
        callbacks[0]()
    await asyncio.sleep(0.01)

    # A replacement group went out even though the first group has not returned.
    assert data_source.num_get_calls == 2

    release.set()
    output = await drain
    assert len(output.samples) == 1


async def test_group_granularity_opts_the_worker_out_of_backfill(monkeypatch):
    callbacks = []
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        callbacks.append(sample_done_callback)
        await release.wait()
        return group

    data_source = FakeDataSource()
    args = make_args(rollout_batch_size=1, rollout_submission_granularity="group")
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1
    # no callback wired at group level
    assert callbacks == [None]

    await asyncio.sleep(0.01)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 1


class TestRolloutFnContract:
    def test_it_is_a_rollout_fn_the_loader_accepts(self):
        """load_rollout_fn gates on issubclass(fn, BaseRolloutFn), so a class that forgets the
        base is rejected at startup no matter how complete its behaviour is."""
        assert issubclass(fully_async.FullyAsyncRolloutFn, BaseRolloutFn)

    def test_the_constructor_input_reaches_the_base(self, monkeypatch):
        """The base stores it as constructor_input; skipping super().__init__ leaves the
        attribute missing on every path that reads it."""
        data_source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

        assert fn.constructor_input.data_source is data_source


async def test_worker_bounds_in_flight_groups(monkeypatch):
    release = asyncio.Event()
    entered = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        entered.set()
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await entered.wait()
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2


def make_checkpointing_args(tmp_path, **overrides) -> Namespace:
    return make_args(save=str(tmp_path), load=str(tmp_path), **overrides)


def owned_sample_indices(fn) -> set[int]:
    holdings = fn.describe_holdings(trainer_model_id=None)
    return {index for indices in holdings.values() for index in indices}


class BlockingPutBuffer(data_buffer.DefaultDataBuffer):
    def __init__(
        self, input: data_buffer.DataBufferConstructorInput, *, entered: asyncio.Event, release: asyncio.Event
    ) -> None:
        super().__init__(input)
        self.entered = entered
        self.release = release

    async def put(self, input: data_buffer.DataBufferInput) -> data_buffer.PutOutcomes:
        self.entered.set()
        await self.release.wait()
        return await super().put(input)


class TestInFlightRegistry:
    async def test_a_submitted_group_is_registered_before_it_starts(self, monkeypatch):
        """A group nobody records between the data source and the buffer is a group a checkpoint loses."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        task = fn._submit_one_group()

        assert fn.describe_holdings(trainer_model_id=None)[SampleOwner.IN_FLIGHT] == [10010, 10011]
        task.cancel()

    async def test_a_generated_group_stays_owned_while_put_blocks_on_a_full_buffer(self, monkeypatch):
        """put may wait for the trainer for minutes, and the group belongs to nobody else meanwhile."""
        release = asyncio.Event()

        entered = asyncio.Event()
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.BlockingPutBuffer")
        monkeypatch.setattr(
            fully_async,
            "load_function",
            lambda path: (lambda input: BlockingPutBuffer(input, entered=entered, release=release)) if path else None,
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0, weight_version=1)))
        await entered.wait()

        assert fn.describe_holdings(trainer_model_id=None)[SampleOwner.OUTPUT_BUFFER] == [10010, 10011]

        release.set()
        await step
        assert fn.describe_holdings(trainer_model_id=None)[SampleOwner.IN_FLIGHT] == []

    async def test_a_group_is_deregistered_only_once_the_buffer_has_it(self, monkeypatch):
        """Deregistering before put returns opens a window where the group has no owner at all."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=1))

        assert fn.describe_holdings(trainer_model_id=None)[SampleOwner.IN_FLIGHT] == []


class TestSaveAndLoad:

    async def test_a_group_blocked_in_put_is_saved_and_delivered_without_regeneration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A prompt held by a producer blocked on a full buffer must survive the checkpoint."""
        release = asyncio.Event()
        entered = asyncio.Event()

        args = make_checkpointing_args(
            tmp_path, rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.BlockingPutBuffer"
        )
        monkeypatch.setattr(
            fully_async,
            "load_function",
            lambda path: (lambda input: BlockingPutBuffer(input, entered=entered, release=release)) if path else None,
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0, weight_version=1)))
        await entered.wait()
        in_flight_indices = owned_sample_indices(fn)
        assert in_flight_indices

        fn.save(0)

        release.set()
        await step
        resumed = make_fn(monkeypatch, make_checkpointing_args(tmp_path, rollout_batch_size=1), FakeDataSource())
        resumed.load(0)
        assert in_flight_indices == set(resumed.describe_holdings(trainer_model_id=None)[SampleOwner.OUTPUT_BUFFER])

    async def test_a_checkpoint_only_resumes_the_policy_still_waiting_for_put(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A consumed policy must not be regenerated when its peer was blocked during save."""
        args = make_checkpointing_args(
            tmp_path, rollout_batch_size=1, megatron_config=encode_megatron_config("a", "b")
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._ensure_output()
        entered, release = asyncio.Event(), asyncio.Event()
        fn._output._inners["b"] = BlockingPutBuffer(
            data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused),
            entered=entered,
            release=release,
        )
        group = make_multi_policy_group(7, "a", "b")
        fn._pending_puts = fn._output.partition(data_buffer.DataBufferInput(prompt_group=group, group=group))
        putting = asyncio.create_task(fn._flush_pending_puts())
        await entered.wait()
        consumed = await fn._output.get(num_groups=1, trainer_model_id="a")
        assert [sample.index for sample in consumed[0].group] == [70]

        fn.save(0)
        putting.cancel()
        await asyncio.gather(putting, return_exceptions=True)
        restored = make_fn(monkeypatch, args, FakeDataSource())
        restored.load(0)
        restored._ensure_output()
        restored._output.restore(restored._pending_restore)
        restored._pending_restore = None
        await restored._flush_pending_puts()

        assert restored._output.snapshot()["a"] == []
        remaining = await restored._output.get(num_groups=1, trainer_model_id="b")
        assert [sample.index for sample in remaining[0].group] == [71]
        assert restored.data_source.num_get_calls == 0

    async def test_a_batch_being_drained_is_saved_with_the_buffer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Between the buffer giving a batch up and the drain returning it, only _in_transit owns it."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._output = make_buffer()[0]
        group = make_group(4)
        await fn._output.put(data_buffer.DataBufferInput(prompt_group=group, group=group))
        batch = await fn._take_batch(num_groups=1, current_version=1, trainer_model_id=None)

        fn.save(0)

        state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=0), weights_only=False)
        assert [entry.prompt_group[0].index for entry in state["output_buffer"][None]] == [
            batch[0].prompt_group[0].index
        ]

    async def test_a_recycled_group_is_saved_from_the_retry_buffer(self, monkeypatch, tmp_path):
        """An aborted group that already went back for a retry has no other record on disk."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1, async_unused_samples_handler="retry")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        recycled = make_group(5)
        fn._recycle(recycled, reason=data_buffer.UnusedReason.ABORTED, trainer_model_id=None)

        fn.save(0)

        resumed = make_fn(monkeypatch, args, FakeDataSource())
        resumed.load(0)
        assert resumed.describe_holdings(trainer_model_id=None)[SampleOwner.RETRY_BUFFER] == [50, 51]

    async def test_an_in_flight_group_is_saved_reset_so_it_reruns_from_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The samples handed to the generate function are written in place, so a half-run turn must be cleared."""
        release = asyncio.Event()
        generate_entered = asyncio.Event()

        async def blocking_generate(
            state: FakeGenerateState,
            group: list[Sample],
            sampling_params: dict[str, Any],
            evaluation: bool = False,
            sample_done_callback: Callable[..., None] | None = None,
        ) -> list[Sample]:
            for sample in group:
                sample.response = "half a turn"
            generate_entered.set()
            await release.wait()
            return group

        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource(), generate=blocking_generate)
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0, weight_version=1)))
        await generate_entered.wait()
        before = owned_sample_indices(fn)
        assert before

        fn.save(0)

        release.set()
        step.cancel()
        resumed = make_fn(monkeypatch, args, FakeDataSource())
        resumed.load(0)
        assert before == set(resumed.describe_holdings(trainer_model_id=None)[SampleOwner.RETRY_BUFFER])
        persisted = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=0), weights_only=False)
        assert all(
            sample.response == "" for pending in persisted["in_flight_prompt_groups"] for sample in pending.samples
        )

    async def test_the_buffered_half_of_a_group_still_in_flight_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A multi policy put fills one policy at a time; keeping both copies would train that half twice."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._output = make_buffer()[0]
        prompt_group = make_group(3)
        await fn._output.put(data_buffer.DataBufferInput(prompt_group=prompt_group, group=prompt_group))
        fn._in_flight[asyncio.Future()] = fully_async._PendingPrompt(samples=prompt_group)
        original_snapshot = fn._output.snapshot
        monkeypatch.setattr(fn._output, "snapshot", lambda: copy.deepcopy(original_snapshot()))

        fn.save(0)

        state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=0), weights_only=False)
        assert state["output_buffer"][None] == []
        assert [pending.samples[0].group_index for pending in state["in_flight_prompt_groups"]] == [3]

    async def test_a_restored_buffer_needs_a_weight_version_before_the_first_step(self, monkeypatch, tmp_path):
        """Groups restored under an unknown version would all be filtered as stale, losing the whole batch."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        group = make_group(4)
        fn._pending_restore = {None: [data_buffer.DataBufferInput(prompt_group=group, group=group)]}

        with pytest.raises(AssertionError, match="has not pushed a weight version"):
            fn._start_worker(weight_version=None)

    async def test_an_empty_restored_buffer_needs_no_weight_version(self, monkeypatch, tmp_path):
        """A run checkpointed with nothing buffered has no group to measure, and must still start."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._pending_restore = {None: []}

        fn._start_worker(weight_version=None)

        assert fn.describe_holdings(trainer_model_id=None)[SampleOwner.OUTPUT_BUFFER] == []
        fn._worker.cancel()

    async def test_a_restored_buffer_is_put_back_before_the_producer_starts(self, monkeypatch, tmp_path):
        """The restored groups are finished work, and regenerating them would waste a whole batch."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        group = make_group(4)
        fn._pending_restore = {None: [data_buffer.DataBufferInput(prompt_group=group, group=group)]}

        output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=1))

        assert output.samples[0][0].group_index == 4

    async def test_loading_after_the_producer_started_is_refused(self, monkeypatch, tmp_path):
        """A late restore would put groups into a buffer the producer is already filling."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=1))

        with pytest.raises(AssertionError, match="before the producer starts"):
            fn.load(0)

    async def test_a_missing_state_file_only_warns(self, monkeypatch, tmp_path):
        """A checkpoint written before this feature existed still has to be resumable."""
        fn = make_fn(monkeypatch, make_checkpointing_args(tmp_path, rollout_batch_size=1), FakeDataSource())

        fn.load(7)

        assert owned_sample_indices(fn) == set()

    async def test_a_run_without_save_writes_nothing(self, monkeypatch, tmp_path):
        """--save is optional, and a run without it must not fail at the checkpoint step."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        fn.save(0)

        assert not list(tmp_path.iterdir())


class TestPutOutcomeEvents:
    def test_each_policy_of_a_group_gets_its_own_ownership_event(self, monkeypatch):
        """A group one policy kept and another dropped needs both moves logged, or the drop looks like a leak."""
        logged = []
        monkeypatch.setattr(
            fully_async.sample_ownership,
            "log_owner_transition",
            lambda samples, **kwargs: logged.append((sorted(s.index for s in samples), kwargs)),
        )
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        prompt_group = make_group(7)
        prompt_group[0].trainer_model_id = "solver"
        prompt_group[1].trainer_model_id = "verifier"

        fn._log_put_outcomes(
            [(prompt_group, {"solver": data_buffer.PutOutcome.KEPT, "verifier": data_buffer.PutOutcome.DROPPED})],
        )

        assert logged == [
            (
                [70],
                dict(
                    from_owner=SampleOwner.IN_FLIGHT,
                    to_owner=SampleOwner.OUTPUT_BUFFER,
                    trainer_model_id="solver",
                    reason=None,
                ),
            ),
            (
                [71],
                dict(
                    from_owner=SampleOwner.IN_FLIGHT,
                    to_owner=SampleOwner.DROPPED,
                    trainer_model_id="verifier",
                    reason="dynamic_filter",
                ),
            ),
        ]

    def test_single_policy_outcomes_combine_all_finished_groups(self, monkeypatch):
        """One producer cycle emits one transition for all finished groups of the same policy."""
        logged = []
        monkeypatch.setattr(
            fully_async.sample_ownership,
            "log_owner_transition",
            lambda samples, **kwargs: logged.append(sorted(s.index for s in samples)),
        )
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        fn._log_put_outcomes(
            [
                (make_group(7), {None: data_buffer.PutOutcome.KEPT}),
                (make_group(8), {None: data_buffer.PutOutcome.KEPT}),
            ]
        )

        assert logged == [[70, 71, 80, 81]]


class TestDescribeHoldings:
    def test_replay_contract_is_available_before_the_restored_worker_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Executor replay snapshots must retain the configured policy replay contract."""
        monkeypatch.setattr(RecordingBuffer, "replays_samples", True)
        args = make_args(
            megatron_config=encode_megatron_config("solver", "verifier"),
            custom_async_data_buffer_path_per_model=[f"verifier={__name__}.RecordingBuffer"],
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())
        group = make_group(1)
        fn._pending_restore = {"verifier": [data_buffer.DataBufferInput(prompt_group=group, group=group)]}

        assert fn.replays_samples(trainer_model_id="solver") is False
        assert fn.replays_samples(trainer_model_id="verifier") is True
        assert fn.describe_holdings(trainer_model_id="verifier")[SampleOwner.OUTPUT_BUFFER] == [10, 11]
        assert fn._worker is None

    async def test_dispose_checks_a_final_snapshot_after_stopping_the_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ownership_event_dir: Path
    ) -> None:
        """Disposal detects a last-step loss without needing a checkpoint or three snapshots."""
        entered = asyncio.Event()

        async def blocked_generate(
            state: FakeGenerateState,
            group: list[Sample],
            sampling_params: dict[str, Any],
            evaluation: bool = False,
            sample_done_callback: Callable[..., None] | None = None,
        ) -> list[Sample]:
            entered.set()
            await asyncio.Event().wait()
            return group

        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource(), generate=blocked_generate)
        executor = make_executor(tmp_path, rollout_fn=fn)
        executor.use_legacy_rollout_v1 = False
        executor._train_parallel_configs_of_model_id = {None: {}}
        executor._metric_checker = None
        executor.args.save_debug_event_data = str(ownership_event_dir)
        executor.args.enable_event_analyzer = False
        executor.rollout_id = 0
        fn._start_worker(weight_version=1)
        await entered.wait()
        fully_async.sample_ownership.log_owner_transition(
            make_group(1), from_owner=SampleOwner.DATA_SOURCE, to_owner=SampleOwner.IN_FLIGHT
        )

        with pytest.raises(ValueError, match="Event analysis found issues"):
            await executor.dispose()

        assert fn._worker.done()
        [snapshot] = [e for e in read_events(ownership_event_dir) if isinstance(e, RolloutHoldingsSnapshotEvent)]
        assert snapshot.reason == "final"
        assert snapshot.holdings[SampleOwner.IN_FLIGHT] == [10010, 10011]
        for task in fn._in_flight:
            task.cancel()
        await asyncio.gather(*fn._in_flight, return_exceptions=True)

    async def test_save_emits_separate_policy_holdings_and_replay_contracts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ownership_event_dir: Path
    ) -> None:
        """A replay policy must not supply holdings or disable duplicate checks for its peer."""
        args = make_checkpointing_args(tmp_path, megatron_config=encode_megatron_config("solver", "verifier"))
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._output = data_buffer.DefaultMultiDataBuffer(
            data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused)
        )
        fn._output._inners["verifier"].replays_samples = True
        group = make_group(1)
        fn._output.restore({"solver": [], "verifier": [data_buffer.DataBufferInput(prompt_group=group, group=group)]})
        executor = make_executor(tmp_path, rollout_fn=fn)
        executor.use_legacy_rollout_v1 = False
        executor._train_parallel_configs_of_model_id = {"solver": {}, "verifier": {}}
        executor._record_last_batch(rollout_id=2, trainer_model_id="verifier", samples=[make_group(4)])

        await executor._save_sample_state(rollout_id=0, rollout_ids={"solver": 0, "verifier": 1})

        snapshots = {
            event.trainer_model_id: event
            for event in read_events(ownership_event_dir)
            if isinstance(event, RolloutHoldingsSnapshotEvent) and event.reason == "save"
        }
        assert set(snapshots) == {"solver", "verifier"}
        assert snapshots["solver"].holdings[SampleOwner.OUTPUT_BUFFER] == []
        assert snapshots["solver"].holdings[SampleOwner.HANDED_TO_TRAINER] == []
        assert snapshots["solver"].replays_samples is False
        assert snapshots["verifier"].holdings[SampleOwner.OUTPUT_BUFFER] == [10, 11]
        assert snapshots["verifier"].holdings[SampleOwner.HANDED_TO_TRAINER] == [40, 41]
        assert snapshots["verifier"].replays_samples is True
        assert snapshots["verifier"].rollout_id == 1

    async def test_policies_share_prompts_but_not_output_holdings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One policy's shared sample index cannot hide another policy's missing output."""
        fn = make_fn(
            monkeypatch,
            make_args(megatron_config=encode_megatron_config("solver", "verifier")),
            FakeDataSource(),
        )
        group = make_group(1)
        fn._pending_restore = {"verifier": [data_buffer.DataBufferInput(prompt_group=group, group=group)]}
        fn._retry_buffer.append(fully_async._PendingPrompt(samples=make_group(2)))
        fn._in_flight[asyncio.Future()] = fully_async._PendingPrompt(samples=make_group(3))

        solver = fn.describe_holdings(trainer_model_id="solver")
        verifier = fn.describe_holdings(trainer_model_id="verifier")

        assert solver[SampleOwner.OUTPUT_BUFFER] == []
        assert verifier[SampleOwner.OUTPUT_BUFFER] == [10, 11]
        assert solver[SampleOwner.RETRY_BUFFER] == verifier[SampleOwner.RETRY_BUFFER] == [20, 21]
        assert solver[SampleOwner.IN_FLIGHT] == verifier[SampleOwner.IN_FLIGHT] == [30, 31]

    async def test_save_reuses_the_collected_buffer_for_holdings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Saving state and reporting its holdings must read the output buffer only once."""
        fn = make_fn(monkeypatch, make_checkpointing_args(tmp_path, rollout_batch_size=1), FakeDataSource())
        fn._output = make_buffer()[0]
        group = make_group(1)
        await fn._output.put(data_buffer.DataBufferInput(prompt_group=group, group=group))
        snapshot = fn._output.snapshot
        reads = 0

        def read_once() -> data_buffer.DataBufferState:
            nonlocal reads
            reads += 1
            assert reads == 1
            return snapshot()

        monkeypatch.setattr(fn._output, "snapshot", read_once)

        holdings = fn.save(0)

        assert holdings[None][SampleOwner.OUTPUT_BUFFER] == [10, 11]

    async def test_it_reports_every_owner_the_rollout_function_has(self, monkeypatch):
        """The checker reads exactly this, so an owner missing here looks like a lost sample."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        fn._output = make_buffer()[0]
        buffered = make_group(1)
        await fn._output.put(data_buffer.DataBufferInput(prompt_group=buffered, group=buffered))
        fn._retry_buffer.append(fully_async._PendingPrompt(samples=make_group(2)))
        fn._in_flight[asyncio.Future()] = fully_async._PendingPrompt(samples=make_group(3))

        holdings = fn.describe_holdings(trainer_model_id=None)

        assert set(holdings[SampleOwner.OUTPUT_BUFFER]) == {10, 11}
        assert set(holdings[SampleOwner.RETRY_BUFFER]) == {20, 21}
        assert set(holdings[SampleOwner.IN_FLIGHT]) == {30, 31}
