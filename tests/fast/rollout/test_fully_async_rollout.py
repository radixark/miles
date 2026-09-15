from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import argparse
import asyncio
import logging
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from tests.fast.rollout.inference_rollout.conftest import (
    StampRecordingGenerate,
    make_eval_args,
    make_eval_prompt_dataset_cache,
)

import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.inference_rollout.inference_rollout_common as rollout_common
from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import ExplicitlyDroppedSamplesEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.function_registry import load_function
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

N_SAMPLES_PER_PROMPT = 2


@pytest.fixture
def sample_flow_event_dir(tmp_path: Path) -> Iterator[Path]:
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor")))
    yield tmp_path
    set_event_logger(None)


class FakeGenerateState(GenerateState):
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False
        self.generate_fn_semaphore = asyncio.Semaphore(2)
        self.generate_function = None


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
        async_max_concurrent_samples=None,
        async_data_buffer_capacity_factor=1000.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        enable_sample_ownership_checker=False,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_request_timeout_secs=14400,
        eval_num_gpus=0,
        namespaced_radix_cache=True,
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

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
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
    await asyncio.sleep(0.05)
    submitted_before_eval = data_source.num_get_calls

    eval_task = asyncio.create_task(fn(RolloutFnEvalInput(rollout_id=0)))
    await eval_started.wait()
    release.set()  # in-flight groups finish and buffer, but no NEW submissions
    await asyncio.sleep(0.05)
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


class TestKvCacheNamespace:
    def _make_eval_fn(self, monkeypatch, recorder: StampRecordingGenerate, *, partition: bool = True, **overrides):
        args = make_args(**make_eval_args(namespaced_radix_cache=partition), **overrides)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn.state.generate_function = recorder
        fn._eval_prompt_dataset_cache.update(make_eval_prompt_dataset_cache(args))
        return fn

    async def test_a_train_call_takes_over_the_namespace_the_producer_stamps_with(self, monkeypatch):
        """The producer follows the rollout id of the call in flight, never a counter of its own."""
        fn = make_fn(monkeypatch, make_args(), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=5))
        assert fn._curr_kv_cache_namespace == "train:-:5"

        await fn(RolloutFnTrainInput(rollout_id=6))
        assert fn._curr_kv_cache_namespace == "train:-:6"

    @pytest.mark.parametrize("producer_namespace", [None, "train:-:1"])
    @pytest.mark.parametrize("partition", [False, True])
    async def test_shared_eval_uses_its_own_namespace_without_changing_the_producer(
        self, monkeypatch: pytest.MonkeyPatch, producer_namespace: str | None, partition: bool
    ) -> None:
        """Initial and final shared evals select eval partitions without changing subsequent training stamps."""
        recorder = StampRecordingGenerate()
        fn = self._make_eval_fn(monkeypatch, recorder, partition=partition)
        fn._curr_kv_cache_namespace = producer_namespace

        for rollout_id in (0, 1):
            await fn(RolloutFnEvalInput(rollout_id=rollout_id))

            assert recorder.take_stamps() == {f"eval:-:{rollout_id}" if partition else None}
            assert fn._curr_kv_cache_namespace == producer_namespace
            assert fn._producer_resumed.is_set()

    async def test_fleet_state_eval_stamps_its_samples_with_the_eval_namespace(self, monkeypatch):
        """Eval on a dedicated fleet opens its own namespace and leaves the producer's alone."""
        recorder = StampRecordingGenerate()
        fn = self._make_eval_fn(monkeypatch, recorder, eval_num_gpus=1, eval_num_gpus_per_engine=1)
        fleet_state = FakeGenerateState(fn.args)
        fleet_state.generate_function = recorder

        await fn(RolloutFnEvalInput(rollout_id=4, generate_state=fleet_state, weight_version="0"))

        assert recorder.take_stamps() == {"eval:-:4"}
        assert fn._curr_kv_cache_namespace is None

    async def test_the_partition_being_off_names_no_namespace_for_a_train_call(self, monkeypatch):
        """With --no-namespaced-radix-cache the producer names no namespace at all."""
        fn = make_fn(monkeypatch, make_args(namespaced_radix_cache=False), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=5))

        assert fn._curr_kv_cache_namespace is None

    async def test_the_partition_being_off_leaves_fleet_eval_samples_unstamped(self, monkeypatch):
        """With the partition off eval on a dedicated fleet stamps nothing, so no request carries an extra_key."""
        recorder = StampRecordingGenerate()
        fn = self._make_eval_fn(monkeypatch, recorder, partition=False, eval_num_gpus=1, eval_num_gpus_per_engine=1)
        fleet_state = FakeGenerateState(fn.args)
        fleet_state.generate_function = recorder

        await fn(RolloutFnEvalInput(rollout_id=4, generate_state=fleet_state, weight_version="0"))

        assert recorder.take_stamps() == {None}

    async def test_calls_of_different_policies_each_open_their_own_namespace(self, monkeypatch):
        """A sample is stamped when the producer takes it, so each policy's rollout id partitions on its own."""
        gate: asyncio.Queue[None] = asyncio.Queue()
        stamps: list[str] = []

        async def gated_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
            stamps.append(group[0].kv_cache_namespace)
            await gate.get()
            return group

        args = make_args(rollout_batch_size=1, megatron_config=encode_megatron_config("a", "b"))
        fn = make_fn(monkeypatch, args, MultiPolicyDataSource(), generate=gated_generate)

        for trainer_model_id, rollout_id in [("a", 10), ("b", 3), ("b", 4)]:
            marker = len(stamps)
            for _ in range(2 * fn.args.rollout_batch_size):
                gate.put_nowait(None)
            await fn(RolloutFnTrainInput(rollout_id=rollout_id, trainer_model_id=trainer_model_id))
            await asyncio.sleep(0.05)

            assert set(stamps[marker:]) == {f"train:{trainer_model_id}:{rollout_id}"}


class TestRetryBuffer:
    async def test_the_next_submission_prefers_retry_prompts_over_the_data_source(self, monkeypatch) -> None:
        """Recycled prompts are retried before advancing the read-only data source."""
        source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), source)
        group = make_group(7)
        fn._recycle(group, data_buffer.UnusedReason.ABORTED)

        entry = await fn._submit_one_group()

        assert entry.prompt_group == group
        assert source.num_get_calls == 0
        assert not fn._retry_buffer


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    for sample in aborted:
        sample.reward = None
    data_source = FakeDataSource(scripted=[aborted])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    calls = 0

    async def abort_once(state, group, **kwargs):
        nonlocal calls
        calls += 1
        for sample in group:
            sample.status = Sample.Status.ABORTED if calls == 1 else Sample.Status.COMPLETED
        return group

    fn = make_fn(monkeypatch, args, data_source, generate=abort_once)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.num_get_calls == 1
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index == 1
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1
    assert "rollout/dynamic_filter/drop_group_has_missing_reward" not in output.metrics


async def test_missing_reward_group_dropped_without_recycling(monkeypatch):
    missing_reward = make_group(1)
    missing_reward[0].reward = None
    data_source = FakeDataSource(scripted=[missing_reward])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == []
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/dynamic_filter/drop_group_has_missing_reward"] == 1


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

    assert data_source.num_get_calls == 1
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


@pytest.mark.parametrize("handler", ["drop", "retry"])
@pytest.mark.parametrize("granularity", ["sample", "group"])
async def test_sample_cancellation_aborts_group_without_stopping_worker(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    handler: str,
    granularity: str,
) -> None:
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def generate_sample(
        state: FakeGenerateState,
        sample: Sample,
        sampling_params: dict,
        evaluation: bool = False,
    ) -> Sample:
        if sample.group_index != 1:
            return sample
        if sample is prompt_group[0]:
            await sibling_started.wait()
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_finished.set()
        return sample

    monkeypatch.setattr(rollout_common, "generate_and_rm", generate_sample)
    monkeypatch.setattr(rollout_common, "policy_uses_routing_key", lambda args: False)
    args = make_args(
        rollout_batch_size=1,
        async_max_concurrent_samples=N_SAMPLES_PER_PROMPT,
        async_unused_samples_handler=handler,
        rollout_submission_granularity=granularity,
        group_rm=False,
    )
    # Exercise the real fan-out/cleanup and scheduler callbacks, not a mocked
    # group result: one cancelled sample also cancels and settles its sibling.
    fn = make_fn(monkeypatch, args, data_source, generate=rollout_common.generate_and_rm_group)

    output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0)), timeout=2)

    assert sibling_finished.is_set()
    assert not fn._worker.done()
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1
    assert all(sample.group_index != 1 for group in output.samples for sample in group)
    assert all(sample.status == Sample.Status.COMPLETED for group in output.samples for sample in group)
    assert "Rollout group was cancelled" in caplog.text
    if handler == "retry":
        assert data_source.recycled == [prompt_group]
        assert all(sample.response == "" and sample.weight_versions == [] for sample in prompt_group)
    else:
        assert data_source.recycled == []
        assert all(sample.status == Sample.Status.COMPLETED for sample in prompt_group)

    # The producer remains usable after the affected batch, not just until the
    # first replacement arrives.
    following = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=1)), timeout=2)
    assert following.samples
    assert following.metrics["rollout/fully_async/aborted_groups_filtered"] == 0


async def test_worker_cancellation_still_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled worker fails the waiting step loudly instead of being filtered away as an abort."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_generate(
        state: FakeGenerateState,
        group: list[Sample],
        sampling_params: dict,
        evaluation: bool = False,
        sample_done_callback: Callable[[], None] | None = None,
    ) -> list[Sample]:
        started.set()
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=blocking_generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.wait_for(started.wait(), timeout=2)
    fn._worker.cancel()

    try:
        with pytest.raises(RuntimeError, match="disposed while a step waited"):
            await asyncio.wait_for(drain, timeout=2)
        assert fn._worker.cancelled()
        assert data_source.recycled == []
        assert fn._output.get_metrics()["rollout/fully_async/aborted_groups_filtered"] == 0
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
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
            expanded.append([aborted, replace(sample)])
        return expanded

    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.num_get_calls == 1
    assert all(isinstance(sample, Sample) for sample in submitted[1])
    assert len(submitted) > 1
    assert len(output.samples) == 1


def reject_group_1(args, group, **kwargs):
    keep = group[0].group_index != 1
    return FilterOutput(keep=keep, reason=None if keep else "rejected")


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
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: unused.append(group))
    )
    return buffer, unused


async def put_group(buffer, group):
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(prompt_group=group, group=group))


async def get_one(buffer, **context):
    """These tests consume one group at a time from a buffer that now hands out whole batches."""
    [entry] = await buffer.get(num_groups=1, **context)
    return entry


async def test_buffer_reports_unfiltered_raw_reward_across_kept_and_dropped():
    """The accepted-only raw_reward is conditioned by the filter, so this mean must still see dropped groups."""
    args = make_args(rollout_batch_size=1, dynamic_sampling_filter_path=f"{__name__}.reject_group_1")
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: None)
    )

    await put_group(buffer, make_group(1, reward=0))
    await put_group(buffer, make_group(2, reward=1))

    metrics = buffer.get_metrics()
    assert metrics["rollout/raw_reward_unfiltered"] == 0.5
    assert metrics["rollout/dynamic_filter/drop_rejected"] == 1
    assert "rollout/raw_reward_unfiltered" not in buffer.get_metrics()


async def test_dynamic_filter_records_the_prompt_as_explicitly_dropped(sample_flow_event_dir: Path) -> None:
    """A rejected generated group resolves its original issued samples."""
    args = make_args(
        enable_sample_ownership_checker=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path=f"{__name__}.reject_group_1",
    )
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: None)
    )

    await put_group(buffer, make_group(1, reward=0))

    [event] = read_events(sample_flow_event_dir)
    assert isinstance(event, ExplicitlyDroppedSamplesEvent)
    assert event.source_sample_indices == [10, 11]
    assert event.reason == "dynamic_filter"


async def test_missing_reward_records_the_prompt_as_explicitly_dropped(sample_flow_event_dir: Path) -> None:
    """A group dropped for a missing reward resolves its issued samples, like every other drop."""
    args = make_args(enable_sample_ownership_checker=True, rollout_batch_size=1)
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: None)
    )

    await put_group(buffer, make_group(1, reward=None))

    [event] = read_events(sample_flow_event_dir)
    assert isinstance(event, ExplicitlyDroppedSamplesEvent)
    assert event.source_sample_indices == [10, 11]
    assert event.reason == "missing_reward"


async def test_stale_group_reports_the_reason_to_the_unused_policy() -> None:
    """The configured policy can distinguish stale output from aborted generation."""
    calls: list[tuple[list[Sample], data_buffer.UnusedReason]] = []
    args = make_args(rollout_batch_size=1, max_weight_staleness=0)
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=args, unused_handler_fn=lambda group, reason: calls.append((group, reason))
        )
    )
    stale = make_group(1, weight_versions=["1"])
    fresh = make_group(2, weight_versions=["2"])
    await put_group(buffer, stale)
    await put_group(buffer, fresh)

    assert (await get_one(buffer, current_version=2)).group == fresh
    assert calls == [(stale, data_buffer.UnusedReason.STALE)]


async def test_buffer_blocks_producer_when_full():
    buffer, _ = make_buffer(max_groups=2)
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    blocked = asyncio.create_task(put_group(buffer, make_group(3)))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    assert (await get_one(buffer)).group[0].group_index == 1
    await blocked
    assert (await get_one(buffer)).group[0].group_index == 2
    assert (await get_one(buffer)).group[0].group_index == 3


async def test_buffer_get_ignores_unknown_context_keys():
    """get(**context) lets the driver add keys without breaking existing buffers."""
    buffer, _ = make_buffer()
    await put_group(buffer, make_group(1))

    assert (await get_one(buffer, current_version=1, some_future_key=2)).group[0].group_index == 1


async def test_buffer_get_skips_groups_stale_at_consumption_time():
    """Both groups were fresh when buffered; only the version passed to get() decides."""
    buffer, unused = make_buffer(max_staleness=2)
    stale = make_group(1, weight_versions=["5"])
    await put_group(buffer, stale)
    await put_group(buffer, make_group(2, weight_versions=["8"]))

    assert (await get_one(buffer, current_version=10)).group[0].group_index == 2
    assert unused == [stale]
    assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1


async def test_buffer_staleness_metrics():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    assert "rollout/fully_async/buffer_avg_staleness" not in buffer.get_metrics()  # engine version never seen

    await put_group(buffer, make_group(2, weight_versions=["6"]))
    await put_group(buffer, make_group(3, weight_versions=["8"]))
    await get_one(buffer, current_version=10)  # pops group 1 and tracks the engine version clock
    metrics = buffer.get_metrics()
    assert metrics["rollout/fully_async/avg_staleness"] == 6.0  # consumed group 1: 10 - 4
    assert metrics["rollout/fully_async/buffer_avg_staleness"] == 3.0  # buffered groups 2, 3: (4 + 2) / 2
    assert metrics["rollout/fully_async/buffer_max_staleness"] == 4


async def test_buffer_reports_selected_policy_provenance():
    buffer, _ = make_buffer(max_groups=8, max_staleness=2)
    await put_group(buffer, make_group(1, weight_versions=["2", "4"]))
    await put_group(buffer, make_group(2, weight_versions=["8", "10"]))

    assert (await get_one(buffer, current_version=10)).group[0].group_index == 2
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/avg_staleness"] == 2
    assert metrics["rollout/fully_async/max_staleness"] == 2
    assert metrics["rollout/fully_async/avg_post_generation_staleness"] == 0
    assert metrics["rollout/fully_async/avg_generation_version_span"] == 2
    assert metrics["rollout/fully_async/token_weighted_staleness"] == 1
    assert metrics["rollout/fully_async/weight_version_sample_coverage"] == 1


async def test_buffer_reports_generation_span_without_current_version():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["2", "4"]))

    await get_one(buffer, current_version=None)
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/avg_generation_version_span"] == 2
    assert "rollout/fully_async/avg_post_generation_staleness" not in metrics
    assert "rollout/fully_async/token_weighted_staleness" not in metrics


async def test_buffer_reports_missing_weight_version_coverage_without_inventing_lag():
    buffer, _ = make_buffer(max_groups=8)
    partial_group = make_group(1, weight_versions=["7"])
    partial_group[1].weight_versions = []
    await put_group(buffer, partial_group)

    await get_one(buffer, current_version=10)
    metrics = buffer.get_metrics()

    assert metrics["rollout/fully_async/weight_version_sample_coverage"] == 0.5
    assert metrics["rollout/fully_async/token_weighted_staleness"] == 3

    metrics = buffer.get_metrics()
    assert "rollout/fully_async/weight_version_sample_coverage" not in metrics
    assert "rollout/fully_async/token_weighted_staleness" not in metrics


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
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: unused.append(group))
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

        entry = await get_one(buffer, trainer_model_id="verifier")

        assert [sample.trainer_model_id for sample in data_buffer.iter_samples(entry.group)] == ["verifier"]

    async def test_a_policy_waits_for_its_own_queue_instead_of_taking_from_another(self):
        """A policy that consumed a queue it does not own would starve the policy that does."""
        buffer, _ = make_multi_buffer("solver", "verifier")
        await put_group(buffer, make_multi_policy_group(1, "solver", "solver"))

        waiting = asyncio.create_task(get_one(buffer, trainer_model_id="verifier"))
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
        drained = asyncio.create_task(get_one(buffer, current_version=9, trainer_model_id="solver"))
        await asyncio.sleep(0.01)

        assert unused == [group]
        drained.cancel()

    async def test_getting_for_a_policy_this_run_does_not_train_is_refused(self):
        """A typo in the trainer's model id would wait forever on a queue that is never fed."""
        buffer, _ = make_multi_buffer("solver", "verifier")

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            await get_one(buffer, trainer_model_id="reviewer")

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

        assert data_buffer._filter_group([solver, verifier], trainer_model_id="solver") == [solver]

    def test_it_keeps_a_sub_group_that_still_has_samples(self):
        """A trajectory whose samples are split across policies survives on both sides, one sample each."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")

        assert data_buffer._filter_group([[solver, verifier]], trainer_model_id="solver") == [[solver]]

    def test_it_drops_a_sub_group_that_lost_every_sample(self):
        """An empty list left in place would be a trajectory that consumers must special-case forever."""
        assert data_buffer._filter_group([[make_tagged_sample(1, "verifier")]], trainer_model_id="solver") == []

    def test_it_leaves_the_group_it_was_given_untouched(self):
        """It runs once per policy over the same group, so a mutating filter would eat the later policies' samples."""
        solver, verifier = make_tagged_sample(1, "solver"), make_tagged_sample(2, "verifier")
        group = [solver, [verifier]]

        data_buffer._filter_group(group, trainer_model_id="solver")

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

        assert RecordingBuffer.constructed_with.unused_handler_fn == unused.append
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

    async def put(self, input: data_buffer.DataBufferInput) -> None:
        await self._never.wait()

    async def get(self, **context) -> list[data_buffer.DataBufferInput]:
        await self._never.wait()
        raise AssertionError("the wedged buffer never hands out a group")

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        return {}


class TestBatchedGet:
    async def test_a_whole_batch_is_handed_over_in_one_get(self):
        """Half a batch held in a caller local would vanish from the checkpoint the buffer still owns."""
        buffer, _ = make_buffer()
        for group_index in (1, 2, 3):
            await put_group(buffer, make_group(group_index))

        entries = await buffer.get(num_groups=2)

        assert [entry.group[0].group_index for entry in entries] == [1, 2]
        assert [entry.group[0].group_index for entry in buffer._buffer] == [3]

    async def test_an_incomplete_batch_keeps_every_group_in_the_buffer(self):
        """A wait that parked groups outside the buffer would lose them on a checkpoint taken meanwhile."""
        buffer, _ = make_buffer()
        await put_group(buffer, make_group(1))

        waiting = asyncio.create_task(buffer.get(num_groups=2))
        await asyncio.sleep(0.01)

        assert not waiting.done()
        assert [entry.group[0].group_index for entry in buffer._buffer] == [1]
        waiting.cancel()

    async def test_stale_groups_are_recycled_while_the_batch_is_still_incomplete(self):
        """A buffer filled with stale groups would block put forever while the get waits for a count."""
        buffer, unused = make_buffer(max_groups=2, max_staleness=0)
        stale = make_group(1, weight_versions=["1"])
        await put_group(buffer, stale)

        waiting = asyncio.create_task(buffer.get(num_groups=2, current_version=9))
        await asyncio.sleep(0.01)

        assert unused == [stale]
        assert buffer._buffer == []
        waiting.cancel()

    async def test_a_capacity_below_one_batch_is_refused(self):
        """A buffer that cannot hold one batch deadlocks: put blocks full while get waits for the count."""
        args = make_args(rollout_batch_size=4, async_data_buffer_capacity_factor=0.5)

        with pytest.raises(AssertionError, match="below the rollout batch"):
            data_buffer.DefaultDataBuffer(
                data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=lambda group, reason: None)
            )

    async def test_a_silent_producer_warns_and_keeps_waiting(self, monkeypatch, caplog):
        """A wait that produces nothing has to say so, and stay a wait rather than fail the step."""
        monkeypatch.setattr(data_buffer, "NO_PROGRESS_WARN_SECS", 0.01)
        buffer, _ = make_buffer()
        caplog.set_level(logging.WARNING, logger=data_buffer.__name__)

        waiting = asyncio.create_task(buffer.get(num_groups=1))
        await asyncio.sleep(0.05)

        assert not waiting.done()
        assert "No completed rollout groups" in caplog.text
        waiting.cancel()


class TestDisposal:
    async def test_disposing_ends_the_producer_before_it_returns(self, monkeypatch):
        """Teardown has to be done when it returns, or the rest of it races the producer's unwinding."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0.05)
        assert not step.done()

        await asyncio.wait_for(fn.dispose(), timeout=5)

        assert fn._worker.cancelled()

    async def test_disposing_fails_a_step_that_is_waiting_for_groups(self, monkeypatch):
        """A run whose producer is parked when it ends must not leave the waiting step there forever."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await asyncio.sleep(0.05)

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

    async def blocking_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2


# ============================== shared helpers ==============================


async def _settle(times: int = 50) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def _parked_task() -> asyncio.Task:
    return asyncio.create_task(asyncio.Event().wait())


class _GatedGenerate:
    def __init__(self) -> None:
        self.gates: list[asyncio.Event] = []
        self.callbacks: list = []
        self.open = False

    async def __call__(self, state, group, sampling_params=None, evaluation=False, sample_done_callback=None):
        gate = asyncio.Event()
        if self.open:
            gate.set()
        self.gates.append(gate)
        self.callbacks.append(sample_done_callback)
        await gate.wait()
        for sample in group:
            sample.status = Sample.Status.COMPLETED
        return group

    def release(self, *indices: int) -> None:
        for index in indices:
            self.gates[index].set()

    def release_all(self) -> None:
        self.open = True
        for gate in self.gates:
            gate.set()


class _RecordingScheduler:
    sample_done_callback = None

    def __init__(self) -> None:
        self.submitted: list[list[list[Sample]]] = []

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return False

    def on_submit(self, groups: list[list[Sample]]) -> None:
        self.submitted.append(groups)

    async def wait_for_progress(self, pendings: set) -> tuple[set, set]:
        return set(), pendings


class _RecordingPutBuffer(data_buffer.DefaultDataBuffer):
    puts: list = []

    async def put(self, input: data_buffer.DataBufferInput) -> None:
        _RecordingPutBuffer.puts.append(input)
        await super().put(input)


class _MetricsRecordingBuffer(data_buffer.DefaultDataBuffer):
    metrics_asked_about: list = []

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        _MetricsRecordingBuffer.metrics_asked_about.append(trainer_model_id)
        return {"asked": 1.0}


class _CancelWatchingBuffer(data_buffer.DataBuffer):
    def __init__(self) -> None:
        self.cancelled = False

    async def put(self, input: data_buffer.DataBufferInput) -> None:
        raise AssertionError("this buffer is only ever waited on")

    async def get(self, **context) -> data_buffer.DataBufferInput:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        return {}


# ======================== submission and concurrency ========================


class TestInFlightBudget:
    def test_a_sample_budget_that_is_not_a_whole_number_of_groups_floors_to_whole_groups(self, monkeypatch) -> None:
        """Whole groups are submitted, so an odd sample budget buys only the groups that fit inside it."""
        fn = make_fn(monkeypatch, make_args(async_max_concurrent_samples=5), FakeDataSource())

        assert fn._max_in_flight_groups() == 2

    def test_a_sample_budget_smaller_than_one_group_still_keeps_one_group_in_flight(self, monkeypatch) -> None:
        """A budget that floors to zero groups would stop the producer from ever submitting anything."""
        fn = make_fn(monkeypatch, make_args(async_max_concurrent_samples=1), FakeDataSource())

        assert fn._max_in_flight_groups() == 1

    def test_an_unset_sample_budget_bounds_in_flight_groups_by_the_rollout_batch_size(self, monkeypatch) -> None:
        """Without --async-max-concurrent-samples the producer keeps one training batch in flight."""
        args = make_args(rollout_batch_size=7, async_max_concurrent_samples=None)
        fn = make_fn(monkeypatch, args, FakeDataSource())

        assert fn._max_in_flight_groups() == 7

    async def test_a_finished_group_frees_exactly_one_submission_slot(self, monkeypatch) -> None:
        """At group granularity the slot a group holds comes back only when that whole group returns."""
        generate = _GatedGenerate()
        source = FakeDataSource()
        args = make_args(rollout_batch_size=2, rollout_submission_granularity="group")
        fn = make_fn(monkeypatch, args, source, generate=generate)

        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await _settle()
        assert source.num_get_calls == 2

        generate.release(0)
        await _settle()
        assert source.num_get_calls == 3

        generate.release_all()
        assert len((await step).samples) == 2

    async def test_the_producer_submits_nothing_more_while_an_eval_pause_is_in_effect(self, monkeypatch) -> None:
        """The shared-engine pause has to hold the producer between groups, not merely stop new eval work."""
        generate = _GatedGenerate()
        source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), source, generate=generate)

        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await _settle()
        assert source.num_get_calls == 1

        fn._producer_resumed.clear()
        generate.release_all()
        await step
        await _settle()
        assert source.num_get_calls == 1

        fn._producer_resumed.set()
        await _settle()
        assert source.num_get_calls == 2

    async def test_every_submitted_group_is_reported_to_the_submission_scheduler(self, monkeypatch) -> None:
        """The scheduler paces on the samples it was told about, and an unreported group is free capacity."""
        group = make_group(7)
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource(scripted=[group]))
        scheduler = _RecordingScheduler()
        fn._scheduler = scheduler

        await fn._submit_one_group()

        assert scheduler.submitted == [[group]]

    async def test_a_retried_prompt_group_is_preferred_over_the_data_source_in_fifo_order(self, monkeypatch) -> None:
        """Recycled prompts are a queue: the oldest failure is the one that has waited longest to be redone."""
        source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), source)
        first, second = make_group(1), make_group(2)
        fn._recycle(first, data_buffer.UnusedReason.ABORTED)
        fn._recycle(second, data_buffer.UnusedReason.ABORTED)

        assert (await fn._submit_one_group()).prompt_group is first
        assert (await fn._submit_one_group()).prompt_group is second
        assert source.num_get_calls == 0

        assert (await fn._submit_one_group()).prompt_group is not second
        assert source.num_get_calls == 1

    async def test_a_finished_group_reaches_the_buffer_with_its_prompt_group_intact(self, monkeypatch) -> None:
        """Recycling resubmits the prompt group, which the buffer can only do if the put carried it along."""
        _RecordingPutBuffer.puts = []
        prompt_group = make_group(1)
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}._RecordingPutBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource(scripted=[prompt_group]))

        output = await fn(RolloutFnTrainInput(rollout_id=0))

        assert len(output.samples) == 1
        assert _RecordingPutBuffer.puts[0].prompt_group is prompt_group

    async def test_a_full_buffer_stops_the_producer_from_submitting_more_groups(self, monkeypatch) -> None:
        """The put lives in the producer loop, so a full buffer wedges submission even with the budget free."""

        async def generate_reporting_samples(
            state, group, sampling_params=None, evaluation=False, sample_done_callback=None
        ):
            for sample in group:
                sample.status = Sample.Status.COMPLETED
                sample_done_callback()
            return group

        source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=8), source, generate=generate_reporting_samples)
        buffer, _ = make_buffer(max_groups=1)
        fn._output = buffer
        await put_group(buffer, make_group(99))

        fn._worker = asyncio.create_task(fn._worker_loop())
        await _settle()

        assert source.num_get_calls == 8
        assert fn._scheduler.samples_in_flight == 0
        assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 1

        await fn.dispose()


# ============================== unused handler ==============================


class TestUnusedHandler:
    def test_a_recycled_prompt_group_is_reset_before_it_is_queued_again(self, monkeypatch) -> None:
        """A prompt still carrying the last attempt's response and versions would be regenerated as stale output."""
        fn = make_fn(monkeypatch, make_args(async_unused_samples_handler="retry"), FakeDataSource())
        group = make_group(1, weight_versions=["3"])

        fn._recycle(group, data_buffer.UnusedReason.ABORTED)

        [queued] = fn._retry_buffer
        assert queued is group
        assert [sample.response for sample in queued] == ["", ""]
        assert [sample.weight_versions for sample in queued] == [[], []]
        assert [sample.reward for sample in queued] == [None, None]

    def test_a_recycled_group_records_no_explicitly_dropped_event(self, monkeypatch, sample_flow_event_dir) -> None:
        """A recycled prompt is regenerated, so calling it dropped would abandon a sample still in flight."""
        args = make_args(enable_sample_ownership_checker=True, async_unused_samples_handler="retry")
        fn = make_fn(monkeypatch, args, FakeDataSource())

        fn._handle_unused(make_group(1), data_buffer.UnusedReason.ABORTED)

        assert read_events(sample_flow_event_dir) == []

    @pytest.mark.parametrize("reason", [data_buffer.UnusedReason.ABORTED, data_buffer.UnusedReason.STALE])
    def test_a_dropped_group_records_the_unused_reason_it_was_dropped_for(
        self, monkeypatch, sample_flow_event_dir, reason
    ) -> None:
        """The ownership audit has to tell an abandoned generation from output that simply arrived too late."""
        args = make_args(enable_sample_ownership_checker=True, async_unused_samples_handler="drop")
        fn = make_fn(monkeypatch, args, FakeDataSource())

        fn._handle_unused(make_group(1), reason)

        [event] = read_events(sample_flow_event_dir)
        assert isinstance(event, ExplicitlyDroppedSamplesEvent)
        assert event.source_sample_indices == [10, 11]
        assert event.reason == reason.value

    def test_the_unused_handler_flag_selects_between_recycling_and_dropping(self, monkeypatch) -> None:
        """--async-unused-samples-handler is the only switch between regenerating a prompt and abandoning it."""
        retrying = make_fn(monkeypatch, make_args(async_unused_samples_handler="retry"), FakeDataSource())
        dropping = make_fn(monkeypatch, make_args(async_unused_samples_handler="drop"), FakeDataSource())

        assert retrying._handle_unused == retrying._recycle
        assert dropping._handle_unused == dropping._drop_unused

    def test_an_unknown_unused_samples_handler_is_refused_at_construction(self, monkeypatch) -> None:
        """A typo would silently pick one of the two policies, and which one is not something to guess."""
        with pytest.raises(AssertionError):
            make_fn(monkeypatch, make_args(async_unused_samples_handler="recycle"), FakeDataSource())


# ========================= dead worker ends the wait ========================


class TestDeadWorkerEndsTheWait:
    async def test_a_worker_that_returns_normally_fails_the_waiting_step(self, monkeypatch) -> None:
        """The producer loop never returns, so a quiet exit is a bug that must not read as an empty buffer."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        fn._output = make_buffer()[0]

        async def quiet():
            return None

        fn._worker = asyncio.create_task(quiet())
        await _settle()

        with pytest.raises(RuntimeError, match="worker exited without an exception"):
            await fn(RolloutFnTrainInput(rollout_id=0))

    async def test_a_cancelled_worker_fails_the_waiting_step_with_the_disposal_message(self, monkeypatch) -> None:
        """A cancelled producer means teardown, and the step has to say so instead of reporting a crash."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        fn._output = make_buffer()[0]
        fn._worker = _parked_task()
        await _settle()
        fn._worker.cancel()
        await _settle()

        with pytest.raises(RuntimeError, match="disposed while a step waited for groups"):
            await fn(RolloutFnTrainInput(rollout_id=0))

    async def test_the_pending_buffer_get_is_cancelled_when_the_step_fails(self, monkeypatch) -> None:
        """A get left running would keep draining groups into a step nobody is waiting on any more."""
        buffer = _CancelWatchingBuffer()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        fn._output = buffer

        async def quiet():
            return None

        fn._worker = asyncio.create_task(quiet())
        await _settle()

        with pytest.raises(RuntimeError, match="worker exited without an exception"):
            await fn(RolloutFnTrainInput(rollout_id=0))
        await _settle()

        assert buffer.cancelled

    async def test_a_step_that_waits_too_long_warns_about_missing_progress(self, monkeypatch, caplog) -> None:
        """A silent stall is the hardest fully-async failure to see, so the wait has to say that it is waiting."""
        monkeypatch.setattr(fully_async, "NO_PROGRESS_WARN_SECS", 0.01)
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        fn._output = make_buffer()[0]
        fn._worker = _parked_task()

        with caplog.at_level(logging.WARNING, logger="miles.rollout.fully_async_rollout"):
            step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
            for _ in range(200):
                await asyncio.sleep(0.01)
                if "No completed rollout groups" in caplog.text:
                    break
            step.cancel()
            await asyncio.gather(step, return_exceptions=True)

        fn._worker.cancel()
        assert "No completed rollout groups" in caplog.text

    async def test_a_worker_death_is_noticed_before_its_buffered_groups_are_drained(self, monkeypatch) -> None:
        """A whole batch already in the buffer would otherwise hide the producer's death for a full step."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())
        buffer, _ = make_buffer()
        fn._output = buffer
        await put_group(buffer, make_group(1))
        await put_group(buffer, make_group(2))

        async def boom():
            raise RuntimeError("generation exploded")

        fn._worker = asyncio.create_task(boom())
        await _settle()

        with pytest.raises(RuntimeError, match="generation exploded"):
            await fn(RolloutFnTrainInput(rollout_id=0))


# ================================== drain ===================================


class TestDrain:
    async def test_a_batch_is_returned_only_when_the_whole_rollout_batch_is_ready(self, monkeypatch) -> None:
        """A step trains on a full batch, so a partly filled one may not be handed back early."""
        generate = _GatedGenerate()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=3), FakeDataSource(), generate=generate)

        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await _settle()
        generate.release(0, 1)
        await _settle()

        assert not step.done()

        generate.release_all()
        assert len((await step).samples) == 3

    async def test_the_batch_is_sorted_by_sample_index_whatever_the_completion_order(self, monkeypatch) -> None:
        """The trainer pairs the batch with its own ordering, and a shuffled batch trains the wrong pairs."""
        generate = _GatedGenerate()
        source = FakeDataSource(scripted=[make_group(5), make_group(3), make_group(1)])
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=3), source, generate=generate)

        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await _settle()
        generate.release(1)
        await _settle()
        generate.release(2)
        await _settle()
        generate.release(0)

        output = await step

        assert [data_buffer.first_sample(group).index for group in output.samples] == [10, 30, 50]

    async def test_a_group_that_does_not_hold_n_samples_per_prompt_is_refused(self, monkeypatch) -> None:
        """A short group would silently train on fewer trajectories than the advantage baseline assumes."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        buffer, _ = make_buffer()
        fn._output = buffer
        short = make_group(1)[:1]
        await buffer.put(data_buffer.DataBufferInput(prompt_group=short, group=short))
        fn._worker = _parked_task()

        with pytest.raises(AssertionError):
            await fn(RolloutFnTrainInput(rollout_id=0))

        fn._worker.cancel()

    async def test_a_run_without_the_global_rollout_dataset_is_refused(self, monkeypatch) -> None:
        """This driver drains one shared buffer, which a per-rank dataset would fill with disjoint prompts."""
        args = make_args(rollout_batch_size=1, rollout_global_dataset=False)
        fn = make_fn(monkeypatch, args, FakeDataSource())

        with pytest.raises(AssertionError):
            await fn(RolloutFnTrainInput(rollout_id=0))

        await fn.dispose()

    async def test_only_the_first_group_of_a_batch_is_previewed_in_the_log(self, monkeypatch, caplog) -> None:
        """One preview per step samples the batch; one per group is a wall of text on every single step."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=3), FakeDataSource())

        with caplog.at_level(logging.INFO, logger="miles.rollout.fully_async_rollout"):
            await fn(RolloutFnTrainInput(rollout_id=0))

        assert sum("First rollout sample:" in record.getMessage() for record in caplog.records) == 1

    async def test_a_sample_filter_that_drops_groups_records_them_as_explicitly_dropped(
        self, monkeypatch, sample_flow_event_dir
    ) -> None:
        """A filter that shortens the batch resolves those samples, and an unresolved one looks lost forever."""

        def drop_the_first_group(args, data):
            data.pop(0)

        args = make_args(rollout_batch_size=2, enable_sample_ownership_checker=True)
        source = FakeDataSource(scripted=[make_group(1), make_group(2)])
        fn = make_fn(monkeypatch, args, source)
        fn._sample_filter = drop_the_first_group

        output = await fn(RolloutFnTrainInput(rollout_id=0))

        assert [group[0].group_index for group in output.samples] == [2]
        [event] = read_events(sample_flow_event_dir)
        assert event.reason == "rollout_sample_filter"
        assert event.source_sample_indices == [10, 11]

    async def test_an_unset_sample_filter_leaves_the_batch_untouched_and_records_no_drop(
        self, monkeypatch, sample_flow_event_dir
    ) -> None:
        """--rollout-sample-filter-path is opt-in, and an unset one may not cost the step a single sample."""
        args = make_args(rollout_batch_size=2, enable_sample_ownership_checker=True)
        fn = make_fn(monkeypatch, args, FakeDataSource())

        output = await fn(RolloutFnTrainInput(rollout_id=0))

        assert fn._sample_filter is None
        assert len(output.samples) == 2
        assert read_events(sample_flow_event_dir) == []

    async def test_the_step_metrics_come_from_the_buffer_of_the_calling_policy(self, monkeypatch) -> None:
        """A multi policy run reports one curve per policy, which a drain that forgets its own name cannot do."""
        _MetricsRecordingBuffer.metrics_asked_about = []
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}._MetricsRecordingBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())

        output = await fn(RolloutFnTrainInput(rollout_id=0, trainer_model_id="solver"))

        assert _MetricsRecordingBuffer.metrics_asked_about == ["solver"]
        assert output.metrics == {"asked": 1.0}


# ========================== lifecycle, eval, buffer =========================


class TestLifecycle:
    async def test_the_worker_starts_once_and_is_reused_by_later_steps(self, monkeypatch) -> None:
        """A second producer would submit its own groups on top of the first one's, past every budget."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=1))
        worker, output = fn._worker, fn._output
        await fn(RolloutFnTrainInput(rollout_id=1, weight_version=1))

        assert fn._worker is worker
        assert fn._output is output

    async def test_the_buffer_is_built_at_construction_so_a_restore_can_fill_it(self, monkeypatch) -> None:
        """load() runs before the first train step, so it has to restore into a buffer that already exists."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        assert type(fn._output) is data_buffer.DefaultDataBuffer
        assert fn._worker is None
        output = fn._output

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert fn._output is output

    async def test_an_evaluation_only_call_never_starts_the_producer(self, monkeypatch) -> None:
        """A checkpoint-eval-only process would otherwise generate training rollouts that nobody drains."""
        source = FakeDataSource()
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), source)

        async def fake_run_eval_datasets(state, cache):
            return {}

        monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

        await fn(RolloutFnEvalInput(rollout_id=0))

        assert fn._worker is None
        assert source.num_get_calls == 0

    async def test_an_eval_that_raises_still_resumes_the_producer(self, monkeypatch) -> None:
        """A failed eval that left the producer paused would stall every later training step."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

        async def failing_run_eval_datasets(state, cache):
            raise RuntimeError("eval exploded")

        monkeypatch.setattr(fully_async, "run_eval_datasets", failing_run_eval_datasets)

        with pytest.raises(RuntimeError, match="eval exploded"):
            await fn(RolloutFnEvalInput(rollout_id=0))

        assert fn._producer_resumed.is_set()

    async def test_the_eval_prompt_dataset_cache_is_reused_across_eval_calls(self, monkeypatch) -> None:
        """Rebuilding the eval datasets every call re-reads and re-tokenizes them on the rollout path."""
        fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
        caches = []

        async def fake_run_eval_datasets(state, cache):
            caches.append(cache)
            return {}

        monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

        await fn(RolloutFnEvalInput(rollout_id=0))
        await fn(RolloutFnEvalInput(rollout_id=1))

        assert caches[0] is fn._eval_prompt_dataset_cache
        assert caches[1] is fn._eval_prompt_dataset_cache

    async def test_a_dedicated_fleet_eval_leaves_the_producer_running(self, monkeypatch) -> None:
        """The fleet has engines of its own, so pausing the shared producer would only waste rollout capacity."""
        args = make_args(rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        resumed_during_eval = []

        async def fake_run_eval_datasets(state, cache):
            resumed_during_eval.append(fn._producer_resumed.is_set())
            return {}

        monkeypatch.setattr(fully_async, "run_eval_datasets", fake_run_eval_datasets)

        await fn(RolloutFnEvalInput(rollout_id=0, generate_state=FakeGenerateState(args)))

        assert resumed_during_eval == [True]

    async def test_disposing_twice_is_quiet(self, monkeypatch) -> None:
        """Teardown runs from several paths, and a second call must not turn a clean shutdown into an error."""
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=f"{__name__}.WedgedBuffer")
        fn = make_fn(monkeypatch, args, FakeDataSource())
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        await _settle()

        await fn.dispose()
        await fn.dispose()

        assert fn._worker.cancelled()
        with pytest.raises(RuntimeError, match="disposed while a step waited for groups"):
            await step

    async def test_the_custom_buffer_is_built_with_the_rollout_functions_unused_handler(self, monkeypatch) -> None:
        """A custom buffer owns every keep-or-recycle decision, so it needs the run's configured policy."""
        args = make_args(
            rollout_batch_size=1,
            async_unused_samples_handler="drop",
            custom_async_data_buffer_path=f"{__name__}.RecordingBuffer",
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert RecordingBuffer.constructed_with.unused_handler_fn == fn._drop_unused
        assert RecordingBuffer.constructed_with.args is args

    async def test_a_custom_buffer_path_that_names_nothing_falls_back_to_the_built_in_buffer(
        self, monkeypatch
    ) -> None:
        """The flag is unset on every default run, so an empty path may not resolve to a missing class."""
        assert load_function(None) is None
        args = make_args(rollout_batch_size=1, custom_async_data_buffer_path=None)
        fn = make_fn(monkeypatch, args, FakeDataSource())

        await fn(RolloutFnTrainInput(rollout_id=0))

        assert type(fn._output) is data_buffer.DefaultDataBuffer
