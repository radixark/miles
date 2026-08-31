from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from collections import deque
from dataclasses import replace

import pytest

import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

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
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        self.recycled.extend(groups)


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=status,
            weight_versions=list(weight_versions or []),
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
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        eval_num_gpus=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await asyncio.sleep(0)
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
    Building/caching the fleet state itself is EvalFleetSession's job, covered in
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


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1
    assert output.metrics["rollout/aborted/drop_unknown"] == 1
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_filtered"] == 1


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
                    sample.weight_versions = list(data_source_fresh_versions)
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    args = make_args(rollout_batch_size=1, max_weight_staleness=2, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 5


async def test_stale_group_dropped_by_default(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    data_source = FakeDataSource(scripted=[stale])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


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
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    args = make_args(rollout_batch_size=1, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
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
    assert data_source.recycled == []
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

    assert data_source.recycled == []
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
        data_buffer.DataBufferConstructorInput(args=args, unused_handler_fn=unused.append)
    )
    return buffer, unused


async def put_group(buffer, group):
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(prompt_group=group, group=group))


async def test_buffer_blocks_producer_when_full():
    buffer, _ = make_buffer(max_groups=2)
    await put_group(buffer, make_group(1))
    await put_group(buffer, make_group(2))

    blocked = asyncio.create_task(put_group(buffer, make_group(3)))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    assert (await buffer.get()).group[0].group_index == 1
    await blocked
    assert (await buffer.get()).group[0].group_index == 2
    assert (await buffer.get()).group[0].group_index == 3


async def test_buffer_get_ignores_unknown_context_keys():
    """get(**context) lets the driver add keys without breaking existing buffers."""
    buffer, _ = make_buffer()
    await put_group(buffer, make_group(1))

    assert (await buffer.get(current_version=1, some_future_key=2)).group[0].group_index == 1


async def test_buffer_get_skips_groups_stale_at_consumption_time():
    """Both groups were fresh when buffered; only the version passed to get() decides."""
    buffer, unused = make_buffer(max_staleness=2)
    stale = make_group(1, weight_versions=["5"])
    await put_group(buffer, stale)
    await put_group(buffer, make_group(2, weight_versions=["9"]))

    assert (await buffer.get(current_version=10)).group[0].group_index == 2
    assert unused == [stale]
    assert buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1


async def test_buffer_staleness_metrics():
    buffer, _ = make_buffer(max_groups=8)
    await put_group(buffer, make_group(1, weight_versions=["4"]))
    assert "rollout/fully_async/buffer_avg_staleness" not in buffer.get_metrics()  # engine version never seen

    await put_group(buffer, make_group(2, weight_versions=["6"]))
    await put_group(buffer, make_group(3, weight_versions=["8"]))
    await buffer.get(current_version=10)  # pops group 1 and tracks the engine version clock
    metrics = buffer.get_metrics()
    assert metrics["rollout/fully_async/avg_staleness"] == 6.0  # consumed group 1: 10 - 4
    assert metrics["rollout/fully_async/buffer_avg_staleness"] == 3.0  # buffered groups 2, 3: (4 + 2) / 2
    assert metrics["rollout/fully_async/buffer_max_staleness"] == 4


class RecordingBuffer(data_buffer.DefaultDataBuffer):
    constructed_with = None

    def __init__(self, input):
        super().__init__(input)
        RecordingBuffer.constructed_with = input


async def test_custom_data_buffer_path_replaces_default(monkeypatch):
    path = f"{__name__}.RecordingBuffer"
    args = make_args(custom_async_data_buffer_path=path, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert type(fn._output) is RecordingBuffer
    assert RecordingBuffer.constructed_with.unused_handler_fn == fn._recycle
    assert len(output.samples) == 2


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
