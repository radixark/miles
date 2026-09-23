import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

import miles.rollout.fully_async_rollout as fully_async
from miles.ray.rollout import rollout_executor as executor_module
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class PromptDataset:
    def __init__(self):
        self.samples = [Sample(prompt=f"prompt {i}", metadata={"original": [i]}) for i in range(32)]

    def __len__(self):
        return len(self.samples)


def make_source(path: Path, **overrides):
    args = SimpleNamespace(
        rollout_global_dataset=False,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        save=str(path),
        load=str(path),
        rollout_batch_size=1,
        async_max_concurrent_samples=6,
        rollout_submission_granularity="group",
        async_data_buffer_capacity_factor=1,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        max_weight_staleness=0,
        reward_key=None,
    )
    vars(args).update(overrides)
    source = RolloutDataSourceWithBuffer(args)
    source.dataset = PromptDataset()
    args.rollout_global_dataset = True
    source.enable_checkpoint_replay()
    return source


def group_ids(groups):
    return [group[0].group_index for group in groups]


def test_resume_replays_original_unconsumed_prompts_once(tmp_path):
    source = make_source(tmp_path)
    groups = source.get_samples(6)
    # Training completes out of issue order; group 2 is deliberately discarded.
    source.acknowledge([3, 1, 2])
    groups[0][0].response = "generated response"
    groups[0][0].metadata["original"].append("mutated")
    source.add_samples([groups[4]])  # a queued retry must also appear only once on load
    source.save(7)

    restored = make_source(tmp_path)
    restored.load(7)
    replay = restored.get_samples(3)
    assert group_ids(replay) == [0, 4, 5]
    assert replay[0][0].response == ""
    assert replay[0][0].metadata == {"original": [0]}
    assert [sample.index for group in replay for sample in group] == [0, 1, 8, 9, 10, 11]

    # A second crash while replaying must preserve the still-unconsumed original inputs.
    replay[0][0].response = "regenerated"
    restored.acknowledge([4])
    restored.save(8)
    twice = make_source(tmp_path)
    twice.load(8)
    assert group_ids(twice.get_samples(3)) == [0, 5, 6]
    assert twice.sample_offset == 7


def test_pipelined_checkpoint_keeps_prefetch_and_retry_but_not_filter_rejects(tmp_path):
    source = make_source(tmp_path)
    source.get_samples(1)  # current training batch
    source.finish_rollout({0})
    lookahead = source.get_samples(3)
    source.add_samples([lookahead[2]])
    source.finish_rollout({1})  # keep batch 1, discard 2, retry 3
    source.acknowledge([0])
    source.save(0)

    restored = make_source(tmp_path)
    restored.load(0)
    assert group_ids(restored.get_samples(3)) == [1, 3, 4]


def test_checkpoint_cannot_split_cursor_advancement_from_pending_registration(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    advanced = Event()
    release = Event()
    saving = Event()
    get_samples = source._get_samples

    def held_get(num_samples):
        groups = get_samples(num_samples)
        advanced.set()
        assert release.wait(10)
        return groups

    def save():
        saving.set()
        source.save(0)

    monkeypatch.setattr(source, "_get_samples", held_get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(source.get_samples, 2)
        assert advanced.wait(10)
        checkpoint = pool.submit(save)
        try:
            assert saving.wait(10)
            assert not checkpoint.done()
        finally:
            release.set()
        generation.result(timeout=10)
        checkpoint.result(timeout=10)

    restored = make_source(tmp_path)
    restored.load(0)
    assert group_ids(restored.get_samples(3)) == [0, 1, 2]


def test_failed_checkpoint_write_preserves_previous_snapshot(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    source.get_samples(1)
    source.save(0)
    source.get_samples(1)

    def interrupted_write(state, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        source.save(0)
    restored = make_source(tmp_path)
    restored.load(0)
    assert restored.sample_offset == 1
    assert group_ids(restored.get_samples(2)) == [0, 1]


@pytest.mark.parametrize("reason", ["aborted", "stale", "missing_reward", "dynamic_filter"])
@pytest.mark.parametrize("handler", ["drop", "retry"])
async def test_resume_respects_drop_and_retry_decisions(tmp_path, monkeypatch, reason, handler):
    source = make_source(tmp_path, async_unused_samples_handler=handler)
    monkeypatch.setattr(fully_async, "GenerateState", lambda args: SimpleNamespace(args=args, sampling_params={}))
    fn = fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=source.args, data_source=source))
    buffer = DefaultDataBuffer(DataBufferConstructorInput(source.args, fn._handle_unused, fn._discard))
    rejected, accepted = source.get_samples(2)
    for group in (rejected, accepted):
        for sample in group:
            sample.status = Sample.Status.COMPLETED
            sample.reward = 1
    if reason == "aborted":
        rejected[0].status = Sample.Status.ABORTED
    elif reason == "stale":
        rejected[0].weight_versions = [WeightVersionsPerCall([WeightVersionSpan("0", 0, 1)])]
    elif reason == "missing_reward":
        rejected[0].reward = None
    else:
        buffer._dynamic_filter = lambda args, group: group[0].group_index != 0
    await buffer.put(DataBufferInput(rejected, rejected))
    # Stale entries are rejected during get(), so let the consumer free the single queue slot.
    drain = asyncio.create_task(buffer.get(current_version=1))
    await buffer.put(DataBufferInput(accepted, accepted))
    assert (await asyncio.wait_for(drain, timeout=10)).prompt_group is accepted
    source.acknowledge([1])
    source.save(0)

    restored = make_source(tmp_path)
    restored.load(0)
    retry = handler == "retry" and reason in {"aborted", "stale"}
    assert group_ids(restored.get_samples(1)) == [0 if retry else 2]


async def test_checkpoint_covers_prefetched_queued_and_inflight_groups(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    started_inflight = asyncio.Event()
    release = asyncio.Event()
    generation_tasks = set()

    async def generate(state, group, **kwargs):
        generation_tasks.add(asyncio.current_task())
        if group[0].group_index >= 3:
            started_inflight.set()
            await release.wait()
        for sample in group:
            sample.reward = 1
            sample.status = Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(fully_async, "GenerateState", lambda args: SimpleNamespace(args=args, sampling_params={}))
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate)
    fn = fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=source.args, data_source=source))
    try:
        current = await asyncio.wait_for(fn(RolloutFnTrainInput(0)), timeout=10)
        trained = set(group_ids(current.samples))
        source.acknowledge(trained)
        prefetched = await asyncio.wait_for(fn(RolloutFnTrainInput(1)), timeout=10)
        await asyncio.wait_for(started_inflight.wait(), timeout=10)
        # One completed group remains queued, alongside a driver batch and active generation.
        assert fn._output._buffer
        assert group_ids(prefetched.samples)[0] not in trained
        issued = source.sample_group_index
        source.save(0)
    finally:
        fn._worker.cancel()
        for task in generation_tasks:
            task.cancel()
        await asyncio.gather(fn._worker, *generation_tasks, return_exceptions=True)

    restored = make_source(tmp_path)
    restored.load(0)
    replay = restored.get_samples(issued - len(trained))
    assert group_ids(replay) == sorted(set(range(issued)) - trained)
    assert len(set(group_ids(replay))) == len(replay)
    assert group_ids(restored.get_samples(1)) == [issued]


@pytest.mark.parametrize("fully_async_mode", [False, True])
async def test_executor_acknowledges_only_the_trained_batch(tmp_path, monkeypatch, fully_async_mode):
    source = make_source(tmp_path, fully_async=fully_async_mode, load_debug_rollout_data=None)
    executor = object.__new__(executor_module.RolloutExecutor.__ray_actor_class__)
    executor.args = source.args
    executor.data_source = executor._checkpoint_source = source
    executor._pending_rollouts = {}
    executor.weight_version = 0
    executor.train_parallel_config = {}
    executor.use_legacy_rollout_v1 = False
    executor.generate_rollout = lambda input: RolloutFnTrainOutput(samples=source.get_samples(2))
    # Exercise the executor's handling of groups dropped by postprocessing as well.
    monkeypatch.setattr(executor_module, "postprocess_rollout_data", lambda args, data, **kw: (data[0], {}))
    monkeypatch.setattr(executor_module, "assert_samples_weight_version_sane", lambda *a, **kw: None)
    monkeypatch.setattr(executor_module.RolloutDataInjectionUtil, "should_inject", lambda *a: False)

    await executor._get_rollout_data(0)
    await executor._get_rollout_data(1)  # driver has prefetched this batch but has not trained it
    executor.acknowledge(0)
    source.save(0)

    restored = make_source(tmp_path)
    restored.load(0)
    assert group_ids(restored.get_samples(2)) == [2, 4]


def test_older_cursor_only_checkpoints_still_load(tmp_path):
    source = make_source(tmp_path)
    source._checkpoint_replay = False
    source.get_samples(2)
    source.save(0)
    restored = make_source(tmp_path)
    restored.load(0)
    assert group_ids(restored.get_samples(1)) == [2]
