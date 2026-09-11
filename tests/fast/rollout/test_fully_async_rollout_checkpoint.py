import asyncio
from argparse import Namespace
from pathlib import Path

import torch
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.types import Sample


def _checkpoint_args(path: Path, **overrides: object) -> Namespace:
    return make_args(save=str(path), load=str(path), rollout_batch_size=1, **overrides)


async def test_in_flight_generation_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A half-written generation is replayed from an untouched prompt after restore."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    group = make_group(3)
    group[0].response = "partial"
    task = asyncio.Future()
    fn._in_flight[task] = group

    fn.save(4)
    restored = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    restored.load(4)

    [pending] = restored._retry_buffer
    assert all(sample.response == "" for sample in pending)


async def test_aborted_group_in_retry_queue_survives_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """An aborted group saved after recycling resumes exactly once from its prompt."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path, async_unused_samples_handler="retry"), FakeDataSource())
    fn._ensure_output()
    group = make_group(6)
    for sample in group:
        sample.status = Sample.Status.ABORTED
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))

    fn.save(3)
    restored = make_fn(monkeypatch, fn.args, FakeDataSource())
    restored.load(3)

    [pending] = restored._retry_buffer
    assert [sample.index for sample in pending] == [60, 61]
    assert all(sample.response == "" for sample in pending)


async def test_pending_admission_is_not_filtered_twice_after_restore(monkeypatch, tmp_path: Path) -> None:
    """A capacity-blocked admission keeps its completed filter decision across restart."""
    args = _checkpoint_args(tmp_path, async_data_buffer_capacity_factor=1)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._ensure_output()
    first, second = make_group(1), make_group(2)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    calls = 0

    def keep_once(args: object, group: object) -> DynamicFilterOutput:
        nonlocal calls
        calls += 1
        return DynamicFilterOutput(keep=calls == 1)

    fn._output._dynamic_filter = keep_once
    fn._pending_outputs.append(DataBufferInput(prompt_group=second, group=second))
    blocked = asyncio.create_task(fn._flush_pending_outputs())
    await asyncio.sleep(0)
    assert calls == 1
    fn.save(1)
    blocked.cancel()
    await asyncio.gather(blocked, return_exceptions=True)

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(1)
    restored._ensure_output()
    restored._output._dynamic_filter = keep_once
    await restored._output.get()
    await restored._flush_pending_outputs()

    assert calls == 1
    assert (await restored._output.get()).group[0].group_index == 2


async def test_stored_put_waiting_for_return_is_saved_only_in_the_buffer(monkeypatch, tmp_path: Path) -> None:
    """A stored put cannot be duplicated while its caller still waits to resume."""
    args = _checkpoint_args(tmp_path)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    stored = asyncio.Event()
    release = asyncio.Event()

    class _AcknowledgingBuffer(DefaultDataBuffer):
        async def put(self, input: DataBufferInput) -> None:
            await super().put(input)
            stored.set()
            await release.wait()

    fn._output = _AcknowledgingBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused))
    group = make_group(8)
    fn._pending_outputs.append(DataBufferInput(prompt_group=group, group=group))
    putting = asyncio.create_task(fn._flush_pending_outputs())
    await stored.wait()

    fn.save(5)
    putting.cancel()
    await asyncio.gather(putting, return_exceptions=True)
    state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=5), weights_only=False)

    assert state["pending_outputs"] == []
    assert [sample.index for sample in state["output"][None][0].group] == [80, 81]
