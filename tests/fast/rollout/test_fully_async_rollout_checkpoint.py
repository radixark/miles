import asyncio
from argparse import Namespace
from pathlib import Path

import torch
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group, train_input

from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.fully_async_rollout import _RunningTask
from miles.utils.types import Sample


def _checkpoint_args(path: Path, **overrides: object) -> Namespace:
    return make_args(save=str(path), load=str(path), rollout_batch_size=1, **overrides)


async def test_running_generation_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A half-written generation is replayed from an untouched prompt after restore."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    group = make_group(3)
    group[0].response = "partial"
    fn._running_tasks.append(_RunningTask(prompt_group=group, task=asyncio.Future()))

    fn.save(tmp_path)
    restored = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert all(sample.response == "" for sample in pending)


async def test_an_incomplete_batch_stays_in_the_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """A drain waiting for a whole batch restores every buffered group exactly once."""
    args = _checkpoint_args(tmp_path)
    args.rollout_batch_size = 2
    fn = make_fn(monkeypatch, args, FakeDataSource())
    first = make_group(5)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    draining = asyncio.create_task(fn._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)))
    try:
        await asyncio.sleep(0.01)
        assert not draining.done()
        fn.save(tmp_path)
    finally:
        draining.cancel()
        fn._worker.cancel()
        await asyncio.gather(draining, fn._worker, return_exceptions=True)

    state = torch.load(tmp_path / "state.pt", weights_only=False)
    assert [sample.index for sample in state.output[0].group] == [50, 51]

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(tmp_path)
    second = make_group(6)
    await restored._output.put(DataBufferInput(prompt_group=second, group=second))
    restored._worker = asyncio.create_task(asyncio.Event().wait())
    try:
        output = await asyncio.wait_for(
            restored._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)), timeout=5
        )
    finally:
        restored._worker.cancel()
        await asyncio.gather(restored._worker, return_exceptions=True)

    assert [[sample.index for sample in group] for group in output.samples] == [[50, 51], [60, 61]]
    assert restored._output.state_dict() == []


async def test_a_save_taken_as_the_drain_wakes_cannot_lose_the_batch(monkeypatch, tmp_path: Path) -> None:
    """A batch that has left the buffer has already reached the drain, so no save sees it nowhere."""
    args = _checkpoint_args(tmp_path)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    draining = asyncio.create_task(fn._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)))
    try:
        await asyncio.sleep(0.01)
        assert not draining.done()
        group = make_group(5)
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        await asyncio.sleep(0)
        fn.save(tmp_path)

        assert draining.done()
        output = await draining
    finally:
        draining.cancel()
        fn._worker.cancel()
        await asyncio.gather(draining, fn._worker, return_exceptions=True)

    state = torch.load(tmp_path / "state.pt", weights_only=False)
    assert state.output == [] and state.running == []
    assert [sample.index for sample in output.samples[0]] == [50, 51]


async def test_aborted_group_in_retry_queue_survives_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """An aborted group saved after recycling resumes exactly once from its prompt."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path, async_unused_samples_handler="retry"), FakeDataSource())
    group = make_group(6)
    for sample in group:
        sample.status = Sample.Status.ABORTED
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))

    fn.save(tmp_path)
    restored = make_fn(monkeypatch, fn.args, FakeDataSource())
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert [sample.index for sample in pending] == [60, 61]
    assert all(sample.response == "" for sample in pending)


async def test_a_group_blocked_in_put_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A group saved while its put waits for buffer capacity is regenerated from its prompt after restore."""
    args = _checkpoint_args(tmp_path, async_data_buffer_capacity_factor=1)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    first, second = make_group(1), make_group(2)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    blocked = asyncio.create_task(fn._output.put(DataBufferInput(prompt_group=second, group=second)))
    fn._running_tasks.append(_RunningTask(prompt_group=second, task=blocked))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    fn.save(tmp_path)
    blocked.cancel()
    await asyncio.gather(blocked, return_exceptions=True)

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert [sample.index for sample in pending] == [20, 21]
    assert all(sample.response == "" for sample in pending)
    assert [entry.group[0].group_index for entry in restored._output.state_dict()] == [1]
    try:
        outputs = [await asyncio.wait_for(restored(train_input(rollout_id=i)), timeout=5) for i in (0, 1)]
    finally:
        await restored.dispose()
    trained = [[sample.index for sample in group] for output in outputs for group in output.samples]
    assert trained == [[10, 11], [20, 21]]
    assert not restored._retry_buffer
