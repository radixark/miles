import asyncio
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group

import miles.rollout.fully_async_rollout as fully_async
from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionSampleConsumptionUtils
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.arguments import _resolve_sample_ownership_check
from miles.utils.audit_utils.event_logger.logger import EventLogger, get_event_logger, set_event_logger
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.checker import SampleOwnershipChecker
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.audit_utils.sample_ownership.step_window import SampleOwnershipStepWindow
from miles.utils.types import Sample, SampleLineage


class TestCheckpointSampleOwnership:
    @pytest.mark.parametrize("lose_prefetched_batch", [False, True], ids=["healthy", "missing-training-step"])
    async def test_ci_checker_detects_a_prefetched_batch_lost_after_restore(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lose_prefetched_batch: bool
    ) -> None:
        """The CI checker rejects a restored batch that the driver never sends to training."""
        event_logger = EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="main"))
        set_event_logger(event_logger)
        args = _checkpoint_args(tmp_path)
        source = FakeDataSource(scripted=[make_group(9)])
        SampleOwnershipRecorder.install_data_source_issue_recorder(source)
        fn = make_fn(monkeypatch, args, source)
        fn._ensure_output()
        [group] = source.get_samples(1)
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        taking = asyncio.create_task(fn._take_group(current_version=1, trainer_model_id=None))
        await asyncio.sleep(0)
        assert taking.done()
        fn.save(6)
        await taking

        restored = make_fn(monkeypatch, args, source)
        restored.load(6)
        restored._ensure_output()
        restored._worker = asyncio.create_task(asyncio.Event().wait())
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            ci_test=True,
            enable_sample_ownership_checker=False,
            sample_ownership_grace_steps=None,
            sample_ownership_check_interval_seconds=0.001,
            custom_convert_samples_to_train_data_path=None,
            save_debug_event_data=str(event_logger.log_dir),
            train_backend="megatron",
            megatron_config=None,
            lora_rank=0,
            lora_adapter_path=None,
            multi_lora=False,
            debug_train_only=False,
            debug_rollout_only=False,
        )
        _resolve_sample_ownership_check(executor.args)
        assert executor.args.enable_sample_ownership_checker
        trainer = _CheckpointTrainer(grace_steps=executor.args.sample_ownership_grace_steps)
        executor.rollout_id = 10
        executor._sample_ownership_checker = SampleOwnershipChecker(args=executor.args)
        try:
            output = await restored._drain(RolloutFnTrainInput(rollout_id=7, weight_version=1))
            assert [[sample.index for sample in batch] for batch in output.samples] == [[90, 91]]
            if not lose_prefetched_batch:
                await trainer.train(rollout_id=7, rollout_data_pack=output.samples)
            for rollout_id in (8, 9):
                await trainer.train(rollout_id=rollout_id, rollout_data_pack=source.get_samples(1))

            if lose_prefetched_batch:
                with pytest.raises(ValueError, match="source sample had no training outcome"):
                    await executor._sample_ownership_checker.check(rollout_id=executor.rollout_id)
            else:
                await executor._sample_ownership_checker.check(rollout_id=executor.rollout_id)
                assert trainer.published == [7, 8, 9]
        finally:
            restored._worker.cancel()
            await asyncio.gather(restored._worker, return_exceptions=True)
            set_event_logger(None)


class _CheckpointTrainer:
    def __init__(self, *, grace_steps: int) -> None:
        self._sample_ownership_steps = SampleOwnershipStepWindow(grace_steps)
        self.model = [torch.nn.Module()]
        self.model[0].add_module(
            "model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        )
        self.published = []

    async def train(
        self, rollout_id: int, rollout_data_pack: list[list[Sample]], external_data: Any = None
    ) -> list[Any]:
        started_at = datetime.now(timezone.utc)
        ModelCompanionSampleConsumptionUtils.record(
            self.model,
            [
                SampleLineage(source_sample_index=sample.index, output_index=0, output_count=1)
                for group in rollout_data_pack
                for sample in group
            ],
        )
        self._sample_ownership_steps.complete_step(started_at=started_at)
        snapshot_id = SampleOwnershipRecorder.publish_cpu_witness(
            self.model,
            rollout_id=rollout_id,
            attempt=0,
            replica_id="cell-0",
            mature_before=self._sample_ownership_steps.mature_before(now=datetime.now(timezone.utc)),
        )
        get_event_logger().log(
            TrainGroupStepEndEvent,
            dict(
                rollout_id=rollout_id,
                attempt=0,
                role="actor",
                cell_outcomes={0: [TrainStepOutcome.NORMAL]},
                sample_ownership_snapshot_ids={0: [snapshot_id]},
            ),
            print_log=False,
        )
        self.published.append(rollout_id)
        return []


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


async def test_partial_batch_removed_from_buffer_remains_in_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """A drain waiting for its next group restores already removed groups exactly once."""
    args = _checkpoint_args(tmp_path)
    args.rollout_batch_size = 2
    fn = make_fn(monkeypatch, args, FakeDataSource())
    waiting = asyncio.Event()
    calls = 0

    class _WaitingBuffer(DefaultDataBuffer):
        async def get(
            self, *, current_version: int | None = None, trainer_model_id: str | None = None
        ) -> DataBufferInput:
            nonlocal calls
            calls += 1
            if calls == 2:
                waiting.set()
            return await super().get(current_version=current_version, trainer_model_id=trainer_model_id)

    fn._output = _WaitingBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused))
    first = make_group(5)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    draining = asyncio.create_task(fn._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        assert not draining.done()
        fn.save(2)
    finally:
        draining.cancel()
        fn._worker.cancel()
        await asyncio.gather(draining, fn._worker, return_exceptions=True)

    state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=2), weights_only=False)
    assert [sample.index for sample in state.output[None][0].group] == [50, 51]

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(2)
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
    assert restored._in_transit == {}
    assert restored._output.snapshot() == {None: []}


async def test_completed_take_batch_is_restored_before_its_parent_resumes(monkeypatch, tmp_path: Path) -> None:
    """A full batch returned by the buffer remains checkpointed until the drain coroutine receives it."""
    args = _checkpoint_args(tmp_path)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._ensure_output()
    group = make_group(9)
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))
    taking = asyncio.create_task(fn._take_group(current_version=1, trainer_model_id=None))
    await asyncio.sleep(0)
    assert taking.done()

    fn.save(6)
    await taking
    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(6)
    restored._ensure_output()
    restored._worker = asyncio.create_task(asyncio.Event().wait())
    try:
        output = await restored._drain(RolloutFnTrainInput(rollout_id=7, weight_version=1))
    finally:
        restored._worker.cancel()
        await asyncio.gather(restored._worker, return_exceptions=True)

    assert [[sample.index for sample in drained] for drained in output.samples] == [[90, 91]]
    assert restored._in_transit == {}


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

    assert state.pending_outputs == []
    assert [sample.index for sample in state.output[None][0].group] == [80, 81]
