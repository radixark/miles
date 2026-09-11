import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group

import miles.rollout.fully_async_rollout as fully_async
from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionUtils, TrainingSampleIdentity
from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.train.group import TrainerController
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.arguments import _resolve_sample_ownership_check
from miles.utils.audit_utils.event_logger.logger import EventLogger, set_event_logger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.flow import record_data_source_issues
from miles.utils.audit_utils.sample_ownership.publication import make_current_cpu_witness_payload
from miles.utils.audit_utils.sample_ownership.step_window import SampleOwnershipStepWindow
from miles.utils.audit_utils.sample_ownership.store import SampleOwnershipEventStore
from miles.utils.types import Sample


class TestCheckpointSampleOwnership:
    @pytest.mark.parametrize("lose_prefetched_batch", [False, True], ids=["healthy", "missing-training-step"])
    async def test_ci_checker_detects_a_prefetched_batch_lost_after_restore(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lose_prefetched_batch: bool
    ) -> None:
        """The periodic CI checker rejects a restored batch that the driver never sends to training."""
        event_logger = EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="main"))
        set_event_logger(event_logger)
        args = _checkpoint_args(tmp_path)
        source = FakeDataSource(scripted=[make_group(9)])
        record_data_source_issues(source)
        fn = make_fn(monkeypatch, args, source)
        fn._ensure_output()
        [group] = source.get_samples(1)
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        taking = asyncio.create_task(fn._take_batch(num_groups=1, current_version=1, trainer_model_id=None))
        await asyncio.sleep(0)
        assert taking.done()
        fn.save(6)
        await taking

        restored = make_fn(monkeypatch, args, source)
        restored.load(6)
        restored._ensure_output()
        restored._output.restore(restored._pending_restore)
        restored._pending_restore = None
        restored._worker = asyncio.create_task(asyncio.Event().wait())
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = SimpleNamespace(
            ci_test=True,
            sample_ownership_check=False,
            sample_ownership_grace_steps=None,
            sample_ownership_check_interval_seconds=0.001,
            sample_ownership_check_timeout_seconds=5,
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
        assert executor.args.sample_ownership_check
        trainer = _CheckpointTrainer(grace_steps=executor.args.sample_ownership_grace_steps)
        executor.rollout_id = 10
        executor._actor_controller = trainer
        executor._sample_ownership_store = SampleOwnershipEventStore(event_logger)
        executor._sample_ownership_task = None
        try:
            output = await restored._drain(RolloutFnTrainInput(rollout_id=7, weight_version=1))
            assert [[sample.index for sample in batch] for batch in output.samples] == [[90, 91]]
            if not lose_prefetched_batch:
                await trainer.train(rollout_id=7, rollout_data_pack=output.samples)
            for rollout_id in (8, 9):
                await trainer.train(rollout_id=rollout_id, rollout_data_pack=source.get_samples(1))

            monkeypatch.setattr(
                rollout_executor_module,
                "compute_trainer_configs",
                lambda _args: [SimpleNamespace(role="actor", trainer_id="actor")],
            )
            monkeypatch.setattr(
                rollout_executor_module, "create_trainer_controller_handle", lambda *_args, **_kwargs: trainer
            )
            monkeypatch.setattr(rollout_executor_module, "get_backend_capability", lambda _args: None)
            executor._start_sample_ownership_checker()
            assert executor._sample_ownership_task is not None
            if lose_prefetched_batch:
                with pytest.raises(ValueError, match="mature issued sample had no training outcome"):
                    await asyncio.wait_for(asyncio.shield(executor._sample_ownership_task), timeout=5)
            else:
                await asyncio.wait_for(trainer.checked.wait(), timeout=5)
                assert not executor._sample_ownership_task.done()
        finally:
            await executor._stop_sample_ownership_checker()
            restored._worker.cancel()
            await asyncio.gather(restored._worker, return_exceptions=True)
            set_event_logger(None)


class _CheckpointTrainer(TrainerController):
    def __init__(self, *, grace_steps: int) -> None:
        self._role = "actor"
        self.args = SimpleNamespace(sample_ownership_check_timeout_seconds=5)
        self._cpu_witness_operation_lock = asyncio.Lock()
        self._sample_ownership_steps = SampleOwnershipStepWindow(grace_steps)
        self._cells_by_id = {"cell-0": SimpleNamespace(cell_index=0, is_alive=True, execute=self._execute)}
        self.model = [torch.nn.Module()]
        self.model[0].add_module("model_companion", ModelCompanion())
        self.checked = asyncio.Event()
        self._snapshot_count = 0

    async def _train(
        self, rollout_id: int, rollout_data_pack: list[list[Sample]], external_data: Any = None
    ) -> list[Any]:
        ModelCompanionUtils.record(
            self.model,
            [
                TrainingSampleIdentity(source_sample_index=sample.index, row_index=0, row_count=1)
                for group in rollout_data_pack
                for sample in group
            ],
        )
        return []

    async def _execute(
        self, method: str, *, rollout_id: int, cohort_id: str, kill_on_failure: bool
    ) -> list[dict[str, object]]:
        assert method == "log_current_cpu_witness"
        payload = make_current_cpu_witness_payload(
            self.model, rollout_id=rollout_id, cohort_id=cohort_id, replica_id="cell-0"
        )
        self._snapshot_count += 1
        if self._snapshot_count == 2:
            self.checked.set()
        return [payload]


def _checkpoint_args(path: Path, **overrides: object) -> Namespace:
    return make_args(save=str(path), load=str(path), rollout_batch_size=1, **overrides)


async def test_in_flight_generation_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A half-written generation is replayed from an untouched prompt after restore."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    group = make_group(3)
    group[0].response = "partial"
    task = asyncio.Future()
    fn._in_flight[task] = fully_async._PendingPrompt(samples=group)

    fn.save(4)
    restored = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    restored.load(4)

    [pending] = restored._retry_buffer
    assert all(sample.response == "" for sample in pending.samples)


async def test_partial_batch_removed_from_buffer_remains_in_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """A batch between buffer removal and caller return is still persisted."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    fn._ensure_output()
    group = make_group(5)
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))
    draining = asyncio.create_task(fn._take_batch(num_groups=2, current_version=1, trainer_model_id=None))
    await asyncio.sleep(0)

    fn.save(2)
    draining.cancel()
    await asyncio.gather(draining, return_exceptions=True)
    state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=2), weights_only=False)

    assert [sample.index for sample in state["output"][None][0].group] == [50, 51]


async def test_completed_take_batch_is_restored_before_its_parent_resumes(monkeypatch, tmp_path: Path) -> None:
    """A full batch returned by the buffer remains checkpointed until the drain coroutine receives it."""
    args = _checkpoint_args(tmp_path)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._ensure_output()
    group = make_group(9)
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))
    taking = asyncio.create_task(fn._take_batch(num_groups=1, current_version=1, trainer_model_id=None))
    await asyncio.sleep(0)
    assert taking.done()

    fn.save(6)
    await taking
    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(6)
    restored._ensure_output()
    restored._output.restore(restored._pending_restore)
    restored._pending_restore = None
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
    assert [sample.index for sample in pending.samples] == [60, 61]
    assert all(sample.response == "" for sample in pending.samples)


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
    fn._pending_put = DataBufferInput(prompt_group=second, group=second)
    blocked = asyncio.create_task(fn._flush_pending_put())
    await asyncio.sleep(0)
    assert calls == 1
    fn.save(1)
    blocked.cancel()
    await asyncio.gather(blocked, return_exceptions=True)

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(1)
    restored._ensure_output()
    restored._output._dynamic_filter = keep_once
    restored._output.restore(restored._pending_restore)
    restored._pending_restore = None
    await restored._output.get(num_groups=1)
    await restored._flush_pending_put()

    assert calls == 1
    assert (await restored._output.get(num_groups=1))[0].group[0].group_index == 2


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
    fn._pending_put = DataBufferInput(prompt_group=group, group=group)
    putting = asyncio.create_task(fn._flush_pending_put())
    await stored.wait()

    fn.save(5)
    putting.cancel()
    await asyncio.gather(putting, return_exceptions=True)
    state = torch.load(fully_async.compute_fully_async_state_path(tmp_path, rollout_id=5), weights_only=False)

    assert state["pending_put"] is None
    assert [sample.index for sample in state["output"][None][0].group] == [80, 81]


def test_restored_output_requires_a_weight_version(monkeypatch, tmp_path: Path) -> None:
    """Restored outputs cannot be measured against an unknown model version."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    group = make_group(7)
    fn._pending_restore = {None: [DataBufferInput(prompt_group=group, group=group)]}

    with pytest.raises(AssertionError, match="requires a weight version"):
        fn._start_worker(weight_version=None)
