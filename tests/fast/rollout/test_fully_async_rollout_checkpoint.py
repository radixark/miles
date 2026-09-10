import asyncio
from pathlib import Path

import pytest
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from tests.fast.rollout.conftest import _AcknowledgingBuffer
from tests.fast.rollout.test_fully_async_rollout import (
    FakeDataSource,
    make_checkpointing_args,
    make_fn,
    make_group,
    make_multi_policy_group,
)

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async_data_buffer import (
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    UnusedReason,
)
from miles.utils.audit_utils.event_logger.models import SampleOwner
from miles.utils.types import Sample


class TestPendingAdmission:
    @pytest.mark.parametrize("consume_before_save", [False, True])
    async def test_stored_put_waiting_for_acknowledgement_is_not_repeated_after_restore(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, consume_before_save: bool
    ) -> None:
        """A buffer owns an admitted group even while its put acknowledgement is pending."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        buffer = _AcknowledgingBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused))
        fn._output = buffer
        group = make_group(7)
        fn._pending_puts = {None: DataBufferInput(prompt_group=group, group=group)}
        putting = asyncio.create_task(fn._flush_pending_puts())
        await buffer.stored.wait()
        if consume_before_save:
            await buffer.get(num_groups=1)

        fn.save(0)
        buffer.acknowledged.set()
        await putting
        restored = make_fn(monkeypatch, args, FakeDataSource())
        restored.load(0)
        restored._ensure_output()
        restored._output.restore(restored._pending_restore)
        restored._pending_restore = None
        await restored._flush_pending_puts()

        saved = restored._output.snapshot()[None]
        assert [sample.index for entry in saved for sample in entry.group] == ([] if consume_before_save else [70, 71])

    async def test_capacity_wait_restores_an_accepted_group_without_repeating_the_dynamic_filter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An admission decision survives a checkpoint taken while buffer capacity blocks the put."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1, async_data_buffer_capacity_factor=1)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._ensure_output()
        first, second = make_group(1), make_group(2)
        await fn._output.put(DataBufferInput(prompt_group=first, group=first))
        calls: list[int] = []

        def dynamic_filter(args: object, group: list) -> DynamicFilterOutput:
            calls.append(group[0].index)
            return DynamicFilterOutput(keep=len(calls) == 1)

        fn._output._dynamic_filter = dynamic_filter
        fn._pending_puts = {None: DataBufferInput(prompt_group=second, group=second)}
        putting = asyncio.create_task(fn._flush_pending_puts())
        await asyncio.sleep(0)
        assert calls == [20]
        assert not putting.done()
        fn.save(0)
        putting.cancel()
        await asyncio.gather(putting, return_exceptions=True)

        restored = make_fn(monkeypatch, args, FakeDataSource())
        restored.load(0)
        restored._ensure_output()
        assert isinstance(restored._output, DefaultDataBuffer)
        restored._output._dynamic_filter = dynamic_filter
        restored._output.restore(restored._pending_restore)
        restored._pending_restore = None
        await restored._output.get(num_groups=1)
        await restored._flush_pending_puts()
        batch = await restored._output.get(num_groups=1)

        assert calls == [20]
        assert [sample.index for sample in batch[0].group] == [20, 21]


class TestPolicyRetry:
    @pytest.mark.parametrize("consume_peer", [False, True])
    async def test_checkpoint_during_aborted_policy_regeneration_preserves_peer_delivery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, consume_peer: bool
    ) -> None:
        """A retry checkpoint preserves its policy target while an unaffected peer stays buffered or consumed."""
        args = make_checkpointing_args(
            tmp_path,
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            megatron_config=encode_megatron_config("a", "b"),
            async_unused_samples_handler="retry",
        )
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        active_fn: list[fully_async.FullyAsyncRolloutFn] = []

        async def regenerate(state: object, prompts: list[Sample], **kwargs: object) -> list[Sample]:
            nonlocal calls
            calls += 1
            active_fn[0]._producer_resumed.clear()
            if calls == 1:
                entered.set()
                await release.wait()
            return make_multi_policy_group(7, "a", "b")

        fn = make_fn(monkeypatch, args, FakeDataSource(), generate=regenerate)
        active_fn.append(fn)
        fn._ensure_output()
        group = make_multi_policy_group(7, "a", "b")
        group[1].status = Sample.Status.ABORTED
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        if consume_peer:
            await fn._output.get(num_groups=1, trainer_model_id="a")
        generating = fn._submit_one_group()
        await entered.wait()
        fn.save(0)
        generating.cancel()
        await asyncio.gather(generating, return_exceptions=True)

        restored = make_fn(monkeypatch, args, FakeDataSource(), generate=regenerate)
        active_fn[0] = restored
        restored.load(0)
        output = await restored(RolloutFnTrainInput(rollout_id=0, weight_version=1, trainer_model_id="b"))

        assert [sample.index for sample in output.samples[0]] == [71]
        assert [sample.index for entry in restored._output.snapshot()["a"] for sample in entry.group] == (
            [] if consume_peer else [70]
        )
        assert group[0].response == "ok"
        assert group[0].trainer_model_id == "a"
        assert restored.data_source.num_get_calls == 0

    @pytest.mark.parametrize("reason", [UnusedReason.STALE, UnusedReason.ABORTED])
    @pytest.mark.parametrize("consume_peer", [False, True])
    async def test_retry_restores_only_its_target_policy_and_preserves_the_peer_samples(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason: UnusedReason, consume_peer: bool
    ) -> None:
        """Retrying one policy never mutates or redelivers the other policy's completed output."""
        args = make_checkpointing_args(
            tmp_path,
            rollout_batch_size=1,
            megatron_config=encode_megatron_config("a", "b"),
            async_unused_samples_handler="retry",
        )
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._ensure_output()
        group = make_multi_policy_group(7, "a", "b")
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        peer = group[0]
        if consume_peer:
            await fn._output.get(num_groups=1, trainer_model_id="a")
        await fn._output.get(num_groups=1, trainer_model_id="b")
        fn._recycle(group, reason=reason, trainer_model_id="b")

        assert peer.response == "ok"
        assert peer.trainer_model_id == "a"
        assert fn.describe_holdings(trainer_model_id="a")[SampleOwner.RETRY_BUFFER] == []
        fn.save(0)
        restored = make_fn(monkeypatch, args, FakeDataSource())
        restored.load(0)
        restored._ensure_output()
        restored._output.restore(restored._pending_restore)
        restored._pending_restore = None

        async def regenerate(state: object, prompts: list, **kwargs: object) -> list:
            return make_multi_policy_group(7, "a", "b")

        monkeypatch.setattr(fully_async, "generate_and_rm_group", regenerate)
        task = restored._submit_one_group()
        entry = await task
        restored._in_flight.pop(task)
        restored._pending_puts = restored._output.partition(entry)
        await restored._flush_pending_puts()

        buffered = restored._output.snapshot()
        assert [sample.index for entry in buffered["a"] for sample in entry.group] == ([] if consume_peer else [70])
        assert [sample.index for entry in buffered["b"] for sample in entry.group] == [71]
        assert restored.data_source.num_get_calls == 0

    async def test_single_policy_sample_tags_do_not_change_the_buffer_owner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A single policy run keeps None ownership even when custom generation stamps a tag."""
        args = make_checkpointing_args(tmp_path, rollout_batch_size=1)
        group = make_group(7)
        for sample in group:
            sample.trainer_model_id = "custom-tag"
        fn = make_fn(monkeypatch, args, FakeDataSource(scripted=[group]))
        output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=1))

        assert [sample.index for sample in output.samples[0]] == [70, 71]
        assert fn._pending_puts == {}
