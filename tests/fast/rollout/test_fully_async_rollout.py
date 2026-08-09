from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
import miles.rollout.fully_async_data_buffer as data_buffer
import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.inference_rollout.fully_async as inference_fully_async
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    TrainAdmissionHold,
    TrainBatchRollbackReason,
)
from miles.rollout.data_source import SourceReservation, SourceReservationId
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async.ownership import ReservationTerminalReceipt
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


class FakeReservationDataSource:
    def __init__(self, reservations: list[SourceReservation], *, failed_requeues: int = 0) -> None:
        self.reservations = deque(reservations)
        self.reserved: list[SourceReservation] = []
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []
        self.failed_requeues = failed_requeues
        self.next_group_index = 1000

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        raise AssertionError("owned fully async rollout must reserve source groups")

    def add_samples(self, groups: list[list[Sample]]) -> None:
        raise AssertionError("owned fully async rollout must settle source reservations")

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        assert num_groups == 1
        if self.reservations:
            reservation = self.reservations.popleft()
        else:
            self.next_group_index += 1
            reservation = make_reservation(self.next_group_index)
        self.reserved.append(reservation)
        return [reservation]

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        self.acknowledged.append((list(reservations), rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        if self.failed_requeues:
            self.failed_requeues -= 1
            raise RuntimeError("scripted requeue failure")
        self.requeued.append(list(reservations))

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass


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


def make_reservation(group_index: int) -> SourceReservation:
    return SourceReservation(
        reservation_id=SourceReservationId(f"source-{group_index}"),
        samples=tuple(make_group(group_index)),
    )


async def wait_until(predicate) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


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
        rollout_health_check_timeout=0.1,
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
    monkeypatch.setattr(inference_fully_async, "generate_and_rm_group", generate or default_generate)
    return fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))


def make_owned_fn(monkeypatch, data_source, generate=None, *, batch_size=1, execution_samples=2):
    return make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=batch_size,
            async_max_concurrent_samples=execution_samples,
            async_data_buffer_capacity_factor=1.0,
            rollout_submission_granularity="group",
        ),
        data_source,
        generate=generate,
    )


async def test_train_call_leases_owned_output_until_settlement(monkeypatch):
    reservation = make_reservation(1)
    first_parent, second_parent = reservation.samples
    first_child = deepcopy(first_parent)
    second_child = deepcopy(first_parent)
    generated_group = [[first_child, second_child], deepcopy(second_parent)]
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return generated_group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    output = await fn(RolloutFnTrainInput(rollout_id=17, weight_version="17"))

    assert isinstance(output, LeasedRolloutFnTrainOutput)
    assert output.samples == [generated_group]
    assert data_source.acknowledged == []
    output.lease.commit()
    assert data_source.acknowledged == [([reservation], 17)]
    assert data_source.requeued == []


async def test_train_admission_hold_blocks_new_owned_reservations_until_release(monkeypatch):
    reservation = make_reservation(2)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)

    hold = await fn.acquire_train_admission_hold()
    assert isinstance(hold, TrainAdmissionHold)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=18, weight_version="18")))
    await asyncio.sleep(0)
    assert data_source.reserved == []

    hold.release()
    output = await drain
    assert data_source.reserved == [reservation]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


async def test_open_owned_lease_blocks_close_until_definitive_settlement(monkeypatch):
    reservation = make_reservation(3)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=19, weight_version="19"))

    with pytest.raises(RuntimeError, match=r"open train batch leases: \[19\]"):
        await fn.close()

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()
    assert fn._closed


async def test_prepare_checkpoint_requires_an_active_admission_hold(monkeypatch):
    fn = make_owned_fn(monkeypatch, FakeReservationDataSource([]))

    with pytest.raises(RuntimeError, match="active train admission hold"):
        await fn.prepare_checkpoint(24)


async def test_prepare_checkpoint_rejects_open_lease_then_accepts_exact_settlement(monkeypatch):
    reservation = make_reservation(24)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=24, weight_version=24))
    hold = await fn.acquire_train_admission_hold()
    await hold.wait_terminal()

    with pytest.raises(RuntimeError, match=r"open train batch leases: \[24\]"):
        await fn.prepare_checkpoint(24)

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.prepare_checkpoint(24)
    assert output.lease not in fn._open_train_batch_leases
    hold.release()


async def test_prepare_checkpoint_rejects_active_frontier_until_hold_waits(monkeypatch):
    release_generation = asyncio.Event()
    reservation = make_reservation(25)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=25, weight_version=25)))
    await wait_until(lambda: data_source.reserved == [reservation])
    hold = await fn.acquire_train_admission_hold()

    with pytest.raises(RuntimeError, match="admission frontier to be terminal"):
        await fn.prepare_checkpoint(25)

    release_generation.set()
    await hold.wait_terminal()
    hold.release()
    output = await drain
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_checkpoint_publication_does_not_wait_for_terminal_result_to_enter_full_buffer(monkeypatch):
    release_generation = asyncio.Event()
    reservation = make_reservation(26)
    data_source = FakeReservationDataSource([reservation])
    events: list[str] = []

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release_generation.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=26, weight_version=26)))
    await wait_until(lambda: data_source.reserved == [reservation])
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain

    output = fn._output
    assert isinstance(output, data_buffer.DefaultDataBuffer)
    buffered_group = make_group(260)
    await output.put(data_buffer.DataBufferInput(source=buffered_group, group=buffered_group))

    manager_class = cast(type[RolloutManager], object.__getattribute__(RolloutManager, "__ray_actor_class__"))
    manager = object.__new__(manager_class)
    manager.args = fn.args
    manager.data_source = data_source
    manager._train_rollout_lifecycle = fn
    manager._rollout_lifecycles_closing = False
    manager._pending_admissions = {}
    monkeypatch.setattr(manager, "_submit_lifecycle_coroutine", lambda coroutine: asyncio.create_task(coroutine))
    monkeypatch.setattr(data_source, "save", lambda rollout_id: events.append(f"source:{rollout_id}"))

    def snapshot(_args, rollout_id):
        assert data_source.reserved == [reservation]
        assert not fn._train_admission_open.is_set()
        events.append(f"event:{rollout_id}")

    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", snapshot)
    save_task = asyncio.create_task(manager.save(26))
    await wait_until(lambda: bool(fn._train_admission_holds))
    release_generation.set()
    await wait_until(lambda: fn._active_executions and all(task.done() for task in fn._active_executions))
    assert fn._worker is not None and not fn._worker.done()

    try:
        done, _ = await asyncio.wait([save_task], timeout=1)
        assert done == {save_task}
        await save_task
        assert events == ["source:26", "event:26"]
        assert output.get_metrics()["rollout/fully_async/queue_size"] == 1
    finally:
        await output.get()
        if not save_task.done():
            await asyncio.wait_for(save_task, timeout=1)
        await fn.close()


async def test_prepare_checkpoint_flushes_discarded_terminal_with_checkpoint_rollout_id(monkeypatch):
    reservation = make_reservation(26)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    ownership = fn._require_ownership()
    retained_slots = fn._require_retained_slots()
    assert fn._completed_slots is not None
    await retained_slots.acquire()
    assert fn._try_acquire_completed_slot()
    [reserved] = ownership.reserve_samples(1)
    [executor_receipt] = ownership.begin_execution([reserved], stage_id="checkpoint-test")
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id="checkpoint-test")
    fn._discard_buffer_source(terminal_receipt)

    hold = await fn.acquire_train_admission_hold()
    await hold.wait_terminal()
    await fn.prepare_checkpoint(27)

    assert data_source.acknowledged == [([reservation], 27)]
    assert not fn._discarded_terminal_receipts
    hold.release()
    await fn.close()


async def test_prepare_checkpoint_retries_retained_terminal_cleanup(monkeypatch):
    reservation = make_reservation(28)
    data_source = FakeReservationDataSource([reservation], failed_requeues=1)
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=28, weight_version=28))
    hold = await fn.acquire_train_admission_hold()
    await hold.wait_terminal()

    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert output.lease not in fn._open_train_batch_leases

    await fn.prepare_checkpoint(28)
    assert data_source.requeued == [[reservation]]
    hold.release()


async def test_prepare_checkpoint_retries_retained_acquisition_and_reservation_cleanup(monkeypatch):
    reservation = make_reservation(29)
    acquisition_attempt = make_reservation(30)
    data_source = FakeReservationDataSource([reservation], failed_requeues=2)
    fn = make_owned_fn(monkeypatch, data_source)
    ownership = fn._require_ownership()
    retained_slots = fn._require_retained_slots()
    await retained_slots.acquire()
    await retained_slots.acquire()
    [reserved] = ownership.reserve_samples(1)
    ownership._pending_acquisition_rollback = [acquisition_attempt]
    fn._pending_acquisition_slot = True
    fn._pending_reserved_rollbacks.append(reserved)
    hold = await fn.acquire_train_admission_hold()

    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        await fn.prepare_checkpoint(29)
    assert ownership.has_pending_acquisition_rollback
    assert fn._pending_reserved_rollbacks == [reserved]

    await fn.prepare_checkpoint(29)
    assert not ownership.has_pending_acquisition_rollback
    assert fn._pending_reserved_rollbacks == []
    assert data_source.requeued == [[acquisition_attempt], [reserved]]
    hold.release()


async def test_admission_epoch_race_rolls_back_prefetched_terminal_before_lease(monkeypatch):
    release = asyncio.Event()
    first = make_reservation(4)
    second = make_reservation(5)
    data_source = FakeReservationDataSource([first, second])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, batch_size=1)
    lease_wait_started = asyncio.Event()
    original_wait = fn._wait_train_batch_lease_admission

    async def wait_for_lease_admission():
        lease_wait_started.set()
        return await original_wait()

    fn._wait_train_batch_lease_admission = wait_for_lease_admission
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=20, weight_version="20")))
    await wait_until(lambda: data_source.reserved == [first])

    hold = await fn.acquire_train_admission_hold()
    frontier = asyncio.create_task(hold.wait_terminal())
    release.set()
    await frontier
    await lease_wait_started.wait()
    assert not drain.done()

    hold.release()
    output = await drain
    assert data_source.requeued == [[first]]
    assert output.samples == [list(second.samples)]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


@pytest.mark.parametrize("failed_requeues", [1, 3])
async def test_epoch_mismatch_rollback_failure_retains_one_close_retry(monkeypatch, failed_requeues):
    release = asyncio.Event()
    reservation = make_reservation(6)
    data_source = FakeReservationDataSource([reservation], failed_requeues=failed_requeues)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, batch_size=1)
    lease_wait_started = asyncio.Event()
    original_wait = fn._wait_train_batch_lease_admission

    async def wait_for_lease_admission():
        lease_wait_started.set()
        return await original_wait()

    fn._wait_train_batch_lease_admission = wait_for_lease_admission
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=21, weight_version="21")))
    await wait_until(lambda: data_source.reserved == [reservation])

    hold = await fn.acquire_train_admission_hold()
    frontier = asyncio.create_task(hold.wait_terminal())
    release.set()
    await frontier
    await lease_wait_started.wait()
    hold.release()

    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        await drain

    for _ in range(failed_requeues + 2):
        try:
            await fn.close()
        except RuntimeError:
            continue
        break
    else:
        pytest.fail("close never settled the retained epoch-mismatch receipt")

    assert fn._closed
    assert data_source.requeued == [[reservation]]


@pytest.mark.parametrize(
    ("max_weight_staleness", "expected_group_indices", "expected_requeued", "expected_stale"),
    [
        (None, [32, 33], [30, 31], 2),
        (0, [32, 33], [30, 31], 2),
        # Receipt 30 was already claimed before the lifecycle hold, so PR5's
        # lease fence replays it. Receipt 31 is claimed after release and
        # remains within the numeric staleness allowance.
        (1, [31, 32], [30], 0),
    ],
)
async def test_recorded_weight_version_revalidates_groups_claimed_before_update(
    monkeypatch,
    max_weight_staleness,
    expected_group_indices,
    expected_requeued,
    expected_stale,
):
    release_second_generation = asyncio.Event()
    reservations = [make_reservation(index) for index in range(30, 34)]
    data_source = FakeReservationDataSource(reservations)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        group_index = group[0].group_index
        if group_index != 30:
            await release_second_generation.wait()
        version = "1" if group_index < 32 else "2"
        return make_group(group_index, weight_versions=[version])

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=2,
            max_weight_staleness=max_weight_staleness,
            async_max_concurrent_samples=2,
            async_data_buffer_capacity_factor=1.0,
            rollout_submission_granularity="group",
        ),
        data_source,
        generate=generate,
    )
    first_group_claimed = asyncio.Event()
    original_next_group = fn._next_group
    claims = 0

    async def observe_claim(input):
        nonlocal claims
        entry = await original_next_group(input)
        claims += 1
        if claims == 1:
            first_group_claimed.set()
        return entry

    fn._next_group = observe_claim
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=30, weight_version=1)))
    await first_group_claimed.wait()
    await wait_until(lambda: len(data_source.reserved) >= 2)

    hold = await fn.acquire_train_admission_hold()
    release_second_generation.set()
    await hold.wait_terminal()
    hold.record_weight_update(2)
    hold.release()

    output = await asyncio.wait_for(drain, timeout=1)
    assert [group[0].group_index for group in output.samples] == expected_group_indices
    assert (
        sorted(reservation.samples[0].group_index for batch in data_source.requeued for reservation in batch)
        == expected_requeued
    )
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == expected_stale

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await fn.close()


async def test_legacy_hold_and_close_do_not_wait_for_full_buffer_publication(monkeypatch):
    buffered_group = make_group(36)
    blocked_group = make_group(37)
    data_source = FakeDataSource([blocked_group])
    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=2,
            async_data_buffer_capacity_factor=1.0,
            async_unused_samples_handler="retry",
        ),
        data_source,
    )
    output = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=fn.args,
            unused_handler_fn=fn._handle_unused_buffer_source,
            discard_handler_fn=fn._discard_buffer_source,
        )
    )
    await output.put(data_buffer.DataBufferInput(source=buffered_group, group=buffered_group, weight_update_epoch=0))
    fn._output = output
    fn._worker = asyncio.create_task(fn._worker_loop())
    await wait_until(lambda: fn._legacy_executions and all(task.done() for task in fn._legacy_executions))

    hold = await fn.acquire_train_admission_hold()
    await asyncio.wait_for(hold.wait_terminal(), timeout=1)
    hold.record_weight_update(1)
    hold.release()
    await fn.close()

    assert sorted(group[0].group_index for group in data_source.recycled) == [36, 37]


async def test_legacy_weight_hold_blocks_new_source_until_update_is_recorded(monkeypatch):
    release_generation = asyncio.Event()
    first_group = make_group(38)
    second_group = make_group(39)
    data_source = FakeDataSource([first_group, second_group])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release_generation.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=2,
            async_data_buffer_capacity_factor=1.0,
            async_unused_samples_handler="retry",
        ),
        data_source,
        generate=generate,
    )
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=38, weight_version=1)))
    await wait_until(lambda: data_source.num_get_calls == 1)

    hold = await fn.acquire_train_admission_hold()
    release_generation.set()
    await hold.wait_terminal()
    await asyncio.sleep(0)
    assert data_source.num_get_calls == 1

    hold.record_weight_update(2)
    hold.release()
    output = await asyncio.wait_for(drain, timeout=1)

    assert output.samples == [second_group]
    assert [group[0].group_index for group in data_source.recycled] == [38]
    await fn.close()


async def test_legacy_hold_without_record_keeps_the_claimed_group(monkeypatch):
    release_generation = asyncio.Event()
    first_group = make_group(40)
    data_source = FakeDataSource([first_group])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release_generation.wait()
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=1,
            async_data_buffer_capacity_factor=1.0,
            async_unused_samples_handler="retry",
        ),
        data_source,
        generate=generate,
    )
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=40, weight_version=1)))
    await wait_until(lambda: data_source.num_get_calls == 1)

    hold = await fn.acquire_train_admission_hold()
    release_generation.set()
    await hold.wait_terminal()
    hold.release()

    output = await asyncio.wait_for(drain, timeout=1)
    assert output.samples == [first_group]
    assert data_source.recycled == []
    await fn.close()


async def test_multiple_admission_holds_reopen_only_after_final_release(monkeypatch):
    reservation = make_reservation(6)
    data_source = FakeReservationDataSource([reservation])
    fn = make_owned_fn(monkeypatch, data_source)
    first_hold = await fn.acquire_train_admission_hold()
    second_hold = await fn.acquire_train_admission_hold()
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=21, weight_version="21")))

    await asyncio.sleep(0)
    assert data_source.reserved == []
    first_hold.release()
    await asyncio.sleep(0)
    assert data_source.reserved == []
    second_hold.release()

    output = await drain
    assert data_source.reserved == [reservation]
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


async def test_cancelled_admission_wait_does_not_cancel_frontier(monkeypatch):
    release = asyncio.Event()
    reservation = make_reservation(7)
    data_source = FakeReservationDataSource([reservation])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        await release.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=22, weight_version="22")))
    await wait_until(lambda: data_source.reserved == [reservation])

    hold = await fn.acquire_train_admission_hold()
    waiter = asyncio.create_task(hold.wait_terminal())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    await hold.wait_terminal()
    hold.release()
    output = await drain
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


@pytest.mark.parametrize("operation", ["commit", "rollback"])
async def test_failed_lease_settlement_removes_open_lease_and_close_retries(monkeypatch, operation):
    reservation = make_reservation(8)
    data_source = FakeReservationDataSource([reservation], failed_requeues=1 if operation == "rollback" else 0)
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=23, weight_version="23"))
    assert output.lease in fn._open_train_batch_leases

    if operation == "rollback":
        with pytest.raises(RuntimeError, match="scripted requeue failure"):
            output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    else:
        # A commit failure retains a pending rollback for close to retry.
        data_source.acknowledge_reservations = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("scripted acknowledge failure")
        )
        with pytest.raises(RuntimeError, match="scripted acknowledge failure"):
            output.lease.commit()

    assert output.lease not in fn._open_train_batch_leases
    await fn.close()
    assert fn._closed


async def test_owned_execution_capacity_bounds_source_reservations(monkeypatch):
    release = asyncio.Event()
    started: list[int] = []
    data_source = FakeReservationDataSource([make_reservation(index) for index in range(6, 10)])

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        started.append(group[0].group_index)
        await release.wait()
        return group

    fn = make_owned_fn(monkeypatch, data_source, generate, execution_samples=4)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=21, weight_version="21")))
    await wait_until(lambda: len(started) == 2)
    for _ in range(10):
        await asyncio.sleep(0)
    assert started == [6, 7]
    assert data_source.reserved == [make_reservation(6), make_reservation(7)]

    release.set()
    output = await drain
    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)


async def test_owned_capacity_reopens_after_lease_settlement(monkeypatch):
    first = make_reservation(10)
    second = make_reservation(11)
    data_source = FakeReservationDataSource([first, second])
    fn = make_owned_fn(monkeypatch, data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=22, weight_version="22"))
    for _ in range(10):
        await asyncio.sleep(0)
    assert data_source.reserved == [first]

    output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    await wait_until(lambda: len(data_source.reserved) == 2)
    assert data_source.requeued == [[first]]
    assert data_source.reserved == [first, second]


async def test_close_retries_failed_lease_rollback(monkeypatch):
    reservation = make_reservation(12)
    data_source = FakeReservationDataSource([reservation], failed_requeues=1)
    fn = make_owned_fn(monkeypatch, data_source)
    output = await fn(RolloutFnTrainInput(rollout_id=23, weight_version="23"))

    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        output.lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    await fn.close()

    assert data_source.requeued == [[reservation]]
    assert fn._closed


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


async def test_owned_stale_callback_failure_is_retried_on_close(monkeypatch):
    reservation = make_reservation(52)
    data_source = FakeReservationDataSource([reservation], failed_requeues=1)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return make_group(52, weight_versions=["1"])

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            max_weight_staleness=0,
            async_max_concurrent_samples=2,
            async_data_buffer_capacity_factor=1.0,
            rollout_submission_granularity="group",
        ),
        data_source,
        generate=generate,
    )
    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        await fn(RolloutFnTrainInput(rollout_id=52, weight_version=2))

    await fn.close()
    assert data_source.requeued == [[reservation]]


async def test_legacy_stale_callback_failure_is_retried_on_close(monkeypatch):
    class FailOnceRecycleDataSource(FakeDataSource):
        def __init__(self, scripted):
            super().__init__(scripted)
            self.fail_recycle = True

        def add_samples(self, groups):
            if self.fail_recycle:
                self.fail_recycle = False
                raise RuntimeError("scripted recycle failure")
            super().add_samples(groups)

    stale = make_group(53, weight_versions=["1"])
    data_source = FailOnceRecycleDataSource([stale])

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=0, async_unused_samples_handler="retry"),
        data_source,
    )
    with pytest.raises(RuntimeError, match="scripted recycle failure"):
        await fn(RolloutFnTrainInput(rollout_id=53, weight_version=2))

    await fn.close()
    assert [group[0].group_index for group in data_source.recycled].count(53) == 1


async def test_legacy_put_callback_failure_has_one_retry_owner(monkeypatch):
    class FailOnceRecycleDataSource(FakeDataSource):
        def __init__(self, scripted):
            super().__init__(scripted)
            self.fail_recycle = True

        def add_samples(self, groups):
            if self.fail_recycle:
                self.fail_recycle = False
                raise RuntimeError("scripted put recycle failure")
            super().add_samples(groups)

    aborted = [replace(sample, status=Sample.Status.ABORTED) for sample in make_group(55)]
    data_source = FailOnceRecycleDataSource([aborted])
    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=N_SAMPLES_PER_PROMPT,
            async_unused_samples_handler="retry",
        ),
        data_source,
    )

    with pytest.raises(RuntimeError, match="scripted put recycle failure"):
        await fn(RolloutFnTrainInput(rollout_id=55))
    with pytest.raises(RuntimeError, match="scripted put recycle failure"):
        await fn.close()

    assert [group[0].group_index for group in data_source.recycled].count(55) == 1
    assert not fn._legacy_close_pending_groups


async def test_owned_put_callback_failure_has_one_retry_owner(monkeypatch):
    reservation = make_reservation(56)
    data_source = FakeReservationDataSource([reservation], failed_requeues=1)

    async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        return [replace(sample, status=Sample.Status.ABORTED) for sample in make_group(56)]

    fn = make_owned_fn(monkeypatch, data_source, generate)
    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        await fn(RolloutFnTrainInput(rollout_id=56))
    with pytest.raises(RuntimeError, match="scripted requeue failure"):
        await fn.close()

    assert data_source.requeued == [[reservation]]
    assert fn._pending_terminal_rollbacks == []


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
        raise RuntimeError("generation exploded")

    source_group = make_group(54)
    data_source = FakeDataSource([source_group])
    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            async_max_concurrent_samples=N_SAMPLES_PER_PROMPT,
            async_unused_samples_handler="retry",
        ),
        data_source,
        generate=failing_generate,
    )

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))
    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn.close()
    assert data_source.recycled == [source_group]


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
    await fn._output.put(data_buffer.DataBufferInput(source=group, group=group))
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


async def test_worker_stays_alive_when_admission_pauses_without_active_work(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
    output = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=fn.args,
            unused_handler_fn=fn._handle_unused_buffer_source,
            discard_handler_fn=fn._discard_buffer_source,
        )
    )
    fn._output = output
    submit_calls = 0

    async def pause_before_submit(active):
        nonlocal submit_calls
        submit_calls += 1
        fn._producer_resumed.clear()
        return False

    monkeypatch.setattr(fn, "_submit_active_group", pause_before_submit)
    worker = asyncio.create_task(fn._worker_loop())
    fn._worker = worker
    await wait_until(lambda: submit_calls == 1)
    await asyncio.sleep(0)
    assert not worker.done()
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)


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
        data_buffer.DataBufferConstructorInput(
            args=args,
            unused_handler_fn=unused.append,
            discard_handler_fn=unused.append,
        )
    )
    return buffer, unused


async def put_group(buffer, group):
    """These tests reuse one group as both the prompt group and the finished group."""
    await buffer.put(data_buffer.DataBufferInput(source=group, group=group))


def make_terminal_receipt() -> ReservationTerminalReceipt:
    return ReservationTerminalReceipt(executor_receipt=object())


async def test_buffer_preserves_terminal_receipt_identity():
    buffer, _ = make_buffer()
    receipt = make_terminal_receipt()
    group = make_group(1)

    await buffer.put(data_buffer.DataBufferInput(source=receipt, group=group))
    entry = await buffer.get()

    assert entry.source is receipt
    assert entry.group is group
    with pytest.raises(RuntimeError, match="does not expose a retryable prompt group"):
        _ = entry.prompt_group


async def test_buffer_stale_filter_settles_the_exact_terminal_receipt():
    buffer, unused = make_buffer(max_staleness=2)
    receipt = make_terminal_receipt()
    await buffer.put(data_buffer.DataBufferInput(source=receipt, group=make_group(1, weight_versions=["5"])))
    fresh = make_group(2, weight_versions=["9"])
    await put_group(buffer, fresh)

    assert (await buffer.get(current_version=10)).source is fresh
    assert unused == [receipt]


async def test_buffer_dynamic_filter_discards_the_exact_terminal_receipt():
    discarded = []
    args = make_args(
        rollout_batch_size=1,
        dynamic_sampling_filter_path=f"{__name__}.reject_group_1",
    )
    buffer = data_buffer.DefaultDataBuffer(
        data_buffer.DataBufferConstructorInput(
            args=args,
            unused_handler_fn=lambda source: None,
            discard_handler_fn=discarded.append,
        )
    )
    receipt = make_terminal_receipt()

    await buffer.put(data_buffer.DataBufferInput(source=receipt, group=make_group(1)))

    assert discarded == [receipt]
    assert buffer.get_metrics()["rollout/fully_async/queue_size"] == 0


async def test_buffer_discard_all_retains_sources_that_fail_settlement():
    buffer, _ = make_buffer()
    retained = make_terminal_receipt()
    discarded = make_terminal_receipt()
    await buffer.put(data_buffer.DataBufferInput(source=retained, group=make_group(1)))
    await buffer.put(data_buffer.DataBufferInput(source=discarded, group=make_group(2)))

    def settle(source):
        if source is retained:
            raise RuntimeError("settlement failed")

    error = await buffer.discard_all(settle)

    assert isinstance(error, RuntimeError)
    assert str(error) == "settlement failed"
    assert (await buffer.get()).source is retained


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


class CompatibleCustomBuffer(data_buffer.DataBuffer):
    async def put(self, input):
        pass

    async def get(self, **context):
        raise RuntimeError("not used")

    def get_metrics(self):
        return {}


async def test_custom_buffer_final_admission_fails_closed_only_after_recorded_update():
    buffer = CompatibleCustomBuffer()
    entry = data_buffer.DataBufferInput(source=make_group(50), group=make_group(50), weight_update_epoch=0)

    assert (
        buffer.validate_final_admission([entry], current_weight_update_epoch=0, current_version=None).rejected_indexes
        == ()
    )
    with pytest.raises(RuntimeError, match="must implement final-admission validation"):
        buffer.validate_final_admission([entry], current_weight_update_epoch=1, current_version=None)

    class ExplicitVerdictBuffer(CompatibleCustomBuffer):
        def validate_final_admission(self, entries, *, current_weight_update_epoch, current_version):
            return data_buffer.DataBufferAdmissionVerdict(rejected_indexes=(0,))

    assert ExplicitVerdictBuffer().validate_final_admission(
        [entry], current_weight_update_epoch=1, current_version=None
    ).rejected_indexes == (0,)


async def test_custom_data_buffer_path_replaces_default(monkeypatch):
    path = f"{__name__}.RecordingBuffer"
    args = make_args(custom_async_data_buffer_path=path, async_unused_samples_handler="retry")
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert type(fn._output) is RecordingBuffer
    assert RecordingBuffer.constructed_with.unused_handler_fn == fn._handle_unused_buffer_source
    assert RecordingBuffer.constructed_with.discard_handler_fn == fn._discard_buffer_source
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
