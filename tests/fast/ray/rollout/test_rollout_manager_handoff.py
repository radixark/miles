import asyncio
import threading
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.train_batch_admission import (
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
)
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnLifecycle,
    RolloutFnTrainOutput,
    TrainAdmissionHold,
    TrainBatchLease,
    TrainBatchRollbackReason,
    WeightUpdateAdmissionHold,
)
from miles.utils.ray_utils import Box


class RecordingTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        rollout_id: int,
        events: list[str],
        commit_error: BaseException | None = None,
        rollback_error: BaseException | None = None,
    ) -> None:
        super().__init__(rollout_id=rollout_id)
        self._events = events
        self._commit_error = commit_error
        self._rollback_error = rollback_error

    def _commit(self) -> None:
        self._events.append("commit")
        if self._commit_error is not None:
            raise self._commit_error

    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        self._events.append(f"rollback:{reason.name}")
        if self._rollback_error is not None:
            raise self._rollback_error


class GatedTrainBatchLease(RecordingTrainBatchLease):
    """Hold awaitable settlement open until the test releases the gate."""

    def __init__(self, rollout_id: int, events: list[str]) -> None:
        super().__init__(rollout_id=rollout_id, events=events)
        self.settlement_started = asyncio.Event()
        self.release_settlement = asyncio.Event()

    async def _commit_async(self) -> None:
        await self._wait_for_gate()
        self._commit()

    async def _rollback_async(self, reason: TrainBatchRollbackReason) -> None:
        await self._wait_for_gate()
        self._rollback(reason)

    async def _wait_for_gate(self) -> None:
        self.settlement_started.set()
        await self.release_settlement.wait()


class RecordingTrainAdmissionHold(WeightUpdateAdmissionHold):
    def __init__(
        self,
        events: list[str],
        *,
        wait_gate: threading.Event | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._events = events
        self._wait_gate = wait_gate
        self._release_error = release_error

    async def _wait_terminal(self) -> None:
        self._events.append("wait")
        if self._wait_gate is not None:
            await asyncio.to_thread(self._wait_gate.wait)

    def _record_weight_update(self, weight_version: int | None = None) -> None:
        self._events.append(f"record:{weight_version}")

    def _release(self) -> None:
        self._events.append("release")
        if self._release_error is not None:
            raise self._release_error


class PlainTrainAdmissionHold(TrainAdmissionHold):
    async def _wait_terminal(self) -> None:
        return None

    def _release(self) -> None:
        return None


class RecordingRolloutLifecycle(RolloutFnLifecycle):
    def __init__(
        self,
        events: list[str],
        *,
        acquire_gate: threading.Event | None = None,
        close_gate: threading.Event | None = None,
        release_error: BaseException | None = None,
        close_errors: list[BaseException] | None = None,
    ) -> None:
        self.events = events
        self.acquire_gate = acquire_gate
        self.close_gate = close_gate
        self.release_error = release_error
        self.close_errors = list(close_errors or [])
        self.acquire_started = threading.Event()
        self.close_started = threading.Event()
        self.holds: list[RecordingTrainAdmissionHold] = []

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        self.events.append(f"prepare:{rollout_id}")

    async def acquire_train_admission_hold(self) -> TrainAdmissionHold:
        self.events.append("acquire")
        self.acquire_started.set()
        if self.acquire_gate is not None:
            await asyncio.to_thread(self.acquire_gate.wait)
        hold = RecordingTrainAdmissionHold(self.events, release_error=self.release_error)
        self.holds.append(hold)
        return hold

    async def close(self) -> None:
        self.events.append("close")
        self.close_started.set()
        if self.close_gate is not None:
            await asyncio.to_thread(self.close_gate.wait)
        if self.close_errors:
            raise self.close_errors.pop(0)


def install_lifecycle(manager: RolloutManager, *lifecycles: RecordingRolloutLifecycle) -> None:
    manager._train_rollout_lifecycle = lifecycles[0] if lifecycles else None
    manager._rollout_lifecycles = rollout_manager_mod._discover_rollout_lifecycles(*lifecycles)
    manager._lifecycle_async_loop = rollout_manager_mod.get_async_loop() if lifecycles else None
    manager._closed_rollout_lifecycles = []
    manager._rollout_lifecycles_closing = False
    manager._dispose_lock = asyncio.Lock()
    manager._manager_resources_disposed = False
    manager._next_train_admission_hold_id = 0
    manager._train_admission_holds = {}
    manager._active_generations = 0
    manager._generations_drained = asyncio.Event()
    manager._generations_drained.set()
    manager._data_source_closed = True
    manager._event_analysis_completed = True
    manager._metric_checker_disposed = True
    manager._checkpoint_eval_disposed = True
    manager._stopped_health_monitors = []
    manager._health_monitors = []
    manager._metric_checker = None
    manager.eval_generate_rollout = None


@pytest.fixture
def manager_env(monkeypatch: pytest.MonkeyPatch) -> tuple[RolloutManager, SimpleNamespace]:
    manager_class = cast(type[RolloutManager], object.__getattribute__(RolloutManager, "__ray_actor_class__"))
    manager = object.__new__(manager_class)
    manager.args = SimpleNamespace(
        ci_test=False,
        ci_inject_rollout_data_path=None,
        use_fault_tolerance=False,
        load_debug_rollout_data=None,
        delay_split_train_data_by_dp=False,
        use_critic=False,
        num_critic_only_steps=0,
    )
    manager.weight_version = None
    manager.rollout_id = -1
    manager.servers = {}
    manager.data_source = SimpleNamespace()
    manager.train_parallel_config = {"dp_size": 1}
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.use_legacy_rollout_v1 = False
    manager._manager_incarnation = "manager-test"
    manager._next_admission_id = 0
    manager._pending_admissions = {}
    monkeypatch.setattr(manager, "_health_monitoring_resume", lambda: None)

    monkeypatch.setattr(rollout_manager_mod.dashboard_hooks, "register_engines", lambda servers: None)
    monkeypatch.setattr(rollout_manager_mod, "timer", lambda name: nullcontext())
    monkeypatch.setattr(
        rollout_manager_mod,
        "postprocess_rollout_data",
        lambda args, data, train_parallel_config: (data, {}),
    )
    monkeypatch.setattr(rollout_manager_mod, "save_debug_rollout_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_manager_mod, "log_rollout_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rollout_manager_mod,
        "convert_samples_to_train_data",
        lambda *args, **kwargs: {"sample_indices": [5]},
    )

    return manager, manager.args


def leased_output(lease: TrainBatchLease) -> LeasedRolloutFnTrainOutput:
    return LeasedRolloutFnTrainOutput(samples=[], metrics={"source": "test"}, lease=lease)


@pytest.mark.asyncio
async def test_weight_update_records_manager_version_before_releasing_hold(manager_env) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    manager.weight_version = 2

    hold_id = await manager.acquire_train_admission_hold()
    await manager.wait_weight_update_admission(hold_id)
    await manager.record_train_weight_update(hold_id)
    await manager.release_train_admission_hold(hold_id)

    assert events == ["acquire", "wait", "record:2", "release"]
    assert manager._train_admission_holds == {}


@pytest.mark.asyncio
async def test_weight_update_fails_closed_for_plain_admission_hold(manager_env) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    hold_id = await manager.acquire_train_admission_hold()
    assert hold_id is not None
    manager._train_admission_holds[hold_id] = PlainTrainAdmissionHold()

    with pytest.raises(RuntimeError, match="does not support weight-update admission"):
        await manager.wait_weight_update_admission(hold_id)

    await manager.release_train_admission_hold(hold_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_weight_update_and_shared_eval_exclude_each_other(manager_env) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    eval_hold = await manager.acquire_train_admission_hold()
    await manager.wait_train_admission_hold(eval_hold)
    assert eval_hold is not None
    await manager._enter_shared_eval(eval_hold)

    update_hold = await manager.acquire_train_admission_hold()
    update_wait = asyncio.create_task(manager.wait_weight_update_admission(update_hold))
    await asyncio.sleep(0)
    assert not update_wait.done()

    manager._leave_shared_eval(eval_hold)
    await manager.release_train_admission_hold(eval_hold)
    await update_wait
    manager.weight_version = 2
    await manager.record_train_weight_update(update_hold)

    next_eval_hold = await manager.acquire_train_admission_hold()
    await manager.wait_train_admission_hold(next_eval_hold)
    assert next_eval_hold is not None
    eval_enter = asyncio.create_task(manager._enter_shared_eval(next_eval_hold))
    await asyncio.sleep(0)
    assert not eval_enter.done()

    await manager.release_train_admission_hold(update_hold)
    await eval_enter
    manager._leave_shared_eval(next_eval_hold)
    await manager.release_train_admission_hold(next_eval_hold)
    assert manager._active_shared_eval_holds == set()


@pytest.mark.asyncio
async def test_dispose_waits_for_shared_eval_before_closing_lifecycle(manager_env) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    hold_id = await manager.acquire_train_admission_hold()
    await manager.wait_train_admission_hold(hold_id)
    assert hold_id is not None
    await manager._enter_shared_eval(hold_id)

    dispose_task = asyncio.create_task(manager.dispose())
    await asyncio.sleep(0)
    assert not lifecycle.close_started.is_set()

    manager._leave_shared_eval(hold_id)
    await manager.release_train_admission_hold(hold_id)
    await dispose_task

    assert events == ["acquire", "wait", "release", "close"]


@pytest.mark.asyncio
async def test_dispose_rejects_active_weight_update_without_closing_manager(manager_env) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    hold_id = await manager.acquire_train_admission_hold()
    await manager.wait_weight_update_admission(hold_id)
    with pytest.raises(RuntimeError, match="weight update owns"):
        await manager.dispose()

    assert not manager._rollout_lifecycles_closing
    manager.weight_version = 3
    await manager.record_train_weight_update(hold_id)
    await manager.release_train_admission_hold(hold_id)
    await manager.dispose()

    assert events == ["acquire", "wait", "record:3", "release", "close"]


@pytest.mark.asyncio
async def test_shared_eval_waits_frontier_and_snapshot_eval_bypasses_it(manager_env, monkeypatch) -> None:
    manager, args = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    args.debug_train_only = False
    args.eval_uses_snapshots = False
    manager.eval_generate_rollout = object()
    manager._metric_checker = None
    monkeypatch.setattr(manager, "_health_monitoring_resume", lambda: events.append("health"))
    monkeypatch.setattr(
        rollout_manager_mod,
        "call_rollout_function",
        lambda rollout_fn, input: events.append("eval") or SimpleNamespace(data=[], metrics={}),
    )
    monkeypatch.setattr(rollout_manager_mod, "log_eval_rollout_data", lambda *args, **kwargs: {})

    await manager.eval(40)
    assert events == ["acquire", "wait", "health", "eval", "release"]

    events.clear()
    args.eval_uses_snapshots = True

    async def eval_checkpoint(*args, **kwargs):
        events.append("snapshot")

    monkeypatch.setattr(manager, "_eval_checkpoint", eval_checkpoint)
    await manager.eval(41, hf_dir="/checkpoint")
    assert events == ["health", "snapshot"]


@pytest.mark.asyncio
async def test_eval_skips_debug_replay_without_eval_rollout_fn(manager_env) -> None:
    manager, args = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    args.debug_train_only = False
    args.eval_uses_snapshots = False
    args.load_debug_rollout_data = "/debug/rollout/data"
    manager.eval_generate_rollout = None

    assert await manager.eval(40) is None
    assert events == []


@pytest.mark.asyncio
async def test_cancelled_public_eval_releases_fence_only_after_worker_finishes(manager_env, monkeypatch) -> None:
    manager, args = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    args.debug_train_only = False
    args.eval_uses_snapshots = False
    manager.eval_generate_rollout = object()
    manager._metric_checker = None
    eval_started = threading.Event()
    release_eval = threading.Event()
    monkeypatch.setattr(manager, "_health_monitoring_resume", lambda: None)

    def blocked_eval(rollout_fn, input):
        eval_started.set()
        release_eval.wait()
        return SimpleNamespace(data=[], metrics={})

    monkeypatch.setattr(rollout_manager_mod, "call_rollout_function", blocked_eval)
    monkeypatch.setattr(rollout_manager_mod, "log_eval_rollout_data", lambda *args, **kwargs: {})

    eval_task = asyncio.create_task(manager.eval(42))
    assert await asyncio.to_thread(eval_started.wait, 1)
    update_hold = await manager.acquire_train_admission_hold()
    update_wait = asyncio.create_task(manager.wait_weight_update_admission(update_hold))
    await asyncio.sleep(0)
    assert not update_wait.done()

    eval_task.cancel()
    await asyncio.sleep(0)
    assert not eval_task.done()
    assert not update_wait.done()

    release_eval.set()
    with pytest.raises(asyncio.CancelledError):
        await eval_task
    await update_wait
    manager.weight_version = 2
    await manager.record_train_weight_update(update_hold)
    await manager.release_train_admission_hold(update_hold)

    assert events == ["acquire", "wait", "acquire", "release", "wait", "record:2", "release"]


async def test_acquire_close_race_releases_unregistered_hold(
    manager_env: tuple[RolloutManager, SimpleNamespace],
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    acquire_gate = threading.Event()
    lifecycle = RecordingRolloutLifecycle(events, acquire_gate=acquire_gate)
    install_lifecycle(manager, lifecycle)

    acquire_task = asyncio.create_task(manager.acquire_train_admission_hold())
    assert await asyncio.to_thread(lifecycle.acquire_started.wait, 1)
    dispose_task = asyncio.create_task(manager.dispose())
    assert await asyncio.to_thread(lifecycle.close_started.wait, 1)

    acquire_gate.set()
    with pytest.raises(RuntimeError, match="closing"):
        await acquire_task
    await dispose_task

    assert events == ["acquire", "close", "release"]
    assert manager._train_admission_holds == {}


async def test_failed_hold_release_retains_handle_for_retry(
    manager_env: tuple[RolloutManager, SimpleNamespace],
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    release_failure = RuntimeError("release failed")
    lifecycle = RecordingRolloutLifecycle(events, release_error=release_failure)
    install_lifecycle(manager, lifecycle)

    hold_id = await manager.acquire_train_admission_hold()
    assert hold_id is not None
    with pytest.raises(RuntimeError, match="release failed"):
        await manager.release_train_admission_hold(hold_id)
    assert hold_id in manager._train_admission_holds

    with pytest.raises(RuntimeError, match="already has a release attempt"):
        await manager.release_train_admission_hold(hold_id)
    assert hold_id in manager._train_admission_holds
    assert events == ["acquire", "release"]


async def test_dispose_waits_for_active_generation(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    resource_events: list[str] = []
    manager.data_source = SimpleNamespace(close=lambda: resource_events.append("resource"))
    manager._data_source_closed = False
    generation_started = threading.Event()
    release_generation = threading.Event()
    lease = RecordingTrainBatchLease(rollout_id=55, events=events)

    def blocked_generate(input):
        generation_started.set()
        assert release_generation.wait(timeout=5)
        return leased_output(lease)

    manager.generate_rollout = blocked_generate
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    generation_task = asyncio.create_task(manager.generate(rollout_id=55))
    assert await asyncio.to_thread(generation_started.wait, 1)
    dispose_task = asyncio.create_task(manager.dispose())
    await asyncio.sleep(0)
    assert not dispose_task.done()

    release_generation.set()
    result = await generation_task
    publication = result["trainer_admission"]
    with pytest.raises(RuntimeError, match="unresolved trainer admissions"):
        await dispose_task
    assert events == []
    assert resource_events == []

    monkeypatch.setattr(
        rollout_manager_mod.object_store, "get_instance", lambda: SimpleNamespace(remove=lambda ref: None)
    )
    assert await manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLED_BACK
    await manager.dispose()
    assert events == ["rollback:TRAINER_ADMISSION_FAILED", "close"]
    assert resource_events == ["resource"]


async def test_lifecycle_close_precedes_resources_and_retries_failed_close(
    manager_env: tuple[RolloutManager, SimpleNamespace],
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    close_failure = RuntimeError("lifecycle close failed")
    lifecycle = RecordingRolloutLifecycle(events, close_errors=[close_failure])
    install_lifecycle(manager, lifecycle)
    manager.data_source = SimpleNamespace(close=lambda: events.append("resource"))
    manager._data_source_closed = False

    with pytest.raises(RuntimeError, match="lifecycle close failed"):
        await manager.dispose()
    assert events == ["close"]

    await manager.dispose()
    assert events == ["close", "close", "resource"]


async def test_checkpoint_fence_orders_prepare_source_and_event(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    manager.data_source = SimpleNamespace(save=lambda rollout_id: events.append(f"source:{rollout_id}"))
    monkeypatch.setattr(
        rollout_manager_mod.event_logger_checkpoint,
        "snapshot",
        lambda _args, rollout_id: events.append(f"event:{rollout_id}"),
    )

    await manager.save(31)

    assert events == ["acquire", "wait", "prepare:31", "source:31", "event:31", "release"]
    assert manager._train_admission_holds == {}


async def test_known_unresolved_admission_skips_prepare_and_publication(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)
    manager._pending_admissions[17] = SimpleNamespace(status=TrainerAdmissionStatus.PENDING)
    manager.data_source = SimpleNamespace(save=lambda _rollout_id: events.append("source"))
    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError, match=r"unresolved trainer admissions \[17\]"):
        await manager.save(32)

    assert events == ["acquire", "wait", "release"]
    assert manager._train_admission_holds == {}


async def test_late_unresolved_admission_blocks_publication_after_prepare(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    async def prepare_checkpoint(rollout_id: int) -> None:
        events.append(f"prepare:{rollout_id}")
        manager._pending_admissions[18] = SimpleNamespace(status=TrainerAdmissionStatus.ROLLBACK_FAILED)

    lifecycle.prepare_checkpoint = prepare_checkpoint
    manager.data_source = SimpleNamespace(save=lambda _rollout_id: events.append("source"))
    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError, match=r"unresolved trainer admissions \[18\]"):
        await manager.save(33)

    assert events == ["acquire", "wait", "prepare:33", "release"]
    assert manager._train_admission_holds == {}


async def test_prepare_failure_releases_hold_without_publishing(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    prepare_failure = RuntimeError("prepare failed")
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    async def fail_prepare(_rollout_id: int) -> None:
        events.append("prepare")
        raise prepare_failure

    lifecycle.prepare_checkpoint = fail_prepare
    manager.data_source = SimpleNamespace(save=lambda _rollout_id: events.append("source"))
    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError) as error:
        await manager.save(34)

    assert error.value is prepare_failure
    assert events == ["acquire", "wait", "prepare", "release"]


async def test_source_failure_preserves_primary_error_when_release_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    source_failure = RuntimeError("source save failed")
    release_failure = RuntimeError("release failed")
    lifecycle = RecordingRolloutLifecycle(events, release_error=release_failure)
    install_lifecycle(manager, lifecycle)
    manager.data_source = SimpleNamespace(save=lambda _rollout_id: (_ for _ in ()).throw(source_failure))
    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", lambda *_args: events.append("event"))

    with pytest.raises(RuntimeError) as error:
        await manager.save(35)

    assert error.value is source_failure
    assert error.value.__cause__ is release_failure
    assert events == ["acquire", "wait", "prepare:35", "release"]


async def test_cancelled_prepare_waits_for_cleanup_and_releases_exact_hold(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.rollout_global_dataset = True
    events: list[str] = []
    prepare_started = threading.Event()
    release_prepare = threading.Event()
    lifecycle = RecordingRolloutLifecycle(events)
    install_lifecycle(manager, lifecycle)

    async def blocked_prepare(_rollout_id: int) -> None:
        events.append("prepare")
        prepare_started.set()
        await asyncio.to_thread(release_prepare.wait)

    lifecycle.prepare_checkpoint = blocked_prepare
    manager.data_source = SimpleNamespace(save=lambda _rollout_id: events.append("source"))
    monkeypatch.setattr(rollout_manager_mod.event_logger_checkpoint, "snapshot", lambda *_args: events.append("event"))

    save_task = asyncio.create_task(manager.save(36))
    assert await asyncio.to_thread(prepare_started.wait, 1)
    save_task.cancel()
    await asyncio.sleep(0)
    assert not save_task.done()
    release_prepare.set()

    with pytest.raises(asyncio.CancelledError):
        await save_task

    assert events == ["acquire", "wait", "prepare", "release"]
    assert manager._train_admission_holds == {}


async def test_ordinary_output_preserves_existing_handoff(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    manager.generate_rollout = lambda input: RolloutFnTrainOutput(samples=[], metrics=None)
    monkeypatch.setattr(
        rollout_manager_mod,
        "split_train_data_by_dp",
        lambda args, data, train_parallel_config: ["published"],
    )

    result = await manager.generate(rollout_id=5)

    assert result == {"sample_indices": [5], "data_ref": ["published"]}


async def test_rejects_a_lease_for_another_rollout(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=99, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)

    def publish(*args, **kwargs):
        events.append("publish")
        return ["published"]

    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)

    with pytest.raises(ValueError) as error:
        await manager.generate(rollout_id=7)

    assert str(error.value) == "Leased train output for rollout 7 carries a lease for rollout 99."
    assert events == ["rollback:HANDOFF_FAILED"]


@pytest.mark.parametrize("rollout_is_async", [False, True])
@pytest.mark.parametrize("delay_split_train_data_by_dp", [False, True])
async def test_keeps_lease_pending_after_train_data_publication(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    delay_split_train_data_by_dp: bool,
    rollout_is_async: bool,
) -> None:
    manager, args = manager_env
    args.delay_split_train_data_by_dp = delay_split_train_data_by_dp
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=7, events=events)
    if rollout_is_async:

        async def generate_rollout(input):
            return leased_output(lease)

        manager.generate_rollout = generate_rollout
    else:
        manager.generate_rollout = lambda input: leased_output(lease)

    published_ref = Box("published") if delay_split_train_data_by_dp else [Box("published")]

    def publish(*args, **kwargs):
        events.append("publish")
        return published_ref

    if delay_split_train_data_by_dp:
        store = SimpleNamespace(put=publish)
        monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: store)
    else:
        monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)

    result = await manager.generate(rollout_id=7)

    assert events == ["publish"]
    publication = result["trainer_admission"]
    assert publication.rollout_id == 7
    assert publication.manager_incarnation == "manager-test"
    assert result == {"sample_indices": [5], "data_ref": published_ref, "trainer_admission": publication}


async def test_publication_construction_failure_rolls_back_and_cleans_published_refs(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication token failed")
    lease = RecordingTrainBatchLease(rollout_id=8, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    monkeypatch.setattr(rollout_manager_mod, "data_ref_ids", lambda data_ref: (_ for _ in ()).throw(handoff_failure))
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda ref: events.append(f"remove:{ref.inner}")),
    )

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=8)

    assert error.value is handoff_failure
    assert events == ["rollback:HANDOFF_FAILED", "remove:published"]
    assert manager._pending_admissions == {}


async def test_publication_registration_failure_rolls_back_and_cleans_published_refs(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication registration failed")
    lease = RecordingTrainBatchLease(rollout_id=9, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])

    class FailingRegistry(dict):
        def __setitem__(self, key, value):
            events.append("register")
            raise handoff_failure

    manager._pending_admissions = FailingRegistry()
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda ref: events.append(f"remove:{ref.inner}")),
    )

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=9)

    assert error.value is handoff_failure
    assert events == ["register", "rollback:HANDOFF_FAILED", "remove:published"]
    assert manager._pending_admissions == {}


async def test_partial_publication_registration_does_not_leave_an_untracked_lease(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication registration failed after insertion")
    lease = RecordingTrainBatchLease(rollout_id=9, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])

    class PartiallyFailingRegistry(dict):
        def __setitem__(self, key, value):
            dict.__setitem__(self, key, value)
            events.append("register")
            raise handoff_failure

    manager._pending_admissions = PartiallyFailingRegistry()
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda ref: events.append(f"remove:{ref.inner}")),
    )

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=9)

    assert error.value is handoff_failure
    assert events == ["register", "rollback:HANDOFF_FAILED", "remove:published"]
    assert manager._pending_admissions == {}


async def test_publication_cleanup_failure_is_chained_under_handoff_failure(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication token failed")
    cleanup_failure = RuntimeError("published ref cleanup failed")
    lease = RecordingTrainBatchLease(rollout_id=10, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    monkeypatch.setattr(rollout_manager_mod, "data_ref_ids", lambda data_ref: (_ for _ in ()).throw(handoff_failure))

    def remove(ref):
        events.append(f"remove:{ref.inner}")
        raise cleanup_failure

    monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: SimpleNamespace(remove=remove))

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=10)

    assert error.value is handoff_failure
    assert error.value.__cause__ is cleanup_failure
    assert events == ["rollback:HANDOFF_FAILED", "remove:published"]


async def test_cancellation_during_handoff_rollback_stays_a_cancellation(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication token failed")
    lease = GatedTrainBatchLease(11, events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    monkeypatch.setattr(rollout_manager_mod, "data_ref_ids", lambda data_ref: (_ for _ in ()).throw(handoff_failure))
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda ref: events.append(f"remove:{ref.inner}")),
    )

    generate = asyncio.create_task(manager.generate(rollout_id=11))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    generate.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not generate.done()

    lease.release_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await generate
    assert events == ["rollback:HANDOFF_FAILED", "remove:published"]


def _actor_receipt(publication):
    return TrainerAdmissionReceipt(
        publication=publication,
        role="actor",
        cohort=TrainerCohort(
            quorum_id=None,
            cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),),
        ),
    )


async def test_commit_validates_exact_roles_and_ref_before_settlement(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=31, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)

    def publish(*args, **kwargs):
        events.append("publish")
        return [Box("published")]

    monkeypatch.setattr(
        rollout_manager_mod,
        "split_train_data_by_dp",
        publish,
    )

    result = await manager.generate(rollout_id=31)
    publication = result["trainer_admission"]
    bad_ref = TrainerAdmissionReceipt(
        publication=publication.__class__(
            manager_incarnation=publication.manager_incarnation,
            admission_id=publication.admission_id,
            rollout_id=publication.rollout_id,
            data_ref_ids=("substituted",),
            required_roles=publication.required_roles,
        ),
        role="actor",
        cohort=_actor_receipt(publication).cohort,
    )

    with pytest.raises(ValueError, match="publication"):
        await manager.commit_trainer_admission(publication, (bad_ref,))
    assert events == ["publish"]

    with pytest.raises(ValueError, match="exactly"):
        await manager.commit_trainer_admission(publication, ())
    assert events == ["publish"]

    assert (
        await manager.commit_trainer_admission(publication, (_actor_receipt(publication),))
        is TrainerAdmissionStatus.COMMITTED
    )
    assert await manager.commit_trainer_admission(publication, ()) is TrainerAdmissionStatus.COMMITTED
    assert events == ["publish", "commit"]


async def test_restart_or_substituted_publication_cannot_settle_lease(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(41, events))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])

    result = await manager.generate(rollout_id=41)
    publication = result["trainer_admission"]
    restarted = publication.__class__(
        manager_incarnation="restarted-manager",
        admission_id=publication.admission_id,
        rollout_id=publication.rollout_id,
        data_ref_ids=publication.data_ref_ids,
        required_roles=publication.required_roles,
    )

    with pytest.raises(ValueError, match="does not match"):
        await manager.commit_trainer_admission(restarted, (_actor_receipt(publication),))
    assert events == []


@pytest.mark.parametrize(
    ("use_critic", "num_critic_only_steps", "rollout_id", "expected"),
    [
        (False, 0, 5, frozenset({"actor"})),
        (True, 2, 1, frozenset({"critic"})),
        (True, 2, 2, frozenset({"actor", "critic"})),
    ],
)
async def test_manager_publication_carries_exact_required_roles(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    use_critic: bool,
    num_critic_only_steps: int,
    rollout_id: int,
    expected: frozenset[str],
) -> None:
    manager, args = manager_env
    args.use_critic = use_critic
    args.num_critic_only_steps = num_critic_only_steps
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(rollout_id, []))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])

    result = await manager.generate(rollout_id=rollout_id)

    assert result["trainer_admission"].required_roles == expected


async def test_role_receipt_validation_happens_before_settlement(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.use_critic = True
    args.num_critic_only_steps = 0
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(43, events))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    publication = (await manager.generate(rollout_id=43))["trainer_admission"]
    actor = _actor_receipt(publication)
    critic = TrainerAdmissionReceipt(publication, "critic", actor.cohort)

    with pytest.raises(ValueError, match="exactly"):
        await manager.commit_trainer_admission(publication, (actor,))
    with pytest.raises(ValueError, match="repeats"):
        await manager.commit_trainer_admission(publication, (critic, critic))
    with pytest.raises(ValueError, match="foreign role"):
        await manager.commit_trainer_admission(
            publication, (actor, TrainerAdmissionReceipt(publication, "other", actor.cohort))
        )
    assert events == []


@pytest.mark.parametrize(
    "cohort",
    [
        TrainerCohort(quorum_id=None, cells=[]),
        TrainerCohort(quorum_id=None, cells=(TrainerCellCohort(cell_index=1, ranks=(0,)),)),
        TrainerCohort(
            quorum_id=None,
            cells=(TrainerCellCohort(cell_index=0, ranks=(0,)), TrainerCellCohort(cell_index=1, ranks=(0,))),
        ),
        TrainerCohort(quorum_id=None, cells=(TrainerCellCohort(cell_index=0, ranks=[0]),)),
        TrainerCohort(quorum_id=5, cells=()),
        TrainerCohort(quorum_id=True, cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=True, ranks=(0,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(True,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(-1,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(1, 0)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(0, 0)),)),
        TrainerCohort(quorum_id=1.0, cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=1.0, ranks=(0,)),)),
        TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(1.0,)),)),
        TrainerCohort(
            quorum_id=5,
            cells=(TrainerCellCohort(cell_index=1, ranks=(0,)), TrainerCellCohort(cell_index=1, ranks=(1,))),
        ),
        TrainerCohort(
            quorum_id=5,
            cells=(TrainerCellCohort(cell_index=1, ranks=(0,)), TrainerCellCohort(cell_index=0, ranks=(0,))),
        ),
    ],
)
async def test_receipt_cohort_structure_is_validated_before_lease_commit(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    cohort: TrainerCohort,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(46, events))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    publication = (await manager.generate(rollout_id=46))["trainer_admission"]

    with pytest.raises(ValueError, match="cohort"):
        await manager.commit_trainer_admission(
            publication,
            (TrainerAdmissionReceipt(publication=publication, role="actor", cohort=cohort),),
        )

    assert events == []


async def test_receipt_cohort_requires_exact_tuple_shapes(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, args = manager_env
    args.use_critic = True
    args.num_critic_only_steps = 0
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(47, events))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    publication = (await manager.generate(rollout_id=47))["trainer_admission"]
    valid_cell = TrainerCellCohort(cell_index=0, ranks=(0,))
    malformed = (
        TrainerAdmissionReceipt(publication, "actor", TrainerCohort(None, (valid_cell,))),
        TrainerAdmissionReceipt(publication, "critic", TrainerCohort(2, [valid_cell])),
    )

    with pytest.raises(ValueError, match="cohort"):
        await manager.commit_trainer_admission(publication, malformed)

    assert events == []


async def test_receipt_cohort_allows_sorted_unique_cell_and_rank_gaps(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(48, events))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    publication = (await manager.generate(rollout_id=48))["trainer_admission"]
    cohort = TrainerCohort(
        quorum_id=5,
        cells=(
            TrainerCellCohort(cell_index=0, ranks=(0,)),
            TrainerCellCohort(cell_index=2, ranks=(0, 2)),
        ),
    )

    status = await manager.commit_trainer_admission(
        publication,
        (TrainerAdmissionReceipt(publication=publication, role="actor", cohort=cohort),),
    )

    assert status is TrainerAdmissionStatus.COMMITTED
    assert events == ["commit"]


async def test_rollback_settles_source_before_deleting_refs_and_is_idempotent(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(44, events))

    def publish(*args, **kwargs):
        events.append("publish")
        return [Box("published")]

    def remove(ref):
        events.append("remove")

    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)
    monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: SimpleNamespace(remove=remove))
    publication = (await manager.generate(rollout_id=44))["trainer_admission"]

    assert await manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLED_BACK
    assert await manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLED_BACK
    assert events == ["publish", "rollback:TRAINER_ADMISSION_FAILED", "remove"]


async def test_rollback_failure_retains_refs_and_fails_closed(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    rollback_failure = RuntimeError("rollback failed")
    manager.generate_rollout = lambda input: leased_output(
        RecordingTrainBatchLease(45, events, rollback_error=rollback_failure)
    )
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda _: events.append("remove")),
    )
    publication = (await manager.generate(rollout_id=45))["trainer_admission"]

    with pytest.raises(RuntimeError, match="rollback failed"):
        await manager.rollback_trainer_admission(publication)
    assert await manager.get_trainer_admission_status(publication) is TrainerAdmissionStatus.ROLLBACK_FAILED
    assert await manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLBACK_FAILED
    assert events == ["rollback:TRAINER_ADMISSION_FAILED"]


async def _wait_for_admission_status(
    manager: RolloutManager,
    publication,
    expected: TrainerAdmissionStatus,
) -> None:
    async def poll() -> None:
        while await manager.get_trainer_admission_status(publication) is not expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1)


async def _publish_gated_admission(
    manager: RolloutManager,
    monkeypatch: pytest.MonkeyPatch,
    rollout_id: int,
    events: list[str],
) -> tuple[GatedTrainBatchLease, object]:
    lease = GatedTrainBatchLease(rollout_id, events)
    manager.generate_rollout = lambda input: leased_output(lease)
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=lambda _: events.append("remove")),
    )
    result = await manager.generate(rollout_id=rollout_id)
    return lease, result["trainer_admission"]


async def _count_ticks_while_settling(lease: GatedTrainBatchLease) -> int:
    """Return how many turns another coroutine got while settlement stayed gated."""
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    ticker = asyncio.create_task(tick())
    for _ in range(3):
        await asyncio.sleep(0)
    ticker.cancel()
    return ticks


async def test_commit_settles_without_blocking_the_manager_loop(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 51, events)

    settle = asyncio.create_task(manager.commit_trainer_admission(publication, (_actor_receipt(publication),)))

    assert await _count_ticks_while_settling(lease) > 0
    assert not settle.done()
    assert events == []

    lease.release_settlement.set()
    assert await settle is TrainerAdmissionStatus.COMMITTED
    assert events == ["commit"]


async def test_rollback_settles_without_blocking_the_manager_loop(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 52, events)

    settle = asyncio.create_task(manager.rollback_trainer_admission(publication))

    assert await _count_ticks_while_settling(lease) > 0
    assert not settle.done()
    assert events == []

    lease.release_settlement.set()
    assert await settle is TrainerAdmissionStatus.ROLLED_BACK
    assert events == ["rollback:TRAINER_ADMISSION_FAILED", "remove"]


async def test_status_query_during_commit_reports_only_the_settled_outcome(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 55, events)

    settle = asyncio.create_task(manager.commit_trainer_admission(publication, (_actor_receipt(publication),)))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    status = asyncio.create_task(manager.get_trainer_admission_status(publication))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not status.done()

    lease.release_settlement.set()
    assert await asyncio.wait_for(status, timeout=1) is TrainerAdmissionStatus.COMMITTED
    assert await settle is TrainerAdmissionStatus.COMMITTED


async def test_status_query_during_rollback_reports_only_the_settled_outcome(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 56, events)

    settle = asyncio.create_task(manager.rollback_trainer_admission(publication))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    status = asyncio.create_task(manager.get_trainer_admission_status(publication))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not status.done()

    lease.release_settlement.set()
    assert await asyncio.wait_for(status, timeout=1) is TrainerAdmissionStatus.ROLLED_BACK
    assert await settle is TrainerAdmissionStatus.ROLLED_BACK


async def test_rollback_during_commit_is_coalesced_into_the_commit_outcome(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 57, events)

    commit = asyncio.create_task(manager.commit_trainer_admission(publication, (_actor_receipt(publication),)))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    rollback = asyncio.create_task(manager.rollback_trainer_admission(publication))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not rollback.done()

    lease.release_settlement.set()
    assert await asyncio.wait_for(rollback, timeout=1) is TrainerAdmissionStatus.COMMITTED
    assert await commit is TrainerAdmissionStatus.COMMITTED
    assert events == ["commit"]


async def test_cancelled_commit_caller_cannot_abandon_settlement(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 53, events)

    settle = asyncio.create_task(manager.commit_trainer_admission(publication, (_actor_receipt(publication),)))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    settle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await settle

    lease.release_settlement.set()
    await _wait_for_admission_status(manager, publication, TrainerAdmissionStatus.COMMITTED)
    assert events == ["commit"]


async def test_cancelled_rollback_caller_cannot_abandon_settlement(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease, publication = await _publish_gated_admission(manager, monkeypatch, 54, events)

    settle = asyncio.create_task(manager.rollback_trainer_admission(publication))
    await asyncio.wait_for(lease.settlement_started.wait(), timeout=1)
    settle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await settle

    lease.release_settlement.set()
    await _wait_for_admission_status(manager, publication, TrainerAdmissionStatus.ROLLED_BACK)
    assert events == ["rollback:TRAINER_ADMISSION_FAILED", "remove"]


async def test_terminal_admission_reconciliation_history_is_bounded(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    monkeypatch.setattr(rollout_manager_mod, "_MAX_RETAINED_TERMINAL_ADMISSIONS", 2)
    manager.generate_rollout = lambda input: leased_output(RecordingTrainBatchLease(input.rollout_id, []))
    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", lambda *args, **kwargs: [Box("published")])

    publications = []
    for rollout_id in range(3):
        publication = (await manager.generate(rollout_id))["trainer_admission"]
        await manager.commit_trainer_admission(publication, (_actor_receipt(publication),))
        publications.append(publication)

    assert list(manager._pending_admissions) == [1, 2]
    with pytest.raises(ValueError, match="Unknown trainer admission"):
        await manager.get_trainer_admission_status(publications[0])
    assert await manager.get_trainer_admission_status(publications[-1]) is TrainerAdmissionStatus.COMMITTED


async def test_rolls_back_when_conversion_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=11, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    failure = RuntimeError("conversion failed")

    def fail_conversion(*args, **kwargs):
        raise failure

    monkeypatch.setattr(rollout_manager_mod, "convert_samples_to_train_data", fail_conversion)

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=11)

    assert error.value is failure
    assert events == ["rollback:HANDOFF_FAILED"]


async def test_rolls_back_when_postprocessing_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=12, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    failure = RuntimeError("postprocessing failed")

    def fail_postprocessing(*args, **kwargs):
        raise failure

    monkeypatch.setattr(rollout_manager_mod, "postprocess_rollout_data", fail_postprocessing)

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=12)

    assert error.value is failure
    assert events == ["rollback:HANDOFF_FAILED"]


@pytest.mark.parametrize("use_legacy_rollout_v1", [True, False])
@pytest.mark.parametrize("cancellation_count", [1, 2])
async def test_cancelled_rollout_invocation_rolls_back_eventual_lease(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    use_legacy_rollout_v1: bool,
    cancellation_count: int,
) -> None:
    manager, _ = manager_env
    manager.use_legacy_rollout_v1 = use_legacy_rollout_v1
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=13, events=events)
    rollout_started = threading.Event()
    release_rollout = threading.Event()
    rollout_finished = threading.Event()

    def complete_rollout() -> LeasedRolloutFnTrainOutput:
        rollout_started.set()
        try:
            assert release_rollout.wait(timeout=5)
            return leased_output(lease)
        finally:
            rollout_finished.set()

    if not use_legacy_rollout_v1:
        manager.generate_rollout = lambda input: complete_rollout()
    else:
        manager.generate_rollout = lambda args, rollout_id, data_source, evaluation: complete_rollout()
    generate_task = asyncio.create_task(manager.generate(rollout_id=13))
    assert await asyncio.to_thread(rollout_started.wait, 5)

    for _ in range(cancellation_count):
        generate_task.cancel()
        await asyncio.sleep(0)
    release_rollout.set()

    with pytest.raises(asyncio.CancelledError):
        await generate_task
    assert await asyncio.to_thread(rollout_finished.wait, 5)
    assert events == ["rollback:HANDOFF_FAILED"]


@pytest.mark.parametrize("delay_split_train_data_by_dp", [False, True])
@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: RuntimeError("publication failed"),
        asyncio.CancelledError,
    ],
)
async def test_rolls_back_when_publication_does_not_complete(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    delay_split_train_data_by_dp: bool,
    failure_factory: Callable[[], BaseException],
) -> None:
    manager, args = manager_env
    args.delay_split_train_data_by_dp = delay_split_train_data_by_dp
    events: list[str] = []
    lease = RecordingTrainBatchLease(rollout_id=13, events=events)
    manager.generate_rollout = lambda input: leased_output(lease)
    failure = failure_factory()

    def fail_publication(*args, **kwargs):
        events.append("publish")
        raise failure

    if delay_split_train_data_by_dp:
        monkeypatch.setattr(
            rollout_manager_mod.object_store,
            "get_instance",
            lambda: SimpleNamespace(put=fail_publication),
        )
    else:
        monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", fail_publication)

    with pytest.raises(type(failure)) as error:
        await manager.generate(rollout_id=13)

    assert error.value is failure
    assert events == ["publish", "rollback:HANDOFF_FAILED"]


async def test_preserves_handoff_error_when_rollback_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    handoff_failure = RuntimeError("publication failed")
    rollback_failure = RuntimeError("rollback failed")
    lease = RecordingTrainBatchLease(
        rollout_id=17,
        events=events,
        rollback_error=rollback_failure,
    )
    manager.generate_rollout = lambda input: leased_output(lease)

    def fail_publication(args, data, train_parallel_config):
        raise handoff_failure

    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", fail_publication)

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=17)

    assert error.value is handoff_failure
    assert error.value.__cause__ is rollback_failure
    assert events == ["rollback:HANDOFF_FAILED"]


@pytest.mark.parametrize("delay_split_train_data_by_dp", [False, True])
async def test_failed_commit_retains_refs_and_fails_closed(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    delay_split_train_data_by_dp: bool,
) -> None:
    manager, args = manager_env
    args.delay_split_train_data_by_dp = delay_split_train_data_by_dp
    events: list[str] = []
    commit_failure = RuntimeError("commit failed")
    lease = RecordingTrainBatchLease(
        rollout_id=19,
        events=events,
        commit_error=commit_failure,
    )
    manager.generate_rollout = lambda input: leased_output(lease)

    published_refs = [Box("published")] if delay_split_train_data_by_dp else [Box("published-0"), Box("published-1")]

    def publish(*args, **kwargs):
        events.append("publish")
        return published_refs[0] if delay_split_train_data_by_dp else published_refs

    store = SimpleNamespace(put=publish)
    monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: store)
    if not delay_split_train_data_by_dp:
        monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)

    result = await manager.generate(rollout_id=19)
    publication = result["trainer_admission"]
    with pytest.raises(RuntimeError) as error:
        await manager.commit_trainer_admission(publication, (_actor_receipt(publication),))

    assert error.value is commit_failure
    assert events == ["publish", "commit"]
    assert await manager.get_trainer_admission_status(publication) is TrainerAdmissionStatus.COMMIT_FAILED


async def test_preserves_commit_error_when_published_data_cleanup_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    commit_failure = RuntimeError("commit failed")
    lease = RecordingTrainBatchLease(
        rollout_id=23,
        events=events,
        commit_error=commit_failure,
    )
    manager.generate_rollout = lambda input: leased_output(lease)

    def publish(*args, **kwargs):
        events.append("publish")
        return [Box("published-0"), Box("published-1")]

    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(),
    )

    result = await manager.generate(rollout_id=23)
    publication = result["trainer_admission"]
    with pytest.raises(RuntimeError) as error:
        await manager.commit_trainer_admission(publication, (_actor_receipt(publication),))

    assert error.value is commit_failure
    assert error.value.__cause__ is None
    assert events == ["publish", "commit"]
