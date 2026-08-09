import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
import train_async


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _RecordingRolloutManager:
    def __init__(self, events, *, leased=False):
        self.events = events
        self.leased = leased
        self.pending = set()
        self._generation = 0
        self.acquire_train_admission_hold = _RemoteMethod(self._acquire_train_admission_hold)
        self.wait_weight_update_admission = _RemoteMethod(self._wait_weight_update_admission)
        self.record_train_weight_update = _RemoteMethod(self._record_train_weight_update)
        self.release_train_admission_hold = _RemoteMethod(self._release_train_admission_hold)
        self.generate = _RemoteMethod(self._generate)
        self.save = _RemoteMethod(self._save)
        self.dispose = _RemoteMethod(self._dispose)

    async def _acquire_train_admission_hold(self):
        return None

    async def _wait_weight_update_admission(self, hold_id):
        assert hold_id is None

    async def _record_train_weight_update(self, hold_id):
        assert hold_id is None

    async def _release_train_admission_hold(self, hold_id):
        assert hold_id is None

    def _generate(self, rollout_id):
        generation = self._generation
        self._generation += 1
        key = (rollout_id, generation)
        self.events.append(f"generate:{rollout_id}:{generation}")
        if self.leased:
            self.pending.add(key)

        async def complete():
            self.events.append(f"handoff:{rollout_id}:{generation}")
            return {"rollout_id": rollout_id, "generation": generation, "leased": self.leased}

        return complete()

    async def _save(self, rollout_id):
        if self.pending:
            raise AssertionError(f"save observed pending publications: {self.pending}")
        self.events.append(f"rollout_save:{rollout_id}")

    async def _dispose(self):
        if self.pending:
            raise AssertionError(f"dispose observed pending publications: {self.pending}")
        self.events.append("dispose")


class _AdmissionRolloutManager(_RecordingRolloutManager):
    def __init__(self, events, *, record_error=None):
        self.active_holds = set()
        self._next_hold_id = 0
        self._record_error = record_error
        super().__init__(events)

    async def _acquire_train_admission_hold(self):
        hold_id = self._next_hold_id
        self._next_hold_id += 1
        self.active_holds.add(hold_id)
        self.events.append(f"acquire:{hold_id}")
        return hold_id

    async def _wait_weight_update_admission(self, hold_id):
        assert hold_id in self.active_holds
        self.events.append(f"wait:{hold_id}")

    async def _record_train_weight_update(self, hold_id):
        assert hold_id in self.active_holds
        self.events.append(f"record:{hold_id}")
        if self._record_error is not None:
            raise self._record_error

    async def _release_train_admission_hold(self, hold_id):
        self.active_holds.remove(hold_id)
        self.events.append(f"release:{hold_id}")


class _BlockingRolloutManager(_RecordingRolloutManager):
    def __init__(self, events):
        super().__init__(events, leased=True)
        self.handoff_started = asyncio.Event()
        self.allow_handoff = asyncio.Event()

    def _generate(self, rollout_id):
        generation = self._generation
        self._generation += 1
        key = (rollout_id, generation)
        self.events.append(f"generate:{rollout_id}:{generation}")
        self.pending.add(key)

        async def complete():
            self.handoff_started.set()
            await self.allow_handoff.wait()
            self.events.append(f"handoff:{rollout_id}:{generation}")
            return {"rollout_id": rollout_id, "generation": generation, "leased": True}

        return complete()


class _RecordingActor:
    def __init__(self, events, *, update_error=None):
        self.events = events
        self.update_error = update_error

    async def update_weights(self, **kwargs):
        label = kwargs.get("rollout_id", "initial")
        self.events.append(f"update:{label}")
        if self.update_error is not None:
            raise self.update_error

    async def save_model(self, rollout_id, *, force_sync):
        self.events.append(f"model_save:{rollout_id}:{force_sync}")


class _RecordingEval:
    def __init__(self, events):
        self.events = events

    async def dispatch(self, rollout_id, **kwargs):
        self.events.append(f"eval:{rollout_id}")

    async def drain(self):
        self.events.append("eval_drain")


class _RecordingCoordinator:
    def __init__(self, *, events, manager, train_error=None, create_sentinel=None, rollback_error=None):
        self.events = events
        self.manager = manager
        self.train_error = train_error
        self.create_sentinel = create_sentinel
        self.rollback_error = rollback_error

    async def train(self, rollout_id, pack):
        self.events.append(f"train:{rollout_id}")
        if pack["leased"]:
            self.manager.pending.remove((pack["rollout_id"], pack["generation"]))
        if self.create_sentinel is not None and rollout_id == 0:
            Path(self.create_sentinel).touch()
        if self.train_error is not None:
            raise self.train_error

    async def rollback_prefetched(self, pack):
        if self.rollback_error is not None:
            raise self.rollback_error
        if not pack["leased"]:
            return False
        key = (pack["rollout_id"], pack["generation"])
        self.events.append(f"rollback:{pack['rollout_id']}:{pack['generation']}")
        self.manager.pending.remove(key)
        return True


def _args(
    *,
    num_rollout=2,
    update_weights_interval=99,
    save_interval=None,
    eval_interval=None,
    debug_exit=None,
    sentinel=None,
):
    return SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        control_server_port=None,
        debug_exit_after_rollout=debug_exit,
        eval_interval=eval_interval,
        ft_components=[],
        hf_checkpoint=None,
        num_rollout=num_rollout,
        num_critic_only_steps=0,
        offload_train=False,
        save_interval=save_interval,
        save_trigger_sentinel=str(sentinel) if sentinel is not None else None,
        skip_eval_before_train=True,
        start_rollout_id=0,
        update_weights_interval=update_weights_interval,
        use_critic=False,
    )


def _patch_train(monkeypatch, manager, actor, coordinator, events):
    monkeypatch.setattr(train_async, "configure_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_async.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_async, "validate_async_off_policy_correction", lambda args: None)
    monkeypatch.setattr(train_async, "remove_rollout_data_refs", lambda *args, **kwargs: events.append("remove_refs"))
    monkeypatch.setattr(train_async, "create_rollout_manager", lambda args, pg: (manager, None))

    async def create_models(args, pgs, rollout_manager):
        return actor, None

    monkeypatch.setattr(train_async, "create_training_models", create_models)
    monkeypatch.setattr(train_async, "maybe_start_mini_ft_controller", lambda args: None)
    monkeypatch.setattr(train_async, "EvalDispatcher", lambda args, actor, manager: _RecordingEval(events))
    monkeypatch.setattr(train_async, "TrainBatchCoordinator", lambda **kwargs: coordinator)


@pytest.mark.asyncio
async def test_weight_updates_record_exact_hold_before_reopening_admission(monkeypatch):
    events = []
    manager = _AdmissionRolloutManager(events)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    await train_async.train(_args(num_rollout=1, update_weights_interval=1))

    assert events == [
        "acquire:0",
        "wait:0",
        "update:initial",
        "record:0",
        "release:0",
        "generate:0:0",
        "handoff:0:0",
        "train:0",
        "acquire:1",
        "wait:1",
        "update:0",
        "record:1",
        "release:1",
        "eval_drain",
        "dispose",
    ]
    assert manager.active_holds == set()


@pytest.mark.asyncio
async def test_weight_update_failure_retains_admission_hold(monkeypatch):
    events = []
    failure = RuntimeError("weight update failed")
    manager = _AdmissionRolloutManager(events)
    actor = _RecordingActor(events, update_error=failure)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    with pytest.raises(RuntimeError) as update_error:
        await train_async.train(_args(num_rollout=1, update_weights_interval=1))

    assert update_error.value is failure
    assert events == ["acquire:0", "wait:0", "update:initial"]
    assert manager.active_holds == {0}


@pytest.mark.asyncio
async def test_weight_update_record_failure_retains_admission_hold(monkeypatch):
    events = []
    failure = RuntimeError("weight update record failed")
    manager = _AdmissionRolloutManager(events, record_error=failure)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    with pytest.raises(RuntimeError) as record_error:
        await train_async.train(_args(num_rollout=1, update_weights_interval=1))

    assert record_error.value is failure
    assert events == ["acquire:0", "wait:0", "update:initial", "record:0"]
    assert manager.active_holds == {0}


@pytest.mark.asyncio
async def test_ordinary_iteration_starts_one_batch_ahead(monkeypatch):
    events = []
    manager = _RecordingRolloutManager(events)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    await train_async.train(_args(num_rollout=2))

    assert events == [
        "update:initial",
        "generate:0:0",
        "handoff:0:0",
        "generate:1:1",
        "train:0",
        "handoff:1:1",
        "train:1",
        "eval_drain",
        "dispose",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"update_weights_interval": 1},
            ["update:initial", "generate:0:0", "handoff:0:0", "train:0", "update:0", "generate:1:1"],
        ),
        (
            {"save_interval": 1},
            [
                "update:initial",
                "generate:0:0",
                "handoff:0:0",
                "train:0",
                "model_save:0:False",
                "rollout_save:0",
                "generate:1:1",
            ],
        ),
        (
            {"eval_interval": 1},
            ["update:initial", "generate:0:0", "handoff:0:0", "train:0", "eval:0", "generate:1:1"],
        ),
        (
            {"debug_exit": 1},
            ["update:initial", "generate:0:0", "handoff:0:0", "train:0", "eval_drain", "dispose"],
        ),
    ],
)
async def test_known_boundary_delays_prefetch_until_after_boundary(monkeypatch, kwargs, expected):
    events = []
    manager = _RecordingRolloutManager(events)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    await train_async.train(_args(**kwargs))

    assert events[: len(expected)] == expected
    if kwargs.get("debug_exit"):
        assert "generate:1:1" not in events


@pytest.mark.asyncio
@pytest.mark.parametrize("leased", [False, True])
async def test_external_save_race_drains_prefetch_and_reuses_legacy_or_replays_leased(monkeypatch, tmp_path, leased):
    events = []
    manager = _RecordingRolloutManager(events, leased=leased)
    actor = _RecordingActor(events)
    sentinel = tmp_path / "save-trigger"
    coordinator = _RecordingCoordinator(
        events=events,
        manager=manager,
        create_sentinel=sentinel,
    )
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    await train_async.train(_args(sentinel=sentinel))

    if leased:
        assert "rollback:1:1" in events
        assert events.index("rollback:1:1") < events.index("rollout_save:0")
        assert events.count("generate:1:1") == 1
        assert events.count("generate:1:2") == 1
    else:
        assert not any(event.startswith("rollback:") for event in events)
        assert events.count("generate:1:1") == 1
        assert events.index("handoff:1:1") < events.index("rollout_save:0")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("train failed"), asyncio.CancelledError()])
async def test_training_failure_drains_leased_prefetch_without_masking_primary(monkeypatch, failure):
    events = []
    manager = _RecordingRolloutManager(events, leased=True)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager, train_error=failure)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    with pytest.raises(type(failure)) as raised:
        await train_async.train(_args())

    assert raised.value is failure
    assert "rollback:1:1" in events
    assert events.index("handoff:1:1") < events.index("rollback:1:1")


@pytest.mark.asyncio
async def test_training_failure_remains_primary_when_prefetch_cleanup_fails(monkeypatch):
    events = []
    manager = _RecordingRolloutManager(events, leased=True)
    actor = _RecordingActor(events)
    train_error = RuntimeError("train failed")
    cleanup_error = RuntimeError("rollback failed")
    coordinator = _RecordingCoordinator(
        events=events,
        manager=manager,
        train_error=train_error,
        rollback_error=cleanup_error,
    )
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    with pytest.raises(RuntimeError) as raised:
        await train_async.train(_args())

    assert raised.value is train_error
    assert raised.value.__cause__ is cleanup_error


@pytest.mark.asyncio
async def test_cancellation_while_syncing_current_handoff_rolls_back_before_exit(monkeypatch):
    events = []
    manager = _BlockingRolloutManager(events)
    actor = _RecordingActor(events)
    coordinator = _RecordingCoordinator(events=events, manager=manager)
    _patch_train(monkeypatch, manager, actor, coordinator, events)

    task = asyncio.create_task(train_async.train(_args(num_rollout=1)))
    await manager.handoff_started.wait()
    task.cancel()
    manager.allow_handoff.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.index("handoff:0:0") < events.index("rollback:0:0")
    assert manager.pending == set()


@pytest.mark.asyncio
async def test_cancelled_remote_handoff_error_keeps_cancellation_primary():
    started = asyncio.Event()
    finish = asyncio.Event()
    remote_error = RuntimeError("handoff failed")

    async def fail_after_cancellation():
        started.set()
        await finish.wait()
        raise remote_error

    waiter = asyncio.create_task(train_async._await_remote_result_with_cancellation(fail_after_cancellation()))
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    finish.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await waiter

    assert raised.value.__cause__ is remote_error
