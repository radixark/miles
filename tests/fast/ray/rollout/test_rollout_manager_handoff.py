import asyncio
import threading
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_mod
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.rollout.base_types import (
    LeasedRolloutFnTrainOutput,
    RolloutFnTrainOutput,
    TrainBatchLease,
    TrainBatchRollbackReason,
)


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
    )
    manager.weight_version = None
    manager.rollout_id = -1
    manager.servers = {}
    manager.data_source = SimpleNamespace()
    manager.train_parallel_config = {"dp_size": 1}
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.use_legacy_rollout_v1 = False
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
async def test_commits_lease_after_train_data_publication(
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

    published_ref = "published" if delay_split_train_data_by_dp else ["published"]

    def publish(*args, **kwargs):
        events.append("publish")
        return published_ref

    if delay_split_train_data_by_dp:
        store = SimpleNamespace(put=publish)
        monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: store)
    else:
        monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)

    result = await manager.generate(rollout_id=7)

    assert events == ["publish", "commit"]
    assert result == {"sample_indices": [5], "data_ref": published_ref}


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
async def test_failed_commit_removes_published_data_without_rollback(
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

    published_refs = ["published"] if delay_split_train_data_by_dp else ["published-0", "published-1"]

    def publish(*args, **kwargs):
        events.append("publish")
        return published_refs[0] if delay_split_train_data_by_dp else published_refs

    def remove(ref):
        events.append(f"remove:{ref}")

    store = SimpleNamespace(put=publish, remove=remove)
    monkeypatch.setattr(rollout_manager_mod.object_store, "get_instance", lambda: store)
    if not delay_split_train_data_by_dp:
        monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=19)

    assert error.value is commit_failure
    expected_events = (
        ["publish", "commit", "remove:published"]
        if delay_split_train_data_by_dp
        else ["publish", "commit", "remove:published-0", "remove:published-1"]
    )
    assert events == expected_events


async def test_preserves_commit_error_when_published_data_cleanup_fails(
    manager_env: tuple[RolloutManager, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = manager_env
    events: list[str] = []
    commit_failure = RuntimeError("commit failed")
    cleanup_failure = RuntimeError("cleanup failed")
    lease = RecordingTrainBatchLease(
        rollout_id=23,
        events=events,
        commit_error=commit_failure,
    )
    manager.generate_rollout = lambda input: leased_output(lease)

    def publish(*args, **kwargs):
        events.append("publish")
        return ["published-0", "published-1"]

    def remove(ref):
        events.append(f"remove:{ref}")
        if ref == "published-0":
            raise cleanup_failure

    monkeypatch.setattr(rollout_manager_mod, "split_train_data_by_dp", publish)
    monkeypatch.setattr(
        rollout_manager_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(remove=remove),
    )

    with pytest.raises(RuntimeError) as error:
        await manager.generate(rollout_id=23)

    assert error.value is commit_failure
    assert error.value.__cause__ is cleanup_failure
    assert events == ["publish", "commit", "remove:published-0", "remove:published-1"]
