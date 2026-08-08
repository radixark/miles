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
    RolloutFnTrainOutput,
    TrainBatchLease,
    TrainBatchRollbackReason,
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
        manager.commit_trainer_admission(publication, (bad_ref,))
    assert events == ["publish"]

    with pytest.raises(ValueError, match="exactly"):
        manager.commit_trainer_admission(publication, ())
    assert events == ["publish"]

    assert (
        manager.commit_trainer_admission(publication, (_actor_receipt(publication),))
        is TrainerAdmissionStatus.COMMITTED
    )
    assert manager.commit_trainer_admission(publication, ()) is TrainerAdmissionStatus.COMMITTED
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
        manager.commit_trainer_admission(restarted, (_actor_receipt(publication),))
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
        manager.commit_trainer_admission(publication, (actor,))
    with pytest.raises(ValueError, match="repeats"):
        manager.commit_trainer_admission(publication, (critic, critic))
    with pytest.raises(ValueError, match="foreign role"):
        manager.commit_trainer_admission(
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
        manager.commit_trainer_admission(
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
        manager.commit_trainer_admission(publication, malformed)

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

    status = manager.commit_trainer_admission(
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

    assert manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLED_BACK
    assert manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLED_BACK
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
        manager.rollback_trainer_admission(publication)
    assert manager.get_trainer_admission_status(publication) is TrainerAdmissionStatus.ROLLBACK_FAILED
    assert manager.rollback_trainer_admission(publication) is TrainerAdmissionStatus.ROLLBACK_FAILED
    assert events == ["rollback:TRAINER_ADMISSION_FAILED"]


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
        manager.commit_trainer_admission(publication, (_actor_receipt(publication),))
        publications.append(publication)

    assert list(manager._pending_admissions) == [1, 2]
    with pytest.raises(ValueError, match="Unknown trainer admission"):
        manager.get_trainer_admission_status(publications[0])
    assert manager.get_trainer_admission_status(publications[-1]) is TrainerAdmissionStatus.COMMITTED


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
        manager.commit_trainer_admission(publication, (_actor_receipt(publication),))

    assert error.value is commit_failure
    assert events == ["publish", "commit"]
    assert manager.get_trainer_admission_status(publication) is TrainerAdmissionStatus.COMMIT_FAILED


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
        manager.commit_trainer_admission(publication, (_actor_receipt(publication),))

    assert error.value is commit_failure
    assert error.value.__cause__ is None
    assert events == ["publish", "commit"]
