from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from miles.ray.train_batch_admission import (
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
)
from miles.ray.train_batch_coordinator import TrainBatchCoordinator
from miles.utils.data import remove_rollout_data_refs


def _args(*, use_critic: bool, num_critic_only_steps: int = 0, offload_train: bool = True):
    return SimpleNamespace(
        use_critic=use_critic,
        num_critic_only_steps=num_critic_only_steps,
        offload_train=offload_train,
    )


def test_rollout_ref_cleanup_attempts_every_shard_before_raising(monkeypatch):
    removed = []
    failure = RuntimeError("first remove failed")

    def remove(ref):
        removed.append(ref)
        if ref == "first":
            raise failure

    monkeypatch.setattr(
        "miles.utils.data.object_store.get_instance",
        lambda: SimpleNamespace(remove=remove),
    )

    with pytest.raises(RuntimeError) as raised:
        remove_rollout_data_refs(None, {"data_ref": ["first", "second"]})

    assert raised.value is failure
    assert removed == ["first", "second"]


def _publication(*roles: str, rollout_id: int = 7) -> TrainBatchPublication:
    return TrainBatchPublication(
        manager_incarnation="manager",
        admission_id=11,
        rollout_id=rollout_id,
        data_ref_ids=("ref",),
        required_roles=frozenset(roles),
    )


def _receipt(publication: TrainBatchPublication, role: str) -> TrainerAdmissionReceipt:
    return TrainerAdmissionReceipt(
        publication=publication,
        role=role,
        cohort=TrainerCohort(
            quorum_id=None,
            cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),),
        ),
    )


@dataclass
class _FakeGroup:
    role: str
    events: list[tuple]
    train_error: BaseException | None = None
    admission_error: BaseException | None = None
    train_result: object = None

    async def admit_train_batch(self, rollout_id, rollout_data_pack):
        self.events.append((self.role, "admit"))
        if self.admission_error is not None:
            raise self.admission_error
        return _receipt(rollout_data_pack["trainer_admission"], self.role)

    async def train(self, rollout_id, rollout_data_pack, external_data=None, admission_receipt=None):
        self.events.append((self.role, "train", external_data, admission_receipt))
        if self.train_error is not None:
            raise self.train_error
        return self.train_result

    async def offload(self):
        self.events.append((self.role, "offload"))

    def discard_train_batch_admission(self, receipt):
        self.events.append((self.role, "discard", receipt))


class _FakeAdmissionAdapter:
    def __init__(
        self,
        events,
        *,
        commit_status=TrainerAdmissionStatus.COMMITTED,
        commit_error=None,
        rollback_error=None,
    ):
        self.events = events
        self.commit_status = commit_status
        self.commit_error = commit_error
        self.rollback_error = rollback_error

    async def commit(self, publication, receipts):
        self.events.append(("manager", "commit", tuple(receipt.role for receipt in receipts)))
        if self.commit_error is not None:
            raise self.commit_error
        return self.commit_status

    async def rollback(self, publication):
        self.events.append(("manager", "rollback"))
        if self.rollback_error is not None:
            raise self.rollback_error
        return TrainerAdmissionStatus.ROLLED_BACK

    async def status(self, publication):
        self.events.append(("manager", "status"))
        return self.commit_status


class _ActorOnlyGroup:
    role = "actor"

    def __init__(self, events):
        self.events = events

    async def admit_train_batch(self, rollout_id, rollout_data_pack):
        self.events.append(("actor", "admit"))
        return _receipt(rollout_data_pack["trainer_admission"], self.role)

    async def train(self, rollout_id, rollout_data_pack, *, admission_receipt=None):
        self.events.append(("actor", "train", admission_receipt))

    def discard_train_batch_admission(self, receipt):
        self.events.append(("actor", "discard", receipt))


@pytest.fixture
def cleanup_refs(monkeypatch):
    removed = []
    monkeypatch.setattr(
        "miles.ray.train_batch_coordinator.remove_rollout_data_refs",
        lambda *args: removed.append(args[-1]),
    )
    return removed


@pytest.mark.asyncio
async def test_legacy_actor_only_trains_then_removes_refs(cleanup_refs):
    events = []
    actor = _ActorOnlyGroup(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
    )
    pack = {"data_ref": object()}

    await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [("actor", "train")]
    assert cleanup_refs == [pack]


@pytest.mark.asyncio
async def test_legacy_critic_only_preserves_critic_offload_order(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, train_result=["values"])
    actor = _FakeGroup("actor", events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True, num_critic_only_steps=10),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
    )
    pack = {"data_ref": object()}

    await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [("critic", "train"), ("critic", "offload")]
    assert cleanup_refs == [pack]


@pytest.mark.asyncio
async def test_legacy_actor_critic_passes_critic_values_and_offloads_in_order(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, train_result=["values"])
    actor = _FakeGroup("actor", events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
    )
    pack = {"data_ref": object()}

    await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("critic", "train"),
        ("critic", "offload"),
        ("actor", "train"),
        ("actor", "offload"),
    ]
    assert events[2][2] == ["values"]
    assert cleanup_refs == [pack]


@pytest.mark.asyncio
async def test_legacy_training_failure_does_not_remove_refs(monkeypatch):
    events = []
    removed = []
    monkeypatch.setattr(
        "miles.ray.train_batch_coordinator.remove_rollout_data_refs",
        lambda *args: removed.append(args[-1]),
    )
    actor = _FakeGroup("actor", events, train_error=ValueError("train failed"))
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False), actor_model=actor, critic_model=None, rollout_manager=None
    )

    with pytest.raises(ValueError, match="train failed"):
        await coordinator.train(7, {"data_ref": object()})

    assert removed == []


@pytest.mark.asyncio
async def test_malformed_admission_cannot_fall_back_to_legacy_training(cleanup_refs):
    events = []
    actor = _ActorOnlyGroup(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
    )

    with pytest.raises(ValueError, match="invalid trainer admission"):
        await coordinator.train(7, {"data_ref": object(), "trainer_admission": "stale"})

    assert events == []
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_leased_actor_critic_admit_commit_then_train_exact_order(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, train_result=["values"])
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor", "critic")
    pack = {"data_ref": object(), "trainer_admission": publication}

    await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("critic", "admit"),
        ("actor", "admit"),
        ("manager", "commit"),
        ("critic", "train"),
        ("critic", "offload"),
        ("actor", "train"),
        ("actor", "offload"),
        ("critic", "discard"),
        ("actor", "discard"),
    ]
    assert cleanup_refs == [pack]


@pytest.mark.asyncio
async def test_rollback_prefetched_leased_batch_returns_source_without_training(cleanup_refs):
    events = []
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor")
    pack = {"data_ref": object(), "trainer_admission": publication}

    assert await coordinator.rollback_prefetched(pack) is True

    assert [event[:2] for event in events] == [("manager", "rollback")]
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_rollback_prefetched_legacy_batch_is_retained(cleanup_refs):
    events = []
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=_ActorOnlyGroup(events),
        critic_model=None,
        rollout_manager=None,
    )
    pack = {"data_ref": object()}

    assert await coordinator.rollback_prefetched(pack) is False

    assert events == []
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_precommit_admission_failure_rolls_back_and_never_optimizes(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, admission_error=ValueError("admission failed"))
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor", "critic")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(ValueError, match="admission failed"):
        await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("critic", "admit"),
        ("manager", "rollback"),
    ]
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_later_role_admission_failure_discards_earlier_role_pin(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events)
    actor = _FakeGroup("actor", events, admission_error=ValueError("actor admission failed"))
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor", "critic")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(ValueError, match="actor admission failed"):
        await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("critic", "admit"),
        ("actor", "admit"),
        ("manager", "rollback"),
        ("critic", "discard"),
    ]
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_pending_commit_rejection_rolls_back_without_training(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, train_result=["values"])
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events, commit_status=TrainerAdmissionStatus.PENDING)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor", "critic")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(RuntimeError, match="commit"):
        await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("critic", "admit"),
        ("actor", "admit"),
        ("manager", "commit"),
        ("manager", "rollback"),
        ("critic", "discard"),
        ("actor", "discard"),
    ]
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_ambiguous_commit_reconciled_to_committed_resumes_training(cleanup_refs):
    events = []
    actor = _ActorOnlyGroup(events)
    adapter = _FakeAdmissionAdapter(events, commit_error=RuntimeError("lost commit response"))
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor")
    pack = {"data_ref": object(), "trainer_admission": publication}

    await coordinator.train(7, pack)

    assert ("manager", "rollback") not in events
    assert any(event[:2] == ("actor", "train") for event in events)
    assert cleanup_refs == [pack]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expect_rollback", "expect_ref_cleanup"),
    [
        (TrainerAdmissionStatus.COMMITTED, False, True),
        (TrainerAdmissionStatus.PENDING, True, False),
    ],
)
async def test_commit_cancellation_settles_without_starting_optimizer(
    cleanup_refs,
    status,
    expect_rollback,
    expect_ref_cleanup,
):
    events = []
    cancellation = asyncio.CancelledError()
    actor = _ActorOnlyGroup(events)
    adapter = _FakeAdmissionAdapter(
        events,
        commit_status=status,
        commit_error=cancellation,
    )
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(asyncio.CancelledError) as raised:
        await coordinator.train(7, pack)

    assert raised.value is cancellation
    assert not any(event[:2] == ("actor", "train") for event in events)
    assert (("manager", "rollback") in events) is expect_rollback
    assert cleanup_refs == ([pack] if expect_ref_cleanup else [])


@pytest.mark.asyncio
async def test_commit_failed_is_fail_closed_and_discards_pins(cleanup_refs):
    events = []
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events, commit_status=TrainerAdmissionStatus.COMMIT_FAILED)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(RuntimeError, match="commit"):
        await coordinator.train(7, pack)

    assert [event[:2] for event in events] == [
        ("actor", "admit"),
        ("manager", "commit"),
        ("actor", "discard"),
    ]
    assert cleanup_refs == []


@pytest.mark.asyncio
async def test_postcommit_failure_never_replays_and_cleanup_error_is_secondary(monkeypatch):
    events = []
    cleanup_error = OSError("cleanup failed")

    def remove(*args):
        raise cleanup_error

    monkeypatch.setattr("miles.ray.train_batch_coordinator.remove_rollout_data_refs", remove)
    actor = _FakeGroup("actor", events, train_error=ValueError("optimizer failed"))
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=False),
        actor_model=actor,
        critic_model=None,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor")

    with pytest.raises(ValueError, match="optimizer failed") as raised:
        await coordinator.train(7, {"data_ref": object(), "trainer_admission": publication})

    assert isinstance(raised.value.__cause__, OSError)
    assert ("manager", "rollback") not in events
    assert ("actor", "discard") in [event[:2] for event in events]


@pytest.mark.asyncio
async def test_postcommit_critic_failure_discards_unstarted_actor_pin(cleanup_refs):
    events = []
    critic = _FakeGroup("critic", events, train_error=ValueError("critic failed"))
    actor = _FakeGroup("actor", events)
    adapter = _FakeAdmissionAdapter(events)
    coordinator = TrainBatchCoordinator(
        args=_args(use_critic=True),
        actor_model=actor,
        critic_model=critic,
        rollout_manager=None,
        admission_adapter=adapter,
    )
    publication = _publication("actor", "critic")
    pack = {"data_ref": object(), "trainer_admission": publication}

    with pytest.raises(ValueError, match="critic failed"):
        await coordinator.train(7, pack)

    assert ("manager", "rollback") not in events
    assert [event[:2] for event in events[-2:]] == [
        ("critic", "discard"),
        ("actor", "discard"),
    ]
    assert cleanup_refs == [pack]
