from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from collections.abc import Callable

import pytest

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
        on_commit: Callable[[], None],
        on_rollback: Callable[[TrainBatchRollbackReason], None],
    ) -> None:
        super().__init__(rollout_id=rollout_id)
        self._on_commit = on_commit
        self._on_rollback = on_rollback

    def _commit(self) -> None:
        self._on_commit()

    def _rollback(self, reason: TrainBatchRollbackReason) -> None:
        self._on_rollback(reason)


def test_leased_output_preserves_the_ordinary_train_output_contract() -> None:
    lease = RecordingTrainBatchLease(
        rollout_id=3,
        on_commit=lambda: None,
        on_rollback=lambda reason: None,
    )

    output = LeasedRolloutFnTrainOutput(samples=[], metrics={"source": "test"}, lease=lease)

    assert isinstance(output, RolloutFnTrainOutput)
    assert output == LeasedRolloutFnTrainOutput(samples=[], metrics={"source": "test"}, lease=lease)


def test_train_batch_lease_commits_once() -> None:
    events: list[str] = []
    lease = RecordingTrainBatchLease(
        rollout_id=7,
        on_commit=lambda: events.append("commit"),
        on_rollback=lambda reason: events.append(f"rollback:{reason.name}"),
    )

    assert lease.rollout_id == 7
    lease.commit()

    assert events == ["commit"]
    with pytest.raises(RuntimeError) as repeated_commit:
        lease.commit()
    assert str(repeated_commit.value) == "Train batch lease for rollout 7 already has a settlement attempt."
    with pytest.raises(RuntimeError) as rollback_after_commit:
        lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert str(rollback_after_commit.value) == "Train batch lease for rollout 7 already has a settlement attempt."


def test_train_batch_lease_rolls_back_once() -> None:
    reasons: list[TrainBatchRollbackReason] = []
    lease = RecordingTrainBatchLease(
        rollout_id=11,
        on_commit=lambda: None,
        on_rollback=reasons.append,
    )

    lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)

    assert reasons == [TrainBatchRollbackReason.HANDOFF_FAILED]
    with pytest.raises(RuntimeError) as repeated_rollback:
        lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert str(repeated_rollback.value) == "Train batch lease for rollout 11 already has a settlement attempt."
    with pytest.raises(RuntimeError) as commit_after_rollback:
        lease.commit()
    assert str(commit_after_rollback.value) == "Train batch lease for rollout 11 already has a settlement attempt."


def test_failed_commit_still_claims_the_only_settlement_attempt() -> None:
    failure = RuntimeError("commit failed")

    def fail_commit() -> None:
        raise failure

    lease = RecordingTrainBatchLease(
        rollout_id=13,
        on_commit=fail_commit,
        on_rollback=lambda reason: None,
    )

    with pytest.raises(RuntimeError) as error:
        lease.commit()
    assert error.value is failure
    with pytest.raises(RuntimeError) as rollback_after_failed_commit:
        lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert (
        str(rollback_after_failed_commit.value) == "Train batch lease for rollout 13 already has a settlement attempt."
    )


def test_failed_rollback_still_claims_the_only_settlement_attempt() -> None:
    failure = RuntimeError("rollback failed")

    def fail_rollback(reason: TrainBatchRollbackReason) -> None:
        raise failure

    lease = RecordingTrainBatchLease(
        rollout_id=17,
        on_commit=lambda: None,
        on_rollback=fail_rollback,
    )

    with pytest.raises(RuntimeError) as error:
        lease.rollback(TrainBatchRollbackReason.HANDOFF_FAILED)
    assert error.value is failure
    with pytest.raises(RuntimeError) as commit_after_failed_rollback:
        lease.commit()
    assert (
        str(commit_after_failed_rollback.value) == "Train batch lease for rollout 17 already has a settlement attempt."
    )
