from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

from collections.abc import Sequence
from dataclasses import replace

import pytest

from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId
from miles.rollout.fully_async.ownership import (
    ReservationAcquisitionRollbackError,
    ReservationIdentityConflictError,
    ReservationOwnership,
    ReservationStageId,
    ReservationTerminalReceipt,
)
from miles.utils.types import Sample


class RecordingDataSource(DataSource):
    def __init__(self, reservations: list[SourceReservation]) -> None:
        self.reservations = list(reservations)
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        raise AssertionError("ownership must reserve source groups")

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise AssertionError("ownership must settle source reservations")

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        reservations = self.reservations[:num_groups]
        del self.reservations[:num_groups]
        return reservations

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        self.acknowledged.append((list(reservations), rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        self.requeued.append(list(reservations))


class FailingSettlementDataSource(RecordingDataSource):
    def __init__(
        self,
        reservations: list[SourceReservation],
        *,
        acknowledge_error: BaseException | None,
        requeue_error: BaseException | None,
    ) -> None:
        super().__init__(reservations)
        self.acknowledge_error = acknowledge_error
        self.requeue_error = requeue_error

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        if self.acknowledge_error is not None:
            raise self.acknowledge_error
        super().acknowledge_reservations(reservations, rollout_id=rollout_id)

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        if self.requeue_error is not None:
            raise self.requeue_error
        super().requeue_reservations(reservations)


def make_reservation(reservation_id: int) -> SourceReservation:
    return SourceReservation(
        reservation_id=SourceReservationId(str(reservation_id)),
        samples=(
            Sample(group_index=reservation_id, index=reservation_id * 2),
            Sample(group_index=reservation_id, index=reservation_id * 2 + 1),
        ),
    )


def test_short_source_acquisition_requeues_every_returned_attempt() -> None:
    source_reservation = make_reservation(0)
    data_source = RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)

    with pytest.raises(RuntimeError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Data source returned 1 reservations for a request of 2."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_duplicate_source_acquisition_requeues_one_exact_attempt() -> None:
    source_reservation = make_reservation(0)
    data_source = RecordingDataSource([source_reservation, source_reservation])
    ownership = ReservationOwnership(data_source)

    with pytest.raises(ValueError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Data source returned duplicate reservation identities: ['0', '0']."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_failed_short_acquisition_rollback_blocks_acquisition_until_retry_succeeds() -> None:
    source_reservation = make_reservation(13)
    requeue_error = RuntimeError("acquisition requeue failed")
    data_source = FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)

    assert not ownership.has_pending_acquisition_rollback

    with pytest.raises(ReservationAcquisitionRollbackError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Failed to requeue reservations after invalid acquisition: acquisition requeue failed"
    assert str(error.value.validation_error) == "Data source returned 1 reservations for a request of 2."
    assert error.value.rollback_error is requeue_error
    assert error.value.__cause__ is requeue_error
    assert ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(RuntimeError) as blocked_error:
        ownership.reserve_samples(1)

    assert str(blocked_error.value) == "Cannot reserve samples while an acquisition rollback is pending."
    assert ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.requeue_error = None
    ownership.retry_failed_acquisition_rollback()

    assert not ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_distinct_duplicate_id_attempts_are_quarantined_and_all_recovered() -> None:
    first_duplicate_attempt = make_reservation(14)
    second_duplicate_attempt = make_reservation(14)
    unique_attempt = make_reservation(15)
    returned_attempts = [first_duplicate_attempt, second_duplicate_attempt, unique_attempt]
    data_source = RecordingDataSource(returned_attempts)
    ownership = ReservationOwnership(data_source)

    with pytest.raises(ReservationIdentityConflictError):
        ownership.reserve_samples(3)

    assert ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.retry_failed_acquisition_rollback()

    assert not ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == [returned_attempts]
    assert all(
        recovered_attempt is returned_attempt
        for recovered_attempt, returned_attempt in zip(data_source.requeued[0], returned_attempts, strict=True)
    )


def test_returned_id_conflict_quarantines_new_attempts_without_losing_active_attempt() -> None:
    active_attempt = make_reservation(16)
    data_source = RecordingDataSource([active_attempt])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    conflicting_attempt = make_reservation(16)
    unrelated_attempt = make_reservation(17)
    returned_attempts = [conflicting_attempt, unrelated_attempt]
    data_source.reservations.extend(returned_attempts)

    with pytest.raises(ReservationIdentityConflictError):
        ownership.reserve_samples(2)

    assert ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.retry_failed_acquisition_rollback()
    stage_id = ReservationStageId("execution-active-conflict")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)

    assert reservation is active_attempt
    assert executor_receipt.reservation_id == active_attempt.reservation_id
    assert not ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == [returned_attempts]
    assert all(
        recovered_attempt is returned_attempt
        for recovered_attempt, returned_attempt in zip(data_source.requeued[0], returned_attempts, strict=True)
    )


def test_mixed_active_and_conflicting_acquisition_recovers_only_new_attempts() -> None:
    active_attempt = make_reservation(18)
    data_source = RecordingDataSource([active_attempt])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    conflicting_attempt = make_reservation(18)
    unrelated_attempt = make_reservation(19)
    newly_acquired_attempts = [conflicting_attempt, unrelated_attempt]
    data_source.reservations.extend([active_attempt, *newly_acquired_attempts])

    with pytest.raises(ReservationIdentityConflictError) as error:
        ownership.reserve_samples(3)

    assert str(error.value) == (
        "Data source returned distinct reservation attempts with conflicting identities: ['18']."
    )
    assert ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.retry_failed_acquisition_rollback()
    stage_id = ReservationStageId("execution-mixed-active-conflict")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)

    assert reservation is active_attempt
    assert executor_receipt.reservation_id == active_attempt.reservation_id
    assert not ownership.has_pending_acquisition_rollback
    assert data_source.acknowledged == []
    assert data_source.requeued == [newly_acquired_attempts]
    assert all(
        recovered_attempt is newly_acquired_attempt
        for recovered_attempt, newly_acquired_attempt in zip(
            data_source.requeued[0],
            newly_acquired_attempts,
            strict=True,
        )
    )
    assert all(recovered_attempt is not active_attempt for recovered_attempt in data_source.requeued[0])


def test_only_exact_receipts_can_commit_terminal_execution() -> None:
    source_reservation = make_reservation(1)
    data_source = RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("execution-1")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    copied_receipt = replace(executor_receipt)
    foreign_ownership = ReservationOwnership(RecordingDataSource([]))

    with pytest.raises(RuntimeError) as copied_error:
        ownership.record_terminal([copied_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as foreign_error:
        foreign_ownership.record_terminal([executor_receipt], stage_id=stage_id)

    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    assert str(copied_error.value) == (
        "Cannot record terminal receipt 0; receipt is not owned by stage 'execution-1'."
    )
    assert str(foreign_error.value) == (
        "Cannot record terminal receipt 0; receipt is not owned by stage 'execution-1'."
    )
    assert terminal_receipt.executor_receipt is executor_receipt
    assert terminal_receipt != ReservationTerminalReceipt(executor_receipt=executor_receipt)
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.commit_batch([terminal_receipt], rollout_id=24)

    assert data_source.acknowledged == [([source_reservation], 24)]
    assert data_source.requeued == []
    with pytest.raises(RuntimeError) as repeated_commit:
        ownership.commit_batch([terminal_receipt], rollout_id=24)
    assert str(repeated_commit.value) == "Cannot commit terminal receipt 0; receipt is not exact terminal ownership."


def test_only_exact_terminal_receipts_can_rollback_execution() -> None:
    source_reservation = make_reservation(2)
    data_source = RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("execution-2")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    copied_receipt = ReservationTerminalReceipt(executor_receipt=terminal_receipt.executor_receipt)

    assert copied_receipt != terminal_receipt
    with pytest.raises(RuntimeError) as copied_error:
        ownership.rollback_batch([copied_receipt])

    assert str(copied_error.value) == ("Cannot roll back terminal receipt 0; receipt is not exact terminal ownership.")
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.rollback_batch([terminal_receipt])

    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]
    with pytest.raises(RuntimeError) as repeated_rollback:
        ownership.rollback_batch([terminal_receipt])
    assert str(repeated_rollback.value) == (
        "Cannot roll back terminal receipt 0; receipt is not exact terminal ownership."
    )


def test_only_exact_reserved_attempt_can_rollback_before_execution() -> None:
    source_reservation = make_reservation(3)
    data_source = RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    copied_reservation = SourceReservation(
        reservation_id=reservation.reservation_id,
        samples=reservation.samples,
    )

    with pytest.raises(RuntimeError) as copied_error:
        ownership.rollback_reserved([copied_reservation])

    assert str(copied_error.value) == "Source reservation 3 is not an exact reserved attempt owned here."
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.rollback_reserved([reservation])

    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_invalid_terminal_receipt_prevents_partial_batch_commit() -> None:
    source_reservations = [make_reservation(4), make_reservation(5)]
    data_source = RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("execution-batch")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)
    copied_second = ReservationTerminalReceipt(
        executor_receipt=terminal_receipts[1].executor_receipt,
    )

    with pytest.raises(RuntimeError) as copied_error:
        ownership.commit_batch([terminal_receipts[0], copied_second], rollout_id=25)

    assert str(copied_error.value) == "Cannot commit terminal receipt 1; receipt is not exact terminal ownership."
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.commit_batch(terminal_receipts, rollout_id=25)

    assert data_source.acknowledged == [(source_reservations, 25)]
    assert data_source.requeued == []


def test_duplicate_terminal_receipt_prevents_partial_batch_rollback() -> None:
    source_reservations = [make_reservation(6), make_reservation(7)]
    data_source = RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("execution-duplicate")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)

    with pytest.raises(ValueError) as duplicate_error:
        ownership.rollback_batch([terminal_receipts[0], terminal_receipts[0]])

    assert str(duplicate_error.value) == "Terminal receipt batch contains duplicate identities: [0, 0]."
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.rollback_batch(terminal_receipts)

    assert data_source.acknowledged == []
    assert data_source.requeued == [source_reservations]


def test_failed_source_acknowledgement_keeps_terminal_batch_retryable() -> None:
    source_reservations = [make_reservation(8), make_reservation(9)]
    acknowledgement_error = RuntimeError("acknowledgement failed")
    data_source = FailingSettlementDataSource(
        source_reservations,
        acknowledge_error=acknowledgement_error,
        requeue_error=None,
    )
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("execution-acknowledgement")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.commit_batch(terminal_receipts, rollout_id=26)

    assert error.value is acknowledgement_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.acknowledge_error = None
    ownership.commit_batch(terminal_receipts, rollout_id=26)

    assert data_source.acknowledged == [(source_reservations, 26)]
    assert data_source.requeued == []


def test_failed_source_requeue_keeps_terminal_batch_retryable() -> None:
    source_reservations = [make_reservation(10), make_reservation(11)]
    requeue_error = RuntimeError("requeue failed")
    data_source = FailingSettlementDataSource(
        source_reservations,
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("execution-requeue")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.rollback_batch(terminal_receipts)

    assert error.value is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.requeue_error = None
    ownership.rollback_batch(terminal_receipts)

    assert data_source.acknowledged == []
    assert data_source.requeued == [source_reservations]


def test_stale_receipt_cannot_settle_reissued_source_identity() -> None:
    first_attempt = make_reservation(12)
    data_source = RecordingDataSource([first_attempt])
    ownership = ReservationOwnership(data_source)
    [first_reservation] = ownership.reserve_samples(1)
    first_stage = ReservationStageId("execution-old")
    [first_executor_receipt] = ownership.begin_execution([first_reservation], stage_id=first_stage)
    [first_terminal_receipt] = ownership.record_terminal([first_executor_receipt], stage_id=first_stage)
    ownership.rollback_batch([first_terminal_receipt])

    second_attempt = make_reservation(12)
    data_source.reservations.append(second_attempt)
    [second_reservation] = ownership.reserve_samples(1)
    second_stage = ReservationStageId("execution-new")
    [second_executor_receipt] = ownership.begin_execution([second_reservation], stage_id=second_stage)
    [second_terminal_receipt] = ownership.record_terminal([second_executor_receipt], stage_id=second_stage)

    with pytest.raises(RuntimeError) as stale_error:
        ownership.commit_batch([first_terminal_receipt], rollout_id=27)

    assert str(stale_error.value) == "Cannot commit terminal receipt 0; receipt is not exact terminal ownership."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[first_attempt]]

    ownership.commit_batch([second_terminal_receipt], rollout_id=27)

    assert first_reservation is first_attempt
    assert second_reservation is second_attempt
    assert data_source.acknowledged == [([second_attempt], 27)]
    assert data_source.requeued == [[first_attempt]]
