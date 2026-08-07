from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NewType

from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId

ReservationStageId = NewType("ReservationStageId", str)
ReservationReceiptId = NewType("ReservationReceiptId", int)


class ReservationAcquisitionRollbackError(RuntimeError):
    """Report a retained source rollback after invalid acquisition.

    Args:
        validation_error: Error that made the acquired batch invalid.
        rollback_error: Source error that prevented compensating requeue.

    Attributes:
        validation_error: Error that made the acquired batch invalid.
        rollback_error: Source error that prevented compensating requeue.
    """

    def __init__(
        self,
        validation_error: ValueError | RuntimeError,
        rollback_error: BaseException,
    ) -> None:
        super().__init__(f"Failed to requeue reservations after invalid acquisition: {rollback_error}")
        self.validation_error = validation_error
        self.rollback_error = rollback_error


class ReservationIdentityConflictError(RuntimeError):
    """Report quarantined source attempts with ambiguous identities."""


class _ReservationState(Enum):
    RESERVED = auto()
    EXECUTING = auto()
    TERMINAL = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class _ReservationRecord:
    reservation: SourceReservation
    state: _ReservationState
    stage_id: ReservationStageId | None = None
    executor_receipt: ReservationExecutorReceipt | None = None
    terminal_receipt: ReservationTerminalReceipt | None = None


@dataclass(frozen=True, eq=False)
class ReservationExecutorReceipt:
    """Identify one exact reservation execution attempt.

    Attributes:
        receipt_id: Owner-local identity for this execution attempt.
        reservation_id: Stable source identity for the prompt group.
        stage_id: Execution stage that submitted the attempt.
    """

    receipt_id: ReservationReceiptId
    reservation_id: SourceReservationId
    stage_id: ReservationStageId
    _owner_token: object = field(repr=False)
    _record: _ReservationRecord = field(repr=False)


@dataclass(frozen=True, eq=False)
class ReservationTerminalReceipt:
    """Identify one exact terminal execution result.

    Attributes:
        executor_receipt: Exact execution attempt that became terminal.
    """

    executor_receipt: ReservationExecutorReceipt


class ReservationOwnership:
    """Track exact source reservations through terminal execution.

    Args:
        data_source: Source that performs durable reservation settlement.
    """

    def __init__(self, data_source: DataSource) -> None:
        self._data_source = data_source
        self._lock = threading.RLock()
        self._owner_token = object()
        self._next_receipt_id = 0
        self._records: dict[SourceReservationId, _ReservationRecord] = {}
        self._pending_acquisition_rollback: list[SourceReservation] | None = None

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        """Reserve and register exact source prompt groups.

        Args:
            num_groups: Number of prompt groups to reserve.

        Returns:
            Exact source reservation attempts in source order.
        """
        with self._lock:
            if self._pending_acquisition_rollback is not None:
                raise RuntimeError("Cannot reserve samples while an acquisition rollback is pending.")
            reservations = self._data_source.reserve_samples(num_groups)
            reservation_ids = [reservation.reservation_id for reservation in reservations]
            first_attempt_by_id: dict[SourceReservationId, SourceReservation] = {}
            ambiguous_ids: set[SourceReservationId] = set()
            for reservation in reservations:
                first_attempt = first_attempt_by_id.setdefault(reservation.reservation_id, reservation)
                active_record = self._records.get(reservation.reservation_id)
                if first_attempt is not reservation or (
                    active_record is not None and active_record.reservation is not reservation
                ):
                    ambiguous_ids.add(reservation.reservation_id)
            if ambiguous_ids:
                self._pending_acquisition_rollback = self._new_acquisition_attempts(reservations)
                raise ReservationIdentityConflictError(
                    "Data source returned distinct reservation attempts with conflicting identities: "
                    f"{sorted(ambiguous_ids)}."
                )

            validation_error: ValueError | RuntimeError | None = None
            if len(reservations) != num_groups:
                validation_error = RuntimeError(
                    f"Data source returned {len(reservations)} reservations for a request of {num_groups}."
                )
            elif len(reservation_ids) != len(set(reservation_ids)):
                validation_error = ValueError(
                    f"Data source returned duplicate reservation identities: {reservation_ids}."
                )
            conflicts = [reservation_id for reservation_id in reservation_ids if reservation_id in self._records]
            if validation_error is None and conflicts:
                validation_error = RuntimeError(f"Source reservations are already owned: {conflicts}.")
            if validation_error is not None:
                rollback_reservations = self._new_acquisition_attempts(reservations)
                if rollback_reservations:
                    self._pending_acquisition_rollback = rollback_reservations
                    try:
                        self._data_source.requeue_reservations(rollback_reservations)
                    except BaseException as rollback_error:
                        if not isinstance(rollback_error, Exception):
                            raise
                        raise ReservationAcquisitionRollbackError(
                            validation_error=validation_error,
                            rollback_error=rollback_error,
                        ) from rollback_error
                    self._pending_acquisition_rollback = None
                raise validation_error
            self._records.update(
                {
                    reservation.reservation_id: _ReservationRecord(
                        reservation=reservation,
                        state=_ReservationState.RESERVED,
                    )
                    for reservation in reservations
                }
            )
            return reservations

    def _new_acquisition_attempts(
        self,
        reservations: Sequence[SourceReservation],
    ) -> list[SourceReservation]:
        attempts: list[SourceReservation] = []
        for reservation in reservations:
            active_record = self._records.get(reservation.reservation_id)
            if active_record is not None and active_record.reservation is reservation:
                continue
            if any(attempt is reservation for attempt in attempts):
                continue
            attempts.append(reservation)
        return attempts

    def retry_failed_acquisition_rollback(self) -> None:
        """Retry a retained acquisition requeue.

        Raises:
            RuntimeError: If no acquisition rollback is pending.
        """
        with self._lock:
            if self._pending_acquisition_rollback is None:
                raise RuntimeError("No failed acquisition rollback is pending.")
            self._data_source.requeue_reservations(self._pending_acquisition_rollback)
            self._pending_acquisition_rollback = None

    @property
    def has_pending_acquisition_rollback(self) -> bool:
        """Return whether acquisition recovery retains exact reservations."""
        with self._lock:
            return self._pending_acquisition_rollback is not None

    def begin_execution(
        self,
        reservations: Sequence[SourceReservation],
        *,
        stage_id: ReservationStageId,
    ) -> list[ReservationExecutorReceipt]:
        """Transfer exact reserved attempts into executor ownership.

        Args:
            reservations: Exact reserved attempts to execute.
            stage_id: Nonempty execution-stage identity.

        Returns:
            Identity-sensitive receipts for terminal executor results.
        """
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError(f"stage_id must be a nonempty string, got {stage_id!r}.")
        attempts = list(reservations)
        with self._lock:
            records = self._require_reservations(attempts, expected_state=_ReservationState.RESERVED)
            receipts = [
                ReservationExecutorReceipt(
                    receipt_id=ReservationReceiptId(self._next_receipt_id + offset),
                    reservation_id=record.reservation.reservation_id,
                    stage_id=stage_id,
                    _owner_token=self._owner_token,
                    _record=record,
                )
                for offset, record in enumerate(records)
            ]
            self._next_receipt_id += len(receipts)
            for record, receipt in zip(records, receipts, strict=True):
                record.state = _ReservationState.EXECUTING
                record.stage_id = stage_id
                record.executor_receipt = receipt
            return receipts

    def rollback_reserved(self, reservations: Sequence[SourceReservation]) -> None:
        """Requeue exact attempts that did not enter execution.

        Args:
            reservations: Exact reserved attempts to return.
        """
        attempts = list(reservations)
        with self._lock:
            records = self._require_reservations(attempts, expected_state=_ReservationState.RESERVED)
            self._data_source.requeue_reservations([record.reservation for record in records])
            for record in records:
                record.state = _ReservationState.ROLLED_BACK
                del self._records[record.reservation.reservation_id]

    def record_terminal(
        self,
        receipts: Sequence[ReservationExecutorReceipt],
        *,
        stage_id: ReservationStageId,
    ) -> list[ReservationTerminalReceipt]:
        """Record exact attempts whose executor work is terminal.

        Args:
            receipts: Exact executor receipts that reached terminal state.
            stage_id: Stage that owns every receipt.

        Returns:
            Exact trainable terminal receipts in input order.
        """
        attempts = list(receipts)
        with self._lock:
            self._validate_unique_executor_receipts(attempts)
            records = [
                self._require_executor_receipt(
                    receipt,
                    stage_id=stage_id,
                )
                for receipt in attempts
            ]
            terminal_receipts = [ReservationTerminalReceipt(executor_receipt=receipt) for receipt in attempts]
            for record, terminal_receipt in zip(records, terminal_receipts, strict=True):
                record.state = _ReservationState.TERMINAL
                record.terminal_receipt = terminal_receipt
            return terminal_receipts

    def commit_batch(
        self,
        receipts: Sequence[ReservationTerminalReceipt],
        *,
        rollout_id: int,
    ) -> None:
        """Acknowledge exact terminal reservations after train handoff.

        Args:
            receipts: Exact trainable terminal receipts in the batch.
            rollout_id: Training rollout that accepted the batch.
        """
        terminal_receipts = list(receipts)
        with self._lock:
            records = self._require_terminal_receipts(terminal_receipts, operation="commit")
            self._data_source.acknowledge_reservations(
                [record.reservation for record in records],
                rollout_id=rollout_id,
            )
            for record in records:
                record.state = _ReservationState.COMMITTED
                del self._records[record.reservation.reservation_id]

    def rollback_batch(self, receipts: Sequence[ReservationTerminalReceipt]) -> None:
        """Requeue exact terminal reservations for pristine replay.

        Args:
            receipts: Exact trainable terminal receipts to return.
        """
        terminal_receipts = list(receipts)
        with self._lock:
            records = self._require_terminal_receipts(terminal_receipts, operation="roll back")
            self._data_source.requeue_reservations([record.reservation for record in records])
            for record in records:
                record.state = _ReservationState.ROLLED_BACK
                del self._records[record.reservation.reservation_id]

    def _require_reservations(
        self,
        reservations: list[SourceReservation],
        *,
        expected_state: _ReservationState,
    ) -> list[_ReservationRecord]:
        reservation_ids = [reservation.reservation_id for reservation in reservations]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError(f"Reservation batch contains duplicate identities: {reservation_ids}.")

        records: list[_ReservationRecord] = []
        for reservation in reservations:
            record = self._records.get(reservation.reservation_id)
            if record is None or record.reservation is not reservation or record.state is not expected_state:
                raise RuntimeError(
                    f"Source reservation {reservation.reservation_id} is not an exact "
                    f"{expected_state.name.lower()} attempt owned here."
                )
            records.append(record)
        return records

    def _require_executor_receipt(
        self,
        receipt: ReservationExecutorReceipt,
        *,
        stage_id: ReservationStageId,
    ) -> _ReservationRecord:
        record = receipt._record
        if (
            receipt._owner_token is not self._owner_token
            or self._records.get(receipt.reservation_id) is not record
            or record.executor_receipt is not receipt
            or receipt.stage_id != stage_id
            or record.stage_id != stage_id
            or record.state is not _ReservationState.EXECUTING
        ):
            raise RuntimeError(
                f"Cannot record terminal receipt {receipt.receipt_id}; receipt is not owned by stage {stage_id!r}."
            )
        return record

    @staticmethod
    def _validate_unique_executor_receipts(receipts: list[ReservationExecutorReceipt]) -> None:
        receipt_ids = [receipt.receipt_id for receipt in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError(f"Executor receipt batch contains duplicate identities: {receipt_ids}.")

    def _require_terminal_receipts(
        self,
        receipts: list[ReservationTerminalReceipt],
        *,
        operation: str,
    ) -> list[_ReservationRecord]:
        receipt_ids = [receipt.executor_receipt.receipt_id for receipt in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError(f"Terminal receipt batch contains duplicate identities: {receipt_ids}.")

        records: list[_ReservationRecord] = []
        for receipt in receipts:
            executor_receipt = receipt.executor_receipt
            record = executor_receipt._record
            if (
                executor_receipt._owner_token is not self._owner_token
                or self._records.get(executor_receipt.reservation_id) is not record
                or record.executor_receipt is not executor_receipt
                or record.terminal_receipt is not receipt
                or record.state is not _ReservationState.TERMINAL
            ):
                raise RuntimeError(
                    f"Cannot {operation} terminal receipt {executor_receipt.receipt_id}; "
                    "receipt is not exact terminal ownership."
                )
            records.append(record)
        return records
