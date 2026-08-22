from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import heapq
import logging
import time

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)

FINISHED_TTL_SECONDS = 12 * 3600.0

MAX_ACTIVE_CALLS = 4096
MAX_UNACKNOWLEDGED_OUTCOME_BYTES = 256 * 1024 * 1024
MAX_QUEUED_REQUEST_BYTES = 64 * 1024 * 1024
MAX_CONTROL_CALLS = 64
MAX_CONTROL_OUTCOME_BYTES = 4 * 1024 * 1024
MAX_CONTROL_QUEUED_REQUEST_BYTES = 1024 * 1024
MAX_TOMBSTONES = 65536
MAX_CONTROL_TOMBSTONES = 65536
MAX_CALL_ID_BYTES = 128
EXPIRY_BATCH_SIZE = 256


class DuplicateCallError(Exception):
    pass


class AcknowledgedCallError(Exception):
    pass


class CallNotFinishedError(Exception):
    pass


class CallStoreCapacityError(Exception):
    pass


class CallIdTooLongError(Exception):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class CallStoreStats:
    active_calls: int
    control_calls: int
    queued_request_bytes: int
    control_queued_request_bytes: int
    reserved_outcome_bytes: int
    control_reserved_outcome_bytes: int
    unacknowledged_outcome_bytes: int
    tombstones: int


class CallStore:
    def __init__(
        self,
        *,
        finished_ttl_seconds: float = FINISHED_TTL_SECONDS,
        max_active_calls: int = MAX_ACTIVE_CALLS,
        max_unacknowledged_outcome_bytes: int = MAX_UNACKNOWLEDGED_OUTCOME_BYTES,
        max_queued_request_bytes: int = MAX_QUEUED_REQUEST_BYTES,
        max_control_calls: int = MAX_CONTROL_CALLS,
        max_control_outcome_bytes: int = MAX_CONTROL_OUTCOME_BYTES,
        max_control_queued_request_bytes: int = MAX_CONTROL_QUEUED_REQUEST_BYTES,
        max_tombstones: int = MAX_TOMBSTONES,
        max_control_tombstones: int = MAX_CONTROL_TOMBSTONES,
        expiry_batch_size: int = EXPIRY_BATCH_SIZE,
    ) -> None:
        self._finished_ttl_seconds = finished_ttl_seconds
        self._max_active_calls = max_active_calls
        self._max_unacknowledged_outcome_bytes = max_unacknowledged_outcome_bytes
        self._max_queued_request_bytes = max_queued_request_bytes
        self._max_control_calls = max_control_calls
        self._max_control_outcome_bytes = max_control_outcome_bytes
        self._max_control_queued_request_bytes = max_control_queued_request_bytes
        self._max_tombstones = max_tombstones
        self._max_control_tombstones = max_control_tombstones
        self._expiry_batch_size = expiry_batch_size
        self._records: dict[str, _CallRecord] = {}
        self._tombstones: dict[str, _CallTombstone] = {}
        self._expiry_heap: list[tuple[float, str]] = []
        self._active_calls = 0
        self._control_calls = 0
        self._queued_request_bytes = 0
        self._reserved_outcome_bytes = 0
        self._unacknowledged_outcome_bytes = 0
        self._control_queued_request_bytes = 0
        self._control_reserved_outcome_bytes = 0
        self._control_unacknowledged_outcome_bytes = 0
        self._data_tombstones = 0
        self._control_tombstones = 0

    @property
    def stats(self) -> CallStoreStats:
        return CallStoreStats(
            active_calls=self._active_calls,
            control_calls=self._control_calls,
            queued_request_bytes=self._queued_request_bytes,
            control_queued_request_bytes=self._control_queued_request_bytes,
            reserved_outcome_bytes=self._reserved_outcome_bytes,
            control_reserved_outcome_bytes=self._control_reserved_outcome_bytes,
            unacknowledged_outcome_bytes=self._unacknowledged_outcome_bytes,
            tombstones=len(self._tombstones),
        )

    def begin(
        self,
        *,
        call_id: str,
        fingerprint: bytes,
        request_reservation_bytes: int = 0,
        outcome_reservation_bytes: int | None = None,
        control_plane: bool = False,
    ) -> None:
        try:
            call_id_bytes = len(call_id.encode())
        except UnicodeEncodeError as e:
            raise CallIdTooLongError("call id must be valid UTF-8") from e
        if call_id_bytes > MAX_CALL_ID_BYTES:
            raise CallIdTooLongError(f"call id exceeds {MAX_CALL_ID_BYTES} UTF-8 bytes")
        self._expire_due(now=time.monotonic())

        if (record := self._records.get(call_id)) is not None:
            self._validate_fingerprint(call_id=call_id, expected=record.fingerprint, actual=fingerprint)
            raise DuplicateCallError(f"call {call_id} already submitted")
        if (tombstone := self._tombstones.get(call_id)) is not None:
            self._validate_fingerprint(call_id=call_id, expected=tombstone.fingerprint, actual=fingerprint)
            raise DuplicateCallError(f"call {call_id} outcome was already acknowledged")

        identity_count = (
            self._control_calls + self._control_tombstones
            if control_plane
            else self._active_calls + self._data_tombstones
        )
        identity_limit = self._max_control_tombstones if control_plane else self._max_tombstones
        admission_class = "control" if control_plane else "active"
        if identity_count >= identity_limit:
            raise CallStoreCapacityError(f"{admission_class} tombstone capacity {identity_limit} is full")
        active_calls = self._control_calls if control_plane else self._active_calls
        active_limit = self._max_control_calls if control_plane else self._max_active_calls
        if active_calls >= active_limit:
            raise CallStoreCapacityError(f"{admission_class} call capacity {active_limit} is full")

        queued_request_bytes = self._control_queued_request_bytes if control_plane else self._queued_request_bytes
        max_queued_request_bytes = (
            self._max_control_queued_request_bytes if control_plane else self._max_queued_request_bytes
        )
        if queued_request_bytes + request_reservation_bytes > max_queued_request_bytes:
            raise CallStoreCapacityError(
                f"{admission_class} request capacity {max_queued_request_bytes} bytes is full"
            )

        reserved_outcome_bytes = (
            self._control_reserved_outcome_bytes if control_plane else self._reserved_outcome_bytes
        )
        max_outcome_bytes = (
            self._max_control_outcome_bytes if control_plane else self._max_unacknowledged_outcome_bytes
        )
        if reserved_outcome_bytes + (outcome_reservation_bytes or 0) > max_outcome_bytes:
            raise CallStoreCapacityError(
                f"{admission_class} outcome retention capacity {max_outcome_bytes} bytes is full"
            )

        self._records[call_id] = _CallRecord(
            fingerprint=fingerprint,
            finished_event=asyncio.Event(),
            request_reservation_bytes=request_reservation_bytes,
            outcome_reservation_bytes=outcome_reservation_bytes,
            control_plane=control_plane,
        )
        self._adjust_call_count(control_plane=control_plane, delta=1)
        self._adjust_reservations(
            control_plane=control_plane,
            request_bytes=request_reservation_bytes,
            outcome_bytes=outcome_reservation_bytes or 0,
        )
        log_structured(
            logger.debug, tag="rpc", op="call_store", phase="accept", call=call_id, tracked=len(self._records)
        )

    def finish(self, *, call_id: str, outcome: CallStatusResponse) -> None:
        record = self._records[call_id]
        if record.outcome is not None:
            raise RuntimeError(f"call {call_id} finished twice")

        outcome_bytes = len(outcome.model_dump_json().encode())
        if record.outcome_reservation_bytes is not None and outcome_bytes > record.outcome_reservation_bytes:
            raise RuntimeError(
                f"call {call_id} serialized outcome is {outcome_bytes} bytes, "
                f"above its {record.outcome_reservation_bytes}-byte reservation"
            )

        record.outcome = outcome
        record.outcome_bytes = outcome_bytes
        record.expires_at = time.monotonic() + self._finished_ttl_seconds
        self._adjust_reservations(
            control_plane=record.control_plane,
            request_bytes=-record.request_reservation_bytes,
            unacknowledged_bytes=outcome_bytes,
        )
        record.request_reservation_bytes = 0
        record.finished_event.set()
        heapq.heappush(self._expiry_heap, (record.expires_at, call_id))
        log_structured(logger.debug, tag="rpc", op="call_store", phase="finish", call=call_id, status=outcome.status)

    def rollback_admission(self, *, call_id: str, fingerprint: bytes) -> None:
        record = self._records[call_id]
        self._validate_fingerprint(call_id=call_id, expected=record.fingerprint, actual=fingerprint)
        if record.outcome is not None:
            raise RuntimeError(f"call {call_id} already finished")

        del self._records[call_id]
        self._adjust_call_count(control_plane=record.control_plane, delta=-1)
        self._adjust_reservations(
            control_plane=record.control_plane,
            request_bytes=-record.request_reservation_bytes,
            outcome_bytes=-(record.outcome_reservation_bytes or 0),
        )

    def acknowledge(self, *, call_id: str, fingerprint: bytes) -> None:
        if (record := self._records.get(call_id)) is None:
            tombstone = self._tombstones[call_id]
            self._validate_fingerprint(call_id=call_id, expected=tombstone.fingerprint, actual=fingerprint)
            return

        self._validate_fingerprint(call_id=call_id, expected=record.fingerprint, actual=fingerprint)
        if record.outcome is None:
            raise CallNotFinishedError(f"call {call_id} is still pending")

        del self._records[call_id]
        self._adjust_call_count(control_plane=record.control_plane, delta=-1)
        assert record.expires_at is not None
        self._tombstones[call_id] = _CallTombstone(
            fingerprint=record.fingerprint,
            control_plane=record.control_plane,
            expires_at=record.expires_at,
        )
        self._adjust_tombstone_count(control_plane=record.control_plane, delta=1)
        self._adjust_reservations(
            control_plane=record.control_plane,
            outcome_bytes=-(record.outcome_reservation_bytes or 0),
            unacknowledged_bytes=-record.outcome_bytes,
        )
        log_structured(logger.debug, tag="rpc", op="call_store", phase="ack", call=call_id)

    async def wait(self, *, call_id: str, timeout: float) -> CallStatusResponse | None:
        if call_id in self._tombstones:
            raise AcknowledgedCallError(f"call {call_id} outcome was already acknowledged")
        record = self._records[call_id]

        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(record.finished_event.wait(), timeout=timeout)

        return record.outcome

    def contains(self, call_id: str) -> bool:
        return call_id in self._records or call_id in self._tombstones

    def in_flight_call_ids(self) -> list[str]:
        return sorted(call_id for call_id, record in self._records.items() if record.outcome is None)

    def _validate_fingerprint(self, *, call_id: str, expected: bytes, actual: bytes) -> None:
        if expected != actual:
            raise DuplicateCallError(f"call {call_id} already belongs to another request")

    def _expire_due(self, *, now: float) -> None:
        purged = 0
        while self._expiry_heap and self._expiry_heap[0][0] < now and purged < self._expiry_batch_size:
            expires_at, call_id = heapq.heappop(self._expiry_heap)
            if (record := self._records.get(call_id)) is not None:
                if record.expires_at != expires_at:
                    continue
                del self._records[call_id]
                self._adjust_call_count(control_plane=record.control_plane, delta=-1)
                self._adjust_reservations(
                    control_plane=record.control_plane,
                    outcome_bytes=-(record.outcome_reservation_bytes or 0),
                    unacknowledged_bytes=-record.outcome_bytes,
                )
                purged += 1
            elif (tombstone := self._tombstones.get(call_id)) is not None and tombstone.expires_at == expires_at:
                del self._tombstones[call_id]
                self._adjust_tombstone_count(control_plane=tombstone.control_plane, delta=-1)
                purged += 1

        if purged:
            log_structured(logger.debug, tag="rpc", op="call_store", phase="purge", purged=purged)

    def _adjust_reservations(
        self,
        *,
        control_plane: bool,
        request_bytes: int = 0,
        outcome_bytes: int = 0,
        unacknowledged_bytes: int = 0,
    ) -> None:
        if control_plane:
            self._control_queued_request_bytes += request_bytes
            self._control_reserved_outcome_bytes += outcome_bytes
            self._control_unacknowledged_outcome_bytes += unacknowledged_bytes
        else:
            self._queued_request_bytes += request_bytes
            self._reserved_outcome_bytes += outcome_bytes
            self._unacknowledged_outcome_bytes += unacknowledged_bytes

    def _adjust_call_count(self, *, control_plane: bool, delta: int) -> None:
        if control_plane:
            self._control_calls += delta
        else:
            self._active_calls += delta

    def _adjust_tombstone_count(self, *, control_plane: bool, delta: int) -> None:
        if control_plane:
            self._control_tombstones += delta
        else:
            self._data_tombstones += delta


@dataclasses.dataclass(slots=True)
class _CallRecord:
    fingerprint: bytes
    finished_event: asyncio.Event
    request_reservation_bytes: int
    outcome_reservation_bytes: int | None
    control_plane: bool
    outcome: CallStatusResponse | None = None
    outcome_bytes: int = 0
    expires_at: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _CallTombstone:
    fingerprint: bytes
    control_plane: bool
    expires_at: float
