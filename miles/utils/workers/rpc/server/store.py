from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)

RETRIEVED_TTL_SECONDS = 300.0
FINISHED_TTL_SECONDS = 12 * 3600.0


class DuplicateCallError(Exception):
    pass


class CallStore:
    def __init__(
        self,
        *,
        retrieved_ttl_seconds: float = RETRIEVED_TTL_SECONDS,
        finished_ttl_seconds: float = FINISHED_TTL_SECONDS,
    ) -> None:
        self._retrieved_ttl_seconds = retrieved_ttl_seconds
        self._finished_ttl_seconds = finished_ttl_seconds
        self._records: dict[str, _CallRecord] = {}

    def begin(self, *, call_id: str) -> None:
        self._purge_expired()

        if call_id in self._records:
            raise DuplicateCallError(f"call {call_id} already submitted")

        self._records[call_id] = _CallRecord(finished_event=asyncio.Event())
        log_structured(
            logger.debug, tag="rpc", op="call_store", phase="accept", call=call_id, tracked=len(self._records)
        )

    def finish(self, *, call_id: str, outcome: CallStatusResponse) -> None:
        record = self._records[call_id]
        if record.outcome is not None:
            raise RuntimeError(f"call {call_id} finished twice")
        record.outcome = outcome
        record.finished_at = time.monotonic()
        record.finished_event.set()
        log_structured(logger.debug, tag="rpc", op="call_store", phase="finish", call=call_id, status=outcome.status)

    async def wait(self, *, call_id: str, timeout: float) -> CallStatusResponse | None:
        record = self._records[call_id]

        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(record.finished_event.wait(), timeout=timeout)

        if record.outcome is not None and record.first_retrieved_at is None:
            record.first_retrieved_at = time.monotonic()
        return record.outcome

    def contains(self, call_id: str) -> bool:
        return call_id in self._records

    def _purge_expired(self) -> None:
        now = time.monotonic()
        retained = {call_id: record for call_id, record in self._records.items() if self._is_live(record, now=now)}
        if len(retained) != len(self._records):
            log_structured(
                logger.debug, tag="rpc", op="call_store", phase="purge", purged=len(self._records) - len(retained)
            )
        self._records = retained

    def _is_live(self, record: _CallRecord, *, now: float) -> bool:
        if record.finished_at is None:
            return True
        if record.first_retrieved_at is not None:
            return now - record.first_retrieved_at <= self._retrieved_ttl_seconds
        return now - record.finished_at <= self._finished_ttl_seconds


@dataclasses.dataclass
class _CallRecord:
    finished_event: asyncio.Event
    outcome: CallStatusResponse | None = None
    finished_at: float | None = None
    first_retrieved_at: float | None = None
