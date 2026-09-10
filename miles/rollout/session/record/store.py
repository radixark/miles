# doc-dev: docs/developer/session_server/00-disk_offload.md

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from miles.rollout.session.record.codec import freeze_record
from miles.rollout.session.record.types import RecordBackend, RecordReadError, RecordRef, RecordStorageError
from miles.rollout.session.types import SessionRecord

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    record: SessionRecord | None
    live: bool = True
    readers: int = 0
    writing: bool = False
    deleting: bool = False
    disk_possible: bool = False


class RecordStore:
    """Retain snapshots until writes succeed and keep actual I/O alive across cancelled waits.

    Call put/delete and start get_many under the serving gate, without awaiting.
    All ownership changes happen on the session-server event loop.
    """

    def __init__(self, *, run_id: str, instance_id: str, backend: RecordBackend | None = None):
        self._run_id = run_id
        self._instance_id = instance_id
        self._backend = backend
        self._entries: dict[RecordRef, _Entry] = {}
        self._puts: dict[RecordRef, None] = {}
        self._tasks: set[asyncio.Task] = set()
        self._writer: asyncio.Task | None = None
        self._shutdown: asyncio.Task | None = None
        self._changed = asyncio.Event()
        self._closing = False
        self._last_warning = float("-inf")

    def put(self, session_id: str, record: SessionRecord) -> RecordRef:
        if self._closing:
            raise RuntimeError("Record store is closing")
        snapshot = freeze_record(record)
        ref = RecordRef((self._run_id, self._instance_id, session_id, uuid.uuid4().hex))
        self._entries[ref] = _Entry(snapshot)
        if self._backend is not None:
            self._puts[ref] = None
            if self._writer is None or self._writer.done():
                self._writer = self._start(self._write_pending())
        return ref

    def get_many(self, refs: list[RecordRef] | tuple[RecordRef, ...], *, timeout: float = 30.0) -> asyncio.Task:
        if self._closing:
            raise RuntimeError("Record store is closing")
        refs = tuple(refs)
        for ref in refs:
            if not self._entries[ref].live:
                raise KeyError(ref)
        for ref in refs:
            self._entries[ref].readers += 1
        actual = self._start(self._read_many(refs, deadline=asyncio.get_running_loop().time() + timeout))
        waiter = asyncio.create_task(self._wait(actual, timeout))
        waiter.add_done_callback(self._observe)
        return waiter

    def delete(self, ref: RecordRef) -> None:
        entry = self._entries.get(ref)
        if entry is not None:
            entry.live = False
            self._puts.pop(ref, None)
            self._retire(ref)

    def in_memory(self, ref: RecordRef) -> bool:
        return self._entries[ref].record is not None

    @staticmethod
    def _observe(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    def _start(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._observe(task)
        self._changed.set()

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= 30.0:
            logger.warning("Session record offload: %s", message)
            self._last_warning = now

    async def _write_pending(self) -> None:
        while self._puts:
            ref = next(iter(self._puts))
            self._puts.pop(ref)
            entry = self._entries[ref]
            entry.writing = entry.disk_possible = True
            try:
                await self._backend.put(ref.key, entry.record)
            except RecordStorageError as exc:
                self._warn(f"put failed; retaining record in memory: {exc}")
            else:
                entry.record = None
            finally:
                entry.writing = False
                self._retire(ref)

    async def _read_many(self, refs: tuple[RecordRef, ...], *, deadline: float) -> list[SessionRecord]:
        try:
            records = []
            for ref in refs:
                entry = self._entries[ref]
                if entry.record is not None:
                    records.append(freeze_record(entry.record))
                else:
                    records.append(await self._read_disk(ref, deadline=deadline))
            return records
        finally:
            for ref in refs:
                self._entries[ref].readers -= 1
                self._retire(ref)

    async def _read_disk(self, ref: RecordRef, *, deadline: float) -> SessionRecord:
        try:
            return await self._backend.get(ref.key)
        except RecordReadError as exc:
            if not exc.retryable or asyncio.get_running_loop().time() >= deadline:
                raise
        # Retry only after the first actual read has finished, within the same pins.
        return await self._backend.get(ref.key)

    def _retire(self, ref: RecordRef) -> None:
        entry = self._entries[ref]
        if entry.live or entry.readers or entry.writing or entry.deleting:
            return
        entry.record = None
        if entry.disk_possible:
            entry.deleting = True
            self._start(self._delete(ref))
        else:
            del self._entries[ref]

    async def _delete(self, ref: RecordRef) -> None:
        try:
            await self._backend.delete(ref.key)
        except RecordStorageError as exc:
            self._warn(f"delete failed for {ref.key}: {exc}")
        finally:
            del self._entries[ref]

    @staticmethod
    async def _wait(task: asyncio.Task, timeout: float | None):
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def _drain(self) -> None:
        while self._tasks:
            self._changed.clear()
            await self._changed.wait()

    async def flush(self, *, timeout: float = 30.0) -> None:
        """Wait for actual work retirement, including writes retained after failure."""
        await asyncio.wait_for(self._drain(), timeout=timeout)

    async def close(self, *, timeout: float | None = 30.0) -> bool:
        if self._shutdown is None:
            self._closing = True
            for ref in tuple(self._entries):
                self.delete(ref)
            self._shutdown = asyncio.create_task(self._close_when_idle())
            self._shutdown.add_done_callback(self._observe)
        try:
            await self._wait(self._shutdown, timeout)
        except (asyncio.TimeoutError, RecordStorageError) as exc:
            self._warn(f"shutdown incomplete; keeping unfinished work and database: {type(exc).__name__}: {exc}")
            return False
        return True

    async def _close_when_idle(self) -> None:
        await self._drain()
        if self._backend is not None:
            await self._backend.close()
