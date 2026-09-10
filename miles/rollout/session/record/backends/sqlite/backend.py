# doc-dev: docs/developer/session_server/00-disk_offload.md

import asyncio
import hashlib
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from sqlite3 import (
    SQLITE_BUSY,
    SQLITE_CANTOPEN,
    SQLITE_FULL,
    SQLITE_IOERR,
    SQLITE_LOCKED,
    SQLITE_PROTOCOL,
    SQLITE_READONLY,
)

import aiosqlite

from miles.rollout.session.record.codec import decode_record, encode_record
from miles.rollout.session.record.types import RecordReadError, RecordStorageError, SessionRecordKey
from miles.rollout.session.types import SessionRecord

logger = logging.getLogger(__name__)
_KEY_WHERE = "run_id=? AND instance_id=? AND session_id=? AND record_id=?"


class SQLiteBackend:
    """Own one temporary database; aiosqlite owns its connection thread."""

    def __init__(self, directory: Path, *, run_id: str, instance_id: str, concurrency: int = 4):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self._directory = directory
        self._namespace = hashlib.sha256(f"{run_id}\0{instance_id}".encode()).hexdigest()[:24]
        self._slots = asyncio.Semaphore(concurrency)
        self._initialization: asyncio.Task[aiosqlite.Connection] | None = None
        self._connection: aiosqlite.Connection | None = None
        self.path: Path | None = None
        self.cleanup_debt: set[SessionRecordKey] = set()

    def _create_path(self) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        namespace = Path(tempfile.mkdtemp(prefix=f"session-records-{self._namespace}-", dir=self._directory))
        return namespace / "records.sqlite3"

    async def _open(self) -> aiosqlite.Connection:
        try:
            self.path = await asyncio.to_thread(self._create_path)
            self._connection = await aiosqlite.connect(self.path, isolation_level=None, timeout=1.0)
            connection = self._connection
            journal = (await connection.execute_fetchall("PRAGMA journal_mode=DELETE"))[0][0]
            await connection.execute_fetchall("PRAGMA synchronous=EXTRA")
            synchronous = (await connection.execute_fetchall("PRAGMA synchronous"))[0][0]
            if journal.lower() != "delete" or synchronous != 3:
                raise RuntimeError(f"Unexpected SQLite settings: journal={journal}, synchronous={synchronous}")
            await connection.execute_fetchall(
                "CREATE TABLE records (run_id TEXT, instance_id TEXT, session_id TEXT, record_id TEXT, "
                "payload BLOB NOT NULL, PRIMARY KEY (run_id, instance_id, session_id, record_id)) WITHOUT ROWID"
            )
            return connection
        except (OSError, aiosqlite.Error, RuntimeError) as exc:
            raise RecordStorageError(f"initialization failed: {type(exc).__name__}: {exc}") from None

    @asynccontextmanager
    async def _operation(self):
        async with self._slots:
            if self._initialization is None:
                self._initialization = asyncio.create_task(self._open())
            try:
                connection = await asyncio.shield(self._initialization)
                yield connection
            except RecordStorageError:
                raise
            except (OSError, aiosqlite.Error, RuntimeError, ValueError, TypeError, LookupError) as exc:
                raise RecordStorageError(f"{type(exc).__name__}: {exc}") from None

    async def put(self, key: SessionRecordKey, record: SessionRecord) -> None:
        async with self._operation() as connection:
            payload = await asyncio.to_thread(encode_record, record)
            await connection.execute_fetchall("INSERT INTO records VALUES (?, ?, ?, ?, ?)", (*key, payload))

    async def get(self, key: SessionRecordKey) -> SessionRecord:
        async with self._operation() as connection:
            try:
                rows = await connection.execute_fetchall(f"SELECT payload FROM records WHERE {_KEY_WHERE}", key)
            except aiosqlite.OperationalError as exc:
                code = exc.sqlite_errorcode & 0xFF
                if code not in (
                    SQLITE_BUSY,
                    SQLITE_LOCKED,
                    SQLITE_IOERR,
                    SQLITE_FULL,
                    SQLITE_CANTOPEN,
                    SQLITE_READONLY,
                    SQLITE_PROTOCOL,
                ):
                    raise
                raise RecordReadError(str(exc), retryable=code in (SQLITE_BUSY, SQLITE_LOCKED)) from None
            except OSError as exc:
                raise RecordReadError(str(exc)) from None
            if not rows:
                raise LookupError(f"Missing session record: {key}")
            return await asyncio.to_thread(decode_record, rows[0][0])

    async def delete(self, key: SessionRecordKey) -> None:
        try:
            async with self._operation() as connection:
                await connection.execute_fetchall(f"DELETE FROM records WHERE {_KEY_WHERE}", key)
        except RecordStorageError:
            self.cleanup_debt.add(key)
            raise

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except (OSError, aiosqlite.Error, RuntimeError) as exc:
                raise RecordStorageError(f"connection close failed: {exc}") from None
            self._connection = None
        if self.path is not None:
            try:
                await asyncio.to_thread(shutil.rmtree, self.path.parent)
            except OSError as exc:
                logger.warning("Session record offload: temporary database cleanup failed: %s", exc)
