import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager

from miles.rollout.session.record.backends.sqlite.backend import SQLiteBackend
from miles.rollout.session.record.store import RecordStore


def make_store(directory):
    backend = SQLiteBackend(directory, run_id="run", instance_id="slot") if directory is not None else None
    return RecordStore(run_id="run", instance_id="slot", backend=backend)


def rows(store):
    if store._backend.path is None:
        return []
    with sqlite3.connect(store._backend.path) as connection:
        return connection.execute("SELECT record_id, payload FROM records").fetchall()


async def wait_event(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=5)


@asynccontextmanager
async def blocked_sql(store, operation="get"):
    entered, release = threading.Event(), threading.Event()
    prefix = {"get": "SELECT payload", "put": "INSERT INTO records", "delete": "DELETE FROM records"}[operation]

    def trace(sql):
        if sql.startswith(prefix):
            entered.set()
            release.wait(timeout=10)

    async with store._backend._operation() as connection:
        await connection.set_trace_callback(trace)
    try:
        yield entered, release
    finally:
        release.set()
        await connection.set_trace_callback(None)
