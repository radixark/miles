import asyncio
import json
import sqlite3
import threading

import aiosqlite
import pytest
from tests.fast.rollout.session.sqlite_helpers import blocked_sql, make_store
from tests.fast.rollout.session.sqlite_helpers import rows as _rows
from tests.fast.rollout.session.sqlite_helpers import wait_event

from miles.rollout.session.record.backends.sqlite.backend import SQLiteBackend
from miles.rollout.session.record.codec import decode_record, encode_record
from miles.rollout.session.record.types import RecordStorageError
from miles.rollout.session.recording import commit_record
from miles.rollout.session.types import SessionRecord


def _record(*, parser_metadata=None):
    return SessionRecord(
        timestamp=1.25,
        request_timestamp=1.0,
        method="POST",
        path="/v1/chat/completions",
        request={
            "messages": [{"role": "user", "content": "你好"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        },
        response={
            "id": "upstream-id-is-not-a-record-key",
            "choices": [
                {"message": {"role": "assistant", "content": "done", "tool_calls": []}, "finish_reason": "stop"}
            ],
            "metadata": parser_metadata,
            "meta_info": {"routed_experts": "AAECAw==", "indexer_topk": "BAUGBw=="},
        },
        status_code=200,
    )


@pytest.fixture
async def store(tmp_path):
    instance = make_store(tmp_path)
    yield instance
    assert await instance.close(timeout=5)


@pytest.mark.parametrize("parser_metadata", [None, {"parser": "inkling", "errors": ["invalid tool syntax"]}])
async def test_real_sqlite_round_trip_and_final_delete(store, parser_metadata):
    original = _record(parser_metadata=parser_metadata)
    ref = store.put("session", original)
    assert ref.key[:3] == ("run", "slot", "session")
    assert store.in_memory(ref)
    await store.flush(timeout=5)
    assert not store.in_memory(ref)
    [(record_id, payload)] = _rows(store)
    assert record_id == ref.key[3] and decode_record(payload) == original
    assert await store.get_many([ref]) == [original]
    store.delete(ref)
    await store.flush(timeout=5)
    assert _rows(store) == []


async def test_memory_snapshot_and_read_results_do_not_alias(tmp_path):
    store = make_store(None)
    try:
        original = _record()
        expected = original.model_copy(deep=True)
        with commit_record(store, "session", original) as checkpoint:
            pass
        original.request["tools"][0]["function"]["name"] = "changed"
        original.response["choices"][0]["message"]["content"] = "changed"
        assert checkpoint.tools == expected.request["tools"]
        checkpoint.tools[0]["function"]["name"] = "hot state changed"
        [first] = await store.get_many([checkpoint.ref])
        assert first == expected
        first.response["meta_info"]["routed_experts"] = "changed"
        assert await store.get_many([checkpoint.ref]) == [expected]
        assert store._backend is None and list(tmp_path.iterdir()) == []
    finally:
        assert await store.close()


async def test_initialization_failure_retains_existing_and_new_records(tmp_path, caplog):
    path = tmp_path / "not-a-directory"
    path.write_text("occupied")
    store = make_store(path)
    try:
        before = store.put("session", _record())
        await store.flush(timeout=5)
        after = store.put("session", _record())
        await store.flush(timeout=5)
        assert store.in_memory(before) and store.in_memory(after)
        assert await store.get_many([before, after]) == [_record(), _record()]
        assert "initialization failed" in caplog.text
    finally:
        assert await store.close()


async def test_keys_are_not_reused_after_delete(store):
    first = store.put("session", _record())
    store.delete(first)
    second = store.put("session", _record())
    assert first.key != second.key and first.key[:3] == second.key[:3]
    await store.flush()
    assert [row[0] for row in _rows(store)] == [second.key[3]]


async def test_process_incarnations_do_not_share_database(tmp_path):
    first, second = make_store(tmp_path), make_store(tmp_path)
    try:
        first.put("session", _record())
        second.put("session", _record())
        await first.flush(timeout=5)
        await second.flush(timeout=5)
        assert first._backend.path != second._backend.path
        assert first._backend.path.exists() and second._backend.path.exists()
    finally:
        assert await first.close()
        assert await second.close()


async def test_duplicate_insert_does_not_overwrite(tmp_path):
    backend = SQLiteBackend(tmp_path, run_id="run", instance_id="slot")
    key = ("run", "slot", "session", "record")
    try:
        await backend.put(key, _record())
        changed = _record()
        changed.response["id"] = "replacement"
        with pytest.raises(RecordStorageError, match="IntegrityError"):
            await backend.put(key, changed)
        assert await backend.get(key) == _record()
        await backend.delete(key)
        with pytest.raises(RecordStorageError, match="Missing session record"):
            await backend.get(key)
    finally:
        await backend.close()


@pytest.mark.parametrize("failure", ["disk-full", "encoding", "unknown-commit"])
async def test_failed_put_retains_memory_without_retry(store, monkeypatch, caplog, failure):
    put = SQLiteBackend.put
    attempts = []

    async def fail(self, key, record):
        attempts.append(key)
        if failure == "disk-full":
            async with self._operation() as connection:
                await connection.execute_fetchall("PRAGMA max_page_count=2")
        await put(self, key, record)
        if failure == "unknown-commit":
            raise RecordStorageError("lost commit acknowledgement")

    monkeypatch.setattr(SQLiteBackend, "put", fail)
    record = _record()
    if failure == "disk-full":
        record.response["padding"] = "x" * 100_000
    elif failure == "encoding":
        record.response["not-json"] = b"\xff"
    ref = store.put("session", record)
    await store.flush(timeout=5)
    assert store.in_memory(ref)
    expected_error = {
        "disk-full": "database or disk is full",
        "encoding": "Unsupported session record JSON value: bytes",
        "unknown-commit": "lost commit acknowledgement",
    }[failure]
    assert expected_error in caplog.text
    assert await store.get_many([ref]) == [record]
    assert len(_rows(store)) == (1 if failure == "unknown-commit" else 0)
    await store.flush(timeout=5)
    assert attempts == [ref.key]
    store.delete(ref)
    await store.flush(timeout=5)
    assert _rows(store) == [] and ref not in store._entries
    monkeypatch.setattr(SQLiteBackend, "put", put)
    healthy = store.put("session", _record())
    await store.flush()
    assert not store.in_memory(healthy)


async def test_pending_writes_stay_in_memory_until_actual_commit(store):
    async with blocked_sql(store, "put") as (entered, proceed):
        refs = [store.put("session", _record()) for _ in range(4)]
        await wait_event(entered)
        assert all(store.in_memory(ref) for ref in refs)
        assert len(store._puts) == 3
        assert _rows(store) == []
        assert await store.get_many(refs) == [_record()] * 4
        with pytest.raises(asyncio.TimeoutError):
            await store.flush(timeout=0.01)
        proceed.set()
    await store.flush(timeout=5)
    assert not any(store.in_memory(ref) for ref in refs)
    assert len(_rows(store)) == 4


async def test_read_starts_at_call_and_result_survives_source_deletion(store):
    ref = store.put("session", _record())
    await store.flush(timeout=5)
    read = store.get_many([ref])
    store.delete(ref)
    with pytest.raises(KeyError):
        store.get_many([ref])
    records = await read
    await store.flush(timeout=5)
    assert _rows(store) == [] and ref not in store._entries
    await asyncio.sleep(0)
    assert records[0].model_dump_json() == _record().model_dump_json()


@pytest.mark.parametrize("mode", ["cancel", "timeout", "cancel-immediately", "unawaited"])
async def test_cancelled_or_unawaited_read_keeps_actual_io_owned(store, mode):
    ref = store.put("session", _record())
    await store.flush(timeout=5)
    path = store._backend.path
    async with blocked_sql(store) as (entered, proceed):
        task = store.get_many([ref], timeout=0.05 if mode == "timeout" else 5)
        if mode == "cancel-immediately":
            task.cancel()
        await wait_event(entered)
        store.delete(ref)
        if mode == "cancel":
            task.cancel()
        if mode != "unawaited":
            with pytest.raises(asyncio.TimeoutError if mode == "timeout" else asyncio.CancelledError):
                await task
        assert store._entries[ref].readers == 1 and len(_rows(store)) == 1
        assert not await store.close(timeout=0.01)
        assert path.exists()
        proceed.set()
    assert await store.close(timeout=5)
    assert not path.parent.exists() and not store._entries


async def test_discard_during_put_cannot_revive_record(store):
    async with blocked_sql(store, "put") as (entered, proceed):
        ref = store.put("session", _record())
        await wait_event(entered)
        store.delete(ref)
        store.delete(ref)
        assert store.in_memory(ref)
        with pytest.raises(KeyError):
            store.get_many([ref])
        proceed.set()
    await store.flush(timeout=5)
    assert ref not in store._entries and _rows(store) == []


async def test_delete_failure_keeps_key_without_payload(store):
    ref = store.put("session", _record())
    await store.flush(timeout=5)
    async with store._backend._operation() as connection:
        await connection.execute_fetchall(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON records BEGIN SELECT RAISE(FAIL, 'delete failed'); END"
        )
    store.delete(ref)
    await store.flush(timeout=5)
    assert ref not in store._entries
    assert store._backend.cleanup_debt == {ref.key} and len(_rows(store)) == 1


async def test_unreadable_disk_record_fails_entire_read(store):
    refs = [store.put("session", _record()) for _ in range(2)]
    await store.flush(timeout=5)
    with sqlite3.connect(store._backend.path) as connection:
        connection.execute("UPDATE records SET payload=? WHERE record_id=?", (b"not JSON", refs[-1].key[-1]))
    with pytest.raises(RecordStorageError, match="JSONDecodeError"):
        await store.get_many(refs)
    assert all(entry.readers == 0 for entry in store._entries.values())


@pytest.mark.parametrize("value", [{1: "coerced-key"}, (1, 2), b"bytes", float("nan")])
def test_codec_rejects_lossy_values(value):
    record = _record()
    record.response["value"] = value
    with pytest.raises((TypeError, ValueError)):
        encode_record(record)


def test_codec_rejects_schema_drift():
    data = json.loads(encode_record(_record()))
    data["extra-field"] = True
    with pytest.raises(ValueError, match="Invalid session record fields"):
        decode_record(json.dumps(data).encode())


@pytest.mark.parametrize("error", [RuntimeError("can't start new thread"), OSError("thread unavailable")])
async def test_connection_start_failure_uses_memory(tmp_path, monkeypatch, caplog, error):
    start = threading.Thread.start

    def fail_start(thread):
        if "_connection_worker_thread" in thread.name:
            raise error
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    store = make_store(tmp_path)
    try:
        assert list(tmp_path.iterdir()) == []
        ref = store.put("session", _record())
        await store.flush(timeout=5)
        assert await store.get_many([ref]) == [_record()]
        assert "initialization failed" in caplog.text
    finally:
        assert await store.close(timeout=5)
        assert await store.close(timeout=5)
        assert list(tmp_path.iterdir()) == []


async def test_sqlite_settings_thread_ownership_and_explicit_close(store, monkeypatch):
    sql_owners, codec_owners = [], []
    before = set(threading.enumerate())
    async with store._backend._operation() as connection:
        assert connection.isolation_level is None
        assert await connection.execute_fetchall("PRAGMA journal_mode") == [("delete",)]
        assert await connection.execute_fetchall("PRAGMA synchronous") == [(3,)]
        await connection.set_trace_callback(lambda sql: sql_owners.append(threading.get_ident()))

    def encode(record):
        codec_owners.append(threading.get_ident())
        return encode_record(record)

    def decode(payload):
        codec_owners.append(threading.get_ident())
        return decode_record(payload)

    monkeypatch.setattr("miles.rollout.session.record.backends.sqlite.backend.encode_record", encode)
    monkeypatch.setattr("miles.rollout.session.record.backends.sqlite.backend.decode_record", decode)
    ref = store.put("session", _record())
    await store.flush(timeout=5)
    assert await store.get_many([ref]) == [_record()]
    store.delete(ref)
    await store.flush(timeout=5)
    assert not connection.in_transaction
    assert len(sql_owners) == 3 and len(set(sql_owners)) == 1
    assert len(codec_owners) == 2
    assert threading.get_ident() not in sql_owners + codec_owners
    workers = [thread for thread in threading.enumerate() if thread not in before and thread.ident in sql_owners]
    assert len(workers) == 1
    assert await store.close()
    await asyncio.to_thread(workers[0].join, 5)
    assert not workers[0].is_alive() and not store._backend.path.parent.exists()


async def test_backend_bounds_concurrent_codec_work(tmp_path, monkeypatch):
    backend = SQLiteBackend(tmp_path, run_id="run", instance_id="slot", concurrency=1)
    entered, release = threading.Event(), threading.Event()
    encoding = []

    def encode(record):
        encoding.append(record)
        entered.set()
        assert release.wait(5)
        return encode_record(record)

    monkeypatch.setattr("miles.rollout.session.record.backends.sqlite.backend.encode_record", encode)
    tasks = [asyncio.create_task(backend.put(("run", "slot", "session", str(i)), _record())) for i in range(3)]
    try:
        await wait_event(entered)
        await asyncio.sleep(0)
        assert len(encoding) == 1
        release.set()
        await asyncio.gather(*tasks)
        assert len(encoding) == 3
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await backend.close()


async def test_connection_close_failure_preserves_database(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.put("session", _record())
    await store.flush()
    connection = store._backend._connection
    close = connection.close

    async def fail_close():
        raise aiosqlite.OperationalError("close failed")

    monkeypatch.setattr(connection, "close", fail_close)
    assert not await store.close()
    assert store._backend.path.exists()
    monkeypatch.setattr(connection, "close", close)
    # The failed shutdown is terminal; clean up the test backend explicitly.
    await store._backend.close()
