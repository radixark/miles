import asyncio
import sqlite3

import httpx
import pytest
from fastapi import FastAPI
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.sqlite_helpers import blocked_sql, make_store, rows, wait_event
from tests.fast.rollout.session.test_record_serving import USER, _chat, _core

import miles.rollout.generate_utils.openai_endpoint_utils as endpoint
import miles.rollout.session.sessions as routes
import miles.utils.http_utils as http_utils
from miles.rollout.session.errors import SessionServerClosingError, TokenizationError
from miles.rollout.session.record.types import RecordReadError, RecordStorageError
from miles.rollout.session.types import SESSION_GENERATION_HEADER, SESSION_RECORD_ERROR_CODE
from miles.utils.types import Sample


@pytest.fixture(params=["v1", "v2"])
async def core(tmp_path, request):
    instance = _core(make_store(tmp_path), request.param)
    yield instance
    assert await instance.close(timeout=5)


async def _collect(core, sid):
    response = await core.collect_samples(sid, max_seq_len=None)
    assert response.status_code == 200, response.body
    return int(response.headers[SESSION_GENERATION_HEADER])


async def _expired(core, sid):
    async with asyncio.timeout(2):
        while sid in core.registry.sessions:
            await asyncio.sleep(0.005)
    await core.registry.record_store.flush()


async def test_stale_collect_and_active_chat_cannot_delete(core):
    sid = core.registry.create_session()
    await _chat(core, sid)
    generation = await _collect(core, sid)
    core.backend.block = asyncio.Event()
    # Drain the earlier completed proxy notification.
    await core.backend.entered.get()
    chat = asyncio.create_task(_chat(core, sid))
    await core.backend.entered.get()
    state = core.registry.get_session(sid)
    assert (await core.delete_session(sid, generation=generation)).status_code == 412
    current = await _collect(core, sid)
    assert (await core.delete_session(sid, generation=current)).status_code == 412
    assert not state.closing and state.activity.active == 1
    core.backend.block.set()
    assert (await chat).status_code == 200
    assert state.activity.generation > current
    assert (await core.delete_session(sid, generation=current)).status_code == 412
    current = await _collect(core, sid)
    assert (await core.delete_session(sid, generation=current)).status_code == 204
    await core.registry.record_store.flush()
    assert rows(core.registry.record_store) == []


async def test_fired_timer_waiting_for_lock_cannot_delete_new_activity(core):
    sid = core.registry.create_session()
    core.lifecycle.idle_timeout = 0.01
    await _collect(core, sid)
    state = core.registry.get_session(sid)
    async with state.lock:
        # Queue accepted work before the timer queues its expiry lock acquisition.
        get = asyncio.create_task(core.get_session(sid))
        await asyncio.sleep(0.03)
        assert core.lifecycle.tasks
    assert (await get).status_code == 200
    assert sid in core.registry.sessions
    await _expired(core, sid)


async def test_rejected_mutating_chat_preserves_existing_expiry(core, monkeypatch):
    sid = core.registry.create_session()
    await _chat(core, sid)
    core.lifecycle.idle_timeout = 0.03
    await _collect(core, sid)
    state = core.registry.get_session(sid)
    timer = state.activity.timer
    generation = state.activity.generation

    def fail_after_rollback(*args, **kwargs):
        raise TokenizationError("render failed after positioning")

    monkeypatch.setattr(core.registry.tito_tokenizer, "apply_chat_template", fail_after_rollback)
    # Replaying the original user message rolls v1 back or positions v2 at the root.
    with pytest.raises(TokenizationError):
        await _chat(core, sid, [USER])
    assert state.activity.generation > generation
    assert state.activity.active == 0 and state.activity.timer is timer
    await _expired(core, sid)


async def test_cancelled_collect_retires_activity_but_pins_actual_read(core):
    sid = core.registry.create_session()
    await _chat(core, sid)
    store = core.registry.record_store
    await store.flush()
    state = core.registry.get_session(sid)
    [ref] = state.record_refs
    core.lifecycle.idle_timeout = 0.01
    async with blocked_sql(store) as (entered, release):
        task = asyncio.create_task(core.collect_samples(sid, max_seq_len=None))
        await wait_event(entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.activity.active == 0 and state.activity.timer is not None
        async with asyncio.timeout(2):
            while sid in core.registry.sessions:
                await asyncio.sleep(0.005)
        assert store._entries[ref].readers == 1 and len(rows(store)) == 1
        release.set()
    await store.flush()
    assert not store._entries and rows(store) == []


async def test_idle_retention_starts_at_collect_only_and_rearms_after_proxy_failure(core, monkeypatch):
    core.lifecycle.idle_timeout = 0.02
    sid = core.registry.create_session()
    await _chat(core, sid)
    state = core.registry.get_session(sid)
    assert state.activity.timer is None
    await _collect(core, sid)
    old = state.activity.timer

    async def unavailable(*args, **kwargs):
        assert state.activity.timer is None and state.activity.active == 1
        return {"status_code": 502, "headers": {}, "response_body": b"upstream unavailable"}

    monkeypatch.setattr(core.backend, "do_proxy", unavailable)
    assert (await _chat(core, sid)).status_code == 502
    assert state.activity.active == 0 and state.activity.timer is not old
    await _expired(core, sid)


async def test_close_deadline_covers_lock_wait_without_cancelling_shutdown(core):
    sid = core.registry.create_session()
    await _chat(core, sid)
    await core.registry.record_store.flush()
    path = core.registry.record_store._backend.path
    state = core.registry.get_session(sid)
    async with state.lock:
        assert not await core.close(timeout=0.01)
        shared = core._shutdown
        assert shared is not None and not shared.done() and path.exists()
        with pytest.raises(SessionServerClosingError):
            await core.create_session()
        assert not await core.close(timeout=0.01)
        assert core._shutdown is shared
    assert await core.close(timeout=5)
    assert not path.exists() and not core.registry.sessions


async def test_sqlite_busy_retries_once_with_same_pins(core, monkeypatch):
    sid = core.registry.create_session()
    await _chat(core, sid)
    store = core.registry.record_store
    await store.flush()
    [ref] = core.registry.get_session(sid).record_refs
    await store._backend._connection.execute_fetchall("PRAGMA busy_timeout=0")
    blocker = sqlite3.connect(store._backend.path)
    blocker.execute("BEGIN EXCLUSIVE")
    get = store._backend.get
    attempts = []

    async def observed(key):
        attempts.append(key)
        assert store._entries[ref].readers == 1
        try:
            return await get(key)
        except RecordReadError as exc:
            assert exc.retryable
            blocker.rollback()
            raise

    monkeypatch.setattr(store._backend, "get", observed)
    try:
        assert len(await store.get_many([ref])) == 1
    finally:
        blocker.close()
    assert attempts == [ref.key, ref.key]
    assert store._entries[ref].readers == 0


async def test_read_deadline_does_not_start_retry(core, monkeypatch):
    sid = core.registry.create_session()
    await _chat(core, sid)
    store = core.registry.record_store
    await store.flush()
    [ref] = core.registry.get_session(sid).record_refs
    calls = []

    async def busy(key):
        calls.append(key)
        await asyncio.sleep(0.03)
        raise RecordReadError("busy", retryable=True)

    monkeypatch.setattr(store._backend, "get", busy)
    with pytest.raises(TimeoutError):
        await store.get_many([ref], timeout=0.005)
    assert store._entries[ref].readers == 1
    await store.flush()
    assert calls == [ref.key] and store._entries[ref].readers == 0


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("disabled", [False, True])
async def test_startup_http_sqlite_collect_and_guarded_cleanup(tmp_path, monkeypatch, version, disabled):
    template = _core(make_store(None), version)
    monkeypatch.setattr(routes, "load_tokenizer", lambda *args, **kwargs: template.registry.tokenizer)
    monkeypatch.setattr(routes, "get_tito_tokenizer", lambda *args, **kwargs: template.registry.tito_tokenizer)
    app = FastAPI()
    config = make_session_server_config(
        hf_checkpoint="test",
        use_session_server=version,
        disk_offload=not disabled,
        disk_offload_dir=str(tmp_path / "records"),
        run_id="run-live",
        instance_id=None,
        num_layers=1,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    routes.setup_session_routes(app, template.backend, config)
    core = app.state.session_core
    try:
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://server"
        ) as client:
            monkeypatch.setattr(http_utils, "_http_client", client)
            sid = (await client.post("/sessions")).json()["session_id"]
            health = (await client.get("/health")).json()
            assert "session_server_instance_id" not in health
            response = await client.post(f"/sessions/{sid}/v1/chat/completions", json={"messages": [USER]})
            assert response.status_code == 200, response.text
            store = core.registry.record_store
            await store.flush()
            [ref] = core.registry.get_session(sid).record_refs
            assert ref.key[:2] == ("run-live", store._instance_id)
            assert store._instance_id and store.in_memory(ref) is disabled
            if disabled:
                assert not (tmp_path / "records").exists()
            else:
                assert store._backend.path.is_file() and len(rows(store)) == 1
            invalid = await client.delete(f"/sessions/{sid}", headers={SESSION_GENERATION_HEADER: "bad"})
            assert invalid.status_code == 422 and sid in core.registry.sessions
            tracer = endpoint.OpenAIEndpointTracer(
                "http://server",
                sid,
                samples_wire_fields=endpoint.COMPUTED_FIELDS_V2 if version == "v2" else endpoint.COMPUTED_FIELDS,
            )
            collected = await tracer.collect_samples(Sample(), max_seq_len=None)
            assert len(collected.reply.samples) == 1
            assert sid in core.registry.sessions
            tracer.schedule_cleanup(collected.generation)
            await asyncio.gather(*endpoint._cleanup_tasks)
            assert sid not in core.registry.sessions
    finally:
        await template.close()
    if not disabled:
        assert list((tmp_path / "records").iterdir()) == []


@pytest.mark.parametrize(
    "failure,status",
    [(RecordReadError("I/O failure"), 503), (RecordStorageError("bad schema"), 500), (TimeoutError(), 503)],
)
async def test_http_classifies_read_failures_and_retains_session(core, monkeypatch, failure, status):
    monkeypatch.setattr(routes, "load_tokenizer", lambda *args, **kwargs: core.registry.tokenizer)
    monkeypatch.setattr(routes, "get_tito_tokenizer", lambda *args, **kwargs: core.registry.tito_tokenizer)
    app = FastAPI()
    routes.setup_session_routes(app, core.backend, make_session_server_config(hf_checkpoint="test"))
    live = app.state.session_core
    sid = live.registry.create_session()

    async def failed():
        raise failure

    monkeypatch.setattr(live.registry.record_store, "get_many", lambda refs: asyncio.create_task(failed()))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://server") as client:
            response = await client.post(f"/sessions/{sid}/samples", json={})
            assert response.status_code == status
            if status == 503:
                assert response.json()["error"]["code"] == SESSION_RECORD_ERROR_CODE
            assert SESSION_GENERATION_HEADER not in response.headers
            state = live.registry.get_session(sid)
            assert state.activity.active == 0 and state.activity.timer is not None
    finally:
        await live.close()


@pytest.mark.parametrize("corruption", ["schema", "payload"])
async def test_sqlite_schema_and_codec_failures_are_not_transient(core, corruption):
    sid = core.registry.create_session()
    await _chat(core, sid)
    store = core.registry.record_store
    await store.flush()
    with sqlite3.connect(store._backend.path) as connection:
        connection.execute("DROP TABLE records" if corruption == "schema" else "UPDATE records SET payload=x'00'")
    with pytest.raises(RecordStorageError) as raised:
        await core.collect_samples(sid, max_seq_len=None)
    assert not isinstance(raised.value, RecordReadError)
    assert sid in core.registry.sessions


async def test_startup_initialization_failure_keeps_serving_in_memory(tmp_path, monkeypatch):
    template = _core(make_store(None), "v1")
    monkeypatch.setattr(routes, "load_tokenizer", lambda *args, **kwargs: template.registry.tokenizer)
    monkeypatch.setattr(routes, "get_tito_tokenizer", lambda *args, **kwargs: template.registry.tito_tokenizer)
    # A file cannot contain a new DB directory; setup itself must not touch it.
    directory = tmp_path / "unavailable"
    directory.write_text("existing file")
    app = FastAPI()
    routes.setup_session_routes(
        app,
        template.backend,
        make_session_server_config(
            hf_checkpoint="test",
            disk_offload=True,
            disk_offload_dir=str(directory),
            num_layers=1,
        ),
    )
    core = app.state.session_core
    try:
        sid = core.registry.create_session()
        assert (await _chat(core, sid)).status_code == 200
        await core.registry.record_store.flush()
        [ref] = core.registry.get_session(sid).record_refs
        assert core.registry.record_store.in_memory(ref)
        assert (await core.get_session(sid)).status_code == 200
        await _collect(core, sid)
        assert directory.read_text() == "existing file"
    finally:
        assert await core.close()
        await template.close()
