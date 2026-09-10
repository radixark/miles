import asyncio
import base64
import json
import sqlite3
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.sqlite_helpers import blocked_sql, make_store
from tests.fast.rollout.session.sqlite_helpers import rows as _rows
from tests.fast.rollout.session.sqlite_helpers import wait_event as _entered
from tests.fast.utils.chat_template_utils.test_inkling_response import FakeInklingTokenizer

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import SessionNotFoundError, TokenizationError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.record.backends.sqlite.backend import SQLiteBackend
from miles.rollout.session.record.codec import decode_record
from miles.rollout.session.record.types import RecordStorageError
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils.tito_tokenizer import InklingTITOTokenizer, TITOTokenizer
from miles.utils.types import Sample

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
USER = {"role": "user", "content": "hello"}


class _Backend:
    def __init__(self, completion):
        self.completion = completion
        self.calls = 0
        self.block = None
        self.entered = asyncio.Queue()

    async def do_proxy(self, _request, _path, *, body, headers):
        request = json.loads(body)
        self.calls += 1
        number = self.calls
        self.entered.put_nowait(number)
        if self.block is not None:
            await self.block.wait()
        rows = len(request["input_ids"]) + len(self.completion) - 1
        replay = base64.b64encode(struct.pack(f"<{rows}i", *range(rows))).decode()
        response = {
            "id": f"response-{number}",
            "choices": [
                {
                    "message": {"role": "assistant", "content": f"answer-{number}"},
                    "finish_reason": "stop",
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, token] for token in self.completion],
                        "completion_tokens": len(self.completion),
                        "routed_experts": replay,
                        "indexer_topk": replay,
                        "indexer_topk_num_layers": 1,
                    },
                }
            ],
        }
        return {"status_code": 200, "headers": {}, "response_body": json.dumps(response).encode()}


def _core(store, version, *, kind="ordinary"):
    tokenizer = FakeInklingTokenizer()
    if kind == "ordinary":
        tito = TITOTokenizer(tokenizer, assistant_start_str="assistant")
        completion = tokenizer.encode("done")
    else:
        tito = InklingTITOTokenizer(tokenizer)
        if kind == "inkling":
            completion = (
                tokenizer.encode("bash") + [149] + tokenizer.encode('{"name":"lookup","args":{}}') + [110, 106]
            )
        else:
            completion = [104, 99, 110, 106]
    tito.create_comparator = lambda: SimpleNamespace(compare_sequences=lambda *args: [])
    tito.apply_chat_template = lambda *args, **kwargs: tokenizer.encode("prompt")
    tito.merge_tokens = lambda **kwargs: kwargs["pretokenized_token_ids"] + tokenizer.encode("next")
    registry_type, core_type = (
        (SessionRegistry, SessionCore) if version == "v1" else (SessionRegistryV2, SessionCoreV2)
    )
    registry = registry_type(tokenizer, tito_tokenizer=tito, record_store=store)
    config = make_session_server_config(
        num_layers=1,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    return core_type(_Backend(completion), registry, config)


async def _chat(core, sid, messages=None, *, tools=TOOLS):
    return await core.chat_completions(
        sid,
        method="POST",
        query="",
        headers={},
        body=json.dumps({"messages": messages if messages is not None else [USER], "tools": tools}).encode(),
    )


def _samples(response):
    assert response.status_code == 200, response.body
    return decode_samples_and_merge_input_sample(response.body, Sample())


@pytest.fixture(params=["v1", "v2"])
def version(request):
    return request.param


@pytest.fixture
async def store(tmp_path):
    instance = make_store(tmp_path)
    yield instance
    assert await instance.close(timeout=5)


@pytest.mark.parametrize("kind", ["ordinary", "inkling", "inkling_error"])
async def test_canonical_chat_get_samples_round_trip(store, version, kind, monkeypatch):
    core = _core(store, version, kind=kind)
    sid = core.registry.create_session()
    committed = []
    put = store.put

    def capture(session_id, record):
        committed.append(record.model_copy(deep=True))
        return put(session_id, record)

    monkeypatch.setattr(store, "put", capture)
    reply = await _chat(core, sid)
    assert reply.status_code == 200
    assert len(committed) == 1
    await store.flush()
    state = core.registry.get_session(sid)
    [ref] = state.record_refs
    assert not store.in_memory(ref)
    [(record_id, payload)] = _rows(store)
    assert record_id == ref.key[3] and decode_record(payload) == committed[0]
    assert (state.record_checkpoints[-1] if version == "v1" else state.active_leaf.record_checkpoint).tools == TOOLS

    def no_parser(**kwargs):
        raise AssertionError("hydrate must not rerun the response parser")

    monkeypatch.setattr(core.registry.tito_tokenizer, "postprocess_completion", no_parser)
    response = json.loads((await core.get_session(sid)).body)
    assert response["records"] == [record.model_dump(mode="json") for record in committed]
    choice = response["records"][0]["response"]["choices"][0]
    if kind != "ordinary":
        assert choice["meta_info"]["miles_response_parser"] == "inkling"
        if kind == "inkling":
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
        else:
            assert choice["meta_info"]["miles_response_parse_error"] == "control_token_inside_open_block"
            assert choice["finish_reason"] == "stop"
    client_choice = json.loads(reply.body)["choices"][0]
    assert "routed_experts" not in client_choice["meta_info"]
    assert "indexer_topk" not in client_choice["meta_info"]
    result = _samples(await core.collect_samples(sid, max_seq_len=None))
    assert len(result.samples) == 1
    assert result.samples[0].tokens == response["metadata"]["accumulated_token_ids"]
    assert result.samples[0].rollout_routed_experts is not None
    assert result.samples[0].rollout_indexer_topk is not None
    await core.delete_session(sid)
    await store.flush()
    assert _rows(store) == [] and not store._entries


@pytest.mark.parametrize("operation", ["get", "samples"])
async def test_read_snapshot_survives_retry_and_delete(store, version, operation, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()
    await _chat(core, sid)
    await store.flush()
    state = core.registry.get_session(sid)
    [old_ref] = state.record_refs
    before = json.loads((await core.get_session(sid)).body)
    async with blocked_sql(store) as (entered, release):
        read = core.get_session(sid) if operation == "get" else core.collect_samples(sid, max_seq_len=None)
        task = asyncio.create_task(read)
        await _entered(entered)
        assert not state.lock.locked()
        await _chat(core, sid, tools=[])
        [new_ref] = state.record_refs
        assert new_ref.key != old_ref.key
        assert (state.record_checkpoints[-1] if version == "v1" else state.active_leaf.record_checkpoint).tools == []
        assert store._entries[old_ref].readers == 1
        await core.delete_session(sid)
        assert sid not in core.registry.sessions
        assert len(_rows(store)) == 1
        release.set()
        reply = await task
    if operation == "get":
        assert json.loads(reply.body) == before
    else:
        result = _samples(reply)
        assert result.samples[0].tokens == before["metadata"]["accumulated_token_ids"]
        if version == "v2":
            assert len(result.session_metadata["tree"]["nodes"]) == 1
            assert result.samples[0].metadata["leaf"]["response_id"] == "response-1"
    await store.flush()
    assert _rows(store) == [] and not store._entries


async def test_concurrent_chat_preserves_each_serving_gate(store, version):
    core = _core(store, version)
    sid = core.registry.create_session()
    core.backend.block = asyncio.Event()
    first, second = asyncio.create_task(_chat(core, sid)), asyncio.create_task(_chat(core, sid))
    await asyncio.wait_for(core.backend.entered.get(), 5)
    await asyncio.wait_for(core.backend.entered.get(), 5)
    core.backend.block.set()
    assert all(reply.status_code == 200 for reply in await asyncio.gather(first, second))
    await store.flush()
    state = core.registry.get_session(sid)
    if version == "v1":
        assert len(state.record_refs) == 1 and len(_rows(store)) == 1
    else:
        assert len(state.tree.nodes) == len(_rows(store)) == 2
        assert all(node.parent is None for node in state.tree.nodes)
        assert len({node.record_checkpoint.ref.key for node in state.tree.nodes}) == 2


@pytest.mark.parametrize("operation", ["get", "samples"])
async def test_read_cancellation_retains_command_ownership(store, version, operation, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()
    await _chat(core, sid)
    await store.flush()
    [ref] = core.registry.get_session(sid).record_refs
    async with blocked_sql(store) as (entered, release):
        read = core.get_session(sid) if operation == "get" else core.collect_samples(sid, max_seq_len=None)
        task = asyncio.create_task(read)
        await _entered(entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await core.delete_session(sid)
        assert store._entries[ref].readers == 1
        assert len(_rows(store)) == 1
        release.set()
    await store.flush()
    assert _rows(store) == [] and not store._entries


async def test_write_failure_keeps_chat_and_reads_usable(store, version, monkeypatch):
    async def disk_full(backend, key, record):
        async with backend._operation() as connection:
            await connection.execute_fetchall("PRAGMA max_page_count=2")
        snapshot = record.model_copy(deep=True)
        snapshot.response["padding"] = "x" * 100_000
        await put(backend, key, snapshot)

    put = SQLiteBackend.put
    monkeypatch.setattr(SQLiteBackend, "put", disk_full)
    core = _core(store, version)
    sid = core.registry.create_session()
    assert (await _chat(core, sid)).status_code == 200
    await store.flush()
    [ref] = core.registry.get_session(sid).record_refs
    assert store.in_memory(ref) and _rows(store) == []
    assert len(json.loads((await core.get_session(sid)).body)["records"]) == 1
    assert len(_samples(await core.collect_samples(sid, max_seq_len=None)).samples) == 1


async def test_unreadable_disk_record_fails_before_hooks_and_keeps_source(store, version, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()
    await _chat(core, sid)
    await store.flush()
    with sqlite3.connect(store._backend.path) as connection:
        connection.execute("UPDATE records SET payload=?", (b"not JSON",))
    if version == "v2":

        def no_hooks(*args):
            pytest.fail("hooks ran after failed hydration")

        core.sample_picker = no_hooks
    with pytest.raises(RecordStorageError):
        await core.collect_samples(sid, max_seq_len=None)
    with pytest.raises(RecordStorageError):
        await core.get_session(sid)
    assert sid in core.registry.sessions and len(_rows(store)) == 1
    assert all(entry.readers == 0 for entry in store._entries.values())


async def test_freeze_failure_does_not_publish_hot_state(store, version, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()

    def cannot_freeze(*args):
        raise ValueError("cannot freeze record")

    monkeypatch.setattr("miles.rollout.session.record.store.freeze_record", cannot_freeze)
    with pytest.raises(ValueError, match="cannot freeze"):
        await _chat(core, sid)
    state = core.registry.get_session(sid)
    assert state.record_refs == [] and not store._entries
    assert state.num_assistant == 0 if version == "v1" else state.tree.nodes == []


async def test_hot_validation_failure_discards_unpublished_record(store, version, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    if version == "v1":

        def invalid_prefix(*args, **kwargs):
            raise TokenizationError("pretokenized prefix mismatch")

        monkeypatch.setattr(state, "update_pretokenized_state", invalid_prefix)
        expected = TokenizationError
    else:
        monkeypatch.setattr("miles.rollout.session.v2.tree_trajectory.MAX_NODES", 0)
        expected = ValueError
    with pytest.raises(expected):
        await _chat(core, sid)
    await store.flush()
    assert state.record_refs == [] and not store._entries and _rows(store) == []


@pytest.mark.parametrize("operation", ["get", "samples"])
async def test_waiting_reader_rechecks_closing_before_starting_read(store, version, operation):
    core = _core(store, version)
    sid = core.registry.create_session()
    await _chat(core, sid)
    state = core.registry.get_session(sid)
    async with state.lock:
        read = core.get_session(sid) if operation == "get" else core.collect_samples(sid, max_seq_len=None)
        task = asyncio.create_task(read)
        await asyncio.sleep(0)
        state.closing = True
    with pytest.raises(SessionNotFoundError):
        await task
    assert all(entry.readers == 0 for entry in store._entries.values())


async def test_memory_store_uses_same_ref_and_read_path(version, tmp_path):
    store = make_store(None)
    try:
        core = _core(store, version)
        sid = core.registry.create_session()
        await _chat(core, sid)
        [ref] = core.registry.get_session(sid).record_refs
        assert store.in_memory(ref) and store._backend is None
        assert len(json.loads((await core.get_session(sid)).body)["records"]) == 1
        assert len(_samples(await core.collect_samples(sid, max_seq_len=None)).samples) == 1
        await core.delete_session(sid)
        assert not store._entries and list(tmp_path.iterdir()) == []
    finally:
        assert await store.close()


async def test_memory_and_disk_return_equal_replies(store, version, monkeypatch):
    core = _core(store, version)
    sid = core.registry.create_session()
    async with blocked_sql(store, operation="put") as (entered, release):
        await _chat(core, sid)
        await _entered(entered)
        [ref] = core.registry.get_session(sid).record_refs
        assert store.in_memory(ref)
        memory_get = await core.get_session(sid)
        memory_samples = _samples(await core.collect_samples(sid, max_seq_len=None))
        release.set()
    await store.flush()
    assert not store.in_memory(ref)
    assert (await core.get_session(sid)).body == memory_get.body
    disk_samples = _samples(await core.collect_samples(sid, max_seq_len=None))
    assert disk_samples.session_metadata == memory_samples.session_metadata
    assert disk_samples.empty_reason == memory_samples.empty_reason
    np.testing.assert_equal(
        [sample.to_dict() for sample in disk_samples.samples],
        [sample.to_dict() for sample in memory_samples.samples],
    )
