"""Collection must not block unrelated sessions or read a changing trajectory."""

import asyncio
import json
import threading

import pytest
from tests.fast.rollout.session.test_samples import _make_record
from tests.fast.router import test_session_samples_op as v1
from tests.fast.router import test_session_samples_op_v2 as v2


@pytest.fixture(params=[v1, v2], ids=["v1", "v2"])
async def collection(request):
    fixtures = request.param
    core = fixtures._build_core()
    sid = await fixtures._make_session(core, fixtures._two_turn_records(), fixtures._ACCUMULATED)
    yield core, sid
    await core.close()


async def test_collection_keeps_its_snapshot_during_commit_and_delete(collection, monkeypatch):
    core, sid = collection
    expected = await core.collect_samples(sid, max_seq_len=None)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    collect = core._collect_samples

    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return collect(*args, **kwargs)

    monkeypatch.setattr(core, "_collect_samples", blocked)
    task = asyncio.create_task(core.collect_samples(sid, max_seq_len=None))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert (await core.health()).status_code == 200
        assert (await core.create_session()).status_code == 200
        session = core.registry.get_session(sid)
        record = _make_record(v1._ACCUMULATED, [40, 41])
        async with session.lock:
            if isinstance(core, v2.SessionCoreV2):
                session.tree.create_node(
                    session.latest(),
                    delta_messages=[],
                    token_ids=v1._ACCUMULATED + [40, 41],
                    completion_span=(9, 11),
                    committed_at=1,
                    response_id="next",
                    record=record,
                    finish_reason="stop",
                )
            else:
                session.append_record(record)
                session.trajectory_token_ids.append(v1._ACCUMULATED + [40, 41])
        assert (await core.delete_session(sid)).status_code == 204
    finally:
        release.set()
        response = await task
    assert response.status_code == expected.status_code == 200
    assert response.body == expected.body


async def test_cancelled_collection_does_not_outlive_server_shutdown(collection, monkeypatch):
    core, sid = collection
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    collect = core._collect_samples
    calls = []

    def blocked(*args, **kwargs):
        calls.append(args[0])
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return collect(*args, **kwargs)

    monkeypatch.setattr(core, "_collect_samples", blocked)
    running = asyncio.create_task(core.collect_samples(sid, max_seq_len=None))
    queued = closing = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert (await core.delete_session(sid)).status_code == 204

        other = await core.create_session()
        other_sid = json.loads(other.body)["session_id"]
        queued = asyncio.create_task(core.collect_samples(other_sid, max_seq_len=None))
        await asyncio.sleep(0)  # Submit before shutting down: queued work must not start.
        closing = asyncio.create_task(core.close())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, timeout=5)
        assert not closing.done()
    finally:
        release.set()
        await core.close()
        await asyncio.gather(
            *(task for task in (running, queued, closing) if task is not None), return_exceptions=True
        )
    assert calls == [sid]
