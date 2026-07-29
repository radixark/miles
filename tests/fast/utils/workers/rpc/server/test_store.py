import asyncio

import pytest

from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server import store as store_module
from miles.utils.workers.rpc.server.store import CallStore, DuplicateCallError


def _make_store(*, ttl: float = 300.0) -> CallStore:
    return CallStore(retrieved_ttl_seconds=ttl)


async def _retrieved_call(*, ttl: float, outcome: CallStatusResponse) -> CallStore:
    store = _make_store(ttl=ttl)
    store.begin(call_id="c1")
    store.finish(call_id="c1", outcome=outcome)
    await store.wait(call_id="c1", timeout=0.01)
    return store


class TestBegin:
    async def test_first_begin_registers_call(self) -> None:
        """The first submission registers its call id."""
        store = _make_store()

        store.begin(call_id="c1")

        assert store.contains("c1")

    async def test_duplicate_pending_call_rejected(self) -> None:
        """Reusing a pending call id fails loudly."""
        store = _make_store()
        store.begin(call_id="c1")

        with pytest.raises(DuplicateCallError, match="already submitted"):
            store.begin(call_id="c1")

    async def test_duplicate_finished_call_rejected(self) -> None:
        """Reusing a finished call id fails loudly."""
        store = _make_store()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        with pytest.raises(DuplicateCallError, match="already submitted"):
            store.begin(call_id="c1")


class TestFinish:
    async def test_double_finish_rejected_and_outcome_preserved(self) -> None:
        """Finishing a call twice raises and preserves the first outcome."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=1)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)

        with pytest.raises(RuntimeError, match="finished twice"):
            store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=2))

        assert await store.wait(call_id="c1", timeout=0.01) == expected


class TestWait:
    async def test_wait_returns_finished_outcome(self) -> None:
        """Waiting on a finished call returns its outcome immediately."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=42)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)

        outcome = await store.wait(call_id="c1", timeout=0.01)

        assert outcome == expected

    async def test_wait_times_out_pending(self) -> None:
        """Waiting on a pending call returns None after the poll timeout."""
        store = _make_store()
        store.begin(call_id="c1")

        assert await store.wait(call_id="c1", timeout=0.01) is None

    async def test_wait_unblocked_by_finish(self) -> None:
        """A concurrent finish unblocks an in-flight wait."""
        store = _make_store()
        expected = CallStatusResponse(status="failed", error="boom")
        store.begin(call_id="c1")

        async def finisher() -> None:
            await asyncio.sleep(0)
            store.finish(call_id="c1", outcome=expected)

        task = asyncio.create_task(finisher())
        outcome = await store.wait(call_id="c1", timeout=1.0)
        await task

        assert outcome == expected

    async def test_wait_unknown_call_raises(self) -> None:
        """Waiting on an unknown call id raises KeyError."""
        store = _make_store()

        with pytest.raises(KeyError):
            await store.wait(call_id="missing", timeout=0.01)

    async def test_wait_after_retrieval_still_returns(self) -> None:
        """A retrieved outcome remains available before its TTL expires."""
        store = _make_store()
        expected = CallStatusResponse(status="success", result=1)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=expected)
        await store.wait(call_id="c1", timeout=0.01)

        assert await store.wait(call_id="c1", timeout=0.01) == expected


class TestPurge:
    async def test_retrieved_record_purged_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A retrieved record is purged after its TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_retrieved_record_kept_within_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A retrieved record remains tracked within its TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_failed_outcome_also_purged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed outcomes follow the same retrieval TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="failed", error="boom"),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_later_retrieval_does_not_extend_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Later retrievals do not extend the first retrieval TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 14.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_never_retrieved_record_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A finished but never retrieved record is retained."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 20.0
        store.begin(call_id="other")

        assert store.contains("c1")
