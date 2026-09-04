import asyncio

import pytest

from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server import store as store_module
from miles.utils.workers.rpc.server.store import FINISHED_TTL_SECONDS, CallStore, DuplicateCallError


def _make_store(*, ttl: float = 300.0, finished_ttl: float = FINISHED_TTL_SECONDS) -> CallStore:
    return CallStore(retrieved_ttl_seconds=ttl, finished_ttl_seconds=finished_ttl)


async def _retrieved_call(
    *, ttl: float, outcome: CallStatusResponse, finished_ttl: float = FINISHED_TTL_SECONDS
) -> CallStore:
    store = _make_store(ttl=ttl, finished_ttl=finished_ttl)
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

    async def test_expired_call_id_can_be_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An expired call id is purged before duplicate detection, so it is accepted again as a fresh call."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="c1")
        reused_outcome = CallStatusResponse(status="success", result=2)
        store.finish(call_id="c1", outcome=reused_outcome)

        assert store.contains("c1")
        assert await store.wait(call_id="c1", timeout=0.01) == reused_outcome


class TestFinish:
    async def test_finish_unknown_call_raises_key_error(self) -> None:
        """Finishing a call id that was never begun raises KeyError instead of creating a record."""
        store = _make_store()

        with pytest.raises(KeyError):
            store.finish(call_id="missing", outcome=CallStatusResponse(status="success", result=1))

        assert not store.contains("missing")

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

    async def test_never_retrieved_record_purged_after_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A finished but never retrieved record is purged once the finished TTL passes."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 110.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_never_retrieved_record_kept_within_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A finished but never retrieved record survives while inside the finished TTL."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 110.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_never_retrieved_failed_outcome_expires_after_finished_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed outcome nobody collected expires on the finished TTL just like a successful one."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="failed", error="boom"))

        now[0] = 110.0
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 110.01
        store.begin(call_id="another")
        assert not store.contains("c1")

    async def test_never_retrieved_records_do_not_accumulate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Many finished calls whose results are never collected do not grow the store."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=0.0, finished_ttl=0.0)

        for index in range(50):
            now[0] += 1.0
            store.begin(call_id=f"c{index}")
            store.finish(call_id=f"c{index}", outcome=CallStatusResponse(status="success", result=index))

        now[0] += 1.0
        store.begin(call_id="last")

        assert [call_id for call_id in ("c0", "c25", "c49", "last") if store.contains(call_id)] == ["last"]

    async def test_unfinished_record_never_purged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A still running call is retained however old it gets, even with both TTLs at zero."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=0.0, finished_ttl=0.0)
        store.begin(call_id="c1")

        now[0] = 1e9
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_retrieved_record_purged_despite_long_finished_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once retrieved, a record expires on the retrieved TTL and a long finished TTL does not save it."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = await _retrieved_call(
            ttl=5.0,
            finished_ttl=1e6,
            outcome=CallStatusResponse(status="success", result=1),
        )

        now[0] = 15.01
        store.begin(call_id="other")

        assert not store.contains("c1")

    async def test_late_retrieval_keeps_record_for_retrieved_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A record retrieved long after finishing is kept for the retrieved TTL counted from that retrieval."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 100.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 105.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_finished_ttl_runs_from_finishing_not_from_submission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A long running call still gets its whole finished TTL, counted from when it finished."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")

        now[0] = 500.0
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 599.0
        store.begin(call_id="other")

        assert store.contains("c1")

    async def test_retrieval_just_before_the_finished_deadline_wins_the_retrieved_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once retrieved, the retrieved TTL alone decides, so a record may outlive its finished deadline."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _make_store(ttl=5.0, finished_ttl=100.0)
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 109.0
        await store.wait(call_id="c1", timeout=0.01)
        now[0] = 112.0
        store.begin(call_id="other")

        assert store.contains("c1")


class TestDefaultTtls:
    async def test_finished_ttl_default_is_twelve_hours(self) -> None:
        """The finished TTL constant is twelve hours."""
        assert FINISHED_TTL_SECONDS == 12 * 3600.0

    async def test_default_store_uses_five_minute_retrieved_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A default store keeps a retrieved outcome for five minutes and purges it right after."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = CallStore()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))
        await store.wait(call_id="c1", timeout=0.01)

        now[0] = 10.0 + 300.0
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 10.0 + 300.01
        store.begin(call_id="another")
        assert not store.contains("c1")

    async def test_default_store_purges_never_retrieved_after_finished_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default store keeps a never retrieved record for the default finished TTL and no longer."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = CallStore()
        store.begin(call_id="c1")
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        now[0] = 10.0 + FINISHED_TTL_SECONDS
        store.begin(call_id="other")
        assert store.contains("c1")

        now[0] = 10.0 + FINISHED_TTL_SECONDS + 0.01
        store.begin(call_id="another")
        assert not store.contains("c1")
