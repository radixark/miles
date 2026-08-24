"""Frontend state stores: fingerprint identity, conflicts, replay retention."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

import pytest

from miles.ray.tinker_frontend.state import (
    ConflictError,
    ExpiredError,
    FutureRecord,
    FutureStore,
    SamplingSessionRecord,
    SamplingSessionStore,
    SessionStore,
    fingerprint_of,
)


def record(request_id="r1", fingerprint="f1", terminal=None):
    rec = FutureRecord(request_id=request_id, kind="operation", fingerprint=fingerprint)
    if terminal is not None:
        rec.resolve(terminal)
    return rec


class TestFutureStore:
    def test_existing_replays_identical_and_conflicts_on_divergence(self):
        store = FutureStore()
        store.put(record())
        assert store.existing("r1", "f1") is not None
        assert store.existing("r2", "f1") is None
        with pytest.raises(ConflictError, match="identical"):
            store.existing("r1", "OTHER")

    def test_delivered_terminal_records_are_evicted_lru(self):
        store = FutureStore(max_delivered=2)
        for i in range(3):
            rec = store.put(record(f"r{i}", terminal={"n": i}))
            store.mark_delivered(rec)
        assert store.get("r0") is None
        assert store.get("r1").terminal == {"n": 1}
        assert store.get("r2").terminal == {"n": 2}

    def test_pending_records_are_never_evicted(self):
        store = FutureStore(max_delivered=1)
        pending = store.put(record("pending"))
        store.mark_delivered(pending)
        for i in range(3):
            store.mark_delivered(store.put(record(f"r{i}", terminal={})))
        assert store.get("pending") is pending

    def test_eviction_leaves_a_typed_tombstone(self):
        store = FutureStore(max_delivered=1)
        store.mark_delivered(store.put(record("r1", "f1", terminal={"n": 1})))
        store.mark_delivered(store.put(record("r2", "f2", terminal={"n": 2})))
        assert store.get("r1") is None
        assert store.expired_fingerprint("r1") == "f1"
        with pytest.raises(ExpiredError, match="already delivered"):
            store.existing("r1", "f1")
        with pytest.raises(ConflictError, match="identical"):
            store.existing("r1", "OTHER")

    def test_tombstones_are_bounded(self):
        store = FutureStore(max_delivered=1, max_expired=2)
        for i in range(4):
            store.mark_delivered(store.put(record(f"r{i}", f"f{i}", terminal={})))
        assert store.expired_fingerprint("r0") is None
        assert store.expired_fingerprint("r2") == "f2"
        assert store.existing("r0", "f0") is None

    def test_resolve_drops_the_forward_payload(self):
        rec = record()
        rec.forward_payload = {"samples": []}
        rec.resolve({"ok": True})
        assert rec.forward_payload is None


class TestSessions:
    def test_heartbeat_only_touches_known_sessions(self):
        store = SessionStore()
        session = store.create("0.24.1", [], None)
        assert store.heartbeat(session.session_id)
        assert not store.heartbeat("sess-nope")

    def test_fingerprints_are_canonical(self):
        assert fingerprint_of({"a": 1, "b": 2}) == fingerprint_of({"b": 2, "a": 1})
        assert fingerprint_of({"a": 1}) != fingerprint_of({"a": 2})

    def test_child_sampler_namespaces_are_retired_in_one_bulk_pass(self):
        store = SamplingSessionStore()
        for session_id in ("sess-a", "sess-b", "sess-live"):
            for suffix in range(2):
                store.add(
                    SamplingSessionRecord(
                        sampling_session_id=f"{session_id}:sample:{suffix}",
                        session_id=session_id,
                        fingerprint=f"fp-{session_id}-{suffix}",
                        base_model="test-model",
                    )
                )

        store.remove_for_sessions({"sess-a", "sess-b"})

        assert set(store.records) == {"sess-live:sample:0", "sess-live:sample:1"}
