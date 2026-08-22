import hashlib
import json
import subprocess
import sys
import textwrap

import pytest

from miles.utils.ft_utils.health_checker import MIN_HEALTH_CHECK_INTERVAL_SECONDS
from miles.utils.workers.rpc.common.protocol import CallStatusResponse, compute_request_digest
from miles.utils.workers.rpc.server import store as store_module


_DIGEST = hashlib.sha256(b"demo-request").digest()
_OTHER_DIGEST = hashlib.sha256(b"other-request").digest()


def _new_store(**kwargs: object) -> store_module.CallStore:
    return store_module.CallStore(**kwargs)


class TestRequestDigest:
    def test_request_digest_is_fixed_length_and_canonical(self) -> None:
        """Equivalent requests share one compact digest while different requests do not."""
        first = compute_request_digest(method_name="demo", query={"b": 2, "a": 1})
        reordered = compute_request_digest(method_name="demo", query={"a": 1, "b": 2})
        different = compute_request_digest(method_name="other", query={"a": 1, "b": 2})

        assert isinstance(first, bytes) and len(first) == hashlib.sha256().digest_size
        assert first == reordered
        assert first != different


class TestAcknowledgement:
    async def test_acknowledgement_releases_the_outcome_but_keeps_a_tombstone(self) -> None:
        """Acknowledgement drops the large outcome while retaining the call identity."""
        store = _new_store()
        store.begin(call_id="c1", fingerprint=_DIGEST)
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result="x" * 1024))

        assert store.stats.unacknowledged_outcome_bytes > 1024
        store.acknowledge(call_id="c1", fingerprint=_DIGEST)

        assert store.stats.unacknowledged_outcome_bytes == 0
        assert store.stats.tombstones == 1
        assert store.contains("c1")
        with pytest.raises(store_module.AcknowledgedCallError):
            await store.wait(call_id="c1", timeout=0.0)

    async def test_acknowledgement_is_idempotent(self) -> None:
        """A lost acknowledgement response can be retried without changing state."""
        store = _new_store()
        store.begin(call_id="c1", fingerprint=_DIGEST)
        store.finish(call_id="c1", outcome=CallStatusResponse(status="failed", error="boom"))

        store.acknowledge(call_id="c1", fingerprint=_DIGEST)
        store.acknowledge(call_id="c1", fingerprint=_DIGEST)

        assert store.stats.tombstones == 1

    async def test_pending_call_cannot_be_acknowledged(self) -> None:
        """A pending call keeps its execution record until it has a terminal outcome."""
        store = _new_store()
        store.begin(call_id="c1", fingerprint=_DIGEST)

        with pytest.raises(store_module.CallNotFinishedError):
            store.acknowledge(call_id="c1", fingerprint=_DIGEST)

    async def test_acknowledged_call_cannot_be_executed_again(self) -> None:
        """A late duplicate submit hits its tombstone and is refused, so the call is never rerun."""
        store = _new_store()
        store.begin(call_id="c1", fingerprint=_DIGEST)
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))
        store.acknowledge(call_id="c1", fingerprint=_DIGEST)

        with pytest.raises(store_module.DuplicateCallError):
            store.begin(call_id="c1", fingerprint=_DIGEST)
        with pytest.raises(store_module.AcknowledgedCallError):
            await store.wait(call_id="c1", timeout=0.0)
        store.acknowledge(call_id="c1", fingerprint=_DIGEST)
        assert store.stats.tombstones == 1
        with pytest.raises(store_module.DuplicateCallError):
            store.begin(call_id="c1", fingerprint=_OTHER_DIGEST)


class TestExpiryScheduling:
    async def test_expiry_purges_at_most_one_batch_per_admission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The batch cap bounds how much expiry work one admission does, and later admissions drain the rest."""
        now = [10.0]
        monkeypatch.setattr(store_module.time, "monotonic", lambda: now[0])
        store = _new_store(finished_ttl_seconds=5.0, expiry_batch_size=2)
        for index in range(5):
            call_id = f"c{index}"
            digest = hashlib.sha256(call_id.encode()).digest()
            store.begin(call_id=call_id, fingerprint=digest)
            store.finish(call_id=call_id, outcome=CallStatusResponse(status="success", result=index))
            store.acknowledge(call_id=call_id, fingerprint=digest)

        assert store.stats.tombstones == 5

        now[0] = 20.0
        store.begin(call_id="first", fingerprint=_DIGEST)

        assert store.stats.tombstones == 3

        store.begin(call_id="second", fingerprint=_OTHER_DIGEST)

        assert store.stats.tombstones == 1


class TestCapacity:
    async def test_active_capacity_is_exact_and_duplicates_do_not_consume_it(self) -> None:
        """A duplicate is refused as a duplicate, not as capacity exhaustion, so it never consumes a slot."""
        store = _new_store(max_active_calls=2)
        for call_id in ("c1", "c2"):
            store.begin(call_id=call_id, fingerprint=_DIGEST)

        with pytest.raises(store_module.DuplicateCallError):
            store.begin(call_id="c1", fingerprint=_DIGEST)
        with pytest.raises(store_module.CallStoreCapacityError, match="active"):
            store.begin(call_id="c3", fingerprint=_DIGEST)
        assert store.stats.active_calls == 2

    async def test_acknowledgement_frees_active_capacity(self) -> None:
        """Acknowledging a terminal outcome lets a new call enter without evicting its tombstone."""
        store = _new_store(max_active_calls=1, max_tombstones=2)
        store.begin(call_id="c1", fingerprint=_DIGEST)
        store.finish(call_id="c1", outcome=CallStatusResponse(status="success", result=1))

        with pytest.raises(store_module.CallStoreCapacityError):
            store.begin(call_id="c2", fingerprint=_DIGEST)
        store.acknowledge(call_id="c1", fingerprint=_DIGEST)

        store.begin(call_id="c2", fingerprint=_OTHER_DIGEST)
        assert store.contains("c1")

    async def test_outcome_capacity_is_reserved_before_admission(self) -> None:
        """A call that cannot reserve its declared result budget is rejected before execution starts."""
        store = _new_store(max_unacknowledged_outcome_bytes=128)
        store.begin(call_id="c1", fingerprint=_DIGEST, outcome_reservation_bytes=96)

        with pytest.raises(store_module.CallStoreCapacityError, match="outcome"):
            store.begin(call_id="c2", fingerprint=_OTHER_DIGEST, outcome_reservation_bytes=64)

        expected = CallStatusResponse(status="success", result="preserved")
        store.finish(call_id="c1", outcome=expected)
        assert await store.wait(call_id="c1", timeout=0.0) == expected
        assert store.stats.reserved_outcome_bytes == 96

    async def test_no_ack_flood_stops_at_the_exact_default_active_cap(self) -> None:
        """Finished calls without ACK cannot grow beyond the configured active-record limit."""
        store = _new_store()
        for index in range(store_module.MAX_ACTIVE_CALLS):
            call_id = f"c{index}"
            store.begin(call_id=call_id, fingerprint=_DIGEST)
            store.finish(call_id=call_id, outcome=CallStatusResponse(status="success"))

        with pytest.raises(store_module.CallStoreCapacityError):
            store.begin(call_id="overflow", fingerprint=_DIGEST)
        assert store.stats.active_calls == store_module.MAX_ACTIVE_CALLS == 4096

    async def test_tombstone_capacity_rejects_new_ids_without_evicting_live_tombstones(self) -> None:
        """A full tombstone budget refuses a duplicate as a duplicate, refuses a new id as capacity, and still ACKs."""
        store = _new_store(max_active_calls=2, max_tombstones=2)
        for call_id in ("c1", "c2"):
            store.begin(call_id=call_id, fingerprint=_DIGEST)
            store.finish(call_id=call_id, outcome=CallStatusResponse(status="success"))
            store.acknowledge(call_id=call_id, fingerprint=_DIGEST)

        with pytest.raises(store_module.DuplicateCallError):
            store.begin(call_id="c1", fingerprint=_DIGEST)
        store.acknowledge(call_id="c2", fingerprint=_DIGEST)
        with pytest.raises(store_module.CallStoreCapacityError, match="tombstone"):
            store.begin(call_id="c3", fingerprint=_DIGEST)
        assert all(store.contains(call_id) for call_id in ("c1", "c2"))

    async def test_queued_request_bytes_are_reserved_and_released_at_completion(self) -> None:
        """Pending decoded kwargs cannot retain more aggregate request bytes than the configured budget."""
        store = _new_store(max_queued_request_bytes=128)
        store.begin(call_id="c1", fingerprint=_DIGEST, request_reservation_bytes=96)

        with pytest.raises(store_module.CallStoreCapacityError, match="request"):
            store.begin(call_id="c2", fingerprint=_OTHER_DIGEST, request_reservation_bytes=64)

        store.finish(call_id="c1", outcome=CallStatusResponse(status="success"))
        assert store.stats.queued_request_bytes == 0
        store.begin(call_id="c2", fingerprint=_OTHER_DIGEST, request_reservation_bytes=64)

    async def test_failed_executor_start_rolls_back_every_admission_reservation(self) -> None:
        """A submit that never starts execution can atomically release its id and byte reservations."""
        store = _new_store(max_active_calls=1, max_queued_request_bytes=128)
        store.begin(
            call_id="c1",
            fingerprint=_DIGEST,
            request_reservation_bytes=96,
            outcome_reservation_bytes=512,
        )

        store.rollback_admission(call_id="c1", fingerprint=_DIGEST)

        assert store.stats.active_calls == 0
        assert store.stats.queued_request_bytes == 0
        assert store.stats.reserved_outcome_bytes == 0
        store.begin(call_id="c1", fingerprint=_OTHER_DIGEST, request_reservation_bytes=128)

    async def test_control_admission_has_a_bounded_reserve_when_data_calls_are_full(self) -> None:
        """Heartbeat-class calls retain a small independent admission budget under data-plane saturation."""
        store = _new_store(max_active_calls=1, max_control_calls=1)
        store.begin(call_id="data", fingerprint=_DIGEST)
        store.begin(call_id="heartbeat", fingerprint=_OTHER_DIGEST, control_plane=True)

        with pytest.raises(store_module.CallStoreCapacityError, match="control"):
            store.begin(call_id="heartbeat-2", fingerprint=hashlib.sha256(b"third").digest(), control_plane=True)

        assert store.stats.active_calls == 1
        assert store.stats.control_calls == 1

    async def test_data_tombstones_cannot_exhaust_control_admission(self) -> None:
        """A saturated data tombstone budget leaves the bounded control-plane lane usable."""
        store = _new_store(max_tombstones=1, max_control_calls=1)
        store.begin(call_id="data", fingerprint=_DIGEST)
        store.finish(call_id="data", outcome=CallStatusResponse(status="success"))
        store.acknowledge(call_id="data", fingerprint=_DIGEST)

        store.begin(call_id="heartbeat", fingerprint=_OTHER_DIGEST, control_plane=True)
        assert store.stats.control_calls == 1

    async def test_default_control_tombstones_cover_the_full_heartbeat_resolution_horizon(self) -> None:
        """The default control lane retains every 30-second heartbeat identity for the 12-hour TTL."""
        heartbeats_per_horizon = int(store_module.FINISHED_TTL_SECONDS // MIN_HEALTH_CHECK_INTERVAL_SECONDS) + 1
        assert store_module.MAX_CONTROL_TOMBSTONES > heartbeats_per_horizon
        store = _new_store()
        for index in range(heartbeats_per_horizon):
            call_id = f"heartbeat-{index}"
            digest = hashlib.sha256(call_id.encode()).digest()
            store.begin(call_id=call_id, fingerprint=digest, control_plane=True)
            store.finish(call_id=call_id, outcome=CallStatusResponse(status="success"))
            store.acknowledge(call_id=call_id, fingerprint=digest)

        store.begin(call_id="next-heartbeat", fingerprint=_OTHER_DIGEST, control_plane=True)

    async def test_call_ids_are_bounded_by_utf8_bytes_before_storage(self) -> None:
        """One oversized logical id cannot retain unaccounted bytes in active records or tombstones."""
        oversized = "€" * (store_module.MAX_CALL_ID_BYTES // len("€".encode()) + 1)
        store = _new_store()

        with pytest.raises(store_module.CallIdTooLongError):
            store.begin(call_id=oversized, fingerprint=_DIGEST)

        assert store.stats.active_calls == 0
        assert store.stats.tombstones == 0


def _run_resource_probe(body: str) -> dict[str, float]:
    script = textwrap.dedent(
        f"""
        import gc
        import hashlib
        import json
        import os
        import time

        from miles.utils.workers.rpc.common.protocol import CallStatusResponse
        from miles.utils.workers.rpc.server.store import CallStore

        def rss_bytes():
            with open('/proc/self/statm') as f:
                return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')

        gc.collect()
        baseline = rss_bytes()
        store = CallStore()
        {textwrap.indent(textwrap.dedent(body), '        ').strip()}
        gc.collect()
        print(json.dumps({{'rss_delta': rss_bytes() - baseline, **metrics}}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return json.loads(completed.stdout.splitlines()[-1])


class TestResourceBounds:
    def test_reporter_ack_workload_stays_within_sixteen_mebibytes(self) -> None:
        """The 8-reporter 64-cell ACK workload has a bounded resident-memory footprint."""
        metrics = _run_resource_probe(
            """
            begin_durations = []
            ack_durations = []
            for cycle in range(45):
                for reporter in range(8):
                    for cell in range(64):
                        call_id = f'{cycle}-{reporter}-{cell}'
                        digest = hashlib.sha256(call_id.encode()).digest()
                        started = time.perf_counter_ns()
                        store.begin(call_id=call_id, fingerprint=digest)
                        begin_durations.append(time.perf_counter_ns() - started)
                        store.finish(call_id=call_id, outcome=CallStatusResponse(status='success'))
                        started = time.perf_counter_ns()
                        store.acknowledge(call_id=call_id, fingerprint=digest)
                        ack_durations.append(time.perf_counter_ns() - started)
            begin_first = sorted(begin_durations[:10000])
            begin_last = sorted(begin_durations[-10000:])
            ack_first = sorted(ack_durations[:10000])
            ack_last = sorted(ack_durations[-10000:])
            metrics = {
                'calls': 45 * 8 * 64,
                'begin_first_p99_ns': begin_first[9900],
                'begin_last_p99_ns': begin_last[9900],
                'ack_first_p99_ns': ack_first[9900],
                'ack_last_p99_ns': ack_last[9900],
            }
            """
        )

        assert metrics["calls"] == 23040
        assert metrics["rss_delta"] <= 16 * 1024 * 1024
        for operation in ("begin", "ack"):
            first = metrics[f"{operation}_first_p99_ns"]
            last = metrics[f"{operation}_last_p99_ns"]
            assert first <= 500_000 and last <= 500_000
            assert last <= max(first * 2, 1)

    def test_sixty_thousand_tombstones_fit_the_memory_and_latency_budget(self) -> None:
        """Sixty thousand tombstones stay compact and ACK latency remains flat."""
        metrics = _run_resource_probe(
            """
            durations = []
            for index in range(60000):
                call_id = f'c{index}'
                digest = hashlib.sha256(call_id.encode()).digest()
                store.begin(call_id=call_id, fingerprint=digest)
                store.finish(call_id=call_id, outcome=CallStatusResponse(status='success'))
                started = time.perf_counter_ns()
                store.acknowledge(call_id=call_id, fingerprint=digest)
                durations.append(time.perf_counter_ns() - started)
            durations.sort()
            same_digest_rejected = False
            try:
                store.begin(call_id='c0', fingerprint=hashlib.sha256(b'c0').digest())
            except Exception as error:
                same_digest_rejected = type(error).__name__ == 'DuplicateCallError'
            different_digest_rejected = False
            try:
                store.begin(call_id='c1', fingerprint=hashlib.sha256(b'different').digest())
            except Exception as error:
                different_digest_rejected = type(error).__name__ == 'DuplicateCallError'
            acknowledged_poll_rejected = False
            try:
                import asyncio
                asyncio.run(store.wait(call_id='c59999', timeout=0.0))
            except Exception as error:
                acknowledged_poll_rejected = type(error).__name__ == 'AcknowledgedCallError'
            metrics = {
                'p99_ns': durations[int(len(durations) * 0.99)],
                'tombstones': store.stats.tombstones,
                'same_digest_rejected': same_digest_rejected,
                'different_digest_rejected': different_digest_rejected,
                'acknowledged_poll_rejected': acknowledged_poll_rejected,
            }
            del durations
            """
        )

        assert metrics["rss_delta"] <= 24 * 1024 * 1024
        assert metrics["p99_ns"] <= 1_000_000
        assert metrics["tombstones"] == 60000
        assert metrics["same_digest_rejected"]
        assert metrics["different_digest_rejected"]
        assert metrics["acknowledged_poll_rejected"]

    def test_default_no_ack_flood_is_stopped_by_the_active_cap_and_reserves_nothing(self) -> None:
        """With no declared outcome cap nothing is reserved, so the flood is stopped by the active-call cap instead."""
        metrics = _run_resource_probe(
            """
            for index in range(4096):
                call_id = f'c{index}'
                digest = hashlib.sha256(call_id.encode()).digest()
                store.begin(call_id=call_id, fingerprint=digest)
                store.finish(
                    call_id=call_id,
                    outcome=CallStatusResponse(status='success', result='x' * (index % 1024)),
                )
            rejected = False
            try:
                store.begin(call_id='overflow', fingerprint=hashlib.sha256(b'overflow').digest())
            except Exception as error:
                rejected = type(error).__name__ == 'CallStoreCapacityError'
            metrics = {
                'active': store.stats.active_calls,
                'reserved': store.stats.reserved_outcome_bytes,
                'rejected': rejected,
            }
            """
        )

        assert metrics["active"] == 4096
        assert metrics["reserved"] == 0
        assert metrics["rejected"]
        assert metrics["rss_delta"] <= 32 * 1024 * 1024

    def test_a_declared_outcome_cap_saturates_the_aggregate_retention_budget(self) -> None:
        """A declared cap still reserves, so the aggregate outcome budget rejects before the active cap can."""
        metrics = _run_resource_probe(
            """
            reservation = 1024 * 1024
            for index in range(256):
                call_id = f'c{index}'
                store.begin(
                    call_id=call_id,
                    fingerprint=hashlib.sha256(call_id.encode()).digest(),
                    outcome_reservation_bytes=reservation,
                )
            detail = ''
            try:
                store.begin(
                    call_id='overflow',
                    fingerprint=hashlib.sha256(b'overflow').digest(),
                    outcome_reservation_bytes=reservation,
                )
            except Exception as error:
                detail = f'{type(error).__name__}: {error}'
            metrics = {
                'active': store.stats.active_calls,
                'reserved': store.stats.reserved_outcome_bytes,
                'detail': detail,
            }
            """
        )

        assert metrics["active"] == 256
        assert metrics["reserved"] == 256 * 1024 * 1024
        assert metrics["detail"].startswith("CallStoreCapacityError")
        assert "outcome retention capacity" in metrics["detail"]

    def test_default_request_reservations_prevent_a_four_gibibyte_queued_workload(self) -> None:
        """Near-limit requests stop at the aggregate queued-byte cap without retaining their payloads in the store."""
        metrics = _run_resource_probe(
            """
            accepted = 0
            request_bytes = 1024 * 1024 - 1024
            for index in range(4096):
                call_id = f'c{index}'
                try:
                    store.begin(
                        call_id=call_id,
                        fingerprint=hashlib.sha256(call_id.encode()).digest(),
                        request_reservation_bytes=request_bytes,
                    )
                except Exception as error:
                    rejected = type(error).__name__ == 'CallStoreCapacityError'
                    break
                accepted += 1
            metrics = {
                'accepted': accepted,
                'queued_request_bytes': store.stats.queued_request_bytes,
                'rejected': rejected,
            }
            """
        )

        assert metrics["accepted"] < 128
        assert metrics["queued_request_bytes"] <= 64 * 1024 * 1024
        assert metrics["rejected"]
        assert metrics["rss_delta"] <= 8 * 1024 * 1024
