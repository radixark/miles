from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.fast.utils.workers.reconcile.utils import (
    FakeSource,
    StreamEnd,
    StreamError,
    make_pod,
    pod_cell,
    replace_of,
    settle,
)

from miles.utils.test_utils.clock import FakeClock

from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, UpsertEvent


class Recorder:
    def __init__(self) -> None:
        self.parent_keys: list[str] = []
        self.snapshots: list[list[str]] = []
        self.concurrency = 0
        self.max_concurrency = 0
        self.gate: asyncio.Event | None = None
        self.fail_parent_keys: set[str] = set()
        self.loop: ReconcileLoop | None = None

    async def __call__(self, parent_key: str) -> None:
        self.concurrency += 1
        self.max_concurrency = max(self.max_concurrency, self.concurrency)
        try:
            self.parent_keys.append(parent_key)
            if self.loop is not None:
                self.snapshots.append([pod.metadata.name for pod in self.loop.get_by_parent(parent_key)])
            if self.gate is not None:
                await self.gate.wait()
            if parent_key in self.fail_parent_keys:
                raise RuntimeError(f"reconcile failed for {parent_key}")
        finally:
            self.concurrency -= 1

    def counts(self) -> dict[str, int]:
        return {parent_key: self.parent_keys.count(parent_key) for parent_key in set(self.parent_keys)}


async def make_loop(
    *,
    source: FakeSource | None = None,
    recorder: Recorder | None = None,
    key_map: Any = pod_cell,
    clock: FakeClock | None = None,
    start: bool = True,
    initial: list[Any] | None = None,
    fail_parent_keys: set[str] | None = None,
    gated: bool = False,
    **kwargs: Any,
) -> tuple[ReconcileLoop, FakeSource, Recorder, FakeClock]:
    source = source or FakeSource()
    recorder = recorder or Recorder()
    clock = clock or FakeClock()
    loop = ReconcileLoop(source=source, reconcile=recorder, key_map=key_map, clock=clock, **kwargs)
    recorder.loop = loop
    recorder.fail_parent_keys = fail_parent_keys or set()
    if gated:
        recorder.gate = asyncio.Event()
    if start:
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(*(initial or [])))
        await settle()
        assert start_task.done()
        await start_task
    return loop, source, recorder, clock


class TestInitialSync:
    async def test_start_blocks_until_the_initial_replace(self):
        """start() returns only once the initial LIST has landed in the store."""
        source = FakeSource()
        loop = ReconcileLoop(source=source, reconcile=Recorder(), clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        assert start_task.done()
        await loop.stop()

    async def test_no_reconcile_before_the_initial_replace(self):
        """No reconcile is dispatched while the initial LIST is still outstanding."""
        source = FakeSource()
        recorder = Recorder()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert recorder.parent_keys == []

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_initial_objects_reconcile_once_per_parent_key(self):
        """Eight pods of one cell collapse into a single reconcile of that cell."""
        pods = [make_pod(f"pod-{i}", cell="cell-a") for i in range(8)]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_initial_objects_reconcile_each_distinct_parent(self):
        """Distinct cells each get exactly one initial reconcile."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_store_is_populated_before_first_reconcile(self):
        """The first reconcile already sees the full initial LIST in the store."""
        pods = [make_pod(f"pod-{i}", cell="cell-a") for i in range(3)]
        loop, _, recorder, _ = await make_loop(initial=pods)
        assert recorder.snapshots == [["pod-0", "pod-1", "pod-2"]]
        await loop.stop()

    async def test_without_key_map_object_key_is_the_reconcile_key(self):
        """key_map=None means the object key is used verbatim as the reconcile key."""
        loop, _, recorder, _ = await make_loop(key_map=None, initial=[make_pod("pod-0"), make_pod("pod-1")])
        assert sorted(recorder.parent_keys) == ["pod-0", "pod-1"]
        await loop.stop()

    async def test_start_retries_when_the_first_stream_fails(self):
        """A source that fails before delivering its initial LIST is reopened after the retry delay."""
        source = FakeSource(fail_opens=1)
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=5.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()
        assert source.open_count == 1

        await clock.elapse(5.0)
        await settle()
        assert source.open_count == 2

        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_a_stream_ending_before_its_replace_is_retried(self):
        """A stream that dies before delivering the initial LIST leaves the barrier closed and nothing cached."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=2.0)
        recorder.loop = loop
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(StreamEnd())
        await settle()
        assert not start_task.done()
        assert recorder.parent_keys == []

        await clock.elapse(2.0)
        await settle()
        source.emit(replace_of(make_pod("pod-fresh")))
        await settle()
        await start_task

        assert recorder.parent_keys == ["cell-a"]
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-fresh"]
        assert source.open_count == 2
        await loop.stop()

    async def test_source_factory_raising_is_retried(self):
        """A source callable that raises synchronously is retried after the delay."""
        source = FakeSource(fail_calls=1)
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=3.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert source.open_count == 1

        await clock.elapse(2.9)
        await settle()
        assert source.open_count == 1

        await clock.elapse(0.1)
        await settle()
        assert source.open_count == 2
        source.emit(replace_of())
        await settle()
        await start_task
        await loop.stop()

    async def test_store_is_updated_before_the_key_is_enqueued(self):
        """The invariant that a handler never reads a world older than the event it woke on."""
        loop, source, _, _ = await make_loop(initial=[make_pod("pod-0")])
        seen_at_enqueue: list[list[str]] = []
        original_add = loop._queue.add

        def recording_add(key: str) -> None:
            seen_at_enqueue.append([pod.metadata.name for pod in loop.get_by_parent(key)])
            original_add(key)

        loop._queue.add = recording_add
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        assert seen_at_enqueue == [["pod-0", "pod-1"]]

        source.emit(DeleteEvent(key="pod-1", last_obj=make_pod("pod-1")))
        await settle()
        assert seen_at_enqueue[-1] == ["pod-0"]
        await loop.stop()

    async def test_empty_initial_list_reconciles_nothing(self):
        """An empty cluster produces no reconcile calls."""
        loop, _, recorder, _ = await make_loop(initial=[])
        assert recorder.parent_keys == []
        await loop.stop()


class TestIncrementalEvents:
    async def test_upsert_enqueues_parent_key(self):
        """A new pod wakes its cell."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_removes_from_store_and_wakes_parent(self):
        """A deleted pod leaves the store and wakes its cell."""
        pod = make_pod("pod-0", cell="cell-a")
        loop, source, recorder, _ = await make_loop(initial=[pod])
        source.emit(DeleteEvent(key="pod-0", last_obj=pod))
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-a"]
        assert loop.get_by_parent("cell-a") == []
        await loop.stop()

    async def test_known_delete_trusts_the_store_over_a_stale_tombstone(self):
        """For a known object the stored copy decides the parent, and the store entry is really dropped."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a")])
        recorder.parent_keys.clear()
        source.emit(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-stale")))
        await settle()

        assert recorder.parent_keys == ["cell-a"]

        source.emit(replace_of())
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_of_unknown_key_falls_back_to_tombstone(self):
        """A delete for an object never seen still wakes the cell via its tombstone."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-9", last_obj=make_pod("pod-9", cell="cell-z")))
        await settle()
        assert recorder.parent_keys == ["cell-z"]
        await loop.stop()

    async def test_delete_without_tombstone_is_ignored(self):
        """A delete with no last_obj for an unknown key cannot be attributed and is dropped."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-9", last_obj=None))
        await settle()
        assert recorder.parent_keys == []

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_delete_with_an_unmappable_tombstone_is_dropped(self):
        """An unknown delete whose tombstone has no parent is discarded, not queued under its own key."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(DeleteEvent(key="pod-orphan", last_obj=make_pod("pod-orphan", cell=None)))
        await settle()

        assert recorder.parent_keys == []
        assert "pod-orphan" not in loop._store

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_reparenting_upsert_wakes_old_and_new_parent(self):
        """A pod whose cell label changed wakes both the old and the new cell."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-b")))
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-b")] == ["pod-0"]
        await loop.stop()

    async def test_repeated_upsert_keeps_one_store_entry(self):
        """Re-delivery of the same object does not duplicate store membership."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", resource_version="1")])
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="2")))
        await settle()
        pods = loop.get_by_parent("cell-a")
        assert len(pods) == 1
        assert pods[0].metadata.resource_version == "2"
        await loop.stop()

    async def test_reconcile_reads_store_snapshot_not_event_payload(self):
        """Reconcile observes the store, so a burst of pods folds into one complete view."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(
            UpsertEvent(key="pod-0", obj=make_pod("pod-0")),
            UpsertEvent(key="pod-1", obj=make_pod("pod-1")),
            UpsertEvent(key="pod-2", obj=make_pod("pod-2")),
        )
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        assert recorder.snapshots[-1] == ["pod-0", "pod-1", "pod-2"]
        await loop.stop()

    async def test_an_unmappable_object_is_dropped_not_stored(self):
        """A key_map that raises drops that one object; the stream keeps running."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell=None)))
        await settle()

        assert "pod-1" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert source.open_count == 1

        source.emit(UpsertEvent(key="pod-2", obj=make_pod("pod-2")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_an_object_that_becomes_unmappable_leaves_its_parent(self):
        """Losing the label is observed as the object leaving the cell."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))
        await settle()

        assert recorder.parent_keys == ["cell-a"]
        assert loop.get_by_parent("cell-a") == []
        assert "pod-0" not in loop._store
        await loop.stop()


class TestWorkQueueDiscipline:
    async def test_a_busy_key_is_never_reconciled_twice_at_once(self):
        """The single worker serializes everything, so a key can never overlap itself."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")], gated=True)
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        source.emit(UpsertEvent(key="pod-2", obj=make_pod("pod-2", cell="cell-b")))
        await settle()
        assert recorder.max_concurrency == 1

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert recorder.max_concurrency == 1
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_repeated_events_while_busy_collapse_into_one_requeue(self):
        """Many events for a busy key produce exactly one follow-up reconcile."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")])
        recorder.parent_keys.clear()
        recorder.gate = asyncio.Event()
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1")))
        await settle()
        for index in range(5):
            source.emit(UpsertEvent(key=f"pod-extra-{index}", obj=make_pod(f"pod-extra-{index}")))
            await settle()

        recorder.gate.set()
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-a"]
        await loop.stop()

    async def test_distinct_keys_are_serialized_by_default(self):
        """The default single worker mirrors controller-runtime's MaxConcurrentReconciles=1."""
        pods = [make_pod("pod-a", cell="cell-a"), make_pod("pod-b", cell="cell-b")]
        loop, _, recorder, _ = await make_loop(initial=pods, gated=True)
        assert recorder.max_concurrency == 1

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_events_before_worker_pickup_collapse(self):
        """Two events for the same key arriving back-to-back yield one reconcile."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(
            UpsertEvent(key="pod-0", obj=make_pod("pod-0")),
            UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="2")),
        )
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_keys_are_dispatched_in_arrival_order(self):
        """The queue is FIFO, not a stack."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")], gated=True)
        source.emit(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-b")))
        await settle()
        source.emit(UpsertEvent(key="pod-c", obj=make_pod("pod-c", cell="cell-c")))
        await settle()

        assert recorder.gate is not None
        recorder.gate.set()
        await settle()
        assert recorder.parent_keys == ["cell-a", "cell-b", "cell-c"]
        await loop.stop()


class TestFailures:
    async def test_a_failing_reconcile_does_not_kill_the_worker(self):
        """A reconcile that raises is logged and the next key is still served."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"})
        source.emit(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-b")))
        await settle()

        assert recorder.counts() == {"cell-a": 1, "cell-b": 1}
        await loop.stop()

    async def test_a_failing_key_is_not_retried_on_its_own(self):
        """Without a retry policy a failure costs one reconcile, not a redelivery."""
        loop, _, recorder, _ = await make_loop(initial=[make_pod("pod-0")], fail_parent_keys={"cell-a"})
        await settle()

        assert recorder.counts() == {"cell-a": 1}
        await loop.stop()


class TestLister:
    async def test_get_by_parent_returns_members_sorted_by_key(self):
        """get_by_parent is the Lister equivalent and is deterministic."""
        pods = [make_pod("pod-2"), make_pod("pod-0"), make_pod("pod-1")]
        loop, _, _, _ = await make_loop(initial=pods)
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0", "pod-1", "pod-2"]
        await loop.stop()

    async def test_get_by_parent_unknown_key_is_empty(self):
        """Unknown parents read as empty, not as an error."""
        loop, _, _, _ = await make_loop(initial=[])
        assert loop.get_by_parent("nope") == []
        await loop.stop()


class TestStop:
    async def test_stop_halts_reconciles(self):
        """After stop() no further events are processed."""
        loop, source, recorder, _ = await make_loop(initial=[])
        await loop.stop()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == []

    async def test_stop_closes_the_source_stream(self):
        """Cancelling the driver closes the stream it was iterating."""
        loop, source, _, _ = await make_loop(initial=[make_pod("pod-0")])
        assert source.closed_count == 0

        await loop.stop()
        assert source.closed_count == 1

    async def test_a_second_stop_is_rejected(self):
        """stop() runs exactly once; a second call is a caller error, not a wait."""
        loop, _, _, _ = await make_loop(initial=[])
        await loop.stop()
        with pytest.raises(AssertionError):
            await loop.stop()

    async def test_stop_before_start_is_rejected(self):
        """stop() has nothing to wait for before start(); calling it early is a caller error."""
        loop = ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock())
        with pytest.raises(AssertionError):
            await loop.stop()

    async def test_start_twice_is_rejected(self):
        """start() is single-shot."""
        loop, _, _, _ = await make_loop(initial=[make_pod("pod-0")])
        with pytest.raises(AssertionError):
            await loop.start()
        await loop.stop()

    async def test_awaiting_stop_inside_reconcile_is_rejected(self):
        """stop() waits for the worker, so awaiting it from inside reconcile is a caller error, not a hang."""
        source = FakeSource()
        rejected: list[BaseException] = []
        loop: ReconcileLoop | None = None

        async def reconcile(key: str) -> None:
            assert loop is not None
            try:
                await loop.stop()
            except AssertionError as error:
                rejected.append(error)

        loop = ReconcileLoop(source=source, reconcile=reconcile, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task

        assert len(rejected) == 1
        await loop.stop()

    async def test_a_reconcile_shuts_the_loop_down_by_spawning_a_task(self):
        """The sanctioned way for reconcile to request shutdown is a task that outlives the worker."""
        source = FakeSource()
        loop: ReconcileLoop | None = None
        shutdown: asyncio.Task[None] | None = None

        async def reconcile(key: str) -> None:
            nonlocal shutdown
            assert loop is not None
            if shutdown is None:
                shutdown = asyncio.create_task(loop.stop())

        loop = ReconcileLoop(source=source, reconcile=reconcile, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        await start_task

        assert shutdown is not None
        await shutdown
        assert source.closed_count == 1


class TestInStreamRelist:
    async def test_relist_replaces_the_store(self):
        """A ReplaceEvent mid-stream swaps the whole world, not just the keys it mentions."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of(pods[0]))
        await settle()
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert loop.get_by_parent("cell-b") == []
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_relist_to_empty_deletes_everything_and_wakes_every_parent(self):
        """A relist that returns nothing must retire every known object."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of())
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert loop.get_by_parent("cell-b") == []
        await loop.stop()

    async def test_relist_refreshes_and_reparents_a_survivor(self):
        """A relisted object that moved cells updates both the store copy and both parents."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0", cell="cell-a", resource_version="1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-0", cell="cell-b", resource_version="2")))
        await settle()
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b"]
        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.resource_version for pod in loop.get_by_parent("cell-b")] == ["2"]
        await loop.stop()

    async def test_incremental_events_resume_after_a_relist(self):
        """The stream returns to incremental mode once the relist has been applied."""
        loop, source, recorder, _ = await make_loop(initial=[])
        source.emit(replace_of())
        await settle()
        recorder.parent_keys.clear()

        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()

    async def test_a_relist_skips_unmappable_objects_and_applies_the_rest(self):
        """One poison object must not stop the whole pool from being updated."""
        pods = [make_pod("pod-a", cell="cell-a"), make_pod("pod-b", cell="cell-b")]
        loop, source, recorder, _ = await make_loop(initial=pods)
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-a", cell="cell-c"), make_pod("pod-bad", cell=None)))
        await settle()

        assert source.open_count == 1
        assert "pod-bad" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-c")] == ["pod-a"]
        assert loop.get_by_parent("cell-a") == []
        assert loop.get_by_parent("cell-b") == []
        assert sorted(recorder.parent_keys) == ["cell-a", "cell-b", "cell-c"]
        await loop.stop()

    async def test_a_relist_removes_a_known_object_that_becomes_unmappable(self):
        """An object still listed but no longer attributable must leave the store, not linger."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0"), make_pod("pod-1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-0", cell=None), make_pod("pod-1")))
        await settle()

        assert "pod-0" not in loop._store
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-1"]
        await loop.stop()

    async def test_consecutive_relists_replace_overlapping_then_disjoint_sets(self):
        """Two relists in a row apply a survivor plus a newcomer, then a completely new world."""
        loop, source, recorder, _ = await make_loop(initial=[make_pod("pod-0"), make_pod("pod-1")])
        recorder.parent_keys.clear()

        source.emit(replace_of(make_pod("pod-1"), make_pod("pod-2")))
        await settle()
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-1", "pod-2"]

        source.emit(replace_of(make_pod("pod-3", cell="cell-b")))
        await settle()

        assert loop.get_by_parent("cell-a") == []
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-b")] == ["pod-3"]
        assert recorder.parent_keys == ["cell-a", "cell-a", "cell-b"]
        await loop.stop()

    async def test_stream_not_opening_with_replace_is_retried(self):
        """A stream whose first event is not ReplaceEvent cannot be trusted to hold the whole world."""
        source = FakeSource()
        recorder = Recorder()
        clock = FakeClock()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=clock, source_retry_delay=1.0)
        start_task = asyncio.create_task(loop.start())
        await settle()
        source.emit(UpsertEvent(key="pod-0", obj=make_pod("pod-0")))
        await settle()
        assert not start_task.done()

        await clock.elapse(1.0)
        await settle()
        assert source.open_count == 2
        source.emit(replace_of())
        await settle()
        await start_task
        await loop.stop()


class TestRelistReplace:
    async def test_reopened_stream_deletes_objects_missing_from_the_new_list(self):
        """A relist that drops a pod synthesizes its deletion (no ghost objects)."""
        pods = [make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")]
        loop, source, recorder, clock = await make_loop(initial=pods, source_retry_delay=1.0)
        recorder.parent_keys.clear()

        source.emit(StreamError(RuntimeError("connection reset")))
        await settle()
        await clock.elapse(1.0)
        await settle()
        assert source.open_count == 2

        source.emit(replace_of(pods[0]))
        await settle()
        assert loop.get_by_parent("cell-b") == []
        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        await loop.stop()

    async def test_reopened_stream_rewakes_surviving_objects(self):
        """Surviving objects are reconciled again after a relist (at-least-once)."""
        pod = make_pod("pod-0", cell="cell-a")
        loop, source, recorder, clock = await make_loop(initial=[pod], source_retry_delay=1.0)
        recorder.parent_keys.clear()

        source.emit(StreamEnd())
        await settle()
        await clock.elapse(1.0)
        await settle()
        source.emit(replace_of(pod))
        await settle()
        assert recorder.parent_keys == ["cell-a"]
        assert [p.metadata.name for p in loop.get_by_parent("cell-a")] == ["pod-0"]
        await loop.stop()

    async def test_stream_error_closes_the_old_stream(self):
        """The failed stream is closed before a new one is opened."""
        loop, source, _, clock = await make_loop(initial=[], source_retry_delay=1.0)
        source.emit(StreamError(RuntimeError("boom")))
        await settle()
        await clock.elapse(1.0)
        await settle()
        assert source.closed_count == 1
        assert source.open_count == 2
        await loop.stop()

    async def test_relist_carries_objects_added_while_disconnected(self):
        """Objects created during the outage appear after the relist."""
        loop, source, recorder, clock = await make_loop(initial=[], source_retry_delay=1.0)
        source.emit(StreamEnd())
        await settle()
        await clock.elapse(1.0)
        await settle()
        source.emit(replace_of(make_pod("pod-new", cell="cell-new")))
        await settle()
        assert recorder.parent_keys == ["cell-new"]
        await loop.stop()

    async def test_cancelling_start_aborts_the_initial_sync(self):
        """A start() hung on the initial LIST is aborted by cancelling its task, which closes the source."""
        source = FakeSource()
        recorder = Recorder()
        loop = ReconcileLoop(source=source, reconcile=recorder, key_map=pod_cell, clock=FakeClock())
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()

        start_task.cancel()
        await asyncio.gather(start_task, return_exceptions=True)
        assert source.closed_count == 1
        await settle()

        assert start_task.cancelled()
        assert source.closed_count == 1
        source.emit(replace_of(make_pod("pod-0")))
        await settle()
        assert recorder.parent_keys == []

    async def test_a_non_positive_source_retry_delay_is_rejected(self):
        """A zero delay would turn the reopen loop into a hot loop, so it never reaches construction."""
        with pytest.raises(AssertionError):
            ReconcileLoop(source=FakeSource(), reconcile=Recorder(), clock=FakeClock(), source_retry_delay=0.0)
