from __future__ import annotations

import pytest
from tests.fast.utils.workers.reconcile.utils import make_pod, pod_cell

from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, UpsertEvent


def make_store() -> ObjectStore:
    return ObjectStore(key_map=pod_cell)


class TestIncrementalEvents:
    def test_an_upsert_stores_the_object_and_reports_its_parent(self):
        """A plain upsert lands in the store and wakes exactly its cell."""
        store = make_store()
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))

        assert affected == {"cell-a"}
        assert "pod-0" in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-0"]

    def test_a_reparenting_upsert_reports_both_parents(self):
        """Moving an object between cells affects the old and the new one."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-b")))

        assert affected == {"cell-a", "cell-b"}
        assert store.get_by_parent("cell-a") == []

    def test_a_delete_reports_the_stored_parent(self):
        """Deleting a known object affects the cell it belonged to."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=None))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_a_delete_of_an_unknown_object_uses_the_tombstone(self):
        """An unknown delete is attributed through last_obj."""
        store = make_store()
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-a")))

        assert affected == {"cell-a"}

    def test_a_delete_whose_tombstone_disagrees_uses_the_stored_parent(self):
        """A relabel we never saw must not crash the delete nor wake the wrong cell."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-b")))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_an_unmappable_upsert_is_dropped_and_removes_any_stored_object(self):
        """A key_map failure turns the upsert into a departure, not a stale entry."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_an_unmappable_upsert_of_an_unknown_object_affects_nobody(self):
        """Nothing was stored under a parent, so there is no departure to report either."""
        store = make_store()

        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))

        assert affected == set()
        assert "pod-0" not in store


class TestReplace:
    def test_replace_swaps_the_whole_store_and_reports_both_sides(self):
        """A replace applies atomically and names the parents it added to and removed from."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-old", obj=make_pod("pod-old", cell="cell-a")))

        pod_new = make_pod("pod-new", cell="cell-b")
        affected = store.handle_event(ReplaceEvent(objects={"pod-new": pod_new}))

        assert affected == {"cell-a", "cell-b"}
        assert "pod-old" not in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-b")] == ["pod-new"]

    def test_replace_synthesizes_deletions_for_objects_that_vanished(self):
        """Objects missing from a relist must be removed, or ghost members persist forever."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-a")))

        pod_0 = make_pod("pod-0", cell="cell-a")
        affected = store.handle_event(ReplaceEvent(objects={"pod-0": pod_0}))

        assert affected == {"cell-a"}
        assert "pod-1" not in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-0"]

    def test_an_empty_replace_clears_the_store(self):
        """A relist that returns nothing means the pool is gone, not that nothing changed."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))

        affected = store.handle_event(ReplaceEvent(objects={}))

        assert affected == {"cell-a"}
        assert store.parent_keys() == set()

    def test_a_relist_refreshes_a_survivor_whose_parent_did_not_change(self):
        """A survivor still in the listing must be replaced by the newly listed copy, not left stale."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="1")))

        affected = store.handle_event(ReplaceEvent(objects={"pod-0": make_pod("pod-0", resource_version="2")}))

        assert affected == {"cell-a"}
        assert [pod.metadata.resource_version for pod in store.get_by_parent("cell-a")] == ["2"]

    def test_an_unmappable_object_in_a_replace_is_dropped(self):
        """One bad object must not stall the rest of the relist."""
        store = make_store()

        affected = store.handle_event(
            ReplaceEvent(objects={"pod-0": make_pod("pod-0", cell=None), "pod-1": make_pod("pod-1", cell="cell-a")})
        )

        assert affected == {"cell-a"}
        assert "pod-0" not in store
        assert "pod-1" in store


class TestQueries:
    def test_get_by_parent_returns_members_sorted_by_key(self):
        """Membership listing is deterministic."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))

        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-a", "pod-b"]

    def test_parent_keys_lists_every_cell_that_still_has_members(self):
        """parent_keys is what a resync re-drives."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-b")))
        store.handle_event(DeleteEvent(key="pod-b", last_obj=None))

        assert store.parent_keys() == {"cell-a"}


class TestUnknownEvent:
    def test_an_event_outside_the_union_fails_fast(self):
        """A source event the store cannot classify must raise rather than be silently ignored."""
        store = make_store()

        with pytest.raises(AssertionError):
            store.handle_event(object())
