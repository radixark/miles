from __future__ import annotations

import pytest

from miles.utils.placement_group_utils import (
    BundleLocationSnapshot,
    PlacementGroupInfo,
    PlacementGroupSlice,
    _bundle_sort_key,
    _partial_resort_snapshots,
)


class _FakePlacementGroup:
    """Minimal stand-in for ray.util.placement_group.PlacementGroup."""
    pass


def _make_pg_info(snapshots: list[BundleLocationSnapshot]) -> PlacementGroupInfo:
    return PlacementGroupInfo(pg=_FakePlacementGroup(), _bundle_location_snapshots=snapshots)


def _make_six_bundle_snapshots() -> list[BundleLocationSnapshot]:
    """6 bundles across 3 nodes (2 GPUs each), already sorted."""
    return [
        BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
        BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
        BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
        BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
        BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
        BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
    ]


class TestBundleSortKey:
    def test_sorts_by_ip_then_gpu_id(self) -> None:
        snapshots = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.2", gpu_id="0"),
        ]
        result = sorted(snapshots, key=_bundle_sort_key)
        assert [(s.node_ip, s.gpu_id) for s in result] == [
            ("10.0.0.1", "0"),
            ("10.0.0.1", "1"),
            ("10.0.0.2", "0"),
            ("10.0.0.2", "1"),
        ]

    def test_hostname_fallback_to_ascii(self) -> None:
        """Non-resolvable hostnames use ASCII char codes for stable sorting."""
        snapshots = [
            BundleLocationSnapshot(bundle_index=0, node_ip="node-b", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="node-a", gpu_id="0"),
        ]
        result = sorted(snapshots, key=_bundle_sort_key)
        assert [s.node_ip for s in result] == ["node-a", "node-b"]

    def test_same_node_sorted_by_gpu_id(self) -> None:
        snapshots = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.1", gpu_id="2"),
        ]
        result = sorted(snapshots, key=_bundle_sort_key)
        assert [s.gpu_id for s in result] == ["1", "2", "3"]


class TestBundleLocationSnapshot:
    def test_fields(self) -> None:
        s = BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="2")
        assert s.bundle_index == 3
        assert s.node_ip == "10.0.0.1"
        assert s.gpu_id == "2"

    def test_frozen(self) -> None:
        s = BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0")
        with pytest.raises(AttributeError):
            s.node_ip = "10.0.0.2"  # type: ignore[misc]


class TestPlacementGroupInfo:
    def test_reordered_bundle_indices(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots()[:3])
        assert info.reordered_bundle_indices == [3, 0, 5]

    def test_reordered_gpu_ids(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots()[:3])
        assert info.reordered_gpu_ids == ["0", "1", "0"]

    def test_empty_bundles(self) -> None:
        info = PlacementGroupInfo(pg=_FakePlacementGroup(), _bundle_location_snapshots=[])
        assert info.reordered_bundle_indices == []
        assert info.reordered_gpu_ids == []

    def test_getitem_rejects_int_index(self) -> None:
        info = _make_pg_info([BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0")])
        with pytest.raises(TypeError, match="slices"):
            info[0]

    def test_getitem_rejects_step(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots()[:2])
        with pytest.raises(ValueError, match="step"):
            info[0:2:2]


class TestPlacementGroupSlice:
    def test_slice_offset_and_count(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[2:4]
        assert s.reordered_bundle_indices == [5, 1]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_from_start(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[0:4]
        assert s.reordered_bundle_indices == [3, 0, 5, 1]
        assert s.reordered_gpu_ids == ["0", "1", "0", "1"]

    def test_slice_to_end(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[4:6]
        assert s.reordered_bundle_indices == [4, 2]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_full(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[0:6]
        assert s.reordered_bundle_indices == info.reordered_bundle_indices
        assert s.reordered_gpu_ids == info.reordered_gpu_ids

    def test_slice_open_end(self) -> None:
        """info[4:] should slice to the end."""
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[4:]
        assert s.reordered_bundle_indices == [4, 2]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_pg_returns_owner_pg(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[0:2]
        assert s.pg is info.pg

    def test_slice_reflects_snapshot_replacement(self) -> None:
        """Slice sees updated data when owner's snapshot list is replaced (refresh)."""
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[2:4]
        assert s.reordered_bundle_indices == [5, 1]

        info._bundle_location_snapshots = _partial_resort_snapshots(
            info._bundle_location_snapshots,
            [
                BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
                BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
                BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
                BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
                BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
                BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
            ],
        )

        assert s.reordered_bundle_indices == [1, 5]
        assert s.reordered_gpu_ids == ["1", "3"]

    def test_multiple_slices_share_owner(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        actor_slice = info[0:2]
        rollout_slice = info[4:6]

        assert actor_slice._owner is rollout_slice._owner
        assert actor_slice.pg is rollout_slice.pg

    def test_single_bundle_slice(self) -> None:
        info = _make_pg_info(_make_six_bundle_snapshots())
        s = info[3:4]
        assert s.reordered_bundle_indices == [1]
        assert s.reordered_gpu_ids == ["1"]


class TestPartialResortSnapshots:
    """Tests for _partial_resort_snapshots (pure function, no Ray)."""

    def test_no_change_preserves_snapshots(self) -> None:
        """No node_ip changes -> output equals input."""
        old = _make_six_bundle_snapshots()
        new = list(old)
        result = _partial_resort_snapshots(old, new)
        assert result == old

    def test_does_not_mutate_inputs(self) -> None:
        old = _make_six_bundle_snapshots()
        old_copy = list(old)
        new = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        new_copy = list(new)
        _partial_resort_snapshots(old, new)
        assert old == old_copy
        assert new == new_copy

    def test_partial_node_change_reorders_only_affected_ranks(self) -> None:
        """node_2 replaced by node_4 -> only ranks 2,3 reordered, rest unchanged."""
        old = _make_six_bundle_snapshots()
        new = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        result = _partial_resort_snapshots(old, new)

        assert result == [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]

    def test_all_nodes_changed_degrades_to_full_sort(self) -> None:
        """All nodes replaced -> all ranks reordered (same as initial sort)."""
        old = _make_six_bundle_snapshots()
        new = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.7", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.8", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.7", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.9", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.8", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.9", gpu_id="0"),
        ]
        result = _partial_resort_snapshots(old, new)

        assert result == [
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.7", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.7", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.8", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.8", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.9", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.9", gpu_id="1"),
        ]

    def test_two_nodes_changed(self) -> None:
        """2 of 3 nodes replaced -> 4 changed ranks re-sorted, 2 unchanged."""
        old = _make_six_bundle_snapshots()
        # node_2 (10.0.0.2) -> node_4 (10.0.0.4), node_3 (10.0.0.3) -> node_5 (10.0.0.5)
        new = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.5", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.5", gpu_id="1"),
        ]
        result = _partial_resort_snapshots(old, new)

        assert result == [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.5", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.5", gpu_id="1"),
        ]

    def test_gpu_id_change_without_node_change_not_treated_as_changed(self) -> None:
        """gpu_id change on the same node_ip is NOT a node replacement."""
        old = _make_six_bundle_snapshots()
        new = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="5"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="6"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        result = _partial_resort_snapshots(old, new)
        assert result == old

    def test_single_rank_change(self) -> None:
        """Only one bundle changes node -> only that rank is updated."""
        old = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="0"),
        ]
        new = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.5", gpu_id="2"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="0"),
        ]
        result = _partial_resort_snapshots(old, new)

        assert result == [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.5", gpu_id="2"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="0"),
        ]
