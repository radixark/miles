from __future__ import annotations

import pytest

from miles.utils.placement_group_utils import (
    BundleLocationSnapshot,
    PlacementGroupInfo,
    PlacementGroupSlice,
    RefreshResult,
    _bundle_sort_key,
)


class _FakePlacementGroup:
    """Minimal stand-in for ray.util.placement_group.PlacementGroup."""
    pass


def _make_pg_info(probes: list[BundleLocationSnapshot]) -> PlacementGroupInfo:
    return PlacementGroupInfo(pg=_FakePlacementGroup(), _bundle_location_snapshots=probes)


class TestBundleSortKey:
    def test_sorts_by_ip_then_gpu_id(self) -> None:
        probes = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.2", gpu_id="0"),
        ]
        result = sorted(probes, key=_bundle_sort_key)
        assert [(p.node_ip, p.gpu_id) for p in result] == [
            ("10.0.0.1", "0"),
            ("10.0.0.1", "1"),
            ("10.0.0.2", "0"),
            ("10.0.0.2", "1"),
        ]

    def test_hostname_fallback_to_ascii(self) -> None:
        """Non-resolvable hostnames use ASCII char codes for stable sorting."""
        probes = [
            BundleLocationSnapshot(bundle_index=0, node_ip="node-b", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="node-a", gpu_id="0"),
        ]
        result = sorted(probes, key=_bundle_sort_key)
        assert [p.node_ip for p in result] == ["node-a", "node-b"]

    def test_same_node_sorted_by_gpu_id(self) -> None:
        probes = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.1", gpu_id="2"),
        ]
        result = sorted(probes, key=_bundle_sort_key)
        assert [p.gpu_id for p in result] == ["1", "2", "3"]


class TestBundleLocationSnapshot:
    def test_fields(self) -> None:
        probe = BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="2")
        assert probe.bundle_index == 3
        assert probe.node_ip == "10.0.0.1"
        assert probe.gpu_id == "2"


class TestPlacementGroupInfo:
    def test_reordered_bundle_indices(self) -> None:
        probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
        ]
        info = _make_pg_info(probes)
        assert info.reordered_bundle_indices == [3, 0, 1]

    def test_reordered_gpu_ids(self) -> None:
        probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
        ]
        info = _make_pg_info(probes)
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
        info = _make_pg_info([
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.1", gpu_id="1"),
        ])
        with pytest.raises(ValueError, match="step"):
            info[0:2:2]


class TestPlacementGroupSlice:
    def _make_six_bundle_info(self) -> PlacementGroupInfo:
        """6 bundles across 3 nodes (2 GPUs each), already sorted."""
        probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        return _make_pg_info(probes)

    def test_slice_offset_and_count(self) -> None:
        info = self._make_six_bundle_info()
        s = info[2:4]
        assert s.reordered_bundle_indices == [5, 1]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_from_start(self) -> None:
        info = self._make_six_bundle_info()
        s = info[0:4]
        assert s.reordered_bundle_indices == [3, 0, 5, 1]
        assert s.reordered_gpu_ids == ["0", "1", "0", "1"]

    def test_slice_to_end(self) -> None:
        info = self._make_six_bundle_info()
        s = info[4:6]
        assert s.reordered_bundle_indices == [4, 2]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_full(self) -> None:
        info = self._make_six_bundle_info()
        s = info[0:6]
        assert s.reordered_bundle_indices == info.reordered_bundle_indices
        assert s.reordered_gpu_ids == info.reordered_gpu_ids

    def test_slice_open_end(self) -> None:
        """info[4:] should slice to the end."""
        info = self._make_six_bundle_info()
        s = info[4:]
        assert s.reordered_bundle_indices == [4, 2]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_pg_returns_owner_pg(self) -> None:
        info = self._make_six_bundle_info()
        s = info[0:2]
        assert s.pg is info.pg

    def test_slice_reflects_bundle_mutation(self) -> None:
        """Slice sees updated data when owner's bundles are mutated in-place (M2 prep)."""
        info = self._make_six_bundle_info()
        s = info[2:4]

        assert s.reordered_bundle_indices == [5, 1]

        # Simulate refresh: replace bundle at rank 2
        info._bundle_location_snapshots[2] = BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3")
        info._bundle_location_snapshots[3] = BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1")

        assert s.reordered_bundle_indices == [5, 1]
        assert s.reordered_gpu_ids == ["3", "1"]
        assert s._owner._bundle_location_snapshots[2].node_ip == "10.0.0.4"

    def test_multiple_slices_share_owner(self) -> None:
        info = self._make_six_bundle_info()
        actor_slice = info[0:2]
        rollout_slice = info[4:6]

        assert actor_slice._owner is rollout_slice._owner
        assert actor_slice.pg is rollout_slice.pg

    def test_single_bundle_slice(self) -> None:
        info = self._make_six_bundle_info()
        s = info[3:4]
        assert s.reordered_bundle_indices == [1]
        assert s.reordered_gpu_ids == ["1"]


class TestRefresh:
    """Tests for PlacementGroupInfo._refresh_from_probes (pure algorithm, no Ray)."""

    def _make_six_bundle_info(self) -> PlacementGroupInfo:
        """6 bundles across 3 nodes (2 GPUs each), already sorted."""
        probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        return _make_pg_info(probes)

    def test_no_change_returns_empty_changed_ranks(self) -> None:
        """No node_ip changes -> bundles completely unchanged."""
        info = self._make_six_bundle_info()
        original_snapshots = list(info._bundle_location_snapshots)

        new_probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        result = info._refresh_from_probes(new_probes)

        assert result.changed_ranks == []
        assert info._bundle_location_snapshots == original_snapshots

    def test_partial_node_change_reorders_only_affected_ranks(self) -> None:
        """node_2 replaced by node_4 -> only ranks 2,3 reordered, rest unchanged."""
        info = self._make_six_bundle_info()

        # node_2 (10.0.0.2) replaced by node_4 (10.0.0.4)
        new_probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        result = info._refresh_from_probes(new_probes)

        assert result.changed_ranks == [2, 3]

        # Unchanged ranks 0,1,4,5 keep exact same values
        assert info._bundle_location_snapshots[0] == BundleLocationSnapshot(
            bundle_index=3, node_ip="10.0.0.1", gpu_id="0"
        )
        assert info._bundle_location_snapshots[1] == BundleLocationSnapshot(
            bundle_index=0, node_ip="10.0.0.1", gpu_id="1"
        )
        assert info._bundle_location_snapshots[4] == BundleLocationSnapshot(
            bundle_index=4, node_ip="10.0.0.3", gpu_id="0"
        )
        assert info._bundle_location_snapshots[5] == BundleLocationSnapshot(
            bundle_index=2, node_ip="10.0.0.3", gpu_id="1"
        )

        # Changed ranks 2,3 sorted by (node_ip, gpu_id): gpu_id=1 before gpu_id=3
        assert info._bundle_location_snapshots[2] == BundleLocationSnapshot(
            bundle_index=1, node_ip="10.0.0.4", gpu_id="1"
        )
        assert info._bundle_location_snapshots[3] == BundleLocationSnapshot(
            bundle_index=5, node_ip="10.0.0.4", gpu_id="3"
        )

    def test_all_nodes_changed_degrades_to_full_sort(self) -> None:
        """All nodes replaced -> all ranks reordered (same as initial sort)."""
        info = self._make_six_bundle_info()

        new_probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.7", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.8", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.7", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.9", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.8", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.9", gpu_id="0"),
        ]
        result = info._refresh_from_probes(new_probes)

        assert result.changed_ranks == [0, 1, 2, 3, 4, 5]

        # After full re-sort by (node_ip, gpu_id):
        # 10.0.0.7:gpu0 (bundle 5), 10.0.0.7:gpu1 (bundle 3)
        # 10.0.0.8:gpu0 (bundle 0), 10.0.0.8:gpu1 (bundle 4)
        # 10.0.0.9:gpu0 (bundle 2), 10.0.0.9:gpu1 (bundle 1)
        assert info.reordered_bundle_indices == [5, 3, 0, 4, 2, 1]
        assert info.reordered_gpu_ids == ["0", "1", "0", "1", "0", "1"]

    def test_slice_sees_refresh_changes(self) -> None:
        """Slices auto-reflect refresh updates (shared _bundle_location_snapshots)."""
        info = self._make_six_bundle_info()
        s = info[2:4]
        assert s.reordered_bundle_indices == [5, 1]

        new_probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.4", gpu_id="3"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.4", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        info._refresh_from_probes(new_probes)

        # Slice at ranks [2,3] should see the re-sorted changed bundles
        assert s.reordered_bundle_indices == [1, 5]
        assert s.reordered_gpu_ids == ["1", "3"]

    def test_gpu_id_change_without_node_change_not_treated_as_changed(self) -> None:
        """gpu_id change on the same node_ip is NOT a node replacement."""
        info = self._make_six_bundle_info()

        new_probes = [
            BundleLocationSnapshot(bundle_index=3, node_ip="10.0.0.1", gpu_id="5"),
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="6"),
            BundleLocationSnapshot(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleLocationSnapshot(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        result = info._refresh_from_probes(new_probes)

        assert result.changed_ranks == []

    def test_single_rank_change(self) -> None:
        """Only one bundle changes node -> only that rank is updated."""
        probes = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="0"),
        ]
        info = _make_pg_info(probes)

        new_probes = [
            BundleLocationSnapshot(bundle_index=0, node_ip="10.0.0.1", gpu_id="0"),
            BundleLocationSnapshot(bundle_index=1, node_ip="10.0.0.5", gpu_id="2"),
            BundleLocationSnapshot(bundle_index=2, node_ip="10.0.0.3", gpu_id="0"),
        ]
        result = info._refresh_from_probes(new_probes)

        assert result.changed_ranks == [1]
        assert info._bundle_location_snapshots[1] == BundleLocationSnapshot(
            bundle_index=1, node_ip="10.0.0.5", gpu_id="2"
        )
