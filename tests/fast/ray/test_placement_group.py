from __future__ import annotations

import pytest

from miles.utils.placement_group_utils import BundleProbe, PlacementGroupInfo, PlacementGroupSlice


class _FakePlacementGroup:
    """Minimal stand-in for ray.util.placement_group.PlacementGroup."""
    pass


def _make_pg_info(probes: list[BundleProbe]) -> PlacementGroupInfo:
    return PlacementGroupInfo(pg=_FakePlacementGroup(), bundles=probes)


class TestBundleProbe:
    def test_fields(self) -> None:
        probe = BundleProbe(bundle_index=3, node_ip="10.0.0.1", gpu_id="2")
        assert probe.bundle_index == 3
        assert probe.node_ip == "10.0.0.1"
        assert probe.gpu_id == "2"


class TestPlacementGroupInfo:
    def test_reordered_bundle_indices(self) -> None:
        probes = [
            BundleProbe(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleProbe(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleProbe(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
        ]
        info = _make_pg_info(probes)
        assert info.reordered_bundle_indices == [3, 0, 1]

    def test_reordered_gpu_ids(self) -> None:
        probes = [
            BundleProbe(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleProbe(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleProbe(bundle_index=1, node_ip="10.0.0.2", gpu_id="0"),
        ]
        info = _make_pg_info(probes)
        assert info.reordered_gpu_ids == ["0", "1", "0"]

    def test_empty_bundles(self) -> None:
        info = PlacementGroupInfo(pg=_FakePlacementGroup(), bundles=[])
        assert info.reordered_bundle_indices == []
        assert info.reordered_gpu_ids == []


class TestPlacementGroupSlice:
    def _make_six_bundle_info(self) -> PlacementGroupInfo:
        """6 bundles across 3 nodes (2 GPUs each), already sorted."""
        probes = [
            BundleProbe(bundle_index=3, node_ip="10.0.0.1", gpu_id="0"),
            BundleProbe(bundle_index=0, node_ip="10.0.0.1", gpu_id="1"),
            BundleProbe(bundle_index=5, node_ip="10.0.0.2", gpu_id="0"),
            BundleProbe(bundle_index=1, node_ip="10.0.0.2", gpu_id="1"),
            BundleProbe(bundle_index=4, node_ip="10.0.0.3", gpu_id="0"),
            BundleProbe(bundle_index=2, node_ip="10.0.0.3", gpu_id="1"),
        ]
        return _make_pg_info(probes)

    def test_slice_offset_and_count(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=2, count=2)
        assert s.reordered_bundle_indices == [5, 1]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_from_start(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=0, count=4)
        assert s.reordered_bundle_indices == [3, 0, 5, 1]
        assert s.reordered_gpu_ids == ["0", "1", "0", "1"]

    def test_slice_to_end(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=4, count=2)
        assert s.reordered_bundle_indices == [4, 2]
        assert s.reordered_gpu_ids == ["0", "1"]

    def test_slice_full(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=0, count=6)
        assert s.reordered_bundle_indices == info.reordered_bundle_indices
        assert s.reordered_gpu_ids == info.reordered_gpu_ids

    def test_slice_pg_returns_owner_pg(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=0, count=2)
        assert s.pg is info.pg

    def test_slice_reflects_bundle_mutation(self) -> None:
        """Slice sees updated data when owner's bundles are mutated in-place (M2 prep)."""
        info = self._make_six_bundle_info()
        s = info.slice(offset=2, count=2)

        assert s.reordered_bundle_indices == [5, 1]

        # Simulate refresh: replace bundle at rank 2
        info.bundles[2] = BundleProbe(bundle_index=5, node_ip="10.0.0.4", gpu_id="3")
        info.bundles[3] = BundleProbe(bundle_index=1, node_ip="10.0.0.4", gpu_id="1")

        assert s.reordered_bundle_indices == [5, 1]
        assert s.reordered_gpu_ids == ["3", "1"]
        assert s.owner.bundles[2].node_ip == "10.0.0.4"

    def test_multiple_slices_share_owner(self) -> None:
        info = self._make_six_bundle_info()
        actor_slice = info.slice(offset=0, count=2)
        rollout_slice = info.slice(offset=4, count=2)

        assert actor_slice.owner is rollout_slice.owner
        assert actor_slice.pg is rollout_slice.pg

    def test_single_bundle_slice(self) -> None:
        info = self._make_six_bundle_info()
        s = info.slice(offset=3, count=1)
        assert s.reordered_bundle_indices == [1]
        assert s.reordered_gpu_ids == ["1"]
