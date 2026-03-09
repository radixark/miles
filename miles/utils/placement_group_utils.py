from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleLocationSnapshot:
    bundle_index: int
    node_ip: str
    gpu_id: str


def _bundle_sort_key(snapshot: BundleLocationSnapshot) -> tuple[list[int], str]:
    node_identifier = snapshot.node_ip
    try:
        node_ip_parts = list(map(int, node_identifier.split(".")))
    except ValueError:
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, snapshot.gpu_id)


def _partial_resort_snapshots(
    old_snapshots: list[BundleLocationSnapshot],
    new_snapshots: list[BundleLocationSnapshot],
) -> list[BundleLocationSnapshot]:
    """Detect changed ranks and partially re-sort.

    Unchanged ranks keep their exact (bundle_index, node_ip, gpu_id).
    Changed ranks (node_ip differs) are re-sorted among themselves.

    Returns a new list — does not mutate inputs.
    """
    new_by_index = {s.bundle_index: s for s in new_snapshots}

    changed_ranks = [
        rank for rank, old in enumerate(old_snapshots)
        if new_by_index[old.bundle_index].node_ip != old.node_ip
    ]

    if not changed_ranks:
        return list(old_snapshots)

    changed_sorted = sorted(
        [new_by_index[old_snapshots[r].bundle_index] for r in changed_ranks],
        key=_bundle_sort_key,
    )

    result = list(old_snapshots)
    for rank, snapshot in zip(changed_ranks, changed_sorted):
        result[rank] = snapshot

    return result


@dataclass
class PlacementGroupInfo:
    pg: PlacementGroup
    _bundle_location_snapshots: list[BundleLocationSnapshot] = field(default_factory=list)

    @property
    def reordered_bundle_indices(self) -> list[int]:
        return [b.bundle_index for b in self._bundle_location_snapshots]

    @property
    def reordered_gpu_ids(self) -> list[str]:
        return [b.gpu_id for b in self._bundle_location_snapshots]

    def __getitem__(self, key: slice) -> PlacementGroupSlice:
        if not isinstance(key, slice):
            raise TypeError(f"PlacementGroupInfo indices must be slices, not {type(key).__name__}")
        start, stop, step = key.indices(len(self._bundle_location_snapshots))
        if step != 1:
            raise ValueError("PlacementGroupInfo does not support step in slicing")
        return PlacementGroupSlice(_owner=self, _offset=start, _count=stop - start)

    def refresh(self) -> None:
        """Re-snapshot all bundles and partially re-sort only changed ranks."""
        new_snapshots = _snapshot_bundles(self.pg, num_bundles=len(self._bundle_location_snapshots))
        self._bundle_location_snapshots = _partial_resort_snapshots(
            self._bundle_location_snapshots, new_snapshots,
        )


@dataclass
class PlacementGroupSlice:
    _owner: PlacementGroupInfo
    _offset: int
    _count: int

    @property
    def pg(self) -> PlacementGroup:
        return self._owner.pg

    @property
    def reordered_bundle_indices(self) -> list[int]:
        return self._owner.reordered_bundle_indices[self._offset : self._offset + self._count]

    @property
    def reordered_gpu_ids(self) -> list[str]:
        return self._owner.reordered_gpu_ids[self._offset : self._offset + self._count]

    def refresh(self) -> None:
        """Delegate to owner: re-snapshot all bundles and partially re-sort changed ranks."""
        self._owner.refresh()


@ray.remote(num_gpus=0.001)
class _InfoActor:
    def get_ip_and_gpu_id(self) -> tuple[str, str]:
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _snapshot_bundles(pg: PlacementGroup, num_bundles: int) -> list[BundleLocationSnapshot]:
    """Snapshot all bundles in a PG to discover current (node_ip, gpu_id) mappings."""
    info_actors = [
        _InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote()
        for i in range(num_bundles)
    ]

    results = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    return [
        BundleLocationSnapshot(bundle_index=i, node_ip=results[i][0], gpu_id=results[i][1])
        for i in range(num_bundles)
    ]


def create_placement_group_info(num_gpus: int) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs, snapshot and sort bundles."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")

    ray.get(pg.ready())

    snapshots = _snapshot_bundles(pg, num_bundles=num_gpus)
    sorted_snapshots = sorted(snapshots, key=_bundle_sort_key)

    for rank, snapshot in enumerate(sorted_snapshots):
        logger.info(
            f"  bundle {rank:4}, actual_bundle_index: {snapshot.bundle_index:4}, "
            f"node: {snapshot.node_ip}, gpu: {snapshot.gpu_id}"
        )

    return PlacementGroupInfo(pg=pg, _bundle_location_snapshots=sorted_snapshots)
