from __future__ import annotations

import json
import logging
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    """Keep bundles at positions where (node_ip, gpu_id) didn't change, re-sort the rest.

    Works for both same-PG refresh (bundles moved after node failure) and
    cross-PG restart (entirely new PG, old snapshots from backup).
    """
    new_by_location: dict[tuple[str, str], BundleLocationSnapshot] = {
        (s.node_ip, s.gpu_id): s for s in new_snapshots
    }

    result: list[BundleLocationSnapshot | None] = [None] * len(old_snapshots)
    used: set[tuple[str, str]] = set()

    for rank, old in enumerate(old_snapshots):
        key = (old.node_ip, old.gpu_id)
        if key in new_by_location:
            result[rank] = new_by_location[key]
            used.add(key)

    unmatched = sorted(
        [s for s in new_snapshots if (s.node_ip, s.gpu_id) not in used],
        key=_bundle_sort_key,
    )
    empty_ranks = [r for r in range(len(result)) if result[r] is None]
    for rank, snapshot in zip(empty_ranks, unmatched):
        result[rank] = snapshot

    return result  # type: ignore[return-value]


def _save_snapshots(path: Path, snapshots: list[BundleLocationSnapshot]) -> None:
    data = [asdict(s) for s in snapshots]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def _load_snapshots(path: Path) -> list[BundleLocationSnapshot] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return [BundleLocationSnapshot(**item) for item in data]
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        logger.warning("Failed to load PG snapshot from %s, ignoring", path, exc_info=True)
        return None


@dataclass
class PlacementGroupInfo:
    pg: PlacementGroup
    _bundle_location_snapshots: list[BundleLocationSnapshot] = field(default_factory=list)
    _snapshot_path: Path | None = field(default=None, repr=False)

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
        if self._snapshot_path:
            _save_snapshots(self._snapshot_path, self._bundle_location_snapshots)


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


def create_placement_group_info(
    num_gpus: int,
    snapshot_path: Path | None = None,
) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs, snapshot and sort bundles."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    new_snapshots = _snapshot_bundles(pg, num_bundles=num_gpus)

    old_snapshots = _load_snapshots(snapshot_path) if snapshot_path else None
    if old_snapshots and len(old_snapshots) == num_gpus:
        sorted_snapshots = _partial_resort_snapshots(old_snapshots, new_snapshots)
    else:
        sorted_snapshots = sorted(new_snapshots, key=_bundle_sort_key)

    for rank, snapshot in enumerate(sorted_snapshots):
        logger.info(
            f"  bundle {rank:4}, actual_bundle_index: {snapshot.bundle_index:4}, "
            f"node: {snapshot.node_ip}, gpu: {snapshot.gpu_id}"
        )

    if snapshot_path:
        _save_snapshots(snapshot_path, sorted_snapshots)

    return PlacementGroupInfo(
        pg=pg, _bundle_location_snapshots=sorted_snapshots, _snapshot_path=snapshot_path,
    )
