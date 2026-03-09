from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


@dataclass
class BundleLocationSnapshot:
    bundle_index: int
    node_ip: str
    gpu_id: str


@dataclass
class RefreshResult:
    changed_ranks: list[int]


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

    def refresh(self) -> RefreshResult:
        """Re-probe all bundles and partially re-sort only changed ranks."""
        new_probes = _probe_bundles(
            self.pg,
            num_bundles=len(self._bundle_location_snapshots),
            num_gpus=0.001,
        )
        return self._refresh_from_probes(new_probes)

    def _refresh_from_probes(self, new_probes: list[BundleLocationSnapshot]) -> RefreshResult:
        """Pure algorithm: detect changed ranks and partially re-sort.

        Unchanged ranks keep their exact (bundle_index, node_ip, gpu_id).
        Changed ranks (node_ip differs) are re-sorted among themselves.
        """
        probe_by_index: dict[int, BundleLocationSnapshot] = {
            p.bundle_index: p for p in new_probes
        }

        # Step 1: find changed logical ranks
        changed_ranks: list[int] = []
        for rank, old_bundle in enumerate(self._bundle_location_snapshots):
            new_probe = probe_by_index[old_bundle.bundle_index]
            if new_probe.node_ip != old_bundle.node_ip:
                changed_ranks.append(rank)

        if not changed_ranks:
            return RefreshResult(changed_ranks=[])

        # Step 2: sort new probes for changed ranks
        new_probes_for_changed = [
            probe_by_index[self._bundle_location_snapshots[r].bundle_index]
            for r in changed_ranks
        ]
        new_probes_for_changed.sort(key=_bundle_sort_key)

        # Step 3: write back in-place
        for rank, new_probe in zip(changed_ranks, new_probes_for_changed):
            self._bundle_location_snapshots[rank] = new_probe

        return RefreshResult(changed_ranks=changed_ranks)


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


def _bundle_sort_key(probe: BundleLocationSnapshot) -> tuple[list[int], str]:
    node_identifier = probe.node_ip
    try:
        node_ip_parts = list(map(int, node_identifier.split(".")))
    except ValueError:
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Convert each character to its ASCII value for stable sorting
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, probe.gpu_id)


@ray.remote(num_gpus=1)
class _InfoActor:
    def get_ip_and_gpu_id(self) -> tuple[str, str]:
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def _probe_bundles(pg: PlacementGroup, num_bundles: int, num_gpus: float = 1) -> list[BundleLocationSnapshot]:
    """Probe all bundles in a PG to discover (node_ip, gpu_id) mappings.

    Args:
        pg: The Ray PlacementGroup to probe.
        num_bundles: Number of bundles in the PG.
        num_gpus: GPU fraction for each _InfoActor (1 for initial probe, tiny for live refresh).
    """
    actor_cls = _InfoActor if num_gpus == 1 else _InfoActor.options(num_gpus=num_gpus)

    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            actor_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )

    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    probes = [
        BundleLocationSnapshot(bundle_index=i, node_ip=gpu_ids[i][0], gpu_id=gpu_ids[i][1])
        for i in range(num_bundles)
    ]

    return probes


def create_placement_group_info(num_gpus: int) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs, probe and sort bundles."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")

    ray.get(pg.ready())

    probes = _probe_bundles(pg=pg, num_bundles=num_gpus)
    sorted_probes = sorted(probes, key=_bundle_sort_key)

    for rank, probe in enumerate(sorted_probes):
        logger.info(
            f"  bundle {rank:4}, actual_bundle_index: {probe.bundle_index:4}, "
            f"node: {probe.node_ip}, gpu: {probe.gpu_id}"
        )

    return PlacementGroupInfo(pg=pg, _bundle_location_snapshots=sorted_probes)
