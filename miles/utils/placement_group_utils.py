from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


@dataclass
class BundleProbe:
    bundle_index: int
    node_ip: str
    gpu_id: str


@dataclass
class PlacementGroupInfo:
    pg: PlacementGroup
    bundles: list[BundleProbe] = field(default_factory=list)

    @property
    def reordered_bundle_indices(self) -> list[int]:
        return [b.bundle_index for b in self.bundles]

    @property
    def reordered_gpu_ids(self) -> list[str]:
        return [b.gpu_id for b in self.bundles]

    def slice(self, offset: int, count: int) -> PlacementGroupSlice:
        return PlacementGroupSlice(owner=self, offset=offset, count=count)


@dataclass
class PlacementGroupSlice:
    owner: PlacementGroupInfo
    offset: int
    count: int

    @property
    def pg(self) -> PlacementGroup:
        return self.owner.pg

    @property
    def reordered_bundle_indices(self) -> list[int]:
        return self.owner.reordered_bundle_indices[self.offset : self.offset + self.count]

    @property
    def reordered_gpu_ids(self) -> list[str]:
        return self.owner.reordered_gpu_ids[self.offset : self.offset + self.count]


def bundle_sort_key(probe: BundleProbe) -> tuple[list[int], str]:
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
class InfoActor:
    def get_ip_and_gpu_id(self) -> tuple[str, str]:
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def probe_bundles(pg: PlacementGroup, num_bundles: int, num_gpus: float = 1) -> list[BundleProbe]:
    """Probe all bundles in a PG to discover (node_ip, gpu_id) mappings.

    Args:
        pg: The Ray PlacementGroup to probe.
        num_bundles: Number of bundles in the PG.
        num_gpus: GPU fraction for each InfoActor (1 for initial probe, tiny for live refresh).
    """
    actor_cls = InfoActor if num_gpus == 1 else InfoActor.options(num_gpus=num_gpus)

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
        BundleProbe(bundle_index=i, node_ip=gpu_ids[i][0], gpu_id=gpu_ids[i][1])
        for i in range(num_bundles)
    ]

    return probes


def create_placement_group_info(num_gpus: int) -> PlacementGroupInfo:
    """Create a placement group with the specified number of GPUs, probe and sort bundles."""
    from ray.util.placement_group import placement_group

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")

    ray.get(pg.ready())

    probes = probe_bundles(pg=pg, num_bundles=num_gpus)
    sorted_probes = sorted(probes, key=bundle_sort_key)

    for rank, probe in enumerate(sorted_probes):
        logger.info(
            f"  bundle {rank:4}, actual_bundle_index: {probe.bundle_index:4}, "
            f"node: {probe.node_ip}, gpu: {probe.gpu_id}"
        )

    return PlacementGroupInfo(pg=pg, bundles=sorted_probes)
