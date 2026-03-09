from __future__ import annotations

from dataclasses import dataclass, field

from ray.util.placement_group import PlacementGroup


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
