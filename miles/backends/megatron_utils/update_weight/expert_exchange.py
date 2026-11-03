"""Deliver converted expert tensors only to the training ranks whose route names them."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.expert_payload import (
    PayloadLayout,
    build_payload_layout,
    pack_units,
    select_units,
    unpack_units,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfNameRoute


@dataclass(frozen=True)
class ExpertExchangePlan:
    """What one rank sends to and receives from every rank of the group in one expert batch."""

    unit_indices_by_dst_rank: tuple[tuple[int, ...], ...]
    send_layouts_by_dst_rank: tuple[PayloadLayout, ...]
    recv_layouts_by_src_rank: tuple[PayloadLayout, ...]
    # every segment is padded to it, so each received payload can be viewed as any of its dtypes
    segment_alignment: int

    @property
    def send_split_sizes(self) -> list[int]:
        return [self._padded(layout.nbytes) for layout in self.send_layouts_by_dst_rank]

    @property
    def recv_split_sizes(self) -> list[int]:
        return [self._padded(layout.nbytes) for layout in self.recv_layouts_by_src_rank]

    def _padded(self, nbytes: int) -> int:
        return (nbytes + self.segment_alignment - 1) // self.segment_alignment * self.segment_alignment


def plan_expert_exchange(
    layouts_by_rank: Sequence[PayloadLayout], route: HfNameRoute, rank: int
) -> ExpertExchangePlan:
    """A unit goes to every rank that needs any of its names. A unit holds every HF tensor converted from one
    training param (a weight and its scales), so it travels whole, and a rank needing only part of it is an error."""
    assert len(layouts_by_rank) == len(
        route.hf_names_by_rank
    ), f"the route names {len(route.hf_names_by_rank)} ranks, the expert exchange spans {len(layouts_by_rank)}"
    unit_indices_by_dst_rank = tuple(
        _destined_unit_indices(layouts_by_rank[rank], needed_hf_names) for needed_hf_names in route.hf_names_by_rank
    )
    return ExpertExchangePlan(
        unit_indices_by_dst_rank=unit_indices_by_dst_rank,
        send_layouts_by_dst_rank=tuple(
            select_units(layouts_by_rank[rank], unit_indices) for unit_indices in unit_indices_by_dst_rank
        ),
        recv_layouts_by_src_rank=tuple(
            select_units(layout, _destined_unit_indices(layout, route.hf_names_by_rank[rank]))
            for layout in layouts_by_rank
        ),
        segment_alignment=max(layout.alignment for layout in layouts_by_rank),
    )


def _destined_unit_indices(layout: PayloadLayout, needed_hf_names: frozenset[str]) -> tuple[int, ...]:
    unit_indices = []
    for index, unit in enumerate(layout.units):
        hf_names = {tensor.name for tensor in unit}
        needed = hf_names & needed_hf_names
        assert not needed or needed == hf_names, (
            f"a rank needs {sorted(needed)} but not {sorted(hf_names - needed)}, all converted from one training "
            "param; a unit travels whole"
        )
        if needed:
            unit_indices.append(index)
    return tuple(unit_indices)


@dataclass
class InFlightExchange:
    """One batch's sends and receives, issued but not yet waited on."""

    # kept alive until the sends complete
    send_payload: torch.Tensor
    recv_payload: torch.Tensor
    works: list


class ExpertExchange:
    """Like ExpertGather, but delivers each unit only to the ranks whose route names it.

    Names, unit boundaries, shapes and dtypes must stay fixed for this object's lifetime; the layouts are exchanged
    and the plan computed on the first call. Received units come in source-rank order. `start` issues a batch's
    transfers without waiting, so the caller can overlap them with other work before `finish`.
    """

    def __init__(self, *, group: dist.ProcessGroup, route: HfNameRoute):
        self._group = group
        self._route = route
        self._plan: ExpertExchangePlan | None = None

    def __call__(
        self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str
    ) -> list[list[tuple[str, torch.Tensor]]]:
        return self.finish(self.start(units, device=device))

    def start(self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str) -> InFlightExchange:
        if self._plan is None:
            layouts_by_rank = [None] * dist.get_world_size(self._group)
            dist.all_gather_object(layouts_by_rank, build_payload_layout(units), group=self._group)
            self._plan = plan_expert_exchange(layouts_by_rank, self._route, dist.get_rank(self._group))
        plan = self._plan
        send_payload = torch.empty(sum(plan.send_split_sizes), dtype=torch.uint8, device=device)
        for unit_indices, layout, segment in zip(
            plan.unit_indices_by_dst_rank,
            plan.send_layouts_by_dst_rank,
            send_payload.split(plan.send_split_sizes),
            strict=True,
        ):
            pack_units([units[index] for index in unit_indices], layout, segment)
        recv_payload = torch.empty(sum(plan.recv_split_sizes), dtype=torch.uint8, device=device)
        works = self._start_exchange_with_peers(
            send_segments=send_payload.split(plan.send_split_sizes),
            recv_segments=recv_payload.split(plan.recv_split_sizes),
        )
        return InFlightExchange(send_payload=send_payload, recv_payload=recv_payload, works=works)

    def finish(self, in_flight: InFlightExchange) -> list[list[tuple[str, torch.Tensor]]]:
        """Makes the current stream wait for the batch's receives; returns the received units."""
        for work in in_flight.works:
            work.wait()
        plan = self._plan
        return [
            unit
            for layout, segment in zip(
                plan.recv_layouts_by_src_rank, in_flight.recv_payload.split(plan.recv_split_sizes), strict=True
            )
            for unit in unpack_units(layout, segment)
        ]

    def _start_exchange_with_peers(
        self, *, send_segments: Sequence[torch.Tensor], recv_segments: Sequence[torch.Tensor]
    ) -> list:
        # pairwise sends and receives with only the peers that exchange bytes: a world-wide all-to-all made every
        # call wait for the last of all ranks, though each rank exchanges with a few
        group_rank = dist.get_rank(self._group)
        ops = []
        for peer, (send_segment, recv_segment) in enumerate(zip(send_segments, recv_segments, strict=True)):
            if peer == group_rank:
                recv_segment.copy_(send_segment)
                continue
            global_peer = dist.get_global_rank(self._group, peer)
            if send_segment.numel():
                ops.append(dist.P2POp(dist.isend, send_segment, global_peer, self._group))
            if recv_segment.numel():
                ops.append(dist.P2POp(dist.irecv, recv_segment, global_peer, self._group))
        return dist.batch_isend_irecv(ops) if ops else []
