"""Pack and gather converted expert tensors and scales across training ranks."""

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.expert_payload import (
    PayloadLayout,
    build_payload_layout,
    pack_units,
    unpack_units,
)


class ExpertGather:
    """Gather one fixed-layout expert batch, exchanging payload layouts once.

    Names, unit boundaries, shapes, dtypes, and group membership must stay fixed
    for this object's lifetime. Recreate it when the model, quantization config,
    or topology changes. Tensor values and storage may change on every call.
    Packing checks the local layout before starting payload collectives.

    The existing process-group reference and layout are reused across updates; Work
    handles belong to individual operations and are waited before returning.
    Each call allocates fresh receive storage, since callers may retain earlier
    outputs. Rank segments are padded only for dtype alignment, never to the
    largest rank's payload. ``device`` selects communication and output storage.
    """

    def __init__(self, *, group: dist.ProcessGroup):
        self._group = group
        self._source_ranks = tuple(dist.get_process_group_ranks(group))
        self._local_index = self._source_ranks.index(dist.get_rank())
        self._layouts: tuple[PayloadLayout, ...] | None = None
        self._payload_sizes: tuple[int, ...] = ()
        self._single_source: int | None = None
        self._total_bytes = 0
        self._uniform = False

    def __call__(
        self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str
    ) -> list[list[tuple[str, torch.Tensor]]]:
        if len(self._source_ranks) == 1:
            return units
        if self._layouts is None:
            layouts = [None] * len(self._source_ranks)
            dist.all_gather_object(layouts, build_payload_layout(units), group=self._group)
            self._layouts = tuple(layouts)
            alignment = max(layout.alignment for layout in self._layouts)
            self._payload_sizes = tuple(
                (layout.nbytes + alignment - 1) // alignment * alignment for layout in self._layouts
            )
            self._total_bytes = sum(self._payload_sizes)
            self._uniform = len(set(self._payload_sizes)) == 1
            sources = [index for index, size in enumerate(self._payload_sizes) if size]
            self._single_source = sources[0] if len(sources) == 1 else None

        local_layout = self._layouts[self._local_index]
        storage = torch.empty(self._total_bytes, dtype=torch.uint8, device=device)
        payloads = list(storage.split(self._payload_sizes))
        local_payload = payloads[self._local_index]
        pack_units(units, local_layout, local_payload)
        handle = self._gather_payloads(storage, payloads, local_payload)
        # Build views on the CPU while the asynchronous transfer is in flight.
        gathered = [
            unit
            for layout, payload in zip(self._layouts, payloads, strict=True)
            for unit in unpack_units(layout, payload)
        ]
        if handle is not None:
            handle.wait()
        return gathered

    def _gather_payloads(self, storage, payloads, local_payload):
        if not self._total_bytes:
            return None
        if self._single_source is not None:
            source = self._single_source
            return dist.broadcast(payloads[source], src=self._source_ranks[source], group=self._group, async_op=True)
        if self._uniform:
            # Native NCCL all-gather, in place: no flattened temporary or copies.
            return dist.all_gather_into_tensor(storage, local_payload, group=self._group, async_op=True)
        # NCCL coalesces uneven all-gather internally into one Work handle.
        return dist.all_gather(payloads, local_payload, group=self._group, async_op=True)
