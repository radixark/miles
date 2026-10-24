"""Pack and gather converted expert tensors and scales across training ranks."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    byte_offset: int
    nbytes: int
    strides: tuple[int, ...]
    storage_offset: int  # Element offset within the payload's typed view.


@dataclass(frozen=True)
class _PayloadLayout:
    units: tuple[tuple[_TensorLayout, ...], ...]
    nbytes: int
    alignment: int
    # Dtype-aligned prefix lengths of the entire payload, not per-dtype totals.
    dtype_view_nbytes: tuple[tuple[torch.dtype, int], ...]


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
        self._layouts: tuple[_PayloadLayout, ...] | None = None
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
            dist.all_gather_object(layouts, _build_payload_layout(units), group=self._group)
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
        _pack_units(units, local_layout, local_payload)
        handle = self._gather_payloads(storage, payloads, local_payload)
        # Build views on the CPU while the asynchronous transfer is in flight.
        gathered = [
            unit
            for layout, payload in zip(self._layouts, payloads, strict=True)
            for unit in _unpack_units(layout, payload)
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


def _build_payload_layout(units):
    unit_layouts = []
    dtype_sizes = {}
    byte_offset = 0
    for unit in units:
        unit_layout = []
        for name, tensor in unit:
            item_size = tensor.element_size()
            dtype_sizes[tensor.dtype] = item_size
            # Typed views require their storage offset to be dtype-aligned.
            byte_offset = (byte_offset + item_size - 1) // item_size * item_size
            nbytes = tensor.numel() * item_size
            shape = tuple(tensor.shape)
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= max(dim, 1)
            unit_layout.append(
                _TensorLayout(
                    name, shape, tensor.dtype, byte_offset, nbytes, tuple(reversed(strides)), byte_offset // item_size
                )
            )
            byte_offset += nbytes
        unit_layouts.append(tuple(unit_layout))
    dtype_view_nbytes = tuple(
        (dtype, byte_offset // item_size * item_size) for dtype, item_size in dtype_sizes.items()
    )
    return _PayloadLayout(tuple(unit_layouts), byte_offset, max(dtype_sizes.values(), default=1), dtype_view_nbytes)


def _pack_units(units, layout, payload):
    assert len(units) == len(layout.units), "Expert output unit count changed; recreate the iterator"
    for unit, unit_layout in zip(units, layout.units, strict=True):
        assert len(unit) == len(unit_layout), "Expert output tensor count changed; recreate the iterator"
        for (name, tensor), tensor_layout in zip(unit, unit_layout, strict=True):
            assert (
                name == tensor_layout.name
                and tensor.shape == tensor_layout.shape
                and tensor.dtype == tensor_layout.dtype
            ), f"Expert output layout changed for {tensor_layout.name}; recreate the iterator"
            payload.narrow(0, tensor_layout.byte_offset, tensor_layout.nbytes).copy_(
                tensor.contiguous().reshape(-1).view(torch.uint8)
            )


def _unpack_units(layout, payload):
    # Crop odd byte tails before reinterpreting the common storage by dtype.
    typed_payloads = {dtype: payload[:nbytes].view(dtype) for dtype, nbytes in layout.dtype_view_nbytes}
    storage_offsets = {dtype: tensor.storage_offset() for dtype, tensor in typed_payloads.items()}
    return [
        [
            (
                tensor_layout.name,
                typed_payloads[tensor_layout.dtype].as_strided(
                    tensor_layout.shape,
                    tensor_layout.strides,
                    storage_offsets[tensor_layout.dtype] + tensor_layout.storage_offset,
                ),
            )
            for tensor_layout in unit_layout
        ]
        for unit_layout in layout.units
    ]
