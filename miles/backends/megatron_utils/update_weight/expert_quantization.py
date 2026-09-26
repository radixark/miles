"""Exchange converted expert tensors without gathering their unquantized weights."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int
    strides: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True)
class _PayloadMetadata:
    units: tuple[tuple[_TensorMetadata, ...], ...]
    nbytes: int
    dtype_bytes: tuple[tuple[torch.dtype, int], ...]


class ExpertGather:
    """Gather one fixed-layout expert batch, exchanging actual metadata once.

    Names, unit boundaries, shapes, dtypes, and group membership must stay fixed
    for this object's lifetime. Recreate it when the model, quantization config,
    or topology changes. Tensor values and storage may change on every call.
    Packing checks the local layout before starting payload collectives.

    Each call allocates fresh exact-size byte buffers. Returned typed views keep
    their receive storage alive; inputs are never modified. ``device`` selects
    the communication and output device, including for custom CUDA backends.
    """

    def __init__(self, *, group: dist.ProcessGroup):
        self._group = group
        self._source_ranks = tuple(dist.get_process_group_ranks(group))
        self._local_index = self._source_ranks.index(dist.get_rank())
        self._metadata: tuple[_PayloadMetadata, ...] | None = None

    def __call__(
        self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str
    ) -> list[list[tuple[str, torch.Tensor]]]:
        if len(self._source_ranks) == 1:
            return units
        if self._metadata is None:
            metadata = [None] * len(self._source_ranks)
            dist.all_gather_object(metadata, _describe_units(units), group=self._group)
            self._metadata = tuple(metadata)

        local_metadata = self._metadata[self._local_index]
        local_payload = torch.empty(local_metadata.nbytes, dtype=torch.uint8, device=device)
        _pack_units(units, local_metadata, local_payload)
        gathered = []
        handles = []
        for index, (rank, metadata) in enumerate(zip(self._source_ranks, self._metadata, strict=True)):
            payload = (
                local_payload
                if index == self._local_index
                else torch.empty(metadata.nbytes, dtype=torch.uint8, device=device)
            )
            if metadata.nbytes:
                handles.append(dist.broadcast(payload, src=rank, group=self._group, async_op=True))
            gathered.extend(_unpack_units(metadata, payload))
        # Queue every source before waiting. The views retain each byte buffer.
        for handle in handles:
            handle.wait()
        return gathered


def _describe_units(units):
    metadata = []
    dtype_sizes = {}
    offset = 0
    for unit in units:
        unit_metadata = []
        for name, tensor in unit:
            item_size = tensor.element_size()
            dtype_sizes[tensor.dtype] = item_size
            # Typed views require their storage offset to be dtype-aligned.
            offset = (offset + item_size - 1) // item_size * item_size
            nbytes = tensor.numel() * item_size
            shape = tuple(tensor.shape)
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= max(dim, 1)
            unit_metadata.append(
                _TensorMetadata(
                    name, shape, tensor.dtype, offset, nbytes, tuple(reversed(strides)), offset // item_size
                )
            )
            offset += nbytes
        metadata.append(tuple(unit_metadata))
    dtype_bytes = tuple((dtype, offset // item_size * item_size) for dtype, item_size in dtype_sizes.items())
    return _PayloadMetadata(tuple(metadata), offset, dtype_bytes)


def _pack_units(units, metadata, payload):
    assert len(units) == len(metadata.units), "Expert output unit count changed; recreate the iterator"
    for unit, unit_metadata in zip(units, metadata.units, strict=True):
        assert len(unit) == len(unit_metadata), "Expert output tensor count changed; recreate the iterator"
        for (name, tensor), info in zip(unit, unit_metadata, strict=True):
            assert (
                name == info.name and tensor.shape == info.shape and tensor.dtype == info.dtype
            ), f"Expert output layout changed for {info.name}; recreate the iterator"
            payload.narrow(0, info.offset, info.nbytes).copy_(tensor.contiguous().reshape(-1).view(torch.uint8))


def _unpack_units(metadata, payload):
    # Crop odd byte tails before reinterpreting the common storage by dtype.
    typed_payloads = {dtype: payload[:nbytes].view(dtype) for dtype, nbytes in metadata.dtype_bytes}
    return [
        [
            (info.name, typed_payloads[info.dtype].as_strided(info.shape, info.strides, info.storage_offset))
            for info in unit_metadata
        ]
        for unit_metadata in metadata.units
    ]
