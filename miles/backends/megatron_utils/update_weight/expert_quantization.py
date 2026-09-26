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
    alignment: int
    dtype_bytes: tuple[tuple[torch.dtype, int], ...]


class ExpertGather:
    """Gather one fixed-layout expert batch, exchanging actual metadata once.

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
        self._metadata: tuple[_PayloadMetadata, ...] | None = None
        self._payload_sizes: tuple[int, ...] = ()
        self._single_source: int | None = None
        self._total_bytes = 0
        self._uniform = False

    def __call__(
        self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str
    ) -> list[list[tuple[str, torch.Tensor]]]:
        if len(self._source_ranks) == 1:
            return units
        if self._metadata is None:
            metadata = [None] * len(self._source_ranks)
            dist.all_gather_object(metadata, _describe_units(units), group=self._group)
            self._metadata = tuple(metadata)
            alignment = max(info.alignment for info in self._metadata)
            self._payload_sizes = tuple(
                (info.nbytes + alignment - 1) // alignment * alignment for info in self._metadata
            )
            self._total_bytes = sum(self._payload_sizes)
            self._uniform = len(set(self._payload_sizes)) == 1
            sources = [index for index, size in enumerate(self._payload_sizes) if size]
            self._single_source = sources[0] if len(sources) == 1 else None

        local_metadata = self._metadata[self._local_index]
        storage = torch.empty(self._total_bytes, dtype=torch.uint8, device=device)
        payloads = list(storage.split(self._payload_sizes))
        local_payload = payloads[self._local_index]
        _pack_units(units, local_metadata, local_payload)
        handle = self._gather_payloads(storage, payloads, local_payload)
        # Build views on the CPU while the asynchronous transfer is in flight.
        gathered = [
            unit
            for metadata, payload in zip(self._metadata, payloads, strict=True)
            for unit in _unpack_units(metadata, payload)
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
    return _PayloadMetadata(tuple(metadata), offset, max(dtype_sizes.values(), default=1), dtype_bytes)


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
    storage_offsets = {dtype: tensor.storage_offset() for dtype, tensor in typed_payloads.items()}
    return [
        [
            (
                info.name,
                typed_payloads[info.dtype].as_strided(
                    info.shape, info.strides, storage_offsets[info.dtype] + info.storage_offset
                ),
            )
            for info in unit_metadata
        ]
        for unit_metadata in metadata.units
    ]
