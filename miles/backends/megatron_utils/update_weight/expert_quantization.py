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


def gather_expert_units(
    units: list[list[tuple[str, torch.Tensor]]],
    *,
    group: dist.ProcessGroup,
    device: torch.device | str,
) -> list[list[tuple[str, torch.Tensor]]]:
    """Gather converted units in group-rank order, preserving each unit's boundary.

    The metadata describes the actual converted tensors, including quantization
    scales. Each source sends one mixed-dtype byte buffer; exact-size broadcasts
    avoid padding every rank to the largest payload. Returned tensors retain
    their receive buffer storage, and input tensors are never modified.
    ``device`` selects the communication and output device; specify CUDA for
    CUDA-capable custom backends or Gloo groups carrying CUDA tensors.
    """
    if dist.get_world_size(group) == 1:
        return units

    local_metadata, local_nbytes = _describe_units(units)
    metadata_by_rank = [None] * dist.get_world_size(group)
    dist.all_gather_object(metadata_by_rank, (local_metadata, local_nbytes), group=group)
    source_ranks = dist.get_process_group_ranks(group)

    gathered = []
    for rank, (metadata, nbytes) in zip(source_ranks, metadata_by_rank, strict=True):
        payload = torch.empty(nbytes, dtype=torch.uint8, device=device)
        if dist.get_rank() == rank:
            _pack_units(units, metadata, payload)
        if nbytes:
            dist.broadcast(payload, src=rank, group=group)
        gathered.extend(_unpack_units(metadata, payload))
    return gathered


def _describe_units(units):
    metadata = []
    offset = 0
    for unit in units:
        unit_metadata = []
        for name, tensor in unit:
            item_size = tensor.element_size()
            # Typed views require their storage offset to be dtype-aligned.
            offset = (offset + item_size - 1) // item_size * item_size
            nbytes = tensor.numel() * item_size
            unit_metadata.append(_TensorMetadata(name, tuple(tensor.shape), tensor.dtype, offset, nbytes))
            offset += nbytes
        metadata.append(unit_metadata)
    return metadata, offset


def _pack_units(units, metadata, payload):
    for unit, unit_metadata in zip(units, metadata, strict=True):
        for (_, tensor), info in zip(unit, unit_metadata, strict=True):
            payload.narrow(0, info.offset, info.nbytes).copy_(tensor.contiguous().reshape(-1).view(torch.uint8))


def _unpack_units(metadata, payload):
    return [
        [
            (info.name, payload.narrow(0, info.offset, info.nbytes).view(info.dtype).reshape(info.shape))
            for info in unit_metadata
        ]
        for unit_metadata in metadata
    ]
