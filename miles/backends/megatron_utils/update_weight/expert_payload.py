"""Encode a list of units of named tensors into one byte payload and decode it back as views."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    byte_offset: int
    nbytes: int
    strides: tuple[int, ...]
    storage_offset: int  # Element offset within the payload's typed view.


@dataclass(frozen=True)
class PayloadLayout:
    units: tuple[tuple[TensorLayout, ...], ...]
    nbytes: int
    alignment: int
    # Dtype-aligned prefix lengths of the entire payload, not per-dtype totals.
    dtype_view_nbytes: tuple[tuple[torch.dtype, int], ...]


def build_payload_layout(units):
    return _lay_out_units([[(name, tuple(tensor.shape), tensor.dtype) for name, tensor in unit] for unit in units])


def select_units(layout, unit_indices):
    """The layout of a payload holding only the chosen units of `layout`, in the given order."""
    return _lay_out_units(
        [[(tensor.name, tensor.shape, tensor.dtype) for tensor in layout.units[index]] for index in unit_indices]
    )


def _lay_out_units(unit_specs):
    unit_layouts = []
    dtype_sizes = {}
    byte_offset = 0
    for unit_spec in unit_specs:
        unit_layout = []
        for name, shape, dtype in unit_spec:
            item_size = dtype.itemsize
            dtype_sizes[dtype] = item_size
            # Typed views require their storage offset to be dtype-aligned.
            byte_offset = (byte_offset + item_size - 1) // item_size * item_size
            nbytes = math.prod(shape) * item_size
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= max(dim, 1)
            unit_layout.append(
                TensorLayout(
                    name, shape, dtype, byte_offset, nbytes, tuple(reversed(strides)), byte_offset // item_size
                )
            )
            byte_offset += nbytes
        unit_layouts.append(tuple(unit_layout))
    dtype_view_nbytes = tuple(
        (dtype, byte_offset // item_size * item_size) for dtype, item_size in dtype_sizes.items()
    )
    return PayloadLayout(tuple(unit_layouts), byte_offset, max(dtype_sizes.values(), default=1), dtype_view_nbytes)


def pack_units(units, layout, payload):
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


def unpack_units(layout, payload):
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
