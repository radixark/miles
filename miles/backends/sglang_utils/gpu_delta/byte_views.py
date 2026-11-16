"""Affine byte views of live storage, shared by bindings, model mappings and kernel backends (CUDA receiver only).

The apply kernel knows only affine byte coordinates. Admission factors an inference backend's
permutations into those coordinates. It never transforms a delta, requantizes a weight, or
allocates a weight image. A segment selects a rectangle of one canonical projection; its
destinations have the same flattened byte order, and may cover several independent images.
Padding is deliberately outside every destination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class _DtypeMismatch(ValueError):
    """A logical or physical dtype mismatch, collected by layout admission."""


@dataclass(frozen=True)
class DestinationSegment:
    # Coordinates are canonical ELEMENTS, not bytes. Only the returned views
    # use byte coordinates, so BF16 source selections retain their element size.
    # Matrix segments hold byte views; raw (scalar/vector) segments hold typed live views.
    slices: tuple[tuple[int, int], ...]
    destinations: tuple[torch.Tensor, ...]
    source_shape: tuple[int, ...] | None = None


def _byte_view(tensor):
    # A strided outer dimension is fine, but dtype reinterpretation requires
    # contiguous elements within each row. Never silently clone a destination.
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.stride(-1) != 1:
        raise ValueError("delta destination requires a contiguous innermost dimension")
    return tensor.detach().view(torch.uint8)


def _storage_key(tensor):
    """The identity of a tensor's storage allocation, equal for every view of it; the one storage comparison."""
    return tensor.device, tensor.untyped_storage().data_ptr()


def _check_served(image, allowed, role):
    if image.dtype not in allowed:
        raise _DtypeMismatch(f"{role} served={image.dtype}, allowed_served={[str(dtype) for dtype in allowed]}")


def _same_image(a, b):
    return (
        _storage_key(a) == _storage_key(b)
        and a.storage_offset() * a.element_size() == b.storage_offset() * b.element_size()
        and a.numel() * a.element_size() == b.numel() * b.element_size()
    )


def _unique_images(images):
    unique = []
    for image in images:
        if not any(_same_image(image, current) for current in unique):
            unique.append(image)
    return unique


def _row_pieces(rows):
    """One full-128-row piece and at most seven power-of-two tail pieces."""
    start = 0
    for unit in (128, 64, 32, 16, 8, 4, 2, 1):
        count = (rows - start) // unit
        if count:
            yield start, start + count * unit, unit
            start += count * unit


def _factored_segments(
    images,
    rows,
    cols,
    element_size,
    row_offset,
    column_block,
    column_stride,
    block_stride,
):
    """Factor fixed row-bit permutations into affine axes, excluding padding.

    A 128-row bulk plus binary tail pieces gives at most eight row rectangles.
    Columns need at most a full-block rectangle and one tail. No tensor data,
    index tensor, or weight-sized intermediate is constructed.
    """
    byte_cols = cols * element_size
    full = byte_cols // column_block * column_block
    columns = [(0, full)] if full else []
    if full < byte_cols:
        columns.append((full, byte_cols))
    images = tuple(_byte_view(image) for image in _unique_images(images))
    result = []
    for r0, r1, unit in _row_pieces(rows):
        bits = [1 << bit for bit in reversed(range(unit.bit_length() - 1))]
        shape = ((r1 - r0) // unit, *(2 for _ in bits))
        origin = row_offset(r0)
        strides = (
            row_offset(r0 + unit) - origin,
            *(row_offset(r0 + bit) - origin for bit in bits),
        )
        for c0, c1 in columns:
            if (c1 - c0) % column_block == 0:
                cshape = ((c1 - c0) // column_block, column_block)
                cstrides = (block_stride, column_stride)
            else:
                cshape, cstrides = (c1 - c0,), (column_stride,)
            offset = origin + c0 // column_block * block_stride
            destinations = tuple(
                image.as_strided(
                    (*shape, *cshape),
                    (*strides, *cstrides),
                    image.storage_offset() + offset,
                )
                for image in images
            )
            result.append(DestinationSegment(((r0, r1), (c0 // element_size, c1 // element_size)), destinations))
    return tuple(result)
