from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket


def flatten_weight_bucket(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> "FlattenedTensorBucket":
    """One SGLang `FlattenedTensorBucket` holding every tensor, whatever its dtype."""
    # Keep SGLang optional for callers using the existing per-tensor protocol.
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    if not getattr(
        FlattenedTensorBucket, "supports_multi_dtypes", False
    ):  # config-access-exempt: tensor bucket versions differ in mixed-dtype support
        raise RuntimeError("Flattened weight buckets require SGLang's mixed-dtype FlattenedTensorBucket")

    # SGLang concatenates raw bytes without padding. Each reconstructed dtype
    # view requires an aligned offset; preserve names/order and reject unsafe
    # layouts before allocating or sending metadata instead of casting values.
    offset = 0
    for name, tensor in named_tensors:
        if offset % tensor.element_size():
            raise ValueError(f"Unaligned flattened weight {name}: byte offset {offset}, dtype {tensor.dtype}")
        offset += tensor.numel() * tensor.element_size()
    return FlattenedTensorBucket(named_tensors=[(name, tensor.contiguous()) for name, tensor in named_tensors])
