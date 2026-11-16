"""Shared GPU-delta receiver test helpers.

FlashInfer's reference row and scale permutations, written independently of the kernel-backend views they check;
routed-expert layer shells whose quantization methods carry SGLang's exact class names; and byte views for bitwise
comparison.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def shuffled_rows(value):
    """FlashInfer's TRTLLM 32-row shuffle: row r of each 32-row block moves to (r % 4) * 8 + r // 4.

    An independent equivalent of its dstToSrcBlk32RowMap index_select.
    """
    return value.reshape(-1, 8, 4, value.shape[-1]).transpose(1, 2).reshape_as(value)


def interleave(value, group):
    """Interleave the two halves of ``value``'s rows in groups of ``group`` rows (FlashInfer's gate/up pairing)."""
    rows, cols = value.shape
    return value.reshape(2, rows // (2 * group), group, cols).transpose(0, 1).reshape(rows, cols)


def blocked_scales_padded(value):
    """FlashInfer's to_blocked, flattened: pad to [128, 4] tiles with 173, then transpose each tile's [4, 32] rows."""
    rows, cols = value.shape
    padded = torch.full(((rows + 127) // 128 * 128, (cols + 3) // 4 * 4), 173, dtype=torch.uint8)
    padded[:rows, :cols] = value
    pr, pc = padded.shape
    return (
        padded.reshape(pr // 128, 128, pc // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .contiguous()
        .flatten()
    )


def apply_segments(segments, mask):
    """XOR ``mask``'s rectangles into their segments' destinations, each canonical byte exactly once."""
    coverage = torch.zeros(mask.shape, dtype=torch.int32, device=mask.device)
    for segment in segments:
        (r0, r1), (c0, c1) = segment.slices
        coverage[r0:r1, c0:c1] += 1
        source = mask[r0:r1, c0:c1].contiguous().view(torch.uint8)
        for target in segment.destinations:
            target.bitwise_xor_(source.reshape(target.shape))
    assert torch.all(coverage == 1)


def quantized_layer(name, **attrs):
    """A routed-expert layer shell whose quantization method has the exact class name ``name``."""
    config = attrs.pop("quant_config", SimpleNamespace())
    method = type(name, (), {})()
    method.__dict__.update(
        runner=None,
        dequant_fp4_to_fp8=False,
        convert_mxfp8_to_block=False,
        use_mxfp8=False,
        is_fp4_expert=False,
    )
    method.__dict__.update(attrs, quant_config=config)
    if not hasattr(config, "scale_fmt"):
        config.scale_fmt = None
        config.weight_block_size = None
        if not hasattr(config, "is_checkpoint_fp8_serialized"):
            config.is_checkpoint_fp8_serialized = False
    return SimpleNamespace(
        quant_method=method,
        quant_config=config,
        moe_runner_config=SimpleNamespace(is_gated=True),
        _cutedsl_wrapper=SimpleNamespace(quant_mode="w4a4"),
    )


def byte_view(tensor):
    """``tensor``'s bytes in its own shape (a scalar as one element), for bitwise comparison."""
    return tensor.detach().contiguous().reshape(tensor.shape or (1,)).view(torch.uint8)
