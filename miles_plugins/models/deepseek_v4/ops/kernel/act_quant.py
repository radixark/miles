"""Per-token UE8M0 FP8 activation quantization — backed by deepseek-ai/TileKernels.

The legacy in-tree kernel is replaced by a thin wrapper around
``tile_kernels.quant.per_token_cast`` so we share the same FP8 cast
implementation as the rest of the DeepSeek stack.
"""

from typing import Optional

import torch

from tile_kernels.quant import per_token_cast


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast ``x`` to FP8 (E4M3) with per-token, per-block scaling factors.

    Returns ``(y, s)`` matching the original kernel's contract:
    - ``y``: same shape as ``x``, ``torch.float8_e4m3fn``
    - ``s``: shape ``(*x.shape[:-1], x.shape[-1] // block_size)``, ``torch.float32``

    ``scale_fmt="ue8m0"`` requests round-to-power-of-two scaling factors
    (the scales are still stored as fp32; storage in UE8M0 format would
    require ``use_packed_ue8m0=True`` which has different consumer
    expectations than the legacy QAT path).
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    N = x.size(-1)
    flat = x.view(-1, N)
    y_flat, s_flat = per_token_cast(
        flat,
        fmt="e4m3",
        num_per_channels=block_size,
        round_sf=scale_fmt is not None,
    )
    y = y_flat.view_as(x)
    s = s_flat.view(*x.shape[:-1], N // block_size)
    return y, s
