"""Backward of ``y = silu(causal_conv1d(x, weight))`` over packed sequences: depthwise, width 4, bf16 activations.

Tiles follow fla's ``chunk_indices``, which cut every sequence on its own into 64-row pieces, so no tile spans a
sequence boundary and the kernel needs no per-row boundary checks (causal-conv1d's ``seq_idx`` backward does them
in every thread, which costs it 208-255 registers per thread; this kernel uses 80). Each block keeps its own
dweight partial; they are summed here in a fixed order, so the result is deterministic. The CUDA source is in
``causal_conv1d_bwd_cuda/``.
"""

import torch

try:
    import causal_conv1d_bwd_cuda
except ImportError:
    causal_conv1d_bwd_cuda = None

TILE_ROWS = 64
WIDTH = 4
_CHANNELS_PER_BLOCK = 64
# Blocks to launch in all: enough to fill an H200 / B200 several times over while keeping the dweight partials
# (blocks x channels x width floats) small.
_TARGET_BLOCKS = 4096


def is_available() -> bool:
    return causal_conv1d_bwd_cuda is not None


def causal_conv1d_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dx, dweight)`` of ``y = silu(causal_conv1d(x, weight))``, each sequence convolved on its own.

    ``x`` and ``dout`` are ``[1, T, C]`` bf16 with contiguous channels and 16-byte aligned rows (``C`` a multiple
    of 8); ``weight`` is ``[C, 4]`` fp32 or bf16; ``cu_seqlens`` is int32 ``[N + 1]`` ending at ``T``;
    ``chunk_indices`` is int32 ``prepare_chunk_indices(cu_seqlens, 64)`` from ``fla.ops.utils``. ``dx`` is bf16
    and ``dweight`` has ``weight``'s dtype.
    """
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    num_tiles = chunk_indices.shape[0]

    if num_tiles == 0:
        return dx, torch.zeros_like(weight)

    strips = -(-x.shape[-1] // _CHANNELS_PER_BLOCK)
    blocks_per_strip = min(num_tiles, -(-_TARGET_BLOCKS // strips))
    partials = torch.empty(blocks_per_strip, x.shape[-1], WIDTH, dtype=torch.float32, device=x.device)

    causal_conv1d_bwd_cuda.causal_conv1d_bwd(x, weight, dout, cu_seqlens, chunk_indices, TILE_ROWS, dx, partials)
    return dx, partials.sum(0).to(weight.dtype)
