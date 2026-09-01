import os

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd
from miles_plugins.models.deepseek_v4.ops.kernel.torch_sparse_mla import sparse_attn_torch

_bwd_fits: dict[tuple, bool] = {}

# Which training-time backward to use: "auto", "kernel", or "torch".
#
# On gfx942 neither option dominates. The tiling the fused kernel has to shrink to (one head and one
# KV slot per tile, a single wave64 per workgroup, 52 KiB of LDS) leaves one wave per CU, and
# measured on MI308X at H=8, D=512, topk=512 it runs ~3x SLOWER than the PyTorch reference at every
# sequence length from 64 to 4096. What it does win is footprint: the reference materializes
# [B, S, topk, D] fp32 gathered KV and keeps it for the backward, so at S=4096 it peaks at 18.9 GiB
# against the kernel's 390 MiB.
#
# So "auto" runs the reference -- the faster of the two, and the path this configuration was
# validated on -- until its gathered-KV allocation gets big enough to threaten the device, and
# switches to the kernel from there.
_BWD_IMPL = os.environ.get("MILES_DSV4_SPARSE_MLA_BWD", "auto").lower()
# Share of FREE device memory the reference backward is allowed to plan for.
_REFERENCE_BUDGET_FRACTION = float(os.environ.get("MILES_DSV4_SPARSE_MLA_BWD_BUDGET", "0.5"))
# The reference peaks well above its gathered-KV tensor: einsum casts it to fp32, autograd keeps it,
# and the score tensors add to that. Measured on MI308X at H=8, D=512, topk=512, peak / (S*topk*D*4)
# was 3.3x at S=1024, 4.6x at S=4096 and 8.6x at S=8192, so 8x is used as the planning figure.
_REFERENCE_PEAK_MULTIPLE = 8
_INT_MAX = 2**31 - 1
_logged_choice: set = set()


class DeepSeekV4SparseAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


def _backward_kernel_fits(q, kv, topk_idxs, sm_scale) -> bool:
    """Can the tilelang backward kernel be built for this shape on this GPU?

    bwd_within_shared_mem retiles until something builds, which on gfx942 means one head and one KV
    slot per tile on a single wave. If even that fails the reference implementation has to take
    over. Probing costs one compile per shape and the answer is cached; the forward kernel is
    unaffected either way.
    """
    B, S, H, D = q.shape
    S_kv = kv.shape[1]
    topk = topk_idxs.shape[-1]
    block_size = 32
    topk = (topk + block_size - 1) // block_size * block_size

    key = (B, S, S_kv, H, D, topk, sm_scale)
    if key not in _bwd_fits:
        try:
            sparse_mla_bwd.bwd_within_shared_mem(B, S, S_kv, H, D, topk, sm_scale)
            _bwd_fits[key] = True
        except Exception as e:
            # bwd_within_shared_mem re-raises the last candidate's error, which is whichever of the
            # three build failures that tiling hit -- not necessarily the shared-memory one.
            if not any(marker in str(e) for marker in sparse_mla_bwd.RETRYABLE_BUILD_ERRORS):
                raise
            _bwd_fits[key] = False
            print(
                f"[sparse_attn] tilelang backward does not fit this GPU's shared memory "
                f"({str(e).strip().splitlines()[0]}); falling back to the reference "
                f"implementation for training. Forward-only passes keep using the kernel.",
                flush=True,
            )
    return _bwd_fits[key]


def _reference_is_affordable(q, topk_idxs) -> bool:
    """Can sparse_attn_torch's activations be planned for out of what is free right now?

    Its footprint is dominated by the [B, S, topk, D] gathered KV, which the fused kernel never
    materializes -- the one axis on which the kernel beats the reference on this GPU.
    """
    B, S, _, D = q.shape
    estimated_peak = _REFERENCE_PEAK_MULTIPLE * B * S * topk_idxs.shape[-1] * D * 4
    free_bytes = torch.cuda.mem_get_info(q.device)[0]
    return estimated_peak <= _REFERENCE_BUDGET_FRACTION * free_bytes


def _reference_backward_is_representable(q, topk_idxs) -> bool:
    """Can gather's backward index the gathered KV without overflowing a 32-bit element count?

    It ends in ``index_put_(accumulate=True)``, which sorts one key per indexed element with a cub
    radix sort. Past INT_MAX elements that sort raises, and B*S*topk*D reaches 5.4e9 for a 128K
    DeepSeek-V4 layer at topk=640. Because ``_reference_is_affordable`` reads free memory at call
    time, whether the run ever gets there depends on the moment -- so without this check the same
    configuration crashes on some runs and not others.
    """
    B, S, _, D = q.shape
    return B * S * topk_idxs.shape[-1] * D <= _INT_MAX


def _log_choice(which, q, topk_idxs, reason):
    """Report the backward each shape settled on, once per shape.

    The two implementations differ by tens of GiB, and only one of the paths into the reference
    announces itself, so without this line two runs of the same configuration are not comparable.
    """
    key = (which, tuple(q.shape), topk_idxs.shape[-1])
    if key in _logged_choice:
        return
    _logged_choice.add(key)
    print(
        f"[sparse_attn] backward={which} for q={tuple(q.shape)} topk={topk_idxs.shape[-1]} ({reason})",
        flush=True,
    )


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    if not torch.is_grad_enabled():
        return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)

    if _BWD_IMPL == "torch":
        return sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale)
    if not _backward_kernel_fits(q, kv, topk_idxs, sm_scale):
        _log_choice("reference", q, topk_idxs, "kernel does not fit shared memory")
        return sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale)
    if (
        _BWD_IMPL == "auto"
        and _reference_is_affordable(q, topk_idxs)
        and _reference_backward_is_representable(q, topk_idxs)
    ):
        _log_choice("reference", q, topk_idxs, "affordable and representable")
        return sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale)
    _log_choice("kernel", q, topk_idxs, f"impl={_BWD_IMPL}")
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
