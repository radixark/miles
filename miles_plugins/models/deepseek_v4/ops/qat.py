import torch

from tile_kernels.quant import per_token_cast_back

from miles_plugins.models.deepseek_v4.arguments import INDEX_QAT_MODES, KV_QAT_MODES
from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import _needs_portable_fp8, act_quant


def resolve_fp8_qat(config, *, is_indexer: bool = False) -> tuple[bool, str | None]:
    """Resolve activation QAT without changing TE's config.fp8 or its recipe.

    Missing policy attributes deliberately preserve the old config.fp8 is not None
    test. A BF16 tensor can still contain quantize/dequantize-simulated FP8 values.
    """
    name = "dsv4_index_qat" if is_indexer else "dsv4_kv_qat"
    mode = getattr(config, name, "legacy")
    choices = INDEX_QAT_MODES if is_indexer else KV_QAT_MODES
    if mode not in choices:
        raise ValueError(f"Invalid {name}={mode!r}; expected one of {choices}.")
    enabled = config.fp8 is not None if mode == "legacy" else mode != "off"
    return enabled, None if mode == "fp8_dynamic" else "ue8m0"


def fp8_simulate(x: torch.Tensor, block_size: int, scale_fmt: str | None = "ue8m0"):
    """Simulate per-token FP8 (E4M3) cast + dequant.

    The default keeps legacy UE8M0 scaling; None uses dynamic FP32 scales.
    Dynamic changes only the scale rule, not the RoPE/Hadamard rounding boundaries.

    gfx942 uses explicit PyTorch FN conversions for quantization and cast-back.
    TileKernels' HIP cast-back interprets FN bytes as FNUZ, halving values.
    This is activation QAT, not the TE recipe that controls training GEMMs.
    """
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, scale_fmt)

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    if _needs_portable_fp8(x):
        # Do not pass genuine FN bytes to the TileLang FNUZ cast-back kernel.
        blocks = y_flat.float().reshape(-1, N // block_size, block_size)
        out_flat = (blocks * scale_flat.unsqueeze(-1)).reshape_as(y_flat)
    else:
        out_flat = per_token_cast_back(
            (y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size
        )
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128, scale_fmt="ue8m0"):
        return fp8_simulate(kv, block_size, scale_fmt)

    @staticmethod
    def backward(ctx, grad_kv):
        # Keep the STE for both the legacy two-argument and explicit-policy calls.
        return (grad_kv,) + (None,) * (len(ctx.needs_input_grad) - 1)


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
