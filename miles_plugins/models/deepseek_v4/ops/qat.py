import torch

from tile_kernels.quant import per_token_cast_back

from miles_plugins.models.deepseek_v4.ops.kernel.act_quant import _needs_portable_fp8, act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    gfx942 uses explicit PyTorch FN conversions for quantization and cast-back.
    TileKernels' HIP cast-back interprets FN bytes as FNUZ, halving values.
    This is activation QAT, not the TE recipe that controls training GEMMs.
    """
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

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
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
