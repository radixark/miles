"""GDN gated RMSNorm: the serving norm-before-gate kernel's forward, an FP32 reference's backward."""

import torch
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd

from miles_plugins.top.kernels.surrogate import reference_mode

# Rows of [T*HV, V] per backward chunk; bounds the FP32 reference's footprint at long context.
_CHUNK_ROWS = 1 << 18


def _gated_norm(x, z, weight, eps, activation):
    out, _, _ = _layer_norm_fwd(
        x.contiguous(),
        weight,
        None,
        eps,
        z=z.contiguous(),
        norm_before_gate=True,
        is_rms_norm=True,
        activation=activation,
    )
    return out


def gated_norm_reference(x, z, weight, eps, activation):
    xf, zf = x.float(), z.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    gate = zf * torch.sigmoid(zf) if activation == "swish" else torch.sigmoid(zf)
    return y * gate


class _GatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps, activation):
        ctx.save_for_backward(x, z, weight)
        ctx.eps, ctx.activation = eps, activation
        return _gated_norm(x, z, weight, eps, activation)

    @staticmethod
    def backward(ctx, grad):
        x, z, weight = ctx.saved_tensors
        dx, dz = torch.empty_like(x), torch.empty_like(z)
        dw = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        for lo in range(0, x.shape[0], _CHUNK_ROWS):
            hi = min(lo + _CHUNK_ROWS, x.shape[0])
            xc = x[lo:hi].detach().requires_grad_()
            zc = z[lo:hi].detach().requires_grad_()
            wc = weight.detach().float().requires_grad_()
            with torch.enable_grad(), reference_mode():
                ref = gated_norm_reference(xc, zc, wc, ctx.eps, ctx.activation)
                gx, gz, gw = torch.autograd.grad(ref, (xc, zc, wc), grad[lo:hi].float())
            dx[lo:hi], dz[lo:hi] = gx, gz
            dw += gw
        return dx, dz, dw.to(weight.dtype), None, None


def delegated_gated_norm(x, z, weight, eps=1e-6, activation="swish"):
    """Serving's RMSNormGated over [..., V]: ``swish`` (SiLU) gate for Qwen3.5/3.6/3.8-27B, ``sigmoid`` for Flash-Next."""
    activation = "swish" if activation == "silu" else activation
    if activation not in ("swish", "sigmoid"):
        raise NotImplementedError(f"[top] gated norm activation {activation!r}")
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    z2 = z.reshape_as(x2)
    if not torch.is_grad_enabled() or not (x.requires_grad or z.requires_grad or weight.requires_grad):
        return _gated_norm(x2, z2, weight, eps, activation).reshape(shape)
    return _GatedNorm.apply(x2, z2, weight, eps, activation).reshape(shape)
