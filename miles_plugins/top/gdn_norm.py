"""Offset (Gemma) RMSNorm for the Qwen GDN hybrids: SGLang's kernel forward, TE's RMSNorm backward.

y = x * rsqrt(mean(x^2) + eps) * (1 + w), so the derivative is RMSNorm's with weight 1 + w.
"""

import torch
from sglang.srt.layers.layernorm import GemmaRMSNorm

from miles_plugins.top.kernels.rms_norm_backward import rms_norm_backward


class _GemmaNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, norm):
        ctx.save_for_backward(x, weight)
        ctx.eps = norm.variance_epsilon
        # The CUDA kernel reads weight.data, so trained weights take effect without a refresh.
        return norm(x.contiguous())

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx, _, dw = rms_norm_backward(x, weight.float() + 1.0, ctx.eps, grad.contiguous())
        return dx, dw.to(weight.dtype), None


class _GemmaResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, norm):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = norm.variance_epsilon
        # The fused kernel writes both buffers; never mutate Megatron's tensors or saved operands.
        return norm(
            x.clone(memory_format=torch.contiguous_format),
            residual.clone(memory_format=torch.contiguous_format),
        )

    @staticmethod
    def backward(ctx, grad, grad_residual):
        x, residual, weight = ctx.saved_tensors
        if grad is None:
            grad = torch.zeros_like(x)
        dx, dr, dw = rms_norm_backward(
            x, weight.float() + 1.0, ctx.eps, grad, residual=residual, grad_residual=grad_residual
        )
        return dx, dr, dw.to(weight.dtype), None


class TopGemmaRMSNorm(torch.nn.Module):
    def __init__(self, config, hidden_size, eps=1e-6, role="", **kwargs):
        super().__init__()
        self.hidden_size, self.role = hidden_size, role
        self._packed_residual = role in ("input_layernorm", "pre_mlp_layernorm", "final_layernorm")
        self.requires_residual_pair = role != "input_layernorm"
        if self._packed_residual and config.pipeline_hidden_size != 2 * config.hidden_size:
            raise ValueError("Qwen GDN requires packed residual transport")
        norm = GemmaRMSNorm(hidden_size, eps)
        object.__setattr__(self, "_sgl", norm)
        self.weight = norm.weight
        self.eps = norm.variance_epsilon

    def rows(self, x, weight):
        """Serving's norm of x [N, hidden_size] with TE's backward, the weight gradient going to ``weight`` (a
        surrogate's reference passes its leaf of ``self.weight``)."""
        return _GemmaNorm.apply(x, weight, self._sgl)

    def forward(self, x):
        residual = None
        if self._packed_residual and x.shape[-1] == 2 * self.hidden_size:
            x, residual = x.split(self.hidden_size, dim=-1)
        elif x.shape[-1] != self.hidden_size or (self._packed_residual and self.requires_residual_pair):
            raise ValueError(f"Invalid packed norm input for {self.role}")
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        if residual is None:
            y = _GemmaNorm.apply(x, self.weight, self._sgl).reshape(shape)
            r = x.reshape(shape)
        else:
            y, r = _GemmaResidualNorm.apply(x, residual.reshape_as(x), self.weight, self._sgl)
            y, r = y.reshape(shape), r.reshape(shape)
        if self.role == "final_layernorm" or not self._packed_residual:
            return y
        return y if residual is None else (y, r)
