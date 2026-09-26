"""The delta-rule short conv (depthwise causal conv1d over q/k/v) through fla."""

import torch

INT32_ELEMENTS = 2**31 - 1
_CHUNK_ELEMENTS = 2**30
_CHANNEL_ALIGN = 128


class _ContiguousGrad(torch.autograd.Function):
    """Identity whose backward hands on a contiguous gradient. The gradient of one channel chunk of a
    concatenation is a strided view whose row stride is the full width; fla's conv backward indexes it
    with int32 and overflows past 2**31 elements."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def short_conv(x, weight, activation, backend, cu_seqlens=None, cp_context=None):
    """Depthwise causal conv of ``x`` ``[b, s, C]`` with ``weight`` ``[C, K]`` through fla. fla indexes with
    int32, so inputs past 2**31 elements (long sequences at low TP) run as channel chunks of at most
    2**30; the conv is per channel, so this is exact."""
    from fla.modules.conv.causal_conv1d import causal_conv1d

    tokens = x.shape[0] * x.shape[1]
    kwargs = {
        "bias": None,
        "activation": activation,
        "backend": backend,
        "cu_seqlens": cu_seqlens,
        "cp_context": cp_context,
    }
    if tokens * x.shape[-1] <= INT32_ELEMENTS:
        return causal_conv1d(x=x, weight=weight, **kwargs)[0]
    width = max(_CHANNEL_ALIGN, _CHUNK_ELEMENTS // tokens // _CHANNEL_ALIGN * _CHANNEL_ALIGN)
    chunks = [
        _ContiguousGrad.apply(causal_conv1d(x=chunk.contiguous(), weight=chunk_weight, **kwargs)[0])
        for chunk, chunk_weight in zip(x.split(width, dim=-1), weight.split(width), strict=True)
    ]
    return torch.cat(chunks, dim=-1)
