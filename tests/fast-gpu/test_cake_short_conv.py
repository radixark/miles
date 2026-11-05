"""--linear-attention-conv-backend cake: the head-sharded short conv on the Cake kernels of the miles-ext-kernels
submodule (miles_ext_kernels.cake.conv1d), in both variants and both weight dtypes (GDN taps train in BF16, KDA taps
in FP32).

The conv matches a float64 torch reference to BF16 rounding (< 5e-3), det repeats bit for bit, and a GDN and a KDA
layer at TP 1 whose short conv runs on cake match the same layers on fla's conv.

    python tests/fast-gpu/test_cake_short_conv.py
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads
from miles_plugins.models import linear_attn

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import ReplicatedGDN, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=600, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the Cake short-conv kernels run on SM100a / SM103a / SM107a",
)

TOKENS, CHANNELS = 2048, 768
HIDDEN = 512
SEQLENS = [300, 1, 777, 970]
TOLERANCE = 2e-2


def _layer_args(variant: str = "det", **backends) -> SimpleNamespace:
    """The miles configuration fields the linear-attention layers read, with the given cake choices."""
    values = {"linear_attention_conv_backend": "fla", "linear_attention_output_norm_backend": "fla"} | backends
    ext = {"cake_conv1d_variant": variant, "cake_conv1d_silu_grad_preactivation": "bf16"}
    return SimpleNamespace(**values, miles_ext_kernels_args=ext)


def _cake_conv(variant: str, dtype: torch.dtype):
    linear_attn.miles_ext("conv1d", _layer_args(variant))
    return linear_attn.ShardedShortConv(CHANNELS, 4, tp_group=None, device="cuda", dtype=dtype, backend="cake")


def _reference(x, weight, cu_seqlens):
    x = x.double().transpose(1, 2)
    weight = weight.double()
    segments = [
        F.silu(F.conv1d(F.pad(x[..., start:end], (weight.shape[-1] - 1, 0)), weight, groups=weight.shape[0]))
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)
    ]
    return torch.cat(segments, dim=-1).transpose(1, 2)


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, TOKENS, CHANNELS, generator=g, device="cuda").bfloat16()
    grad = torch.randn(1, TOKENS, CHANNELS, generator=g, device="cuda").bfloat16()
    cu_seqlens = torch.tensor([0, 700, 701, TOKENS], device="cuda", dtype=torch.int32)
    return x, grad, cu_seqlens


def _run_conv(conv, x, grad, cu_seqlens):
    conv.weight.grad = None
    xi = x.clone().requires_grad_()
    out = conv(xi, cu_seqlens=cu_seqlens)
    out.backward(grad)
    torch.cuda.synchronize()
    return out.detach(), xi.grad, conv.weight.grad.clone()


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_cake_conv_matches_float64_reference(variant, weight_dtype):
    torch.manual_seed(0)
    conv = _cake_conv(variant, weight_dtype)
    x, grad, cu_seqlens = _inputs()

    out, dx, dweight = _run_conv(conv, x, grad, cu_seqlens)
    xr = x.double().requires_grad_()
    wr = conv.weight.detach()[:, 0].double().requires_grad_()
    out_ref = _reference(xr, wr.unsqueeze(1), cu_seqlens)
    out_ref.backward(grad.double())

    assert dweight.dtype == weight_dtype
    for actual, expected in ((out, out_ref), (dx, xr.grad), (dweight[:, 0], wr.grad)):
        assert ((actual.double() - expected).norm() / expected.norm()).item() < 5e-3


def test_det_conv_repeats_bit_for_bit():
    torch.manual_seed(0)
    conv = _cake_conv("det", torch.bfloat16)
    x, grad, cu_seqlens = _inputs()

    first = _run_conv(conv, x, grad, cu_seqlens)
    second = _run_conv(conv, x, grad, cu_seqlens)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.fixture
def megatron_tp1(nccl_world):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()


def _run_layer(layer, x, grad, cu_seqlens) -> dict[str, torch.Tensor]:
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out, _ = layer(x, packed_seq_params=packed(cu_seqlens))
    out.backward(grad)
    torch.cuda.synchronize()
    return {"out": out.detach(), "dx": x.grad, **{name: p.grad for name, p in layer.named_parameters()}}


@pytest.mark.parametrize("family", ["qwen3_5", "kda"])
def test_a_layer_with_the_cake_conv_matches_the_fla_conv(megatron_tp1, family):
    torch.manual_seed(1)
    if family == "kda":
        ref = ReplicatedKDA(HIDDEN, LinearAttnHeads(4, 4, 128, 128), torch.bfloat16).cuda()
    else:
        ref = ReplicatedGDN(family, HIDDEN, LinearAttnHeads(4, 8, 128, 128), torch.bfloat16).cuda()
    config = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=8, params_dtype=torch.bfloat16, bf16=True
    )
    fla_layer = build_layer(ref, config, args=_layer_args())
    cake_layer = build_layer(ref, config, args=_layer_args(linear_attention_conv_backend="cake"))
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(sum(SEQLENS), 1, HIDDEN, generator=g, device="cuda").bfloat16()
    grad = torch.randn(sum(SEQLENS), 1, HIDDEN, generator=g, device="cuda").bfloat16()
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)

    fla = _run_layer(fla_layer, x, grad, cu_seqlens)
    cake = _run_layer(cake_layer, x, grad, cu_seqlens)

    errors = {name: rel_err(fla[name], cake[name]) for name in fla}
    assert all(error <= TOLERANCE for error in errors.values()), errors


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
