"""--linear-attention-output-norm-backend cake: the SiLU-gated output RMSNorm (RMSNorm(x) * w * SiLU(z)) of the GDN
layers on the Cake kernels of the miles-ext-kernels submodule (miles_ext_kernels.gated_rmsnorm).

On ``[T, H, 128]`` rows whose gate is a column slice of a wider projection, as the layer passes them, both variants
stay within NORM_OVER_FLA times fla's error against a float64 reference (plus ERROR_FLOOR) for the output and every
gradient, and det repeats bit for bit. A GDN layer at TP 1 with the cake output norm matches the same layer on fla's
norm, and a KDA layer (sigmoid gate) refuses the cake output norm when it is built.

    python tests/fast-gpu/test_cake_gated_norm.py
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
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
    reason="the Cake gated RMSNorm kernels run on SM100a / SM103a / SM107a",
)

TOKENS, HEADS, HEAD_DIM = 3001, 8, 128
EPS = 1e-6
NORM_OVER_FLA = 2.0
ERROR_FLOOR = 1e-3
HIDDEN = 512
SEQLENS = [300, 1, 777, 970]
TOLERANCE = 2e-2


def _layer_args(variant: str = "det", **backends) -> SimpleNamespace:
    """The miles configuration fields the linear-attention layers read, with the given cake choices."""
    values = {"linear_attention_conv_backend": "fla", "linear_attention_output_norm_backend": "fla"} | backends
    return SimpleNamespace(**values, miles_ext_kernels_args={"cake_gated_rmsnorm_variant": variant})


def _inputs():
    """x ``[T, H, 128]``, the gate as a column slice of a wider projection, the BF16 weight, the output gradient."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(TOKENS, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
    projection = torch.randn(TOKENS, 3 * HEADS * HEAD_DIM, generator=g, device="cuda").bfloat16()
    gate = projection[:, HEADS * HEAD_DIM : 2 * HEADS * HEAD_DIM].view(TOKENS, HEADS, HEAD_DIM)
    weight = (1 + 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")).bfloat16()
    grad = torch.randn(TOKENS, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
    return x, gate, weight, grad


def _reference(x, gate, weight):
    x, gate, weight = x.double(), gate.double(), weight.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight * torch.nn.functional.silu(gate)


def _run(norm, x, gate, weight, grad):
    leaves = [t.detach().clone().requires_grad_() for t in (x, gate, weight)]
    out = norm(*leaves)
    out.backward(grad.to(out.dtype))
    torch.cuda.synchronize()
    return [out.detach(), *(leaf.grad for leaf in leaves)]


def _fla(x, gate, weight):
    from fla.modules.fused_norm_gate import rms_norm_gated

    rows = x.reshape(-1, HEAD_DIM), gate.reshape(-1, HEAD_DIM)
    return rms_norm_gated(*rows, weight, None, "silu", eps=EPS).view_as(x)


def _cake(variant):
    norm = linear_attn.miles_ext("gated_rmsnorm", _layer_args(variant))
    return lambda x, gate, weight: norm.rmsnorm_silu_gate(x, gate, weight, EPS)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_cake_gated_norm_is_as_accurate_as_fla(variant):
    x, gate, weight, grad = _inputs()

    reference = _run(_reference, x, gate, weight, grad)
    fla = _run(_fla, x, gate, weight, grad)
    cake = _run(_cake(variant), x, gate, weight, grad)

    for name, ref, fla_value, cake_value in zip(("out", "dx", "dgate", "dweight"), reference, fla, cake, strict=True):
        fla_error, cake_error = rel_err(ref, fla_value), rel_err(ref, cake_value)
        assert cake_error <= NORM_OVER_FLA * fla_error + ERROR_FLOOR, (name, cake_error, fla_error)


def test_det_gated_norm_repeats_bit_for_bit():
    x, gate, weight, grad = _inputs()
    cake = _cake("det")

    first, second = _run(cake, x, gate, weight, grad), _run(cake, x, gate, weight, grad)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.fixture
def megatron_tp1(nccl_world):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=8, params_dtype=torch.bfloat16, bf16=True
    )


def _run_layer(layer, x, grad, cu_seqlens) -> dict[str, torch.Tensor]:
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out, _ = layer(x, packed_seq_params=packed(cu_seqlens))
    out.backward(grad)
    torch.cuda.synchronize()
    return {"out": out.detach(), "dx": x.grad, **{name: p.grad for name, p in layer.named_parameters()}}


def test_a_gdn_layer_with_the_cake_output_norm_matches_the_fla_norm(megatron_tp1):
    torch.manual_seed(1)
    ref = ReplicatedGDN("qwen3_5", HIDDEN, LinearAttnHeads(4, 8, 128, 128), torch.bfloat16).cuda()
    fla_layer = build_layer(ref, _config(), args=_layer_args())
    cake_layer = build_layer(ref, _config(), args=_layer_args(linear_attention_output_norm_backend="cake"))
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(sum(SEQLENS), 1, HIDDEN, generator=g, device="cuda").bfloat16()
    grad = torch.randn(sum(SEQLENS), 1, HIDDEN, generator=g, device="cuda").bfloat16()
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)

    fla = _run_layer(fla_layer, x, grad, cu_seqlens)
    cake = _run_layer(cake_layer, x, grad, cu_seqlens)

    errors = {name: rel_err(fla[name], cake[name]) for name in fla}
    assert all(error <= TOLERANCE for error in errors.values()), errors


def test_a_kda_layer_refuses_the_cake_output_norm(megatron_tp1):
    ref = ReplicatedKDA(HIDDEN, LinearAttnHeads(4, 4, 128, 128), torch.bfloat16).cuda()

    with pytest.raises(ValueError, match="silu|SiLU"):
        build_layer(ref, _config(), args=_layer_args(linear_attention_output_norm_backend="cake"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
