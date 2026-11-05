"""--kda-backend cake: the KDA layers of Kimi-K3 (head-sharded KimiDeltaAttention) and GLM-5.3-Flash (Glm5NextKDA)
on the deterministic Cake kernels of the miles-ext-kernels submodule (miles_ext_kernels.kda), against the same layers
on fla and the replicated fla reference of linear_attn_reference.py.

At TP 1 with a ragged packed batch: the cake layer's output, input gradient and every parameter gradient stay within
TOLERANCE of the fla layer (the cake kernels take a BF16 beta where fla takes FP32; GATE_PARAM_TOLERANCE for the
token-summed A_log and dt_bias gradients), a second pass repeats every one of them bit for bit, and more than 64 heads
per rank is refused when the layer is built.

    python tests/fast-gpu/test_cake_kda_layer.py
"""

import os
import sys

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import KDA_LOWER_BOUND, ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the Cake KDA kernels run on SM100a / SM103a / SM107a",
)

HIDDEN = 512
HEAD_DIM = 128
SEQLENS = [300, 1, 777, 64]
TOLERANCE = 2e-2
# A_log and dt_bias get the gate gradient summed over every token of the batch. With BF16 operands fla and cake are
# each 1-4 % (relative norm) from the exact FP64 recurrence there, in different directions (B200, heads 4 and 12; fla:
# dt_bias 3.4 %, A_log 2.0-3.2 %; cake: dt_bias 3.5-3.6 %, A_log 1.1-2.1 %), so the layers differ by up to 5.1 % on
# B200 and 5.6 % on GB300. The bound leaves a margin of 1.8x over the largest.
GATE_PARAM_TOLERANCE = 1e-1
GATE_PARAMS = ("linear_attn.A_log", "linear_attn.dt_bias")


@pytest.fixture
def megatron_tp1(nccl_world):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()


def _config(heads: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=heads,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=1,
    )


def _batch(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(SEQLENS)
    x = torch.randn(total, 1, HIDDEN, generator=g, device="cuda").bfloat16()
    grad = torch.randn(total, 1, HIDDEN, generator=g, device="cuda").bfloat16()
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return x, grad, cu_seqlens


def _run_layer(layer, x, grad, cu_seqlens) -> dict[str, torch.Tensor]:
    """Output, input gradient and every parameter gradient of one forward and backward."""
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out, _ = layer(x, packed_seq_params=packed(cu_seqlens))
    out.backward(grad)
    torch.cuda.synchronize()
    return {"out": out.detach(), "dx": x.grad, **{name: p.grad for name, p in layer.named_parameters()}}


@pytest.mark.parametrize("heads", [4, 12])
def test_kimi_kda_layer_on_cake_matches_fla_and_repeats_bit_for_bit(megatron_tp1, heads):
    torch.manual_seed(heads)
    ref = ReplicatedKDA(HIDDEN, LinearAttnHeads(heads, heads, HEAD_DIM, HEAD_DIM), torch.bfloat16).cuda()
    config = _config(heads)
    fla_layer = build_layer(ref, config, backend="fla")
    cake_layer = build_layer(ref, config, backend="cake")
    x, grad, cu_seqlens = _batch(seed=heads)

    fla = _run_layer(fla_layer, x, grad, cu_seqlens)
    cake = _run_layer(cake_layer, x, grad, cu_seqlens)
    again = _run_layer(cake_layer, x, grad, cu_seqlens)

    assert fla.keys() == cake.keys()
    errors = {name: rel_err(fla[name], cake[name]) for name in fla}
    bounds = {name: GATE_PARAM_TOLERANCE if name in GATE_PARAMS else TOLERANCE for name in fla}
    assert all(errors[name] <= bounds[name] for name in fla), errors
    assert all(torch.equal(cake[name], again[name]) for name in cake), "the cake layer is not bit-reproducible"


def test_kimi_kda_layer_refuses_more_than_64_heads_per_rank_on_cake(megatron_tp1):
    ref = ReplicatedKDA(HIDDEN, LinearAttnHeads(96, 96, HEAD_DIM, HEAD_DIM), torch.bfloat16).cuda()

    with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
        build_layer(ref, _config(96), backend="cake")


def _glm_kda(backend: str, heads: int, state=None):
    from miles_plugins.models.glm5_next.kda import Glm5NextKDA

    layer = Glm5NextKDA(HIDDEN, heads, HEAD_DIM, 4, KDA_LOWER_BOUND, 1e-6, kda_backend=backend).cuda()
    layer.to(torch.bfloat16)
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    if state is None:
        torch.manual_seed(0)
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if name == "A_log":
                    param.copy_(torch.log(torch.empty_like(param).uniform_(1, 16)))
                elif name == "dt_bias":
                    param.uniform_(0, 1)
                elif param.dim() > 1:
                    param.normal_(0, param.shape[-1] ** -0.5)
    else:
        layer.load_state_dict(state)
    return layer


def _run_glm(layer, x, grad, cu_seqlens) -> dict[str, torch.Tensor]:
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out = layer(x, cu_seqlens=cu_seqlens)
    out.backward(grad)
    torch.cuda.synchronize()
    return {"out": out.detach(), "dx": x.grad, **{name: p.grad for name, p in layer.named_parameters()}}


def test_glm_kda_on_cake_matches_fla_and_repeats_bit_for_bit():
    """64 heads, not head-sharded; q / k reach the kernels as strided split views of the conv output."""
    heads = 64
    fla_layer = _glm_kda("fla", heads)
    cake_layer = _glm_kda("cake", heads, state=fla_layer.state_dict())
    x, grad, cu_seqlens = _batch(seed=5)
    x, grad = x.transpose(0, 1), grad.transpose(0, 1)  # [1, T, hidden]

    fla = _run_glm(fla_layer, x, grad, cu_seqlens)
    cake = _run_glm(cake_layer, x, grad, cu_seqlens)
    again = _run_glm(cake_layer, x, grad, cu_seqlens)

    errors = {name: rel_err(fla[name], cake[name]) for name in fla}
    assert all(error <= TOLERANCE for error in errors.values()), errors
    assert all(torch.equal(cake[name], again[name]) for name in cake), "the cake layer is not bit-reproducible"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
