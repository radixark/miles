"""Megatron's SelfAttention with the Qwen3.5 full-attention settings (output gate, zero-centered q/k RMSNorm, partial
RoPE) on CakeGQACoreAttention (--full-attention-backend cake) against Transformer Engine's core attention.

One packed micro-batch, its alignment pad as the last segment, runs through three layers with the same weights that
differ only in their core attention: Transformer Engine's, the Cake kernels' and an FP64 one. At the tensor-parallel
local head pairs 8/2 and 6/1, the Cake layer's output, input gradient and every parameter gradient stay within
CAKE_OVER_TE times the Transformer Engine layer's error against the FP64 layer, plus ERROR_FLOOR. Also: the Cake layer
loads Transformer Engine's state dict as is and has its sharded checkpoint keys; selective core-attention recompute
leaves the layer unchanged; the MTP layer, which Megatron builds from the decoder's last (full-attention) layer spec,
runs the Cake core attention. On SM103 the Transformer Engine layer runs on its unfused attention
(TE_UNFUSED_CAPABILITIES).

    python tests/fast-gpu/test_cake_full_attention.py
"""

import dataclasses
import itertools
import sys

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.cake_full_attention import CakeGQACoreAttention, apply_cake_full_attention

register_cuda_ci(est_time=600, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the Cake GQA kernels run on SM100a / SM103a / SM107a",
)

HIDDEN = 1024
HEAD_DIM = 256
# three sequences, then the pad that aligns the packed tokens to a multiple of 128 as its own segment
SEGMENTS = (1000, 517, 2049, 18)
ROTARY_PERCENT = 0.25
ROTARY_BASE = 10_000_000
LAYER_NUMBER = 4  # a full-attention layer of Qwen3.5
# Calibration budget against the FP64 layer: the Cake layer's relative L2 error over Transformer Engine's.
CAKE_OVER_TE = 1.25
ERROR_FLOOR = 1e-3
# Two Cake runs differ only by the FP32 summation order of dQ before its single BF16 rounding.
DQ_ORDER_REL_L2 = 2**-7
# Transformer Engine runs this layer's core attention on FlashAttention 4 (its cuDNN fused attention does not take
# it). On SM103 the head-dim-256 backward of FlashAttention 4.0.0b37 rejects the causal window Transformer Engine passes
# as local attention, so there the reference layer runs on Transformer Engine's unfused attention (NVTE_FLASH_ATTN=0),
# whose error against FP64 is about 1.5x FlashAttention's.
TE_UNFUSED_CAPABILITIES = ((10, 3),)


@pytest.fixture
def megatron_tp1(nccl_world):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()


def _config(heads: int, groups: int, **overrides) -> TransformerConfig:
    values = dict(
        num_layers=4,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=heads,
        num_query_groups=groups,
        kv_channels=HEAD_DIM,
        attention_output_gate=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    return TransformerConfig(**(values | overrides))


class Fp64CoreAttention(torch.nn.Module):
    """Causal GQA per packed segment in FP64, rounded to the operands' dtype on the way out."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type, softmax_scale=None, **kwargs):
        super().__init__()
        self.softmax_scale = config.kv_channels**-0.5 if softmax_scale is None else softmax_scale

    def forward(
        self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None
    ):
        bounds = packed_seq_params.cu_seqlens_q.tolist()
        group = query.shape[1] // key.shape[1]
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            q = query[start:end].double()
            k = key[start:end].double().repeat_interleave(group, dim=1)
            v = value[start:end].double().repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) * self.softmax_scale
            causal = torch.ones(end - start, end - start, dtype=torch.bool, device=q.device).tril()
            probs = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            outs.append(torch.einsum("hqk,khd->qhd", probs, v))
        return torch.cat(outs).to(query.dtype).reshape(query.shape[0], -1)


def _layers(config: TransformerConfig) -> dict[str, torch.nn.Module]:
    """The three SelfAttention layers, the Cake and FP64 ones loading the Transformer Engine one's weights."""
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    te_spec = layer_spec.submodules.self_attention
    fp64_spec = dataclasses.replace(
        te_spec, submodules=dataclasses.replace(te_spec.submodules, core_attention=Fp64CoreAttention)
    )
    te = build_module(te_spec, config=config, layer_number=LAYER_NUMBER)
    cake = build_module(
        apply_cake_full_attention(layer_spec).submodules.self_attention, config=config, layer_number=LAYER_NUMBER
    )
    fp64 = build_module(fp64_spec, config=config, layer_number=LAYER_NUMBER)
    state = te.state_dict()
    cake.load_state_dict(state)  # strict: one set of keys, Transformer Engine's extra state included
    loaded = fp64.load_state_dict(state, strict=False)
    assert loaded.missing_keys == [] and set(loaded.unexpected_keys) <= {"core_attention._extra_state"}
    return {"te": te, "cake": cake, "fp64": fp64}


def _batch():
    """Hidden states ``[T, 1, HIDDEN]``, their output gradient, the packed boundaries and the RoPE frequencies."""
    total = sum(SEGMENTS)
    g = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(total, 1, HIDDEN, generator=g, device="cuda").bfloat16()
    grad = torch.randn(total, 1, HIDDEN, generator=g, device="cuda").bfloat16()
    cu = torch.tensor([0, *itertools.accumulate(SEGMENTS)], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(SEGMENTS), max_seqlen_kv=max(SEGMENTS)
    )
    rotary = RotaryEmbedding(kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROTARY_BASE)(total)
    return hidden, grad, packed, rotary


def _run(layer, batch) -> dict[str, torch.Tensor]:
    """The output, the input gradient and every parameter gradient of one forward and backward."""
    hidden, grad, packed, rotary = batch
    x = hidden.clone().requires_grad_()
    out, _ = layer(x, attention_mask=None, rotary_pos_emb=rotary, packed_seq_params=packed)
    out.backward(grad)
    torch.cuda.synchronize()
    return {"out": out.detach(), "dx": x.grad, **{name: p.grad for name, p in layer.named_parameters()}}


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.double() - ref.double()).norm() / ref.double().norm())


@pytest.mark.parametrize("heads, groups", [(8, 2), (6, 1)])
def test_the_cake_layer_matches_transformer_engine_against_fp64(heads, groups, megatron_tp1, monkeypatch):
    if torch.cuda.get_device_capability() in TE_UNFUSED_CAPABILITIES:
        monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    layers = _layers(_config(heads, groups))
    assert type(layers["te"].core_attention) is TEDotProductAttention
    assert type(layers["cake"].core_attention) is CakeGQACoreAttention
    assert layers["cake"].sharded_state_dict().keys() == layers["te"].sharded_state_dict().keys()
    batch = _batch()

    results = {name: _run(layer, batch) for name, layer in layers.items()}

    ref = results.pop("fp64")
    assert results["te"].keys() == results["cake"].keys() == ref.keys()
    errors = {
        name: (_rel_l2(results["te"][name], ref[name]), _rel_l2(results["cake"][name], ref[name])) for name in ref
    }
    print(
        f"REL_L2 vs fp64 {heads}/{groups} " + " ".join(f"{n}=te:{t:.2e},cake:{c:.2e}" for n, (t, c) in errors.items())
    )
    over = {name: (te, cake) for name, (te, cake) in errors.items() if cake > CAKE_OVER_TE * te + ERROR_FLOOR}
    assert over == {}, f"the Cake layer exceeds {CAKE_OVER_TE} x the Transformer Engine error + {ERROR_FLOOR}: {over}"


def test_selective_core_attention_recompute_leaves_the_layer_unchanged(megatron_tp1):
    spec = apply_cake_full_attention(get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True))
    spec = spec.submodules.self_attention
    plain = build_module(spec, config=_config(8, 2), layer_number=LAYER_NUMBER)
    recomputed = build_module(
        spec,
        config=_config(8, 2, recompute_granularity="selective", recompute_modules=["core_attn"]),
        layer_number=LAYER_NUMBER,
    )
    recomputed.load_state_dict(plain.state_dict())
    assert recomputed.checkpoint_core_attention and not plain.checkpoint_core_attention
    batch = _batch()

    expected, got = _run(plain, batch), _run(recomputed, batch)

    assert torch.equal(got.pop("out"), expected.pop("out"))
    # every gradient reaches the input and the weights through dQ, whose summation order varies run to run
    errors = {name: _rel_l2(got[name], expected[name]) for name in expected}
    assert max(errors.values()) <= DQ_ORDER_REL_L2, errors


def test_the_mtp_layer_inherits_the_cake_core_attention(megatron_tp1):
    config = _config(8, 2, mtp_num_layers=1)
    block = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    # the Qwen3.5 / Qwen3-Next spec swaps every full-attention layer, the decoder's last layer among them
    block.layer_specs[-1] = apply_cake_full_attention(block.layer_specs[-1])

    mtp = get_gpt_mtp_block_spec(config, block, use_transformer_engine=True)

    inner = mtp.layer_specs[0].submodules.mtp_model_layer
    assert inner.submodules.self_attention.submodules.core_attention is CakeGQACoreAttention
    layer = build_module(inner, config=config, layer_number=1, is_mtp_layer=True)
    assert type(layer.self_attention.core_attention) is CakeGQACoreAttention
    hidden, grad, packed, rotary = _batch()
    x = hidden.clone().requires_grad_()
    out, _ = layer(x, attention_mask=None, rotary_pos_emb=rotary, packed_seq_params=packed)
    out.backward(grad)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
