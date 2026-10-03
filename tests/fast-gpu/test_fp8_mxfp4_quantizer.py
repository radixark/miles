"""Weight sync to an MXFP4 MiMo-V2 engine must emit the tensors of the official checkpoint.

`quant_method: fp8` with `store_dtype: mxfp4`: routed experts packed as MXFP4 in the checkpoint's
nibble and scale convention, `ignored_layers` left BF16, and a fused `qkv_proj` block-quantized per
kv-head shard, which the converter (the checkpoint's reader) decodes back. On Blackwell the FP8
scales take DeepGEMM's UE8M0 layout; each test pins the scale format so both run on either GPU.
"""

import json
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cuda_ci

# The FP8 block cast is a triton kernel; the UE8M0 layout comes from DeepGEMM's layout utils.
register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

import torch
from tools.convert_mimo_v2_to_bf16 import dequant_fused_qkv, dequant_mxfp4, split_fused_qkv

from miles.backends.megatron_utils.megatron_to_hf.processors import quantize_params, quantizer_fp8
from miles_plugins.megatron_bridge.mimo_v2 import fuse_qkv

QCFG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "store_dtype": "mxfp4",
    "mxfp4_block_size": 32,
    "ignored_layers": ["model.layers.0.self_attn.o_proj"],
}
# num_attention_heads 8 over 4 shards: a GA shard is 2*192 + 192 + 128 = 704 rows, not whole 128-row blocks.
MODEL = {
    "num_key_value_heads": 4,
    "num_attention_heads": 8,
    "head_dim": 192,
    "v_head_dim": 128,
    "swa_num_key_value_heads": 8,
    "swa_num_attention_heads": 8,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
}
HIDDEN = 256


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    # fp32 block scales, as on Hopper; test_fused_qkv_ue8m0_* switch to the Blackwell layout.
    monkeypatch.setattr(quantizer_fp8, "_get_scale_format", lambda *_: None)
    (tmp_path / "config.json").write_text(json.dumps(MODEL))
    return SimpleNamespace(
        hf_checkpoint=str(tmp_path), sglang_moe_runner_backend="marlin", sglang_moe_a2a_backend="none"
    )


def _checkpoint_mxfp4(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random MXFP4 blocks whose largest code is 4 or 6, like every block of the official checkpoint."""
    codes = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    blocks = codes.view(rows, cols // 32, 32)
    top = torch.randint(6, 8, (rows, cols // 32), dtype=torch.uint8)  # e2m1 codes 6 and 7 are 4.0 and 6.0
    blocks[..., 0] = top | (blocks[..., 0] & 0x8)
    blocks[..., 1:] = torch.minimum(blocks[..., 1:] & 0x7, top[..., None]) | (blocks[..., 1:] & 0x8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale = torch.randint(110, 125, (rows, cols // 32), dtype=torch.uint8)
    return packed, scale


def test_routed_experts_repack_to_the_checkpoint_bytes(args):
    packed, scale = _checkpoint_mxfp4(512, HIDDEN)
    weight = dequant_mxfp4(packed, scale).to(torch.bfloat16).cuda()
    name = "model.layers.1.mlp.experts.3.gate_proj.weight"

    out = dict(
        quantize_params(args, "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight3", [(name, weight)], QCFG)
    )

    assert set(out) == {name, "model.layers.1.mlp.experts.3.gate_proj.weight_scale"}
    torch.testing.assert_close(out[name].cpu(), packed, rtol=0, atol=0)
    torch.testing.assert_close(out["model.layers.1.mlp.experts.3.gate_proj.weight_scale"].cpu(), scale, rtol=0, atol=0)


def test_ignored_layers_stay_bf16_and_dense_weights_become_fp8(args):
    for megatron_name, hf_name, shape in (
        (
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            (HIDDEN, HIDDEN),
        ),
        ("decoder.layers.1.mlp.router.weight", "model.layers.1.mlp.gate.weight", (8, HIDDEN)),
    ):
        weight = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        ((name, out),) = quantize_params(args, f"module.module.{megatron_name}", [(hf_name, weight)], QCFG)
        assert name == hf_name and out is weight

    dense = [("model.layers.0.mlp.down_proj.weight", torch.randn(HIDDEN, 512, device="cuda", dtype=torch.bfloat16))]
    out = dict(quantize_params(args, "module.module.decoder.layers.0.mlp.linear_fc2.weight", dense, QCFG))
    assert out["model.layers.0.mlp.down_proj.weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.mlp.down_proj.weight_scale_inv"].shape == (2, 4)


@pytest.mark.parametrize("is_swa", [False, True])
def test_fused_qkv_matches_the_checkpoint_layout(args, is_swa):
    prefix = "swa_" if is_swa else ""
    heads, kv_heads = MODEL[f"{prefix}num_attention_heads"], MODEL[f"{prefix}num_key_value_heads"]
    q = torch.randn(heads * 192, HIDDEN)
    k = torch.randn(kv_heads * 192, HIDDEN)
    v = torch.randn(kv_heads * 128, HIDDEN)
    fused = fuse_qkv(q, k, v, MODEL["num_key_value_heads"])
    for part, expected in zip(split_fused_qkv(fused, MODEL, is_swa), (q, k, v)):
        torch.testing.assert_close(part, expected, rtol=0, atol=0)

    name = "model.layers.1.self_attn.qkv_proj.weight"
    bf16 = fused.to(torch.bfloat16).cuda()
    out = dict(
        quantize_params(args, "module.module.decoder.layers.1.self_attention.linear_qkv.weight", [(name, bf16)], QCFG)
    )
    qweight, scale = out[name], out["model.layers.1.self_attn.qkv_proj.weight_scale_inv"]
    shard_rows = fused.shape[0] // 4
    assert qweight.dtype == torch.float8_e4m3fn and scale.shape == (4 * -(-shard_rows // 128), HIDDEN // 128)
    decoded = dequant_fused_qkv(qweight.cpu(), scale.cpu(), shards=4)
    torch.testing.assert_close(decoded.float(), bf16.float().cpu(), rtol=0.07, atol=0.02)

    # A checkpoint tensor comes back to itself: decode, round to BF16 as the trainer holds it, re-encode.
    requant = dict(
        quantize_params(
            args,
            "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
            [(name, decoded.to(torch.bfloat16).cuda())],
            QCFG,
        )
    )
    assert torch.equal(requant[name].view(torch.uint8), qweight.view(torch.uint8))


def _dequant_ue8m0_rows(qweight: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Per-row UE8M0 scales (four E8M0 bytes per int32, one row per weight row) applied per 128 columns."""
    rows, cols = qweight.shape
    exponents = packed.contiguous().view(torch.uint8).view(rows, -1)[:, : -(-cols // 128)]
    scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    return qweight.float() * scale.repeat_interleave(128, dim=1)[:, :cols]


@pytest.mark.parametrize("is_swa", [False, True])
def test_fused_qkv_ue8m0_scales_stack_per_shard(args, monkeypatch, is_swa):
    """Blackwell: every shard carries DeepGEMM's per-row UE8M0 scales; they stack along rows."""
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0

    monkeypatch.setattr(quantizer_fp8, "_get_scale_format", lambda *_: "ue8m0")
    prefix = "swa_" if is_swa else ""
    heads, kv_heads = MODEL[f"{prefix}num_attention_heads"], MODEL[f"{prefix}num_key_value_heads"]
    fused = fuse_qkv(
        torch.randn(heads * 192, HIDDEN), torch.randn(kv_heads * 192, HIDDEN), torch.randn(kv_heads * 128, HIDDEN), 4
    )
    bf16 = fused.to(torch.bfloat16).cuda()
    name = "model.layers.1.self_attn.qkv_proj.weight"
    out = dict(
        quantize_params(args, "module.module.decoder.layers.1.self_attention.linear_qkv.weight", [(name, bf16)], QCFG)
    )
    qweight, scale = out[name], out["model.layers.1.self_attn.qkv_proj.weight_scale_inv"]
    shard_rows = fused.shape[0] // 4
    assert qweight.dtype == torch.float8_e4m3fn and scale.dtype == torch.int32 and scale.shape[0] == 4 * shard_rows

    # Each slice SGLang's loader cuts out equals what its load-time requant makes of that shard.
    for bf16_shard, w_shard, s_shard in zip(bf16.chunk(4), qweight.chunk(4), scale.chunk(4)):
        ref_w, ref_s = quant_weight_ue8m0(bf16_shard, [128, 128])
        assert torch.equal(w_shard.view(torch.uint8), ref_w.view(torch.uint8))
        assert torch.equal(s_shard, transform_scale_ue8m0(ref_s, mn=shard_rows))
        torch.testing.assert_close(_dequant_ue8m0_rows(w_shard, s_shard), bf16_shard.float(), rtol=0.15, atol=0.02)
