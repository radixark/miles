from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_dsv41 import block_cast_fp32

RELEASE_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}
LAYER = "module.module.decoder.layers.20."
ARGS = SimpleNamespace(backend=SimpleNamespace(vocab_size=129280))


def _sync(name, param):
    return dict(convert_to_hf(ARGS, "deepseekv41", LAYER + name, param, RELEASE_CONFIG))


def _on_grid_master(shape, seed=0):
    """A master that is the exact BF16 dequant of E4M3 codes with 32x32 power-of-two scales, as the release loads."""
    g = torch.Generator().manual_seed(seed)
    codes = (torch.randn(shape, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    exps = torch.randint(-14, -10, (shape[0] // 32, shape[1] // 32), generator=g).float()
    scale = torch.exp2(exps).repeat_interleave(32, 0).repeat_interleave(32, 1)
    master = (codes * scale).to(torch.bfloat16)
    assert torch.equal(master.float(), codes * scale)
    return master


def _dequant(codes, scale, block):
    return codes.float() * scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)


def _assert_dequant_close(codes, scale, block, master):
    """Within E4M3's rounding: half a step relative, or half a subnormal step of the block's scale."""
    expanded = scale.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    err = (_dequant(codes, scale, block) - master.float()).abs()
    assert torch.all(err <= torch.maximum(master.float().abs() * 2**-4, expanded * 2**-10)), float(err.max())


def _release_rule(weight):
    """The release's requant rule: 32x32 blocks, scale 2^ceil(log2(amax / 448))."""
    x = weight.float()
    n, k = x.shape
    amax = x.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / 448)))
    return ((x.view(n // 32, 32, k // 32, 32) / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).view(
        n, k
    )


def test_dense_projections_sync_as_32x32_fp8_with_fp32_scales():
    master = _on_grid_master((64, 96))
    for name, hf in [
        ("self_attention.linear_q_down_proj.weight", "layers.20.attn.wq_a"),
        ("self_attention.linear_q_up_proj.weight", "layers.20.attn.wq_b"),
        ("self_attention.linear_kv_proj.weight", "layers.20.attn.wkv"),
        ("self_attention.linear_proj.weight", "layers.20.attn.wo_b"),
        ("self_attention.core_attention.indexer.linear_wq_b.weight", "layers.20.attn.indexer.wq_b"),
        ("v41_engram.linear_wkv.weight", "layers.20.engram.wkv"),
        ("mlp.shared_experts.linear_fc2.weight", "layers.20.ffn.shared_experts.w2"),
    ]:
        out = _sync(name, master)
        assert set(out) == {hf + ".weight", hf + ".weight_scale_inv"}, name
        codes, scale = out[hf + ".weight"], out[hf + ".weight_scale_inv"]
        assert codes.dtype == torch.float8_e4m3fn and codes.shape == master.shape
        assert scale.dtype == torch.float32 and scale.shape == (2, 3)
        _assert_dequant_close(codes, scale, (32, 32), master)


def test_shared_expert_fc1_splits_into_w1_and_w3():
    master = _on_grid_master((128, 64))
    out = _sync("mlp.shared_experts.linear_fc1.weight", master)
    p = "layers.20.ffn.shared_experts."
    assert set(out) == {p + "w1.weight", p + "w1.weight_scale_inv", p + "w3.weight", p + "w3.weight_scale_inv"}
    _assert_dequant_close(out[p + "w3.weight"], out[p + "w3.weight_scale_inv"], (32, 32), master[64:])


def test_tensors_serving_holds_in_bf16_pass_through():
    """wo_a is served dequantized to BF16, and the indexer's wk and the compressor are BF16 in the release."""
    for name, hf in [
        ("self_attention.linear_o_group_proj", "layers.20.attn.wo_a.weight"),
        ("self_attention.core_attention.indexer.linear_wk.weight", "layers.20.attn.indexer.wk.weight"),
        ("self_attention.core_attention.compressor.linear_wkv.weight", "layers.20.attn.compressor.wkv.weight"),
        ("mlp.router.weight", "layers.20.ffn.gate.weight"),
    ]:
        param = torch.randn(64, 64, dtype=torch.bfloat16)
        out = _sync(name, param)
        assert list(out) == [hf], name
        assert out[hf] is param


def test_routed_experts_sync_in_the_fp4_dequant_cast_layout(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_FP4_DEQUANT", "1")
    master = _on_grid_master((512, 256))
    out = _sync("mlp.experts.linear_fc1.weight7", master)
    e = "layers.20.ffn.experts.7."
    assert set(out) == {e + "w1.weight", e + "w1.weight_scale_inv", e + "w3.weight", e + "w3.weight_scale_inv"}
    assert out[e + "w1.weight"].dtype == torch.float8_e4m3fn and out[e + "w1.weight"].shape == (256, 256)
    assert out[e + "w1.weight_scale_inv"].dtype == torch.float32 and out[e + "w1.weight_scale_inv"].shape == (2, 2)
    _assert_dequant_close(out[e + "w1.weight"], out[e + "w1.weight_scale_inv"], (128, 128), master[:256])
    out = _sync("mlp.experts.linear_fc2.weight7", master)
    assert out[e + "w2.weight_scale_inv"].shape == (4, 2)


def test_routed_experts_refuse_a_serving_without_the_cast(monkeypatch):
    monkeypatch.delenv("SGLANG_DSV4_FP4_DEQUANT", raising=False)
    with pytest.raises(RuntimeError, match="SGLANG_DSV4_FP4_DEQUANT"):
        _sync("mlp.experts.linear_fc2.weight0", _on_grid_master((256, 128)))


def test_fp32_scales_carry_small_updates_that_the_release_rule_rounds_away():
    """Regression: a master on the release grid, requantized with the release's power-of-two rule, maps small updates
    back to the release bytes; the sync's FP32 scales must pass them through."""
    master = _on_grid_master((256, 256))
    update = 100 * 1e-6 * torch.randn(master.shape, generator=torch.Generator().manual_seed(1)).sign()
    stepped = (master.float() + update).to(torch.bfloat16)

    def carry(quantize):
        moved = quantize(stepped) - quantize(master)
        return float((moved * update).sum() / (update * update).sum())

    native = carry(lambda w: _dequant(*block_cast_fp32(w, (32, 32)), (32, 32)))
    release = carry(_release_rule)
    assert native > 0.2, native
    assert release < native / 2, (release, native)
