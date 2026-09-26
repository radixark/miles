import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.convert_mimo_v2_to_bf16 import (
    FP4_TABLE,
    dequant_fp8_block,
    dequant_fused_qkv,
    dequant_mxfp4,
    main,
    split_fused_qkv,
)

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _pack_mxfp4(codes: torch.Tensor) -> torch.Tensor:
    """codes: [out, in] 4-bit indices into FP4_TABLE -> [out, in / 2] uint8, element 2i in the low nibble."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _quant_fp8_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(128, 128) block quantization of one matrix, rows padded to whole blocks."""
    rows, cols = weight.shape
    padded = torch.zeros(-(-rows // 128) * 128, -(-cols // 128) * 128)
    padded[:rows, :cols] = weight
    blocks = padded.view(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / E4M3_MAX
    quant = (blocks / scale[:, None, :, None]).view_as(padded)[:rows, :cols].to(torch.float8_e4m3fn)
    return quant, scale


def _interleave_qkv(q, k, v, shards):
    return torch.cat(
        [torch.cat(parts) for parts in zip(q.chunk(shards), k.chunk(shards), v.chunk(shards), strict=True)]
    )


def test_dequant_mxfp4_is_exact():
    codes = torch.randint(0, 16, (8, 64))
    exponents = torch.randint(120, 134, (8, 2), dtype=torch.uint8)
    expected = FP4_TABLE[codes] * torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
    assert torch.equal(dequant_mxfp4(_pack_mxfp4(codes), exponents).float(), expected.to(torch.bfloat16).float())


def test_fused_qkv_scales_are_per_shard():
    # Shards of 224 rows are not 128-aligned: each one carries its own 2 scale rows.
    shards, rows = 2, 224
    weight = torch.randn(shards * rows, 256)
    quant_scale = [_quant_fp8_block(part) for part in weight.chunk(shards)]
    fp8 = torch.cat([q for q, _ in quant_scale])
    scale = torch.cat([s for _, s in quant_scale])
    assert scale.shape == (shards * 2, 2)
    expected = torch.cat(
        [q.float() * s.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1) for q, s in quant_scale]
    )
    out = dequant_fused_qkv(fp8, scale, shards)
    assert torch.equal(out, expected.to(torch.bfloat16))
    assert (out.float() - weight).abs().max() < 0.07 * weight.abs().max()


def test_split_fused_qkv_restores_head_order():
    config = {
        "num_key_value_heads": 2,
        "swa_num_attention_heads": 8,
        "swa_num_key_value_heads": 4,
        "swa_head_dim": 3,
        "swa_v_head_dim": 2,
    }
    q, k, v = torch.randn(8 * 3, 5), torch.randn(4 * 3, 5), torch.randn(4 * 2, 5)
    out = split_fused_qkv(_interleave_qkv(q, k, v, shards=2), config, is_swa=True)
    for got, want in zip(out, (q, k, v), strict=True):
        assert torch.equal(got, want)


def _tiny_checkpoint(path, mtp=False):
    """Two layers in the official MiMo-V2 format: global+dense (0) and SWA+MoE (1), optionally one MTP layer."""
    config = {
        "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv",
        "num_hidden_layers": 2,
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 1],
        "hybrid_block_size": None,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "v_head_dim": 32,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 4,
        "swa_head_dim": 64,
        "swa_v_head_dim": 32,
        "add_swa_attention_sink_bias": True,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "quantization_config": {
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "ignored_layers": ["model.layers.1.self_attn.o_proj"],
        },
    }
    expected, tensors = {}, {"model.embed_tokens.weight": torch.randn(16, 128).bfloat16()}
    for layer, (kv_heads, is_swa) in enumerate(((2, False), (4, True))):
        p = f"model.layers.{layer}."
        q, k, v = torch.randn(4 * 64, 128), torch.randn(kv_heads * 64, 128), torch.randn(kv_heads * 32, 128)
        fused = _interleave_qkv(q, k, v, shards=2)
        quant_scale = [_quant_fp8_block(part) for part in fused.chunk(2)]
        tensors[p + "self_attn.qkv_proj.weight"] = torch.cat([qq for qq, _ in quant_scale])
        tensors[p + "self_attn.qkv_proj.weight_scale_inv"] = torch.cat([s for _, s in quant_scale])
        deq = dequant_fused_qkv(
            tensors[p + "self_attn.qkv_proj.weight"], tensors[p + "self_attn.qkv_proj.weight_scale_inv"], 2
        )
        for proj, part in zip("qkv", split_fused_qkv(deq, config, is_swa), strict=True):
            expected[p + f"self_attn.{proj}_proj.weight"] = part
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            tensors[p + name] = torch.ones(128).bfloat16()
        tensors[p + "self_attn.o_proj.weight"] = torch.randn(128, 4 * 32).bfloat16()
    tensors["model.layers.1.self_attn.attention_sink_bias"] = torch.zeros(4).bfloat16()
    for proj, shape in (("gate_proj", (256, 128)), ("up_proj", (256, 128)), ("down_proj", (128, 256))):
        quant, scale = _quant_fp8_block(torch.randn(shape))
        tensors[f"model.layers.0.mlp.{proj}.weight"] = quant
        tensors[f"model.layers.0.mlp.{proj}.weight_scale_inv"] = scale
    tensors["model.layers.1.mlp.gate.weight"] = torch.randn(4, 128).bfloat16()
    tensors["model.layers.1.mlp.gate.e_score_correction_bias"] = torch.randn(4)
    for expert in range(4):
        for proj, (rows, cols) in (("gate_proj", (64, 128)), ("up_proj", (64, 128)), ("down_proj", (128, 64))):
            name = f"model.layers.1.mlp.experts.{expert}.{proj}."
            codes = torch.randint(0, 16, (rows, cols))
            exps = torch.randint(124, 130, (rows, cols // 32), dtype=torch.uint8)
            tensors[name + "weight"], tensors[name + "weight_scale"] = _pack_mxfp4(codes), exps
            expected[name + "weight"] = dequant_mxfp4(tensors[name + "weight"], exps)
    if mtp:
        quant_scale = [_quant_fp8_block(part) for part in torch.randn(640, 128).chunk(2)]
        tensors["model.mtp.layers.0.self_attn.qkv_proj.weight"] = torch.cat([q for q, _ in quant_scale])
        tensors["model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv"] = torch.cat([s for _, s in quant_scale])
        quant, scale = _quant_fp8_block(torch.randn(256, 128))
        tensors["model.mtp.layers.0.mlp.gate_proj.weight"] = quant
        tensors["model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"] = scale
    save_file(tensors, str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")})
    )
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")
    return expected


@pytest.mark.parametrize("layers, num_experts", [(None, None), ([1], 2)])
def test_convert_end_to_end(tmp_path, layers, num_experts):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    expected = _tiny_checkpoint(src)
    main(str(src), str(dst), layers, num_experts, keep_quant=False, device="cpu")

    config = json.loads((dst / "config.json").read_text())
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    out = {name: t for f in set(index.values()) for name, t in load_file(str(dst / f)).items()}
    assert "quantization_config" not in config and "attention_projection_layout" not in config
    assert (dst / "tokenizer_config.json").exists()
    assert not any(name.endswith(("weight_scale", "weight_scale_inv", "qkv_proj.weight")) for name in out)
    remap = {old: new for new, old in enumerate(layers)} if layers else {0: 0, 1: 1}
    assert config["num_hidden_layers"] == len(remap)
    assert config["hybrid_layer_pattern"] == config["moe_layer_freq"] == list(remap)
    num_kept_experts = num_experts or 4
    kept = {}
    for name, want in expected.items():
        _, _, old, *rest = name.split(".")
        if int(old) in remap and not (rest[1] == "experts" and int(rest[2]) >= num_kept_experts):
            kept[f"model.layers.{remap[int(old)]}.{'.'.join(rest)}"] = want
    assert len(kept) == (3 if 0 in remap else 0) + 3 + 3 * num_kept_experts
    for name, want in kept.items():
        assert torch.equal(out[name], want), name
    experts = {name for name in out if ".experts." in name}
    assert len(experts) == 3 * num_kept_experts
    assert config["n_routed_experts"] == num_kept_experts
    assert out[f"model.layers.{remap[1]}.mlp.gate.weight"].shape[0] == num_kept_experts
    assert out[f"model.layers.{remap[1]}.mlp.gate.e_score_correction_bias"].dtype == torch.float32


def test_keep_quant_reindexes_ignored_layers(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _tiny_checkpoint(src)
    main(str(src), str(dst), [1], None, keep_quant=True, device="cpu")
    config = json.loads((dst / "config.json").read_text())
    assert config["quantization_config"]["ignored_layers"] == ["model.layers.0.self_attn.o_proj"]
    assert config["attention_projection_layout"] == "fused_qkv"


@pytest.mark.parametrize("layers", [None, [1]])
def test_bf16_linears_decode_only_the_decoder_fp8_linears(tmp_path, layers):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _tiny_checkpoint(src, mtp=True)
    source = load_file(str(src / "model-00001.safetensors"))
    main(str(src), str(dst), layers, None, keep_quant=True, device="cpu", bf16_linears=True)

    config = json.loads((dst / "config.json").read_text())
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    out = {name: t for f in set(index.values()) for name, t in load_file(str(dst / f)).items()}
    remap = {old: new for new, old in enumerate(layers)} if layers else {0: 0, 1: 1}
    expected = {}
    for name, tensor in source.items():
        _, group, *rest = name.split(".")
        if group == "layers":
            if int(rest[0]) not in remap or name.endswith("weight_scale_inv"):
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                scale = source[name + "_scale_inv"]
                # the fused qkv stays fused and kv-head interleaved
                tensor = (
                    dequant_fused_qkv(tensor, scale, 2) if "qkv_proj" in name else dequant_fp8_block(tensor, scale)
                )
            name = f"model.layers.{remap[int(rest[0])]}.{'.'.join(rest[1:])}"
        expected[name] = tensor
    assert out.keys() == expected.keys()
    for name, want in expected.items():
        assert out[name].dtype == want.dtype and torch.equal(out[name].view(torch.uint8), want.view(torch.uint8)), name
    # MXFP4 experts and the MTP layers keep their source format.
    assert out[f"model.layers.{remap[1]}.mlp.experts.0.gate_proj.weight"].dtype == torch.uint8
    assert out["model.mtp.layers.0.self_attn.qkv_proj.weight"].dtype == torch.float8_e4m3fn

    assert config["attention_projection_layout"] == "fused_qkv"
    assert config["quantization_config"]["store_dtype"] == "mxfp4"
    # SGLang skips a fused qkv_proj by its q/k/v shard names, Miles' weight sync by the fused name.
    ignored = {f"model.layers.{remap[1]}.self_attn.o_proj"}
    for new in remap.values():
        ignored |= {f"model.layers.{new}.self_attn.{proj}" for proj in ("qkv_proj", "q_proj", "k_proj", "v_proj")}
    if 0 in remap:
        ignored |= {f"model.layers.0.mlp.{proj}" for proj in ("gate_proj", "up_proj", "down_proj")}
    assert set(config["quantization_config"]["ignored_layers"]) == ignored
