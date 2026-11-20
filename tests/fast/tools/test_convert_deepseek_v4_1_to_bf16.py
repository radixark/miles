import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tools.convert_deepseek_v4_1_to_bf16 import convert_config, dequant_fp8_block, main

from miles.utils.mxfp4 import dequantize_mxfp4

E8M0 = torch.float8_e8m0fnu


def _fp8(rows: int, cols: int) -> torch.Tensor:
    """Random finite e4m3 bit patterns (0x7f/0xff are NaN)."""
    bits = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8) | (
        torch.randint(0, 2, (rows, cols), dtype=torch.uint8) << 7
    )
    return bits.view(torch.float8_e4m3fn)


def _e8m0(rows: int, cols: int) -> torch.Tensor:
    return torch.randint(120, 134, (rows, cols), dtype=torch.uint8).view(E8M0)


def _fp4(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed e2m1 as int8 [rows, cols / 2] with one scale per 32 inputs, the V4.1 expert layout."""
    return torch.randint(-128, 128, (rows, cols // 2), dtype=torch.int8), _e8m0(rows, cols // 32)


def _tiny_checkpoint(path):
    """Two decoder layers in the official V4.1 layout, plus the MTP/vision tensors the converter drops."""
    hidden, inter, experts = 64, 32, 2
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": hidden,
            "num_hidden_layers": 2,
            "n_routed_experts": experts,
            "engram_layer_ids": [1],
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 5,
        },
        "vision_config": {"model_type": "deepseek_v41_vision"},
    }
    tensors = {
        "embed.weight": torch.randn(16, hidden).bfloat16(),
        "head.weight": torch.randn(16, hidden).bfloat16(),
        "norm.weight": torch.randn(hidden).bfloat16(),
        "image_start": torch.randn(hidden).bfloat16(),
        "vision.norm.weight": torch.randn(8).bfloat16(),
        "mtp.0.attn.wq_a.weight": _fp8(32, hidden),
        "mtp.0.attn.wq_a.scale": _e8m0(1, 2),
    }
    for layer in range(2):
        p = f"layers.{layer}."
        # 48 rows: a partial trailing scale block.
        for name, shape in (("attn.wq_a", (48, hidden)), ("attn.wkv", (32, hidden)), ("attn.wo_a", (hidden, 32))):
            tensors[p + name + ".weight"] = _fp8(*shape)
            tensors[p + name + ".scale"] = _e8m0(-(-shape[0] // 32), -(-shape[1] // 32))
        for name in ("attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
            tensors[p + name + ".weight"] = _fp8(hidden, hidden)
            tensors[p + name + ".scale"] = _e8m0(2, 2)
        tensors[p + "attn.attn_sink"] = torch.randn(4)
        tensors[p + "hc_attn_fn"] = torch.randn(24, 4 * hidden)
        tensors[p + "ffn.gate.weight"] = torch.randn(experts, hidden).bfloat16()
        tensors[p + "ffn.gate.bias"] = torch.randn(experts)
        tensors[p + "ffn.gate.bias_vl"] = torch.randn(experts)
        for e in range(experts):
            for proj, (rows, cols) in (("w1", (inter, hidden)), ("w2", (hidden, inter)), ("w3", (inter, hidden))):
                name = f"{p}ffn.experts.{e}.{proj}."
                tensors[name + "weight"], tensors[name + "scale"] = _fp4(rows, cols)
    tensors["layers.1.engram.embed.weight"] = _fp8(10, 256)
    tensors["layers.1.engram.embed.scale"] = _e8m0(10, 8)
    tensors["layers.1.engram.wkv.weight"] = _fp8(64, 96)
    tensors["layers.1.engram.wkv.scale"] = _e8m0(2, 3)
    tensors["layers.1.engram.q_weight"] = torch.randn(4, hidden).bfloat16()

    files = {
        "model-00001.safetensors": [n for n in tensors if n.startswith(("vision.", "image_"))],
        "model-00002.safetensors": [n for n in tensors if n.startswith("mtp.")],
        "model-00003.safetensors": [n for n in tensors if ".engram." in n],
    }
    files["model-00004.safetensors"] = [n for n in tensors if n not in sum(files.values(), [])]
    weight_map = {}
    for file_name, names in files.items():
        save_file({n: tensors[n] for n in names}, str(path / file_name))
        weight_map.update(dict.fromkeys(names, file_name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    return tensors


def _load_dir(path) -> dict[str, torch.Tensor]:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    out = {name: t for f in set(index["weight_map"].values()) for name, t in load_file(str(path / f)).items()}
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in out.values())
    return out


def test_dequant_fp8_block_is_exact():
    weight, scale = _fp8(48, 64), _e8m0(2, 2)
    expected = weight.float() * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(32, 0)[
        :48
    ].repeat_interleave(32, 1)
    assert torch.equal(dequant_fp8_block(weight, scale.view(torch.uint8), (32, 32)), expected.bfloat16())


def test_convert_config_flattens_text_config():
    config = convert_config(
        {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "quantization_config": {},
            "text_config": {
                "model_type": "deepseek_v41_text",
                "hidden_size": 8,
                "num_nextn_predict_layers": 3,
                "dspark_block_size": 5,
            },
            "vision_config": {},
        }
    )
    assert config == {
        "hidden_size": 8,
        "num_nextn_predict_layers": 0,
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
    }


@pytest.mark.parametrize("num_shards", [None, 2])
def test_convert_end_to_end(tmp_path, num_shards):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    source = _tiny_checkpoint(src)
    if num_shards is None:
        main(str(src), str(dst), "cpu", None, None, finalize_only=False, overwrite=False)
    else:
        for rank in range(num_shards):
            main(str(src), str(dst), "cpu", rank, num_shards, finalize_only=False, overwrite=False)
        assert not (dst / "model.safetensors.index.json").exists()
        main(str(src), str(dst), "cpu", None, None, finalize_only=True, overwrite=False)

    out = _load_dir(dst)
    config = json.loads((dst / "config.json").read_text())
    assert config["model_type"] == "deepseek_v41" and config["num_hidden_layers"] == 2
    assert "quantization_config" not in config and "text_config" not in config
    assert (dst / "tokenizer.json").exists()
    assert not (dst / "model-00001.safetensors").exists() and not (dst / "model-00002.safetensors").exists()

    dropped = {n for n in source if n.startswith(("mtp.", "vision.", "image_")) or n.endswith("bias_vl")}
    scales = {n for n in source if n.endswith(".scale") and ".engram.embed." not in n}
    assert out.keys() == source.keys() - dropped - scales
    for name, got in out.items():
        want = source[name]
        if name.endswith(".scale") or ".engram.embed." in name or want.dtype in (torch.bfloat16, torch.float32):
            assert got.dtype == want.dtype and torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name
        elif want.dtype == torch.int8:
            scale = source[name.removesuffix("weight") + "scale"].view(torch.uint8)
            assert torch.equal(got, dequantize_mxfp4(want.view(torch.uint8), scale, 32)), name
        else:
            scale = source[name.removesuffix("weight") + "scale"].view(torch.uint8)
            assert torch.equal(got, dequant_fp8_block(want, scale, (32, 32))), name
            assert got.dtype == torch.bfloat16

    # The trainer memory-maps engram rows straight out of the file, so the bytes must sit where the header says.
    with safe_open(str(dst / "model-00003.safetensors"), framework="pt") as f:
        assert f.metadata() == {"format": "pt"}


def test_existing_outputs_are_skipped(tmp_path, capsys):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _tiny_checkpoint(src)
    main(str(src), str(dst), "cpu", None, None, finalize_only=False, overwrite=False)
    before = (dst / "model-00004.safetensors").stat().st_mtime_ns
    main(str(src), str(dst), "cpu", None, None, finalize_only=False, overwrite=False)
    assert (dst / "model-00004.safetensors").stat().st_mtime_ns == before
    assert "Skipping existing" in capsys.readouterr().out
