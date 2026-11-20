import json
import sys

import torch
from safetensors.torch import load_file, save_file
from tools.convert_inkling2_to_bf16 import E2M1_TABLE, main

_NVFP4 = "model.llm.layers.0.mlp.experts.w13_weight"
_MXFP8 = "model.llm.layers.0.attn.wq_du.weight"
_BF16 = "model.llm.embed.weight"


def _write_release_checkpoint(root):
    torch.manual_seed(0)
    experts, rows, cols = 2, 4, 32
    codes = torch.randint(0, 16, (experts, rows, cols), dtype=torch.uint8)
    block_scale = torch.rand(experts, rows, cols // 16) + 0.5
    amax = torch.tensor([2.0, 3.0])
    denominator = torch.tensor(6.0)
    nvfp4 = (
        E2M1_TABLE[(codes & 7).long()]
        * torch.where((codes & 8) != 0, -1.0, 1.0)
        * block_scale.to(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        * (amax / denominator).view(-1, 1, 1)
    )
    mxfp8_codes = torch.randn(rows, cols).to(torch.float8_e4m3fn)
    exponent = torch.randint(120, 130, (rows, cols // 32), dtype=torch.uint8)
    mxfp8 = mxfp8_codes.float() * torch.exp2(exponent.float() - 127).repeat_interleave(32, -1)
    embed = torch.randn(8, cols, dtype=torch.bfloat16)

    root.mkdir()
    save_file(
        {
            _NVFP4: codes[..., 0::2] | (codes[..., 1::2] << 4),
            _NVFP4 + "_scale": block_scale.to(torch.float8_e4m3fn),
            _NVFP4 + "_amax": amax,
            _NVFP4 + "_outer_scale_denominator": denominator,
            _MXFP8: mxfp8_codes,
            _MXFP8 + "_scale": exponent,
            _BF16: embed,
        },
        str(root / "model-00001-of-00001.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {_NVFP4: "model-00001-of-00001.safetensors"}})
    )
    groups = [
        {"compute_dtype": "nvfp4", "tensors": [_NVFP4]},
        {"compute_dtype": "mxfp8", "tensors": [_MXFP8]},
        {"compute_dtype": "bfloat16", "tensors": [_BF16]},
        {"compute_dtype": "bfloat16", "fwd_output_dtype": "float32", "tensors": []},
    ]
    (root / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}))
    save_file({"mtp.proj": torch.ones(3)}, str(root / "mtp.safetensors"))
    (root / "config.json").write_text("{}")
    return {_NVFP4: nvfp4, _MXFP8: mxfp8, _BF16: embed}


def test_the_release_checkpoint_dequantizes_to_bf16_with_its_side_files(tmp_path, monkeypatch):
    expected = _write_release_checkpoint(tmp_path / "src")
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert", "--model-dir", str(tmp_path / "src"), "--save-dir", str(tmp_path / "dst"), "--device", "cpu"],
    )

    main()

    dst = tmp_path / "dst"
    weight_map = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(weight_map) == set(expected)
    tensors = {}
    for shard in set(weight_map.values()):
        tensors |= load_file(str(dst / shard))
    for name, value in expected.items():
        assert tensors[name].dtype == torch.bfloat16
        torch.testing.assert_close(tensors[name].float(), value.bfloat16().float())
    groups = json.loads((dst / "hf_quant_config.json").read_text())["model"]["parameter_groups"]
    assert {g["compute_dtype"] for g in groups} == {"bfloat16"}
    assert sorted(groups[0]["tensors"]) == sorted(expected)
    assert (dst / "mtp.safetensors").read_bytes() == (tmp_path / "src" / "mtp.safetensors").read_bytes()
    assert (dst / "config.json").exists()
