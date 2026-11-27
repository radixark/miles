import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tools.convert_fp8_to_bf16_reference import _scaled_weights, convert

from miles_plugins.top.fp8_masters import master_from_checkpoint


def _checkpoint(tmp_path, scale_suffix):
    g = torch.Generator().manual_seed(0)
    payload = (torch.randn(256, 256, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    tensors = {
        "layers.0.attn.wq_b.weight": payload,
        f"layers.0.attn.wq_b{scale_suffix}": torch.full((2, 2), 2.0**-10),
        "layers.0.attn_norm.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text("{}")
    return tensors


def test_scaled_weights_reads_both_scale_names():
    weight_map = dict.fromkeys(
        ["a.weight", "a.weight_scale_inv", "b.weight", "b.scale", "c.weight", "d.scale", "e.weight"], "x"
    )
    assert _scaled_weights(weight_map) == {"a.weight": "a.weight_scale_inv", "b.weight": "b.scale"}


@pytest.mark.parametrize("scale_suffix", [".scale", ".weight_scale_inv"])
def test_masters_reference_is_the_trainers_masters(tmp_path, scale_suffix):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    tensors = _checkpoint(src, scale_suffix)
    convert(str(src), str(dst), "masters", "cpu")
    out = load_file(dst / "model.safetensors")
    assert set(out) == {"layers.0.attn.wq_b.weight", "layers.0.attn_norm.weight"}
    expected = master_from_checkpoint(
        tensors["layers.0.attn.wq_b.weight"], tensors[f"layers.0.attn.wq_b{scale_suffix}"]
    )
    assert torch.equal(out["layers.0.attn.wq_b.weight"], expected)
    assert torch.equal(out["layers.0.attn_norm.weight"], tensors["layers.0.attn_norm.weight"])
    assert (dst / "config.json").exists()
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(out)


def test_served_reference_is_the_sync_value(tmp_path):
    pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    from miles_plugins.top.fp8_masters import dequantize_block_fp8

    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    tensors = _checkpoint(src, ".scale")
    convert(str(src), str(dst), "served", "cpu")
    served = load_file(dst / "model.safetensors")["layers.0.attn.wq_b.weight"]
    # Power-of-two scales: serving computes with the checkpoint's own values, not the off-grid masters.
    expected = dequantize_block_fp8(tensors["layers.0.attn.wq_b.weight"], tensors["layers.0.attn.wq_b.scale"])
    assert torch.equal(served, expected)


@pytest.mark.parametrize("weights", ["served", "masters"])
def test_a_checkpoint_that_is_not_block_fp8_is_refused(tmp_path, weights):
    """An FP4-expert release pairs int8 codes with 1x32 scales; the reference must not be garbage BF16."""
    src = tmp_path / "src"
    src.mkdir()
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(256, 128, dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.ones(256, 8),
    }
    save_file(tensors, src / "model.safetensors")
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match=r"\[top\].*128x128 block scales"):
        convert(str(src), str(tmp_path / "dst"), weights, "cpu")
