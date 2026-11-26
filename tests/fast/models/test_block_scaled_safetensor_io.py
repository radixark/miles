import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("mbridge")
from miles_plugins.mbridge.block_scaled_safetensor_io import BlockScaledSafeTensorIO  # noqa: E402

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _bf16_bits(values):
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _reference(values, scale, block):
    """float32 value times its block's scale, rounded to nearest-even BF16, as raw bits."""
    rows, cols = values.shape
    scale = np.repeat(np.repeat(scale, block[0], axis=0)[:rows], block[1], axis=1)[:, :cols]
    return _bf16_bits(values * scale)


def _e8m0(raw):
    return (2.0 ** (raw.numpy().astype(int) - 127)).astype(np.float32)


def _bits(tensor):
    return tensor.view(torch.int16).numpy().view(np.uint16)


def _write_checkpoint(path, tensors):
    save_file(tensors, path / "model.safetensors")
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return BlockScaledSafeTensorIO(str(path), "cpu")


def _random_bytes(generator, shape, low=0, high=256):
    return torch.randint(low, high, shape, dtype=torch.uint8, generator=generator)


@pytest.mark.parametrize(
    "shape, block, scale_suffix",
    [((64, 96), 32, ".scale"), ((256, 384), 128, ".scale"), ((200, 300), 128, ".weight_scale_inv")],
)
def test_fp8_blocks_match_bit_level_decode(tmp_path, shape, block, scale_suffix):
    generator = torch.Generator().manual_seed(block)
    raw = _random_bytes(generator, shape)
    raw[(raw & 0x7F) == 0x7F] = 0  # e4m3 NaN encodings
    scale_shape = (-(-shape[0] // block), -(-shape[1] // block))
    if scale_suffix == ".scale":
        scale_raw = _random_bytes(generator, scale_shape, 110, 140)
        scale, expected_scale = scale_raw.view(torch.float8_e8m0fnu), _e8m0(scale_raw)
    else:
        scale = torch.rand(scale_shape, generator=generator) * 1e-3
        expected_scale = scale.numpy()
    io = _write_checkpoint(
        tmp_path, {"attn.wq_a.weight": raw.view(torch.float8_e4m3fn), f"attn.wq_a{scale_suffix}": scale}
    )

    out = io.load_some_hf_weight(["attn.wq_a.weight"])

    assert list(out) == ["attn.wq_a.weight"]
    expected = _reference(raw.view(torch.float8_e4m3fn).float().numpy(), expected_scale, (block, block))
    np.testing.assert_array_equal(_bits(out["attn.wq_a.weight"]), expected)


def test_packed_fp4_experts_match_bit_level_decode(tmp_path):
    generator = torch.Generator().manual_seed(0)
    packed = _random_bytes(generator, (8, 64))
    scale = _random_bytes(generator, (8, 4), 110, 140)
    io = _write_checkpoint(
        tmp_path,
        {"experts.0.w1.weight": packed.view(torch.int8), "experts.0.w1.scale": scale.view(torch.float8_e8m0fnu)},
    )

    weight = io.load_one_hf_weight("experts.0.w1.weight")

    values = np.empty((8, 128), dtype=np.float32)
    values[:, 0::2], values[:, 1::2] = _E2M1[packed.numpy() & 0xF], _E2M1[packed.numpy() >> 4]
    np.testing.assert_array_equal(_bits(weight), _reference(values, _e8m0(scale), (1, 32)))


def test_mismatched_scale_shape_raises(tmp_path):
    raw = torch.zeros(64, 96, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale = torch.full((3, 3), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    io = _write_checkpoint(tmp_path, {"attn.wq_a.weight": raw, "attn.wq_a.scale": scale})

    with pytest.raises(ValueError, match="does not fit"):
        io.load_some_hf_weight(["attn.wq_a.weight"])


def test_bf16_checkpoint_passes_through(tmp_path):
    weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(0)).bfloat16()
    io = _write_checkpoint(tmp_path, {"attn.wq_a.weight": weight, "attn.attn_sink": weight[0].float()})

    out = io.load_some_hf_weight(["attn.wq_a.weight", "attn.attn_sink"])

    assert torch.equal(out["attn.wq_a.weight"], weight)
    assert out["attn.attn_sink"].dtype == torch.float32
