"""A source's base model is matched to the run's by the HF files' norms and first attention projections, whatever
their dtype, on the tensors both hold unquantized."""

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.training_utils.checkpoint.adapter.base_model import assert_same_base_model, hf_fingerprint_tensors

NORM, O_PROJ = "model.norm.weight", "model.layers.0.self_attn.o_proj.weight"


def _hf_dir(path, **tensors):
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    return str(path)


def test_a_merged_lora_is_another_base(tmp_path):
    base = _hf_dir(tmp_path / "base", **{NORM: torch.ones(4), O_PROJ: torch.ones(4, 4)})
    merged = _hf_dir(tmp_path / "merged", **{NORM: torch.ones(4), O_PROJ: torch.ones(4, 4) * 2})
    with pytest.raises(AssertionError, match="another base model"):
        assert_same_base_model("src", [base], merged)


def test_other_layers_linear_weights_and_the_base_dtype_do_not_count(tmp_path):
    norm = torch.randn(4).bfloat16()
    a = _hf_dir(tmp_path / "a", **{NORM: norm, "model.layers.0.mlp.down_proj.weight": torch.randn(4, 4)})
    b = _hf_dir(tmp_path / "b", **{NORM: norm.float(), "model.layers.0.mlp.down_proj.weight": torch.randn(4, 4)})
    assert hf_fingerprint_tensors(tmp_path / "a").keys() == {NORM}
    assert_same_base_model("src", [a], b)


def test_a_quantized_copy_of_the_base_compares_by_its_unquantized_tensors(tmp_path):
    base = _hf_dir(tmp_path / "bf16", **{NORM: torch.ones(4), O_PROJ: torch.ones(4, 4)})
    fp8 = _hf_dir(tmp_path / "fp8", **{NORM: torch.ones(4), O_PROJ: torch.ones(4, 4).to(torch.float8_e4m3fn)})
    assert hf_fingerprint_tensors(tmp_path / "fp8").keys() == {NORM}
    assert_same_base_model("src", [base], fp8)


def test_another_base_or_an_unresolvable_one_is_refused(tmp_path):
    a = _hf_dir(tmp_path / "a", **{NORM: torch.ones(4)})
    b = _hf_dir(tmp_path / "b", **{NORM: torch.ones(4) * 2})
    with pytest.raises(AssertionError, match="another base model"):
        assert_same_base_model("src", [a], b)
    with pytest.raises(AssertionError, match="has no local HF files"):
        assert_same_base_model("src", [None, str(tmp_path / "missing")], b)
    with pytest.raises(AssertionError, match="--hf-checkpoint org/model has no local HF weights"):
        assert_same_base_model("src", [a], "org/model")
