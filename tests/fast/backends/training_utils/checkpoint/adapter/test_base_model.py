"""A source's base model is matched to the run's by the HF files' norms and first attention projections, whatever
their dtype."""

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.training_utils.checkpoint.adapter.base_model import (
    assert_same_base_model,
    fingerprint_named_tensors,
    hf_base_fingerprint,
)


def _hf_dir(path, norm: torch.Tensor):
    path.mkdir()
    save_file(
        {"model.norm.weight": norm, "model.layers.0.mlp.down_proj.weight": torch.randn(4, 4)},
        path / "model.safetensors",
    )
    return path


def test_a_merged_lora_changes_the_fingerprint(tmp_path):
    base = _hf_dir(tmp_path / "base", torch.ones(4))
    merged = tmp_path / "merged"
    merged.mkdir()
    tensors = {"model.norm.weight": torch.ones(4), "model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4)}
    save_file(tensors, merged / "model.safetensors")
    save_file(tensors | {"model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4) * 2}, base / "model.safetensors")
    assert hf_base_fingerprint(base) != hf_base_fingerprint(merged)
    shared = ["model.norm.weight"]
    assert hf_base_fingerprint(base, shared) == hf_base_fingerprint(merged, shared), "names restrict the hash"


def test_the_fingerprint_ignores_other_layers_linear_weights(tmp_path):
    norm = torch.randn(4).bfloat16()
    a, b = _hf_dir(tmp_path / "a", norm), _hf_dir(tmp_path / "b", norm)
    assert hf_base_fingerprint(a) == hf_base_fingerprint(b), "other tensors differ, the norms do not"
    assert hf_base_fingerprint(a) == fingerprint_named_tensors([("model.norm.weight", norm.float())])
    assert_same_base_model("src", [str(a)], str(b))


def test_another_base_or_an_unresolvable_one_is_refused(tmp_path):
    a = _hf_dir(tmp_path / "a", torch.ones(4))
    b = _hf_dir(tmp_path / "b", torch.ones(4) * 2)
    with pytest.raises(AssertionError, match="whose weights differ"):
        assert_same_base_model("src", [str(a)], str(b))
    with pytest.raises(AssertionError, match="has no local HF files"):
        assert_same_base_model("src", [None, str(tmp_path / "missing")], str(b))
