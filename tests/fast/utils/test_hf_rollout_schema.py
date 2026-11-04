"""Tests for metadata-only Hugging Face rollout schemas."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from miles.utils.hf_rollout_schema import build_mxfp8_quantization_config, create_mxfp8_rollout_schema


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": torch.zeros(32, 64, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.zeros(64, dtype=torch.bfloat16),
        "model.layers.0.self_attn.wq_a.weight": torch.zeros(64, 64, dtype=torch.float8_e4m3fn),
        "model.layers.0.self_attn.wo_a.weight": torch.zeros(64, 64, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(64, 32, dtype=torch.int8),
        "lm_head.weight": torch.zeros(32, 64, dtype=torch.bfloat16),
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, path / shard_name)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TestForCausalLM"],
                "expert_dtype": "fp4",
                "quantization_config": {"quant_method": "fp8"},
            }
        )
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard_name for name in tensors}})
    )
    (path / "tokenizer.json").write_text("{}")


def test_mxfp8_config_uses_headers_without_treating_packed_experts_as_unquantized(tmp_path):
    source = tmp_path / "source"
    _write_checkpoint(source)

    quantization_config = build_mxfp8_quantization_config(source)

    assert quantization_config["quant_method"] == "mxfp8"
    assert quantization_config["weight_block_size"] == [1, 32]
    ignored = quantization_config["modules_to_not_convert"]
    assert "model.embed_tokens" in ignored
    assert "model.layers.0.input_layernorm" in ignored
    assert "model.layers.0.self_attn.wo_a" in ignored
    assert "lm_head" in ignored
    assert "model.layers.0.self_attn.wq_a" not in ignored
    assert not any("experts.0.gate_proj" in name for name in ignored)


def test_schema_contains_model_metadata_but_no_weight_index_or_payload(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "schema"
    _write_checkpoint(source)

    created = create_mxfp8_rollout_schema(source, destination)

    assert created == destination.resolve()
    assert (destination / "tokenizer.json").is_file()
    assert not (destination / "model.safetensors.index.json").exists()
    assert not list(destination.glob("*.safetensors"))
    config = json.loads((destination / "config.json").read_text())
    assert config["expert_dtype"] == "fp8"
    assert config["quantization_config"]["quant_method"] == "mxfp8"


def test_schema_refuses_directory_with_weight_payloads(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "schema"
    _write_checkpoint(source)
    destination.mkdir()
    (destination / "unexpected.safetensors").write_bytes(b"payload")

    with pytest.raises(ValueError, match="containing weight payloads"):
        create_mxfp8_rollout_schema(source, destination)


@pytest.mark.parametrize(
    "dtype,width,valid",
    [(torch.int8, 16, True), (torch.int8, 24, False), (torch.bfloat16, 48, False), (torch.int32, 32, False)],
)
def test_schema_validates_unpacked_expert_layout(tmp_path, dtype, width, valid):
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["TestForCausalLM"]}))
    save_file({name: torch.zeros(32, width, dtype=dtype)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors"}}))
    if valid:
        assert build_mxfp8_quantization_config(tmp_path)["modules_to_not_convert"] == []
    else:
        with pytest.raises(ValueError, match="Unsupported MXFP8 expert"):
            build_mxfp8_quantization_config(tmp_path)


def test_schema_supports_unindexed_checkpoint(tmp_path):
    name = "model.embed_tokens.weight"
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["TestForCausalLM"]}))
    save_file({name: torch.zeros(32, 32, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")

    schema = create_mxfp8_rollout_schema(tmp_path, tmp_path / "schema")

    config = json.loads((schema / "config.json").read_text())
    assert config["quantization_config"]["modules_to_not_convert"] == ["model.embed_tokens"]
    assert not list(schema.glob("*.safetensors"))
