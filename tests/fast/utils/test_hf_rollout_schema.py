"""Tests for metadata-only Hugging Face rollout schemas."""

import json
import struct
from pathlib import Path

import pytest

from miles.utils.hf_rollout_schema import build_mxfp8_quantization_config, create_mxfp8_rollout_schema


def _write_safetensors_header(path: Path, tensors: dict[str, dict]) -> None:
    header = json.dumps(tensors).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [32, 64], "data_offsets": [0, 1]},
        "model.layers.0.input_layernorm.weight": {
            "dtype": "BF16",
            "shape": [64],
            "data_offsets": [1, 2],
        },
        "model.layers.0.self_attn.wq_a.weight": {
            "dtype": "F8_E4M3",
            "shape": [64, 64],
            "data_offsets": [2, 3],
        },
        "model.layers.0.self_attn.wo_a.weight": {
            "dtype": "F8_E4M3",
            "shape": [64, 64],
            "data_offsets": [3, 4],
        },
        "model.layers.0.mlp.experts.0.gate_proj.weight": {
            "dtype": "I8",
            "shape": [64, 32],
            "data_offsets": [4, 5],
        },
        "lm_head.weight": {"dtype": "BF16", "shape": [32, 64], "data_offsets": [5, 6]},
    }
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors_header(path / shard_name, tensors)
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
    marker = json.loads((destination / ".miles-rollout-schema.json").read_text())
    assert marker["weight_payloads"] is False


def test_schema_refuses_directory_with_weight_payloads(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "schema"
    _write_checkpoint(source)
    destination.mkdir()
    (destination / "unexpected.safetensors").write_bytes(b"payload")

    with pytest.raises(ValueError, match="containing weight payloads"):
        create_mxfp8_rollout_schema(source, destination)
