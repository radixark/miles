"""Tests for weightless Hugging Face rollout schemas."""

import json
from pathlib import Path

import pytest

from miles.utils.hf_rollout_schema import create_mxfp8_expert_rollout_schema


def _write_checkpoint(path: Path, *, quant_method: str = "fp8") -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TestForCausalLM"],
                "expert_dtype": "fp4",
                "quantization_config": {"quant_method": quant_method, "weight_block_size": [128, 128]},
            }
        )
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"payload")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (path / "tokenizer.json").write_text("{}")


def test_schema_overrides_only_the_routed_experts_and_drops_weights(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "schema"
    _write_checkpoint(source)

    created = create_mxfp8_expert_rollout_schema(source, destination)

    assert created == destination.resolve()
    assert (destination / "tokenizer.json").is_file()
    assert not (destination / "model.safetensors.index.json").exists()
    assert not list(destination.glob("*.safetensors"))
    config = json.loads((destination / "config.json").read_text())
    assert config["expert_dtype"] == "fp8"
    assert config["quantization_config"] == {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "routed_experts_quant_method": "mxfp8",
    }


def test_schema_requires_a_block_wise_fp8_source(tmp_path):
    source = tmp_path / "source"
    _write_checkpoint(source, quant_method="mxfp8")

    with pytest.raises(ValueError, match="block-wise FP8"):
        create_mxfp8_expert_rollout_schema(source, tmp_path / "schema")


def test_schema_refuses_directory_with_weight_payloads(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "schema"
    _write_checkpoint(source)
    destination.mkdir()
    (destination / "unexpected.safetensors").write_bytes(b"payload")

    with pytest.raises(ValueError, match="containing weight payloads"):
        create_mxfp8_expert_rollout_schema(source, destination)
