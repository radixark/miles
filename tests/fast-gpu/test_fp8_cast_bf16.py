from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"])


import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPT_PATH = Path(__file__).parents[2] / "tools" / "fp8_cast_bf16.py"


def _run_converter(input_path: Path, output_path: Path, check: bool = True):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-fp8-hf-path",
            str(input_path),
            "--output-bf16-hf-path",
            str(output_path),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def _mxfp8_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.view(torch.float8_e8m0fnu).float()
    scale = scale.repeat_interleave(32, dim=-1)
    return (weight.float() * scale).to(torch.bfloat16)


def test_mxfp8_checkpoint_casts_to_bf16(tmp_path):
    input_path = tmp_path / "mxfp8"
    output_path = tmp_path / "bf16"
    input_path.mkdir()

    weight_name = "model.layers.0.self_attn.q_a_proj.weight"
    scale_name = f"{weight_name}_scale_inv"
    bf16_name = "model.layers.0.input_layernorm.weight"
    shard_name = "model-00001-of-00001.safetensors"

    weight = torch.linspace(-4.0, 4.0, steps=33 * 64, device="cuda").reshape(33, 64)
    weight = weight.to(torch.float8_e4m3fn)
    scale_codes = torch.tensor([88, 96, 104, 112, 119, 127], dtype=torch.uint8, device="cuda")
    scale = scale_codes.repeat((33 * 2 + scale_codes.numel() - 1) // scale_codes.numel())[: 33 * 2]
    scale = scale.reshape(33, 2)
    bf16_weight = torch.arange(33, dtype=torch.bfloat16)
    expected = _mxfp8_reference(weight, scale).cpu()

    save_file(
        {
            weight_name: weight.cpu(),
            scale_name: scale.cpu(),
            bf16_name: bf16_weight,
        },
        input_path / shard_name,
    )
    config = {
        "architectures": ["TestForCausalLM"],
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "scale_fmt": "ue8m0",
        },
    }
    (input_path / "config.json").write_text(json.dumps(config))
    (input_path / "tokenizer_config.json").write_text("{}")
    (input_path / "chat_template.jinja").write_text("")
    (input_path / "modeling_test.py").write_text("")
    index = {
        "metadata": {},
        "weight_map": {
            weight_name: shard_name,
            scale_name: shard_name,
            bf16_name: shard_name,
        },
    }
    (input_path / "model.safetensors.index.json").write_text(json.dumps(index))

    _run_converter(input_path, output_path)

    output = load_file(output_path / shard_name)
    assert set(output) == {weight_name, bf16_name}
    assert output[weight_name].dtype == torch.bfloat16
    torch.testing.assert_close(output[weight_name], expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(output[bf16_name], bf16_weight, rtol=0, atol=0)

    output_config = json.loads((output_path / "config.json").read_text())
    assert output_config["torch_dtype"] == "bfloat16"
    assert "quantization_config" not in output_config

    output_index = json.loads((output_path / "model.safetensors.index.json").read_text())
    assert output_index["weight_map"] == {
        weight_name: shard_name,
        bf16_name: shard_name,
    }


def test_mxfp8_checkpoint_rejects_missing_scale(tmp_path):
    input_path = tmp_path / "mxfp8"
    output_path = tmp_path / "bf16"
    input_path.mkdir()

    weight_name = "model.layers.0.self_attn.q_a_proj.weight"
    shard_name = "model-00001-of-00001.safetensors"
    weight = torch.zeros((1, 32), device="cuda").to(torch.float8_e4m3fn).cpu()
    save_file({weight_name: weight}, input_path / shard_name)
    config = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "scale_fmt": "ue8m0",
        },
    }
    (input_path / "config.json").write_text(json.dumps(config))
    index = {"metadata": {}, "weight_map": {weight_name: shard_name}}
    (input_path / "model.safetensors.index.json").write_text(json.dumps(index))

    result = _run_converter(input_path, output_path, check=False)
    assert result.returncode != 0
    assert f"Missing scale_inv tensor for MXFP8 weight {weight_name}" in result.stderr


def test_block_fp8_checkpoint_still_casts_to_bf16(tmp_path):
    input_path = tmp_path / "fp8"
    output_path = tmp_path / "bf16"
    input_path.mkdir()

    weight_name = "model.layers.0.self_attn.q_a_proj.weight"
    scale_name = f"{weight_name}_scale_inv"
    shard_name = "model-00001-of-00001.safetensors"
    weight = torch.linspace(-4.0, 4.0, steps=33 * 64, device="cuda").reshape(33, 64)
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5]], dtype=torch.float32)
    expected = (weight.float() * scale.cuda()).to(torch.bfloat16).cpu()
    save_file({weight_name: weight.cpu(), scale_name: scale}, input_path / shard_name)
    config = {
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    (input_path / "config.json").write_text(json.dumps(config))
    index = {
        "metadata": {},
        "weight_map": {weight_name: shard_name, scale_name: shard_name},
    }
    (input_path / "model.safetensors.index.json").write_text(json.dumps(index))

    _run_converter(input_path, output_path)

    output = load_file(output_path / shard_name)
    torch.testing.assert_close(output[weight_name], expected, rtol=0, atol=0)
    output_config = json.loads((output_path / "config.json").read_text())
    assert output_config["torch_dtype"] == "bfloat16"
    assert "quantization_config" not in output_config


@pytest.mark.parametrize(
    ("weight_block_size", "scale_fmt"),
    [([32, 32], "ue8m0"), ([1, 32], "float32")],
)
def test_mxfp8_checkpoint_rejects_unsupported_format(tmp_path, weight_block_size, scale_fmt):
    input_path = tmp_path / "mxfp8"
    output_path = tmp_path / "bf16"
    input_path.mkdir()
    config = {
        "quantization_config": {
            "quant_method": "mxfp8",
            "weight_block_size": weight_block_size,
            "scale_fmt": scale_fmt,
        }
    }
    (input_path / "config.json").write_text(json.dumps(config))

    result = _run_converter(input_path, output_path, check=False)
    assert result.returncode != 0
    assert "Unsupported MXFP8 format" in result.stderr
