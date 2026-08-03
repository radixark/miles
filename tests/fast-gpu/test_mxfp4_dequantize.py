from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
)


import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch

from miles.utils.mxfp4 import MXFP4_GROUP_SIZE, mxfp4_dequantize

FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _make_packed_weight(rows: int, logical_cols: int, device: str) -> torch.Tensor:
    encoded = torch.arange(rows * logical_cols, dtype=torch.int64, device=device).reshape(rows, logical_cols)
    encoded = (encoded % 16).to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).view(torch.int8).contiguous()


def _make_scale(rows: int, logical_cols: int, device: str) -> torch.Tensor:
    scale_cols = logical_cols // MXFP4_GROUP_SIZE
    encoded = torch.arange(rows * scale_cols, dtype=torch.int64, device=device).reshape(rows, scale_cols)
    encoded = (125 + encoded % 6).to(torch.uint8)
    return encoded.view(torch.float8_e8m0fnu)


def _dequantize_reference(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    table = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=weight.device)
    unpacked = torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(1)
    expanded_scale = scale.view(torch.float8_e8m0fnu).float().repeat_interleave(MXFP4_GROUP_SIZE, dim=1)
    return (unpacked * expanded_scale).to(dtype)


def _make_block_fp8_weight(rows: int, cols: int) -> torch.Tensor:
    values = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    return ((values % 15).float() / 2 - 3.5).to(torch.float8_e4m3fn)


def _make_block_scale(rows: int, cols: int) -> torch.Tensor:
    shape = (rows // 128, cols // 128)
    return torch.full(shape, 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _run_fp8_cast_bf16(model_dir: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root), env.get("PYTHONPATH", "")))
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "fp8_cast_bf16.py"),
            "--input-fp8-hf-path",
            str(model_dir),
            "--output-bf16-hf-path",
            str(output_dir),
            *extra_args,
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("shape", [(1, 32), (3, 64), (128, 4096)])
@pytest.mark.parametrize("scale_storage", [torch.float8_e8m0fnu, torch.uint8])
def test_mxfp4_dequantize_matches_reference(shape, scale_storage):
    rows, logical_cols = shape
    weight = _make_packed_weight(rows, logical_cols, device="cuda")
    scale = _make_scale(rows, logical_cols, device="cuda")
    if scale_storage == torch.uint8:
        scale = scale.view(torch.uint8)

    actual = mxfp4_dequantize(weight, scale, dtype=torch.bfloat16)
    expected = _dequantize_reference(weight, scale, dtype=torch.bfloat16)

    assert actual.shape == (rows, logical_cols)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


def test_mxfp4_dequantize_rejects_wrong_scale_shape():
    weight = _make_packed_weight(2, 64, device="cuda")
    scale = _make_scale(2, 32, device="cuda")

    with pytest.raises(ValueError, match="Expected MXFP4 scale shape"):
        mxfp4_dequantize(weight, scale, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    ("weight_dtype", "scale_dtype", "match"),
    [
        (torch.bfloat16, torch.float8_e8m0fnu, "packed weights"),
        (torch.int8, torch.float32, "scales must use"),
    ],
)
def test_mxfp4_dequantize_rejects_wrong_dtypes(weight_dtype, scale_dtype, match):
    weight = torch.zeros((2, 16), dtype=weight_dtype, device="cuda")
    scale = torch.ones((2, 1), dtype=scale_dtype, device="cuda")

    with pytest.raises(ValueError, match=match):
        mxfp4_dequantize(weight, scale, dtype=torch.bfloat16)


def test_fp8_cast_bf16_converts_native_dsv4_mxfp4_experts(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "bf16"
    model_dir.mkdir()

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    main_weight_name = "layers.0.ffn.experts.0.w1.weight"
    main_scale_name = "layers.0.ffn.experts.0.w1.scale"
    mtp_weight_name = "mtp.0.ffn.experts.0.w2.weight"
    mtp_scale_name = "mtp.0.ffn.experts.0.w2.scale"
    shared_weight_name = "layers.0.ffn.shared_experts.w1.weight"
    shared_scale_name = "layers.0.ffn.shared_experts.w1.scale"
    main_proj_weight_name = "mtp.0.main_proj.weight"
    main_proj_scale_name = "mtp.0.main_proj.scale"

    shard_1 = {
        "embed.weight": torch.ones((4, 4), dtype=torch.bfloat16),
        main_weight_name: _make_packed_weight(2, 64, device="cpu"),
        mtp_scale_name: _make_scale(1, 32, device="cpu"),
        shared_weight_name: _make_block_fp8_weight(128, 128),
        main_proj_scale_name: _make_block_scale(128, 128),
    }
    shard_2 = {
        main_scale_name: _make_scale(2, 64, device="cpu"),
        mtp_weight_name: _make_packed_weight(1, 32, device="cpu"),
        shared_scale_name: _make_block_scale(128, 128),
        main_proj_weight_name: _make_block_fp8_weight(128, 128),
    }
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    safetensors.torch.save_file(shard_1, model_dir / shard_names[0], metadata={"format": "pt"})
    safetensors.torch.save_file(shard_2, model_dir / shard_names[1], metadata={"format": "pt"})
    index = {
        "metadata": {},
        "weight_map": {
            **{name: shard_names[0] for name in shard_1},
            **{name: shard_names[1] for name in shard_2},
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    result = _run_fp8_cast_bf16(model_dir, output_dir)
    assert result.returncode == 0, result.stderr

    main_output_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    mtp_output_name = "mtp.0.mlp.experts.0.down_proj.weight"
    shared_output_name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
    main_proj_output_name = "mtp.0.main_proj.weight"

    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text())

    def load_output(name):
        output_shard = output_index["weight_map"][name]
        with safetensors.safe_open(output_dir / output_shard, framework="pt", device="cuda") as f:
            return f.get_tensor(name)

    main_output = load_output(main_output_name)
    mtp_output = load_output(mtp_output_name)
    shared_output = load_output(shared_output_name)
    main_proj_output = load_output(main_proj_output_name)

    expected_main = _dequantize_reference(
        shard_1[main_weight_name].cuda(),
        shard_2[main_scale_name].cuda(),
        dtype=torch.bfloat16,
    )
    expected_mtp = _dequantize_reference(
        shard_2[mtp_weight_name].cuda(),
        shard_1[mtp_scale_name].cuda(),
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(main_output, expected_main, rtol=0, atol=0)
    torch.testing.assert_close(mtp_output, expected_mtp, rtol=0, atol=0)
    expected_shared = (
        shard_1[shared_weight_name].float()
        * shard_2[shared_scale_name].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    ).to(torch.bfloat16)
    expected_main_proj = (
        shard_2[main_proj_weight_name].float()
        * shard_1[main_proj_scale_name].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(shared_output, expected_shared.cuda(), rtol=0, atol=0)
    torch.testing.assert_close(main_proj_output, expected_main_proj.cuda(), rtol=0, atol=0)
    assert all(tensor.dtype == torch.bfloat16 for tensor in (main_output, mtp_output, shared_output, main_proj_output))

    output_config = json.loads((output_dir / "config.json").read_text())
    assert "expert_dtype" not in output_config
    assert "quantization_config" not in output_config

    assert set(output_index["weight_map"]) == {
        "model.embed_tokens.weight",
        main_output_name,
        mtp_output_name,
        shared_output_name,
        main_proj_output_name,
    }
    assert not any(name.endswith((".scale", "_scale_inv")) for name in output_index["weight_map"])


def test_fp8_cast_bf16_preserves_native_dspark_shards_byte_for_byte(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "bf16"
    model_dir.mkdir()

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "tokenizer.json").write_bytes(b"native tokenizer bytes\n")

    target_shard_name = "model-00001-of-00002.safetensors"
    draft_shard_name = "model-00002-of-00002.safetensors"
    target_weights = {"embed.weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    draft_weights = {
        "mtp.0.ffn.experts.0.w1.weight": _make_packed_weight(2, 64, device="cpu"),
        "mtp.0.ffn.experts.0.w1.scale": _make_scale(2, 64, device="cpu"),
    }
    safetensors.torch.save_file(target_weights, model_dir / target_shard_name, metadata={"format": "pt"})
    safetensors.torch.save_file(draft_weights, model_dir / draft_shard_name, metadata={"format": "pt"})
    index = {
        "metadata": {"source": "test", "total_size": 1234},
        "weight_map": {
            **{name: target_shard_name for name in target_weights},
            **{name: draft_shard_name for name in draft_weights},
        },
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    result = _run_fp8_cast_bf16(model_dir, output_dir, "--preserve-native-dspark-checkpoint")
    assert result.returncode == 0, result.stderr

    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert output_index["weight_map"] == {"model.embed_tokens.weight": target_shard_name}
    assert not (output_dir / draft_shard_name).exists()

    draft_dir = output_dir / "dspark"
    assert (draft_dir / draft_shard_name).read_bytes() == (model_dir / draft_shard_name).read_bytes()
    assert (draft_dir / "config.json").read_bytes() == (model_dir / "config.json").read_bytes()
    assert (draft_dir / "tokenizer.json").read_bytes() == (model_dir / "tokenizer.json").read_bytes()
    draft_index = json.loads((draft_dir / "model.safetensors.index.json").read_text())
    assert draft_index == {
        "metadata": {"source": "test"},
        "weight_map": {name: draft_shard_name for name in draft_weights},
    }
    with safetensors.safe_open(draft_dir / draft_shard_name, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(draft_weights)
        assert handle.get_tensor("mtp.0.ffn.experts.0.w1.weight").dtype == torch.int8


def test_fp8_cast_bf16_rejects_mixed_target_and_dspark_shard(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
            }
        )
    )

    shard_name = "model.safetensors"
    weights = {
        "embed.weight": torch.ones((4, 4), dtype=torch.bfloat16),
        "mtp.0.ffn.experts.0.w1.weight": _make_packed_weight(2, 64, device="cpu"),
    }
    safetensors.torch.save_file(weights, model_dir / shard_name, metadata={"format": "pt"})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard_name for name in weights}})
    )

    result = _run_fp8_cast_bf16(model_dir, output_dir, "--preserve-native-dspark-checkpoint")

    assert result.returncode != 0
    assert "MTP and target tensors share shards" in result.stderr


def test_fp8_cast_bf16_retrofits_existing_output_without_rewriting_target_shards(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "bf16"
    model_dir.mkdir()
    output_dir.mkdir()

    source_config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
    }
    (model_dir / "config.json").write_text(json.dumps(source_config))
    source_target_shard = "model-00001-of-00002.safetensors"
    source_draft_shard = "model-00002-of-00002.safetensors"
    source_target_weights = {"embed.weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    source_draft_weights = {
        "mtp.0.ffn.experts.0.w1.weight": _make_packed_weight(2, 64, device="cpu"),
        "mtp.0.ffn.experts.0.w1.scale": _make_scale(2, 64, device="cpu"),
    }
    safetensors.torch.save_file(
        source_target_weights,
        model_dir / source_target_shard,
        metadata={"format": "pt"},
    )
    safetensors.torch.save_file(
        source_draft_weights,
        model_dir / source_draft_shard,
        metadata={"format": "pt"},
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    **{name: source_target_shard for name in source_target_weights},
                    **{name: source_draft_shard for name in source_draft_weights},
                },
            }
        )
    )

    output_config_bytes = b'{"num_hidden_layers":1,"torch_dtype":"bfloat16"}\n'
    (output_dir / "config.json").write_bytes(output_config_bytes)
    output_target_weights = {"model.embed_tokens.weight": torch.full((4, 4), 2, dtype=torch.bfloat16)}
    output_draft_weights = {"mtp.0.mlp.experts.0.gate_proj.weight": torch.full((2, 64), 3, dtype=torch.bfloat16)}
    safetensors.torch.save_file(
        output_target_weights,
        output_dir / source_target_shard,
        metadata={"format": "pt"},
    )
    safetensors.torch.save_file(
        output_draft_weights,
        output_dir / source_draft_shard,
        metadata={"format": "pt"},
    )
    output_index = {
        "metadata": {"total_size": 999},
        "weight_map": {
            **{name: source_target_shard for name in output_target_weights},
            **{name: source_draft_shard for name in output_draft_weights},
        },
    }
    output_index_bytes = json.dumps(output_index).encode()
    (output_dir / "model.safetensors.index.json").write_bytes(output_index_bytes)
    target_shard_bytes = (output_dir / source_target_shard).read_bytes()
    converted_draft_bytes = (output_dir / source_draft_shard).read_bytes()
    converted_draft_inode = (output_dir / source_draft_shard).stat().st_ino

    result = _run_fp8_cast_bf16(model_dir, output_dir, "--retrofit-native-dspark-checkpoint")
    assert result.returncode == 0, result.stderr

    assert (output_dir / "config.json").read_bytes() == output_config_bytes
    assert (output_dir / source_target_shard).read_bytes() == target_shard_bytes
    assert not (output_dir / source_draft_shard).exists()
    assert json.loads((output_dir / "model.safetensors.index.json").read_text()) == {
        "metadata": {},
        "weight_map": {"model.embed_tokens.weight": source_target_shard},
    }

    backup_dir = output_dir / "dspark-converted-backup"
    assert (backup_dir / source_draft_shard).read_bytes() == converted_draft_bytes
    assert (backup_dir / source_draft_shard).stat().st_ino == converted_draft_inode
    assert (backup_dir / "model.safetensors.index.json").read_bytes() == output_index_bytes
    assert (output_dir / "dspark" / source_draft_shard).read_bytes() == (model_dir / source_draft_shard).read_bytes()
    assert (output_dir / "dspark" / "config.json").read_bytes() == (model_dir / "config.json").read_bytes()
