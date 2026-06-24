from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tools" / "convert_torch_dist_to_hf_ray.py"


def load_converter_module():
    tools_dir = str(SCRIPT_PATH.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    spec = importlib.util.spec_from_file_location("convert_torch_dist_to_hf_ray", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_task_result(module, **overrides):
    values = {
        "task_id": 0,
        "actor_id": 0,
        "node": "node",
        "pid": 1,
        "shards": (),
        "source_bytes": 0,
        "output_bytes": 0,
        "weights": 0,
        "source_keys": ("source",),
        "dcp_read_items": 0,
        "dcp_files": 0,
        "dcp_storage_bytes": 0,
        "cuda_device_id": None,
        "ray_node_id": "local",
    }
    values.update(overrides)
    return module.TaskResult(**values)


def make_args(module, input_dir: Path, output_dir: Path, **overrides):
    values = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "origin_hf_dir": None,
        "model_name": "qwen2",
        "force": False,
        "max_file_bytes": 20 * 1024**3,
        "concurrency": 1,
        "task_group_bytes": 0,
        "source_key_regex": None,
        "dry_run_plan": False,
        "sha1sum_output": False,
        "progress": False,
        "progress_interval_seconds": 5.0,
    }
    values.update(overrides)
    return module.Args(**values)


def write_common_pt(path: Path, **overrides):
    args = SimpleNamespace(
        original_hf_model_name="qwen2",
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=None,
        num_layers=1,
        num_experts=2,
        q_lora_rank=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    torch.save({"args": args}, path / "common.pt")


def write_dcp_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dist_cp.save(tensors, checkpoint_id=str(path))
    write_common_pt(path)


def test_plan_whole_source_tasks_keeps_mla_pairs_together():
    module = load_converter_module()
    tensor_metadata = {
        "decoder.layers.0.self_attention.linear_q_down_proj.weight": (torch.Size([2, 4]), torch.bfloat16),
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight": (torch.Size([2, 4]), torch.bfloat16),
        "decoder.layers.0.mlp.linear_fc2.weight": (torch.Size([2, 2]), torch.bfloat16),
    }

    tasks = module.plan_whole_source_tasks(tensor_metadata, q_lora_rank=128, task_group_bytes=0)

    paired_tasks = [
        task
        for task in tasks
        if "decoder.layers.0.self_attention.linear_q_down_proj.weight" in task.keys
        or "decoder.layers.0.self_attention.linear_kv_down_proj.weight" in task.keys
    ]
    assert len(paired_tasks) == 1
    assert paired_tasks[0].keys == (
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight",
        "decoder.layers.0.self_attention.linear_q_down_proj.weight",
    )


def test_converted_moe_tensors_from_full_fc1_chunk_names_gate_and_up():
    module = load_converter_module()
    read_item = SimpleNamespace(
        storage_index=SimpleNamespace(offset=torch.Size([2, 0, 0])),
        lengths=torch.Size([1, 4, 3]),
    )
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    groups = module.converted_moe_tensors_from_chunk(
        "decoder.layers.7.mlp.experts.experts.linear_fc1.weight",
        layer_idx=7,
        linear_name="linear_fc1",
        hf_prefix="",
        read_item=read_item,
        tensor=tensor,
        tensor_size=torch.Size([4, 4, 3]),
    )

    assert [group.source_name for group in groups] == ["module.module.decoder.layers.7.mlp.experts.linear_fc1.weight2"]
    assert [name for name, _ in groups[0].tensors] == [
        "model.layers.7.mlp.experts.2.gate_proj.weight",
        "model.layers.7.mlp.experts.2.up_proj.weight",
    ]
    assert groups[0].tensors[0][1].shape == torch.Size([2, 3])
    assert groups[0].tensors[1][1].shape == torch.Size([2, 3])


def test_converted_moe_tensors_from_fc2_chunk_names_down_projection():
    module = load_converter_module()
    read_item = SimpleNamespace(
        storage_index=SimpleNamespace(offset=torch.Size([3, 0, 0])),
        lengths=torch.Size([1, 2, 3]),
    )
    tensor = torch.ones((1, 2, 3), dtype=torch.float32)

    groups = module.converted_moe_tensors_from_chunk(
        "language_model.decoder.layers.5.mlp.experts.experts.linear_fc2.weight",
        layer_idx=5,
        linear_name="linear_fc2",
        hf_prefix="language_model.",
        read_item=read_item,
        tensor=tensor,
        tensor_size=torch.Size([8, 2, 3]),
    )

    assert [name for name, _ in groups[0].tensors] == ["language_model.model.layers.5.mlp.experts.3.down_proj.weight"]


def test_plan_global_shards_is_deterministic_and_rejects_duplicates():
    module = load_converter_module()
    result = make_task_result(
        module,
        shards=(
            module.ShardManifest("tmp-b.safetensors", None, ("b",), 8, 1),
            module.ShardManifest("tmp-a.safetensors", None, ("a",), 4, 1),
        ),
        source_bytes=12,
        output_bytes=12,
        weights=2,
    )

    planned, index = module.plan_global_shards([result])

    assert [shard.final_filename for shard in planned] == [
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
    ]
    assert index == {
        "metadata": {"total_size": 12},
        "weight_map": {
            "b": "model-00000-of-00002.safetensors",
            "a": "model-00001-of-00002.safetensors",
        },
    }

    duplicate = make_task_result(
        module,
        shards=(
            module.ShardManifest("tmp-1.safetensors", None, ("dup",), 1, 1),
            module.ShardManifest("tmp-2.safetensors", None, ("dup",), 1, 1),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate HF tensor"):
        module.plan_global_shards([duplicate])

    with pytest.raises(ValueError, match="No HF tensor shards"):
        module.plan_global_shards([])


def test_cloud_paths_are_rejected():
    module = load_converter_module()

    with pytest.raises(ValueError, match="local filesystem path"):
        module.reject_cloud_path("scheme://bucket/checkpoint", "input_dir")


def test_tiny_dense_checkpoint_conversion_writes_checksums_when_requested(tmp_path):
    module = load_converter_module()
    input_dir = tmp_path / "ckpt"
    output_dir = tmp_path / "out"
    source_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    write_dcp_checkpoint(input_dir, {"decoder.layers.0.mlp.linear_fc2.weight": source_tensor})

    try:
        module.convert_torch_dist_to_hf_ray(
            make_args(
                module,
                input_dir,
                output_dir,
                max_file_bytes=128,
                sha1sum_output=True,
            )
        )
    finally:
        module.ray.shutdown()

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"model.layers.0.mlp.down_proj.weight": "model-00000-of-00001.safetensors"}
    shard = safetensors.torch.load_file(output_dir / "model-00000-of-00001.safetensors")
    assert torch.equal(shard["model.layers.0.mlp.down_proj.weight"], source_tensor)

    checksum = json.loads((output_dir / "checksum.json").read_text())
    shard_path = output_dir / "model-00000-of-00001.safetensors"
    assert checksum == {
        "algorithm": "sha1",
        "files": {
            "model-00000-of-00001.safetensors": {
                "bytes": shard_path.stat().st_size,
                "sha1": module.sha1sum_file(str(shard_path)),
                "tensor_bytes": 16,
            }
        },
    }
    assert not (output_dir / ".ray-convert-staging").exists()


def test_tiny_direct_moe_checkpoint_conversion(tmp_path):
    module = load_converter_module()
    input_dir = tmp_path / "ckpt"
    output_dir = tmp_path / "out"
    source_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    write_dcp_checkpoint(input_dir, {"decoder.layers.0.mlp.experts.experts.linear_fc1.weight": source_tensor})

    try:
        module.convert_torch_dist_to_hf_ray(
            make_args(
                module,
                input_dir,
                output_dir,
            )
        )
    finally:
        module.ray.shutdown()

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    assert not (output_dir / "checksum.json").exists()
