import argparse
import json
from pathlib import Path
from types import ModuleType

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp


@pytest.mark.parametrize("model_name", ["inkling", "qwen3"])
def test_ray_conversion_uses_explicit_hf_assets_with_real_model_dispatch(
    local_ray_converter: ModuleType, tmp_path: Path, model_name: str
) -> None:
    """Workers use current HF assets while real Inkling and ordinary tensor conversion preserve values."""
    input_dir = tmp_path / "checkpoint"
    current_hf = tmp_path / "current-hf"
    stale_hf = tmp_path / "stale-hf"
    for directory, intermediate_size in [(current_hf, 2), (stale_hf, 99)]:
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "llama", "intermediate_size": intermediate_size, "vocab_size": 8})
        )
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    source_name = (
        "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        if model_name == "inkling"
        else "embedding.word_embeddings.weight"
    )
    dist_cp.save({source_name: tensor}, checkpoint_id=str(input_dir))
    torch.save(
        {"args": argparse.Namespace(num_layers=1, num_experts=1, hf_checkpoint=str(stale_hf))},
        input_dir / "common.pt",
    )
    output_dir = tmp_path / "output"
    args = local_ray_converter.Args(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        origin_hf_dir=str(current_hf),
        model_name=model_name,
        force=False,
        max_file_bytes=1024,
        concurrency=1,
        task_group_bytes=1024,
        source_key_regex=None,
        dry_run_plan=False,
        progress=False,
        progress_interval_seconds=1,
    )

    assert local_ray_converter.convert_torch_dist_to_hf_ray(args) == str(output_dir)

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors.update(safetensors.torch.load_file(output_dir / shard))
    if model_name == "inkling":
        assert set(tensors) == {
            "model.llm.layers.0.mlp.experts.0.gate_proj.weight",
            "model.llm.layers.0.mlp.experts.0.up_proj.weight",
        }
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.gate_proj.weight"], tensor[:2])
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.up_proj.weight"], tensor[2:])
    else:
        assert set(tensors) == {"model.embed_tokens.weight"}
        torch.testing.assert_close(tensors["model.embed_tokens.weight"], tensor)
    assert (output_dir / "config.json").read_bytes() == (current_hf / "config.json").read_bytes()
