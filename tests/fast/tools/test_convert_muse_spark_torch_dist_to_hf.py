"""A Muse Spark torch_dist checkpoint converts back into the released HF layout: unchanged weights reproduce every file
byte for byte with any number of workers, the towers come from the HF checkpoint, and a trained expert lands in its
slice of the stacked expert tensor."""

import json
import math

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from tools.convert_muse_spark_torch_dist_to_hf import _source, convert

from miles_plugins.models.muse_spark import checkpoint
from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.weight_mapping import build_mappings, hf_tensor_specs

_TOWER = "model.vision_tower.patch_embed.weight"
_HF_CONFIG = {
    "architectures": ["MuseSparkForConditionalGeneration"],
    "model_type": "muse_spark",
    "dtype": "bfloat16",
    "text_config": {
        "model_type": "muse_spark_text",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "intermediate_size": 32,
        "layer_types": ["full_attention", "full_attention"],
        "layer_rope_theta": [500000.0, 500000.0],
        "mlp_layer_types": ["dense", "sparse"],
        "moe_hidden_size": 8,
        "moe_intermediate_size": 16,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "output_multiplier": (16 / 256) ** -0.5,
        "qk_scale_factor": 43.7840518911 / 2.0,
        "residual_gate_temperature": 0.3,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    },
}


def _write_hf_checkpoint(path, spark):
    """Two shards with a weight index, like the released checkpoint, plus a vision tower tensor."""
    tensors = {}
    for index, (name, spec) in enumerate(hf_tensor_specs(spark).items()):
        weight = torch.arange(math.prod(spec.shape), dtype=torch.float32).reshape(spec.shape)
        tensors[name] = (weight / 113 + index / 1000).to(spec.dtype)
    tensors[_TOWER] = torch.arange(12, dtype=torch.float32).reshape(3, 4).bfloat16()
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_HF_CONFIG))
    weight_map = {}
    for shard_index in range(2):
        shard = f"model-{shard_index + 1:05d}-of-00002.safetensors"
        names = sorted(tensors)[shard_index::2]
        save_file({name: tensors[name] for name in names}, path / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(names, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _write_torch_dist(path, hf_checkpoint, spark):
    """The torch_dist layout Megatron saves: global tensors, one layer's experts stacked under one key."""
    source = checkpoint.SafetensorsCheckpoint(hf_checkpoint)
    state, experts = {}, {}
    for mapping in build_mappings(spark):
        key, expert = _source(mapping.megatron_param)
        weight = checkpoint._read_mapped_weight(source, mapping)
        if expert is None:
            state[key] = weight
        else:
            experts.setdefault(key, {})[expert] = weight
    for key, by_expert in experts.items():
        state[key] = torch.stack([by_expert[expert] for expert in sorted(by_expert)])
    dcp.save(state, storage_writer=dcp.FileSystemWriter(str(path)), no_dist=True)
    return state


@pytest.fixture
def checkpoints(tmp_path):
    spark = parse_muse_spark_1x_config(_HF_CONFIG)
    hf_checkpoint = tmp_path / "hf"
    _write_hf_checkpoint(hf_checkpoint, spark)
    return spark, hf_checkpoint


@pytest.mark.parametrize("num_workers", [1, 3])
def test_unchanged_weights_reproduce_the_hf_checkpoint(tmp_path, checkpoints, num_workers):
    spark, hf_checkpoint = checkpoints
    _write_torch_dist(tmp_path / "torch_dist", hf_checkpoint, spark)
    convert(tmp_path / "torch_dist", hf_checkpoint, tmp_path / "out", num_workers=num_workers)
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        path.name for path in hf_checkpoint.iterdir()
    )
    for path in hf_checkpoint.iterdir():
        assert (tmp_path / "out" / path.name).read_bytes() == path.read_bytes(), path.name


def test_trained_expert_lands_in_its_stacked_slice(tmp_path, checkpoints):
    spark, hf_checkpoint = checkpoints
    state = _write_torch_dist(tmp_path / "base_dist", hf_checkpoint, spark)
    trained_key = "decoder.layers.1.mlp.experts.experts.linear_fc1.weight"
    state[trained_key][2] += 1
    dcp.save(state, storage_writer=dcp.FileSystemWriter(str(tmp_path / "torch_dist")), no_dist=True)
    convert(tmp_path / "torch_dist", hf_checkpoint, tmp_path / "out", num_workers=2)
    base, out = {}, {}
    for shard in sorted(path.name for path in hf_checkpoint.glob("*.safetensors")):
        base.update(load_file(hf_checkpoint / shard))
        out.update(load_file(tmp_path / "out" / shard))
    (stacked,) = [name for name in base if ".layers.1." in name and name.endswith(".experts.gate_up_proj")]
    expected = base[stacked].clone()
    expected[2] += 1
    assert torch.equal(out[stacked], expected)
    assert all(torch.equal(out[name], base[name]) for name in base if name != stacked)
