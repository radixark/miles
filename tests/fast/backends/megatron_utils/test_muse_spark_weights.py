import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tests.fast.backends.megatron_utils.muse_spark_test_utils import build_model_config

from mbridge import AutoBridge
from miles_plugins.mbridge.muse_spark import MuseSpark1xBridge
from miles_plugins.models.muse_spark import checkpoint
from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.weight_mapping import (
    NON_TEXT_PREFIXES,
    build_mappings,
    hf_tensor_specs,
    rope_pairs_hf_to_kernel,
)


def _build_hf_config(dense_prefix=0, num_layers=2):
    """Small ``MuseSparkForConditionalGeneration`` config (all layers RoPE + global attention, MoE)."""
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "text_config": {
            "model_type": "muse_spark_text",
            "hidden_size": 16,
            "num_hidden_layers": num_layers,
            "vocab_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "intermediate_size": 32,
            "layer_types": ["full_attention"] * num_layers,
            "layer_rope_theta": [500000.0] * num_layers,
            "mlp_layer_types": ["dense"] * dense_prefix + ["sparse"] * (num_layers - dense_prefix),
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


def _build_hf_weights(hf_config):
    spark = hf_config if not isinstance(hf_config, dict) else parse_muse_spark_1x_config(hf_config)
    hf_weights = {}
    for index, (name, spec) in enumerate(hf_tensor_specs(spark).items()):
        weight = torch.arange(math.prod(spec.shape), dtype=torch.float32).reshape(spec.shape)
        hf_weights[name] = (weight / 113 + index / 1000).to(spec.dtype)
    return hf_weights


def _write_checkpoint(path, hf_config, hf_weights=None):
    path.mkdir(parents=True, exist_ok=True)
    hf_weights = _build_hf_weights(hf_config) if hf_weights is None else hf_weights
    (path / "config.json").write_text(json.dumps(hf_config))
    save_file(hf_weights, path / "model.safetensors")
    return hf_weights


class _FakeSingleRankGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _WeightOnlyModel(torch.nn.Module):
    """CPU destination exposing the actual state and process-group mapping contract."""

    def __init__(self, hf_config):
        super().__init__()
        spark_config = parse_muse_spark_1x_config(hf_config)
        self.config = build_model_config(spark_config)
        self.pre_process = self.post_process = True
        self.pg_collection = SimpleNamespace(tp=_FakeSingleRankGroup(), expt_tp=_FakeSingleRankGroup())
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(spark_config.num_hidden_layers)])
        for mapping in build_mappings(spark_config):
            parts = mapping.megatron_param.split(".")
            module = self
            for part in parts[:-1]:
                if part not in module._modules:
                    module.add_module(part, torch.nn.Module())
                module = module.get_submodule(part)
            spec = checkpoint._get_megatron_tensor_spec(mapping, hf_tensor_specs(spark_config))
            weight = torch.zeros(spec.shape, dtype=spec.dtype)
            if parts[-1] == "expert_bias":
                module.register_buffer(parts[-1], weight)
            else:
                module.register_parameter(parts[-1], torch.nn.Parameter(weight))
        for index, layer in enumerate(self.decoder.layers):
            layer.layer_number = index + 1
            layer.mlp.local_expert_indices = list(range(spark_config.num_local_experts))


@pytest.fixture
def weight_model(monkeypatch):
    # The CPU fixture exercises loading and mapping without constructing a GPU model.
    monkeypatch.setattr(checkpoint, "MuseSpark1xGPTModel", _WeightOnlyModel)
    return _WeightOnlyModel(_build_hf_config())


@pytest.mark.parametrize("indexed", [False, True], ids=["single-shard", "indexed-shards"])
def test_slice_loader_preserves_weights_and_fp32_buffers(tmp_path, weight_model, monkeypatch, indexed):
    expected_weights = _write_checkpoint(tmp_path, _build_hf_config())
    if indexed:
        weight_map = {}
        for index in range(2):
            filename = f"model-{index}.safetensors"
            tensors = dict(list(expected_weights.items())[index::2])
            save_file(tensors, tmp_path / filename)
            weight_map.update(dict.fromkeys(tensors, filename))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (tmp_path / "model.safetensors").unlink()
        save_file({"unrelated": torch.ones(1)}, tmp_path / "unrelated.safetensors")
    hf_checkpoint = checkpoint.SafetensorsCheckpoint(tmp_path)
    stacked_expert_names = {
        mapping.hf_param for mapping in build_mappings(weight_model.config.spark) if mapping.expert_index is not None
    }
    slice_reads = []
    read_slice = hf_checkpoint.read_slice

    def record_slice(name, selection):
        slice_reads.append((name, selection))
        if name in stacked_expert_names:
            assert isinstance(selection[0], int)
        return read_slice(name, selection)

    monkeypatch.setattr(hf_checkpoint, "read_slice", record_slice)
    checkpoint.load_hf_checkpoint([weight_model], hf_checkpoint)
    assert {name for name, _ in slice_reads} == set(expected_weights)
    assert len(slice_reads) > len(expected_weights)
    model_state = weight_model.state_dict()
    for mapping in build_mappings(weight_model.config.spark):
        for name, weight in mapping.megatron_to_hf(model_state[mapping.megatron_param]).items():
            expected_weight = expected_weights[name]
            if mapping.expert_index is not None:
                expected_weight = expected_weight[mapping.expert_index]
            assert weight.dtype == expected_weight.dtype
            assert torch.equal(weight, expected_weight), name
    router = model_state["decoder.layers.0.mlp.router.weight"]
    bias = model_state["decoder.layers.0.mlp.router.expert_bias"]
    assert router.dtype == bias.dtype == torch.float32
    assert not torch.equal(router, router.bfloat16().float())


def test_offline_mbridge_entry_uses_slice_loader(tmp_path, weight_model):
    expected_weights = _write_checkpoint(tmp_path, _build_hf_config())
    bridge = AutoBridge.from_pretrained(str(tmp_path), trust_remote_code=False)
    assert isinstance(bridge, MuseSpark1xBridge)
    bridge.load_weights([weight_model], str(tmp_path), memory_efficient=True)
    assert torch.equal(
        weight_model.embedding.word_embeddings.weight, expected_weights["model.language_model.embed_tokens.weight"]
    )


def test_omni_source_loads_complete_text_without_reading_non_text_payload(tmp_path, weight_model, monkeypatch):
    hf_weights = _build_hf_weights(_build_hf_config())
    auxiliary_names = tuple(prefix + "encoder.layers.0.weight" for prefix in NON_TEXT_PREFIXES)
    hf_weights.update({name: torch.ones(2, dtype=torch.bfloat16) for name in auxiliary_names})
    _write_checkpoint(tmp_path, _build_hf_config(), hf_weights)
    hf_checkpoint = checkpoint.SafetensorsCheckpoint(tmp_path)
    read_slice = hf_checkpoint.read_slice

    def text_slice(name, selection):
        assert name not in auxiliary_names
        return read_slice(name, selection)

    monkeypatch.setattr(hf_checkpoint, "read_slice", text_slice)
    checkpoint.load_hf_checkpoint(weight_model, hf_checkpoint)
    assert torch.equal(
        weight_model.embedding.word_embeddings.weight, hf_weights["model.language_model.embed_tokens.weight"]
    )


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "dtype"])
def test_hf_contract_fails_before_target_mutation(tmp_path, weight_model, problem):
    hf_weights = _build_hf_weights(_build_hf_config())
    hf_weights["model.vision_tower.encoder.weight"] = torch.ones(1)
    name = "model.language_model.layers.0.mlp.gate.weight"
    if problem == "missing":
        del hf_weights[name]
    elif problem == "unexpected":
        hf_weights["unmapped.weight"] = torch.ones(1)
    elif problem == "shape":
        hf_weights[name] = hf_weights[name].T.contiguous()
    else:
        hf_weights[name] = hf_weights[name].bfloat16()
    _write_checkpoint(tmp_path, _build_hf_config(), hf_weights)
    weights_before = {name: weight.clone() for name, weight in weight_model.state_dict().items()}
    with pytest.raises(ValueError, match="coverage|dtype|shape"):
        checkpoint.load_hf_checkpoint(weight_model, tmp_path)
    assert all(torch.equal(weight, weights_before[name]) for name, weight in weight_model.state_dict().items())


@pytest.mark.parametrize("problem", ["parameter", "buffer", "router_dtype", "extra_state"])
def test_destination_contract_rejects_orphan_and_lossy_state(tmp_path, weight_model, problem):
    _write_checkpoint(tmp_path, _build_hf_config())
    if problem == "parameter":
        weight_model.register_parameter("orphan", torch.nn.Parameter(torch.ones(1)))
    elif problem == "buffer":
        weight_model.register_buffer("orphan", torch.ones(1))
    elif problem == "extra_state":
        weight_model.register_buffer("_extra_state", torch.empty(0, dtype=torch.uint8))
    else:
        weight_model.decoder.layers[0].mlp.router.weight.data = weight_model.decoder.layers[
            0
        ].mlp.router.weight.data.bfloat16()
    with pytest.raises(ValueError, match="coverage|dtype"):
        checkpoint.load_hf_checkpoint(weight_model, tmp_path)


@pytest.mark.parametrize("gated", [False, True], ids=["qkv", "qgkv"])
def test_qgkv_local_slice_layout_matches_grouped_global_layout(tmp_path, gated):
    # The checkpoint always carries the attention output gate; the ungated variant is exercised on the mapping only.
    spark_config = replace(parse_muse_spark_1x_config(_build_hf_config()), use_attn_o_gate=gated)
    hf_weights = _write_checkpoint(tmp_path, _build_hf_config(), _build_hf_weights(spark_config))
    mapping = next(candidate for candidate in build_mappings(spark_config) if candidate.kind == "qgkv")
    projections = ("q", "g", "k", "v") if gated else ("q", "k", "v")

    def kernel_rows(projection):
        weight = hf_weights[mapping.hf_param[projection]]
        if projection in ("q", "k"):
            # the loader turns the checkpoint's half-split RoPE rows into interleaved pairs
            weight = rope_pairs_hf_to_kernel(weight, spark_config.head_dim)
        return weight.reshape(spark_config.num_key_value_heads, -1, spark_config.hidden_size)

    global_weight = torch.cat([kernel_rows(projection) for projection in projections], dim=1).flatten(0, 1)
    mapping.partition_size = 2
    mapping.partition_dim = 0
    hf_checkpoint = checkpoint.SafetensorsCheckpoint(tmp_path)
    for rank in range(2):
        mapping.partition_rank = rank
        local_weight = checkpoint._read_mapped_weight(hf_checkpoint, mapping)
        assert torch.equal(local_weight, global_weight.chunk(2)[rank])
        restored_weights = mapping.megatron_to_hf(local_weight)
        for name, weight in restored_weights.items():
            assert torch.equal(weight, hf_weights[name].chunk(2)[rank])


def test_dense_glu_local_slices_preserve_gate_up_stride(tmp_path):
    hf_config = _build_hf_config(dense_prefix=1)  # layer 0 is a dense GLU layer
    hf_weights = _write_checkpoint(tmp_path, hf_config)
    spark_config = parse_muse_spark_1x_config(hf_config)
    mapping = next(candidate for candidate in build_mappings(spark_config) if candidate.kind == "gated_mlp")
    mapping.partition_size = 2
    mapping.partition_dim = 0
    mapping.partition_stride = 2
    hf_checkpoint = checkpoint.SafetensorsCheckpoint(tmp_path)
    for rank in range(2):
        mapping.partition_rank = rank
        expected_weights = torch.cat(
            [hf_weights[mapping.hf_param[projection]].chunk(2)[rank] for projection in ("gate", "up")]
        )
        assert torch.equal(checkpoint._read_mapped_weight(hf_checkpoint, mapping), expected_weights)


def test_mapping_names_use_partial_local_layers_without_original_offset():
    spark_config = parse_muse_spark_1x_config(_build_hf_config(num_layers=4))
    mappings = build_mappings(spark_config)
    layer_indices = {
        int(name.split(".")[3])
        for mapping in mappings
        for name in (mapping.hf_param.values() if isinstance(mapping.hf_param, dict) else (mapping.hf_param,))
        if name.startswith("model.language_model.layers.")
    }
    assert layer_indices == {0, 1, 2, 3}
