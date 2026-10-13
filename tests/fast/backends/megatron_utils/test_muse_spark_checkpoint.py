"""The Muse Spark checkpoint (``MuseSparkForConditionalGeneration``, ``model_type: muse_spark``) must load into
and export from the Megatron tensors exactly: nested config semantics, tensor names, router orientation, expert
stacks and the rotary-pair row order."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from miles_plugins.models.muse_spark import checkpoint
from miles_plugins.models.muse_spark.config import map_to_megatron_fields, parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.hf_config import MuseSparkHFConfig
from miles_plugins.models.muse_spark.weight_mapping import (
    build_mappings,
    hf_tensor_specs,
    rope_pairs_hf_to_kernel,
    rope_pairs_kernel_to_hf,
)

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB, EXPERTS = 16, 4, 4, 2, 4, 32, 4


def checkpoint_config():
    # Layers 0 and 2: RoPE + sliding window 8; layers 1 and 3: NoPE + global attention. All layers MoE.
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "transformers_version": "5.18.0.dev0",
        "image_token_id": 7,
        "video_token_id": 8,
        "audio_token_id": 9,
        "out_hidden_size": 32,
        "projector_hidden_size": 8,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_spark_text",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "final_logit_softcapping": 20.0,
            "head_dim": HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": HIDDEN,
            "initializer_range": 0.02,
            "intermediate_size": 2 * HIDDEN,
            "layer_rope_theta": [500000.0, 0, 500000.0, 0],
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "max_position_embeddings": 64,
            "mlp_layer_types": ["sparse"] * LAYERS,
            "moe_hidden_size": HIDDEN // 2,
            "moe_intermediate_size": 12,
            "num_attention_heads": HEADS,
            "num_experts_per_tok": 2,
            "num_hidden_layers": LAYERS,
            "num_key_value_heads": KV_HEADS,
            "num_local_experts": EXPERTS,
            "num_shared_experts": 0,
            "output_multiplier": (HIDDEN / 256) ** -0.5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 43.7840518911 / math.sqrt(HEAD_DIM),
            "residual_gate_temperature": 0.3,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "routed_scaling_factor": 1.0,
            "sliding_window": 8,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": VOCAB,
        },
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_act": "gelu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "layer_norm_eps": 1e-5,
            "layer_types": ["window_attention", "full_attention"],
            "max_position_embeddings": 16,
            "merge_size": 2,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "patch_size": 2,
            "patch_temporal": 1,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
        },
        "audio_config": {"model_type": "muse_spark_audio", "out_hidden_size": HIDDEN},
    }


class _DictCheckpoint:
    """In-memory stand-in for ``SafetensorsCheckpoint`` (metadata + slice reads)."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.tensor_metadata = {
            name: checkpoint.TensorSpec(tuple(tensor.shape), tensor.dtype) for name, tensor in tensors.items()
        }
        self.weight_map = dict.fromkeys(tensors, "model.safetensors")

    def read_slice(self, name, selection):
        return self.tensors[name][selection]


def _random_hf_tensors(spark, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(spec.shape, generator=generator).to(spec.dtype)
        for name, spec in hf_tensor_specs(spark).items()
    }


def _megatron_tensors(spark, hf_tensors):
    source = _DictCheckpoint(hf_tensors)
    return {
        mapping.megatron_param: checkpoint._read_mapped_weight(source, mapping) for mapping in build_mappings(spark)
    }


def _export(spark, megatron_tensors):
    exported = {}
    for mapping in build_mappings(spark):
        for name, tensor in mapping.megatron_to_hf(megatron_tensors[mapping.megatron_param]).items():
            if mapping.expert_index is None:
                exported[name] = tensor
            else:
                exported.setdefault(name, {})[mapping.expert_index] = tensor
    return {
        name: torch.stack([value[index] for index in sorted(value)]) if isinstance(value, dict) else value
        for name, value in exported.items()
    }


def test_config_parses_schedules_and_moe_semantics():
    spark = parse_muse_spark_1x_config(checkpoint_config())
    assert (spark.hidden_size, spark.num_hidden_layers, spark.head_dim, spark.vocab_size) == (
        HIDDEN,
        LAYERS,
        HEAD_DIM,
        VOCAB,
    )
    assert spark.rope_layer_schedule == (1, 0, 1, 0) and spark.global_attn_cfg == (8, 0, 8, 0)
    assert [spark.layer_uses_rope(index) for index in range(LAYERS)] == [True, False, True, False]
    assert [spark.layer_window_size(index) for index in range(LAYERS)] == [7, None, 7, None]
    assert all(spark.layer_is_moe(index) for index in range(LAYERS))
    assert (spark.num_local_experts, spark.num_experts_per_tok, spark.moe_compression_factor) == (EXPERTS, 2, 2)
    assert (spark.moe_override_routed_hidden_dim, spark.expert_hidden_size, spark.moe_intermediate_size) == (12, 8, 12)
    assert spark.ffn_dim_multiplier == 2.0 and spark.rope_theta == 500000.0 and spark.output_soft_cap_temp == 20.0
    assert spark.use_attn_o_gate and spark.use_residual_gate and spark.metap_enabled and spark.moe_router_in_fp32
    assert math.isclose(spark.query_multiplier, 43.7840518911 / math.sqrt(HEAD_DIM))
    assert map_to_megatron_fields(spark)["no_rope_freq"] == [0, 1, 0, 1]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.__setitem__("qk_scale_factor", 1.0), "qk_scale_factor"),
        (lambda t: t.__setitem__("output_multiplier", 1.0), "output_multiplier"),
        (lambda t: t.__setitem__("residual_gate_temperature", 0.5), "residual_gate_temperature"),
        (lambda t: t.__setitem__("mlp_layer_types", ["sparse", "dense", "sparse", "sparse"]), "prefix"),
        (lambda t: t.__setitem__("layer_rope_theta", [1.0, 0, 2.0, 0]), "one RoPE theta"),
        (lambda t: t.__setitem__("brand_new_knob", 3), "unsupported text_config fields"),
    ],
)
def test_config_rejects_semantics_miles_does_not_implement(mutate, message):
    source = checkpoint_config()
    mutate(source["text_config"])
    with pytest.raises(ValueError, match=message):
        parse_muse_spark_1x_config(source)


def test_config_requires_the_nested_text_config():
    flat = {"model_type": "musespark1x_omni", "hidden_size": HIDDEN, "num_hidden_layers": LAYERS}
    with pytest.raises(ValueError, match="unsupported Spark config fields|text_config"):
        parse_muse_spark_1x_config(flat)
    with pytest.raises(ValueError, match="text_config"):
        parse_muse_spark_1x_config({"model_type": "muse_spark"})


def test_checkpoint_names_and_orientations():
    spark = parse_muse_spark_1x_config(checkpoint_config())
    specs = hf_tensor_specs(spark)
    layer0 = {name.removeprefix("model.language_model.layers.0.") for name in specs if ".layers.0." in name}
    assert layer0 == {
        "input_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_attention_residual_gate.gate",
        "post_feedforward_residual_gate.gate",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.pre_expert_proj.weight",
        "mlp.post_expert_proj.weight",
        "mlp.pre_expert_norm.weight",
        "mlp.experts.post_expert_norm.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    assert {name for name in specs if ".layers." not in name} == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    # The router is stored [experts, hidden] (Megatron's orientation, no transpose) in float32 like its bias.
    assert specs["model.language_model.layers.0.mlp.gate.weight"] == checkpoint.TensorSpec(
        (EXPERTS, HIDDEN), torch.float32
    )
    assert specs["model.language_model.layers.0.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    assert specs["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (EXPERTS, 24, HIDDEN // 2)
    assert specs["model.language_model.layers.0.mlp.experts.down_proj"].shape == (EXPERTS, HIDDEN // 2, 12)
    router = next(m for m in build_mappings(spark) if m.megatron_param == "decoder.layers.0.mlp.router.weight")
    assert router.kind == "direct"


def test_rope_pair_permutation_matches_sglang_loader():
    heads, head_dim = 4, 8
    kernel = torch.arange(heads * head_dim * 3).reshape(heads * head_dim, 3)
    # SGLang load_weights: hf.reshape(heads, 2, head_dim // 2, ...).transpose(1, 2) == kernel
    hf = kernel.reshape(heads, head_dim // 2, 2, 3).transpose(1, 2).reshape(-1, 3)
    torch.testing.assert_close(rope_pairs_hf_to_kernel(hf, head_dim), kernel, rtol=0, atol=0)
    torch.testing.assert_close(rope_pairs_kernel_to_hf(kernel, head_dim), hf, rtol=0, atol=0)
    bias = torch.arange(heads * head_dim)
    torch.testing.assert_close(
        rope_pairs_hf_to_kernel(rope_pairs_kernel_to_hf(bias, head_dim), head_dim), bias, rtol=0, atol=0
    )


def test_load_and_export_round_trip_bit_exact():
    spark = parse_muse_spark_1x_config(checkpoint_config())
    hf_tensors = _random_hf_tensors(spark)
    megatron = _megatron_tensors(spark, hf_tensors)
    assert set(megatron) == {mapping.megatron_param for mapping in build_mappings(spark)}
    # q/k rows are re-interleaved on the way in: the kernel row order differs from the checkpoint rows.
    qkv = megatron["decoder.layers.0.self_attention.linear_qkv.weight"]
    q_rows = HEADS // KV_HEADS * HEAD_DIM
    assert not torch.equal(qkv[:q_rows], hf_tensors["model.language_model.layers.0.self_attn.q_proj.weight"][:q_rows])
    exported = _export(spark, megatron)
    assert set(exported) == set(hf_tensors)
    for name, tensor in hf_tensors.items():
        torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)


class _FakeSingleRankGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _WeightOnlyModel(torch.nn.Module):
    """CPU destination with the real state / process-group mapping contract, no GPU model."""

    def __init__(self, spark):
        super().__init__()
        self.config = SimpleNamespace(spark=spark)
        self.pre_process = self.post_process = True
        self.pg_collection = SimpleNamespace(tp=_FakeSingleRankGroup(), expt_tp=_FakeSingleRankGroup())
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(spark.num_hidden_layers)])
        specs = hf_tensor_specs(spark)
        for mapping in build_mappings(spark):
            parts = mapping.megatron_param.split(".")
            module = self
            for part in parts[:-1]:
                if part not in module._modules:
                    module.add_module(part, torch.nn.Module())
                module = module.get_submodule(part)
            spec = checkpoint._get_megatron_tensor_spec(mapping, specs)
            weight = torch.zeros(spec.shape, dtype=spec.dtype)
            if parts[-1] == "expert_bias":
                module.register_buffer(parts[-1], weight)
            else:
                module.register_parameter(parts[-1], torch.nn.Parameter(weight))
        for index, layer in enumerate(self.decoder.layers):
            layer.layer_number = index + 1
            layer.mlp.local_expert_indices = list(range(spark.num_local_experts))


def test_load_hf_checkpoint_ignores_non_text_tensors_and_rejects_wrong_names(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "MuseSpark1xGPTModel", _WeightOnlyModel)
    monkeypatch.setattr(checkpoint, "validate_model_config", lambda config: None)
    spark = parse_muse_spark_1x_config(checkpoint_config())
    tensors = _random_hf_tensors(spark, seed=3)
    megatron = _megatron_tensors(spark, tensors)
    # Non-text tensors of the omni checkpoint are ignored by the text coverage check.
    tensors["model.vision_tower.ln_pre.weight"] = torch.ones(8, dtype=torch.bfloat16)
    tensors["model.audio_tower.stem.weight"] = torch.ones(2, dtype=torch.bfloat16)
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(checkpoint_config()))
    save_file(tensors, path / "model.safetensors")
    model = _WeightOnlyModel(spark)
    checkpoint.load_hf_checkpoint(model, path)
    state = model.state_dict()
    assert set(state) == set(megatron)
    for name, tensor in megatron.items():
        torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)
    # A text tensor under a wrong name is rejected before any copy.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps(checkpoint_config()))
    wrong = dict(tensors)
    wrong["model.layers.0.self_attn.q_proj.weight"] = wrong.pop(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    save_file(wrong, bad / "model.safetensors")
    with pytest.raises(ValueError, match="HF tensor coverage differs"):
        checkpoint.load_hf_checkpoint(_WeightOnlyModel(spark), bad)


def test_hf_config_class_exposes_flat_view(tmp_path):
    config = MuseSparkHFConfig(**checkpoint_config())
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (HIDDEN, LAYERS, VOCAB)
    assert (config.num_attention_heads, config.head_dim, config.rope_theta) == (HEADS, HEAD_DIM, 500000.0)
    assert not hasattr(config, "text_config")
    assert config.to_dict()["text_config"]["layer_types"][0] == "sliding_attention"
    assert config.to_dict()["model_type"] == "muse_spark"
    assert config.architectures == ["MuseSparkForConditionalGeneration"]
    assert parse_muse_spark_1x_config(config.to_dict()) == parse_muse_spark_1x_config(checkpoint_config())
    assert MuseSparkHFConfig().model_type == "muse_spark"  # transformers builds a default instance for diffs
    from miles.utils.chat_template_utils.muse_spark_omni import is_muse_spark_omni_checkpoint

    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config()))
    assert is_muse_spark_omni_checkpoint(str(tmp_path))


def test_raw_export_uses_checkpoint_names(tmp_path):
    from miles.backends.megatron_utils.megatron_to_hf.muse_spark import _checkpoint_layout, convert_muse_spark_1x_to_hf

    _checkpoint_layout.cache_clear()
    spark = parse_muse_spark_1x_config(checkpoint_config())
    hf_tensors = _random_hf_tensors(spark, seed=5)
    megatron = _megatron_tensors(spark, hf_tensors)
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config()))
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))
    layer = "model.language_model.layers.1."
    experts = dict(
        convert_muse_spark_1x_to_hf(
            args,
            "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight2",
            megatron["decoder.layers.1.mlp.experts.linear_fc1.weight2"],
        )
    )
    assert set(experts) == {layer + "mlp.experts.2.gate_proj.weight", layer + "mlp.experts.2.up_proj.weight"}
    stacked = hf_tensors[layer + "mlp.experts.gate_up_proj"][2]
    torch.testing.assert_close(
        torch.cat(
            [experts[layer + "mlp.experts.2.gate_proj.weight"], experts[layer + "mlp.experts.2.up_proj.weight"]]
        ),
        stacked,
        rtol=0,
        atol=0,
    )
    down = dict(
        convert_muse_spark_1x_to_hf(
            args,
            "decoder.layers.1.mlp.experts.linear_fc2.weight0",
            megatron["decoder.layers.1.mlp.experts.linear_fc2.weight0"],
        )
    )
    assert set(down) == {layer + "mlp.experts.0.down_proj.weight"}
    attention = dict(
        convert_muse_spark_1x_to_hf(
            args,
            "decoder.layers.1.self_attention.linear_qkv.weight",
            megatron["decoder.layers.1.self_attention.linear_qkv.weight"],
        )
    )
    assert set(attention) == {layer + f"self_attn.{key}_proj.weight" for key in ("q", "k", "v", "gate")}
    for key in ("q", "k", "v", "gate"):
        torch.testing.assert_close(
            attention[layer + f"self_attn.{key}_proj.weight"],
            hf_tensors[layer + f"self_attn.{key}_proj.weight"],
            rtol=0,
            atol=0,
        )
    ((router_name, router_tensor),) = dict(
        convert_muse_spark_1x_to_hf(
            args, "decoder.layers.1.mlp.router.weight", megatron["decoder.layers.1.mlp.router.weight"]
        )
    ).items()
    assert router_name == layer + "mlp.gate.weight"
    torch.testing.assert_close(router_tensor, megatron["decoder.layers.1.mlp.router.weight"], rtol=0, atol=0)
