import json
from argparse import Namespace

import pytest

pytest.importorskip("torchtitan")

# RadixArk/Muse-Spark-Open-last4-partial's text_config: layers 58-61 of meta-models/Muse-Spark-Open-816B-A42B
_HF_TEXT_CONFIG = dict(
    model_type="muse_spark_text",
    hidden_size=8192,
    intermediate_size=16384,
    num_hidden_layers=4,
    num_attention_heads=128,
    num_key_value_heads=16,
    head_dim=64,
    vocab_size=202048,
    max_position_embeddings=1048576,
    sliding_window=2048,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
    mlp_layer_types=["sparse", "sparse", "sparse", "sparse"],
    num_local_experts=256,
    num_experts_per_tok=8,
    num_shared_experts=0,
    moe_hidden_size=4096,
    moe_intermediate_size=4096,
    routed_scaling_factor=1.0,
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    final_logit_softcapping=20.0,
    output_multiplier=0.17677669529663687,
    qk_scale_factor=5.4730064863875,
    residual_gate_temperature=0.3,
    tie_word_embeddings=False,
)

_LAYER_KEYS = [
    *(f"self_attn.{p}.weight" for p in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]),
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.pre_expert_proj.weight",
    "mlp.post_expert_proj.weight",
    "mlp.pre_expert_norm.weight",
    "mlp.experts.post_expert_norm.weight",
    "post_attention_residual_gate.gate",
    "post_feedforward_residual_gate.gate",
    *(f"{p}.weight" for p in ["input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]),
]


def _write_checkpoint_config(tmp_path, **text_overrides) -> str:
    config = {"model_type": "muse_spark", "text_config": {**_HF_TEXT_CONFIG, **text_overrides}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _model_config(tmp_path, **text_overrides):
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.config import build_model_config

    checkpoint = _write_checkpoint_config(tmp_path, **text_overrides)
    return build_model_config(
        Namespace(backend=Namespace(titan_model_name="muse_spark", titan_model_flavor="816B-A42B", seq_length=4096)),
        parallelism=ParallelismConfig(),
        hf_assets_path=checkpoint,
    )


def test_the_model_is_sized_from_the_checkpoint_config(tmp_path):
    from torchtitan.models.common.moe import MoE

    model = _model_config(tmp_path)
    assert (model.dim, model.vocab_size, len(model.layers), model.max_context_length) == (8192, 202048, 4, 4096)
    assert model.dataset_modalities == ()
    assert model.vision_encoder is not None and model.audio_encoder is not None
    assert [layer.attention.rope is not None for layer in model.layers] == [True, True, True, False]
    assert [layer.attention.inner_attention.window_size for layer in model.layers[:3]] == [2048] * 3
    moe = model.layers[0].feed_forward
    assert isinstance(moe, MoE.Config)
    assert (moe.num_experts, moe.router.top_k) == (256, 8)
    # miles keeps the rollout engine's routing: no aux loss, frozen expert bias
    assert (moe.router.aux_loss, moe.freeze_expert_bias) == (None, True)
    assert model.lm_head.output_soft_cap_temp == 20.0


def test_global_layers_follow_the_checkpoint_schedule_not_torchtitans_count(tmp_path):
    """torchtitan's released flavor puts global attention at layers 1, 5, 9, ...; a slice keeps the
    original schedule, so the layer types decide."""
    model = _model_config(
        tmp_path,
        layer_types=["full_attention"] + ["sliding_attention"] * 3,
        layer_rope_theta=[0, 500000.0, 500000.0, 500000.0],
    )
    assert [layer.attention.rope is None for layer in model.layers] == [True, False, False, False]


def test_a_global_layer_with_rope_or_other_constants_is_refused(tmp_path):
    with pytest.raises(ValueError, match="global layers have no RoPE"):
        _model_config(tmp_path, layer_rope_theta=[500000.0] * 4)
    with pytest.raises(ValueError, match="qk_scale_factor"):
        _model_config(tmp_path, qk_scale_factor=3.87)
    with pytest.raises(ValueError, match="dense layers"):
        _model_config(tmp_path, mlp_layer_types=["dense"] + ["sparse"] * 3)


def test_the_adapter_names_every_decoder_tensor_of_the_checkpoint(tmp_path):
    model = _model_config(tmp_path)
    adapter = type(model)._owner.state_dict_adapter_cls(model, None)
    named = set(adapter.from_hf_map)
    expected = {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    for i in range(4):
        expected |= {f"model.language_model.layers.{i}.{s}" for s in _LAYER_KEYS}
    assert expected <= named


def test_all_linear_targets_attention_only(tmp_path):
    """Routed experts are grouped weights the torchtitan backend does not adapt."""
    from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

    _write_checkpoint_config(tmp_path)
    hf_config = json.loads((tmp_path / "config.json").read_text())
    prefix = "model.language_model.layers.*.self_attn."
    assert resolve_hf_lora_targets(hf_config, target_modules=["all-linear"]) == [
        f"{prefix}{p}" for p in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]


def test_attention_lora_adapts_the_decoder_projections_only(tmp_path):
    """The HF-name probe walks every Linear, towers included; torchtitan builds the vision ``conv1`` config as
    the module ``conv1_linear``, a key the adapter rejects, and that must leave the projection untargeted
    rather than fail the run."""
    from torchtitan.models.common.linear import Linear

    from miles.backends.torchtitan_utils.lora import convert_model_config
    from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, resolve_hf_lora_targets

    hf_config = {"model_type": "muse_spark", "text_config": _HF_TEXT_CONFIG}
    targets = resolve_hf_lora_targets(hf_config, target_modules=["attn"])
    layout = resolve_hf_lora_targets(hf_config, target_modules=list(LORA_TARGET_GROUPS))
    config = convert_model_config(_model_config(tmp_path), hf_targets=targets, layout=layout, rank=8, alpha=16)
    adapted = {fqn for fqn, cfg, _, _ in config.traverse(Linear.Config) if hasattr(cfg, "rank")}
    assert adapted == {f"layers.{i}.attention.{p}" for i in range(4) for p in ("qkv_linear.wqkv", "wo")}


def test_the_hf_config_loads_without_a_transformers_class(tmp_path):
    from miles.utils.hf_utils.config import hf_text_config, load_hf_config

    checkpoint = _write_checkpoint_config(tmp_path)
    text = hf_text_config(load_hf_config(checkpoint))
    assert (text.model_type, text.num_hidden_layers, text.moe_hidden_size) == ("muse_spark_text", 4, 4096)


def test_the_adapter_converts_a_streamed_part_and_a_rank_wide_factor(tmp_path):
    """miles exports the dense weights, each grouped expert tensor and the LoRA factors as separate parts;
    torchtitan's adapter wants a layer complete and a factor the weight's full shape."""
    import torch

    model = _model_config(tmp_path)
    adapter = type(model)._owner.state_dict_adapter_cls(model, None)
    attention = model.layers[0].attention
    rows = (attention.n_heads + 2 * attention.n_kv_heads) * attention.head_dim
    rank = 4
    factor = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, rank).contiguous()
    weight = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, model.dim).contiguous()

    exported = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": factor})
    reference = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": weight})
    prefix = "model.language_model.layers.0.self_attn."
    assert {name: tuple(tensor.shape) for name, tensor in exported.items()} == {
        f"{prefix}q_proj.weight": (attention.n_heads * attention.head_dim, rank),
        f"{prefix}k_proj.weight": (attention.n_kv_heads * attention.head_dim, rank),
        f"{prefix}v_proj.weight": (attention.n_kv_heads * attention.head_dim, rank),
    }
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor[:, 0], reference[name][:, 0])

    experts = model.layers[0].feed_forward.routed_experts
    w13 = torch.empty(
        (experts.w13.group_size, 2, experts.w13.out_features // 2, experts.w13.in_features), device="meta"
    )
    assert set(adapter.to_hf({"layers.0.feed_forward.routed_experts.w13.weight": w13})) == {
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    }
    with pytest.raises(ValueError, match="unexpected native keys"):
        adapter.to_hf({"layers.0.nope.weight": weight})
