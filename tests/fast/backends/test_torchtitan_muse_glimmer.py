import json
from argparse import Namespace

import pytest
import torch

pytest.importorskip("torchtitan")
pytest.importorskip("transformers.models.muse_glimmer")

# meta-models/Muse-Glimmer-30B's text_config, cut to the last four layers of its cyclic pattern
_HF_TEXT_CONFIG = dict(
    model_type="muse_glimmer_text",
    hidden_size=6656,
    intermediate_size=19968,
    num_hidden_layers=4,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
    vocab_size=202048,
    max_position_embeddings=131072,
    sliding_window=2048,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
    rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    final_logit_softcapping=20.0,
    output_multiplier=0.19611613513818404,
    qk_scale_factor=3.87,
    tie_word_embeddings=False,
)

_LAYER_KEYS = [
    *(f"self_attn.{p}.weight" for p in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]),
    *(f"mlp.{p}.weight" for p in ["gate_proj", "up_proj", "down_proj"]),
    *(
        f"{p}.weight"
        for p in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
    ),
]


def _checkpoint_keys() -> set[str]:
    keys = {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    for i in range(_HF_TEXT_CONFIG["num_hidden_layers"]):
        keys |= {f"model.language_model.layers.{i}.{s}" for s in _LAYER_KEYS}
    return keys


def _write_checkpoint_config(tmp_path, **text_overrides) -> str:
    config = {"model_type": "muse_glimmer", "text_config": {**_HF_TEXT_CONFIG, **text_overrides}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _model_config(tmp_path, **text_overrides):
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.config import build_model_config

    checkpoint = _write_checkpoint_config(tmp_path, **text_overrides)
    return build_model_config(
        Namespace(backend=Namespace(titan_model_name="muse_glimmer", titan_model_flavor="30B", seq_length=4096)),
        parallelism=ParallelismConfig(),
        hf_assets_path=checkpoint,
    )


def test_the_model_is_sized_from_the_checkpoint_config(tmp_path):
    from torchtitan.models.common.attention import SlidingWindowFlexInnerAttention

    from miles.backends.torchtitan_utils.models.muse_glimmer import MilesMuseGlimmerModel

    model = _model_config(tmp_path)
    assert type(model) is MilesMuseGlimmerModel.Config
    assert (model.dim, model.vocab_size, len(model.layers), model.max_context_length) == (6656, 202048, 4, 4096)
    assert model.vision_encoder is None
    windows = [
        (
            layer.attention.inner_attention.window_size
            if isinstance(layer.attention.inner_attention, SlidingWindowFlexInnerAttention.Config)
            else None
        )
        for layer in model.layers
    ]
    assert windows == [2048, 2048, 2048, None]
    assert [layer.attention.rope is not None for layer in model.layers] == [True, True, True, False]
    assert model.layers[0].feed_forward.w2.in_features == 19968
    assert (model.lm_head.output_multiplier, model.lm_head.output_soft_cap_temp) == (pytest.approx(0.1961161), 20.0)


def test_a_checkpoint_off_torchtitans_layer_pattern_is_refused(tmp_path):
    """torchtitan repeats the last four layers' pattern backward through the stack."""
    with pytest.raises(ValueError, match="cyclic window pattern"):
        _model_config(
            tmp_path,
            num_hidden_layers=8,
            layer_types=["full_attention"] + ["sliding_attention"] * 7,
            layer_rope_theta=[500000.0, 500000.0, 500000.0, 0] * 2,
        )
    with pytest.raises(ValueError, match="NoPE phase"):
        _model_config(tmp_path, layer_rope_theta=[0, 500000.0, 500000.0, 500000.0])
    with pytest.raises(ValueError, match="rope theta 10000.0"):
        _model_config(tmp_path, layer_rope_theta=[10000.0, 500000.0, 500000.0, 0])
    with pytest.raises(ValueError, match="unscaled RoPE"):
        _model_config(tmp_path, rope_parameters={"rope_theta": 500000.0, "rope_type": "yarn", "factor": 4.0})


def test_a_checkpoint_with_other_constants_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rms_norm_eps"):
        _model_config(tmp_path, rms_norm_eps=1e-6)
    with pytest.raises(ValueError, match="intermediate_size"):
        _model_config(tmp_path, intermediate_size=16384)


def test_the_adapter_names_every_decoder_tensor_of_the_checkpoint(tmp_path):
    model = _model_config(tmp_path)
    adapter = type(model)._owner.state_dict_adapter_cls(model, None)
    named = {key.format(i) for key in adapter.from_hf_map for i in range(_HF_TEXT_CONFIG["num_hidden_layers"])}
    assert _checkpoint_keys() <= named


def test_a_lora_factor_exports_through_the_fused_qkv_split_and_the_rope_permute(tmp_path):
    """torchtitan's adapter sizes the k permute from the model; a rank-r lora_B has to go through
    the same split and row order as its weight."""
    model = _model_config(tmp_path)
    adapter = type(model)._owner.state_dict_adapter_cls(model, None)
    rows = (32 + 2 * 2) * 128
    rank = 8
    lora_b = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, rank).contiguous()
    weight = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, 6656).contiguous()

    exported = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": lora_b})
    reference = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": weight})

    prefix = "model.language_model.layers.0.self_attn."
    assert {k: tuple(v.shape) for k, v in exported.items()} == {
        f"{prefix}q_proj.weight": (4096, rank),
        f"{prefix}k_proj.weight": (256, rank),
        f"{prefix}v_proj.weight": (256, rank),
    }
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor[:, 0], reference[name][:, 0])


def test_all_linear_targets_the_decoder_projections_under_the_language_model(tmp_path):
    from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

    _write_checkpoint_config(tmp_path)
    hf_config = json.loads((tmp_path / "config.json").read_text())
    targets = resolve_hf_lora_targets(hf_config, target_modules=["all-linear"])
    prefix = "model.language_model.layers.*."
    assert targets == [f"{prefix}self_attn.{p}" for p in ["q_proj", "k_proj", "v_proj", "o_proj"]] + [
        f"{prefix}mlp.{p}" for p in ["gate_proj", "up_proj", "down_proj"]
    ]


def test_a_bare_gate_proj_target_cannot_reach_the_attention_output_gate(tmp_path):
    """The layout leaves self_attn.gate_proj out because sglang's LoRA cannot serve it; a bare selector that
    matches it by leaf name is refused rather than trained into an adapter the engine then rejects."""
    from miles.backends.torchtitan_utils.lora import convert_model_config
    from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, resolve_hf_lora_targets

    hf_config = {"model_type": "muse_glimmer", "text_config": _HF_TEXT_CONFIG}
    layout = resolve_hf_lora_targets(hf_config, target_modules=list(LORA_TARGET_GROUPS))
    with pytest.raises(ValueError, match="self_attn.gate_proj"):
        convert_model_config(
            _model_config(tmp_path), hf_targets=["gate_proj", "up_proj"], layout=layout, rank=8, alpha=16
        )
    attention = resolve_hf_lora_targets(hf_config, target_modules=["attn"])
    convert_model_config(_model_config(tmp_path), hf_targets=attention, layout=layout, rank=8, alpha=16)


def test_the_sequence_may_not_exceed_the_checkpoints_rotary_table(tmp_path):
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _model_config(tmp_path, max_position_embeddings=2048)
