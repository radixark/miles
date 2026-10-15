from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=20, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import pytest
import torch
from torchtitan.models.common.linear import Linear
from torchtitan.models.qwen3 import model_registry

from miles.backends.torchtitan_utils.lora import convert_model_config
from miles.backends.torchtitan_utils.lora_names import iter_hf_adapter_tensors
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

_QWEN3_ALL_LINEAR = resolve_hf_lora_targets({"model_type": "qwen3"})
_QWEN3_MOE_HF_CONFIG = {"model_type": "qwen3_moe", "num_experts": 128, "num_hidden_layers": 48}


def _qwen3_spec(flavor: str = "0.6B"):
    return model_registry(flavor, attn_backend="flex")


def _lora_ranks(model_config) -> dict[str, int]:
    return {fqn: getattr(cfg, "rank", 0) for fqn, cfg, _, _ in model_config.traverse(Linear.Config)}


def test_all_linear_adapts_every_attention_and_mlp_projection_and_nothing_else():
    spec = _qwen3_spec()
    converted = convert_model_config(spec, hf_targets=_QWEN3_ALL_LINEAR, rank=8, alpha=16)

    ranks = _lora_ranks(converted)
    adapted = {fqn for fqn, rank in ranks.items() if rank}
    layer0 = {fqn.removeprefix("layers.0.") for fqn in adapted if fqn.startswith("layers.0.")}
    # wkv is the one config torchtitan builds both wk and wv from
    assert layer0 == {
        "attention.qkv_linear.wq",
        "attention.qkv_linear.wkv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    }
    assert len(adapted) == 6 * len(converted.layers)
    assert ranks["lm_head"] == 0


def test_a_target_set_the_converter_cannot_express_is_refused_before_the_build():
    spec = _qwen3_spec()
    # k_proj and v_proj come from one shared config, so adapting one adapts both
    with pytest.raises(ValueError, match="also adapt model.layers.0.self_attn.v_proj"):
        convert_model_config(spec, hf_targets=["model.layers.*.self_attn.k_proj"], rank=8, alpha=16)
    # the converter matches by name in every layer, never one layer
    with pytest.raises(ValueError, match="also adapt model.layers.1.self_attn.q_proj"):
        convert_model_config(spec, hf_targets=["model.layers.0.self_attn.q_proj"], rank=8, alpha=16)


def test_moe_experts_are_refused_but_moe_attention_is_adapted():
    spec = _qwen3_spec("30B-A3B")
    with pytest.raises(ValueError, match="name no Linear projection"):
        convert_model_config(spec, hf_targets=resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG), rank=8, alpha=16)

    attention = resolve_hf_lora_targets(_QWEN3_MOE_HF_CONFIG, target_modules=["attn"])
    converted = convert_model_config(spec, hf_targets=attention, rank=8, alpha=16)
    adapted = {fqn for fqn, rank in _lora_ranks(converted).items() if rank}
    assert all(".attention." in fqn for fqn in adapted)
    assert len(adapted) == 3 * len(converted.layers)


def test_adapters_export_under_hf_peft_names_and_base_weights_are_skipped():
    spec = _qwen3_spec()
    sd_adapter = spec.state_dict_adapter(spec.model, None)
    a, b, base = torch.zeros(8, 16), torch.zeros(32, 8), torch.zeros(32, 16)
    state = {
        "layers.3.attention.qkv_linear.wv.lora_a.weight": a,
        "layers.3.attention.qkv_linear.wv.lora_b.weight": b,
        "layers.3.attention.qkv_linear.wv.weight": base,
        "layers.3.feed_forward.w2.lora_a.weight": a,
    }
    assert list(iter_hf_adapter_tensors(sd_adapter, state)) == [
        ("model.layers.3.self_attn.v_proj.lora_A.weight", a),
        ("model.layers.3.self_attn.v_proj.lora_B.weight", b),
        ("model.layers.3.mlp.down_proj.lora_A.weight", a),
    ]
