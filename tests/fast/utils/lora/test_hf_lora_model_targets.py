import argparse
import importlib
import shlex
from fnmatch import fnmatchcase

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig

from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import (
    _HF_LORA_MODELS,
    exclude_hf_lora_targets,
    get_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)

_NATIVE_MODELS = (
    "llama",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_5",
    "qwen3_5_moe",
    "gpt_oss",
    "deepseek_v2",
    "deepseek_v3",
    "glm4_moe",
    "glm_moe_dsa",
)
_MULTIMODAL_MODELS = {"qwen3_5", "qwen3_5_moe"}


def _small_config(model_type, overrides):
    text = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    if model_type in {"qwen3_moe", "qwen3_next", "qwen3_5_moe_text", "qwen3_5_moe"}:
        text.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16)
    if model_type in {"qwen3_moe", "qwen3_next"}:
        text.update(decoder_sparse_step=2, mlp_only_layers=[])
    if model_type in {"qwen3_next", "qwen3_5_moe_text", "qwen3_5_moe"}:
        text.update(shared_expert_intermediate_size=16)
    if model_type == "qwen3_next" or model_type.startswith("qwen3_5"):
        text.update(
            layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
    if model_type == "gpt_oss":
        text.update(num_local_experts=2, num_experts_per_tok=1)
    if model_type in {"deepseek_v2", "deepseek_v3", "glm4_moe", "glm_moe_dsa"}:
        text.update(
            first_k_dense_replace=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=16,
            n_group=1,
            topk_group=1,
        )
    if model_type in {"deepseek_v2", "deepseek_v3", "glm_moe_dsa"}:
        text.update(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8)
    if model_type == "glm_moe_dsa":
        text.update(mlp_layer_types=["dense", "sparse"], index_n_heads=2, index_head_dim=8, index_topk=4)
    text.update(overrides)
    if model_type in _MULTIMODAL_MODELS:
        return AutoConfig.for_model(
            model_type,
            text_config=text,
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=4,
                out_hidden_size=32,
                num_position_embeddings=16,
                patch_size=2,
                temporal_patch_size=1,
                spatial_merge_size=1,
            ),
        )
    return AutoConfig.for_model(model_type, **text)


def test_native_model_coverage():
    # These entries use custom code, a Miles alias, or a version newer than the pinned Transformers.
    non_native = {
        "deepseek_v32",
        "kimi_k2",
        "kimi_k25",
        "kimi_k3",
        "inkling_model",
        "inkling_mm_model",
        "inkling_text",
    }
    assert set(_NATIVE_MODELS) == set(_HF_LORA_MODELS) - non_native


@pytest.mark.parametrize(
    "model_type,overrides",
    [pytest.param(name, {}, id=name) for name in _NATIVE_MODELS]
    + [
        pytest.param(name, {"q_lora_rank": None}, id=f"{name}-uncompressed-query")
        for name in ("deepseek_v2", "deepseek_v3")
    ]
    + [
        pytest.param(name, {"first_k_dense_replace": count}, id=f"{name}-{kind}")
        for name in ("deepseek_v2", "deepseek_v3", "glm4_moe")
        for kind, count in (("dense", 2), ("moe", 0))
    ]
    + [
        pytest.param(name, {"decoder_sparse_step": 1, "mlp_only_layers": layers}, id=f"{name}-{kind}")
        for name in ("qwen3_moe", "qwen3_next")
        for kind, layers in (("dense", [0, 1]), ("moe", []))
    ]
    + [
        pytest.param(name, {"layer_types": [kind, kind]}, id=f"{name}-{kind}")
        for name in ("qwen3_next", "qwen3_5_text", "qwen3_5_moe_text")
        for kind in ("full_attention", "linear_attention")
    ]
    + [
        pytest.param("glm_moe_dsa", {"mlp_layer_types": [kind, kind]}, id=f"glm-dsa-{kind}")
        for kind in ("dense", "sparse")
    ],
)
def test_targets_match_native_hf_model(model_type, overrides):
    config = _small_config(model_type, overrides)
    model_cls = AutoModelForImageTextToText if model_type in _MULTIMODAL_MODELS else AutoModelForCausalLM
    with torch.device("meta"):
        model = model_cls.from_config(config, attn_implementation="eager")

    # Keep tied output heads; packed expert projections are parameters without a .weight suffix.
    projections = {
        name.removesuffix(".weight"): param
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.ndim in (2, 3)
    }
    assert projections
    assert all(param.is_meta for param in model.parameters())
    hf_mapping = HfWeightMapping.from_config(config)
    assert hf_mapping.parameter_names == {
        name for name, param in model.named_parameters(remove_duplicate=False) if param.ndim in (2, 3)
    }
    layout = get_hf_lora_targets(config.to_dict())
    for group in (layout.attention, layout.mlp, layout.unembed):
        assert group
        for target in group:
            matches = [name for name in projections if fnmatchcase(name, target)]
            assert matches, f"{model_type}: HF model has no projection matching {target!r}"
            assert all(projections[name].numel() > 0 for name in matches)
            assert all(".indexer." not in name for name in matches)

    defaults = resolve_hf_lora_targets(config.to_dict())
    assert set(defaults) == set(layout.attention + layout.mlp)
    all_groups = resolve_hf_lora_targets(config.to_dict(), target_modules=["attn", "mlp", "unembed"])
    assert set(all_groups) == set(layout.attention + layout.mlp + layout.unembed)


@pytest.mark.parametrize("launcher", ["run_glm5_1_744b_a40b_lora", "run_glm5_2_744b_a40b_lora"])
def test_glm_launcher_ablation_selects_only_attention(monkeypatch, launcher):
    module = importlib.import_module(f"scripts.{launcher}")
    train_commands = []
    monkeypatch.setenv("KEEP_MOE_LORA", "0")
    monkeypatch.delenv("MOE_LORA_LAYERS", raising=False)
    monkeypatch.setattr(module.U, "execute_train", lambda **kwargs: train_commands.append(kwargs["train_args"]))
    module._train(module.ScriptArgs(enable_wandb=False))

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-modules")
    parser.add_argument("--exclude-modules")
    assert len(train_commands) == 1
    args, _ = parser.parse_known_args(shlex.split(train_commands[0]))
    config = _small_config("glm_moe_dsa", {})
    selected = exclude_hf_lora_targets(
        resolve_hf_lora_targets(config.to_dict(), target_modules=parse_lora_targets(args.target_modules)),
        parse_lora_targets(args.exclude_modules) or [],
    )
    assert set(selected) == set(get_hf_lora_targets(config.to_dict()).attention)
    selected_parameters = {
        name
        for name in HfWeightMapping.from_config(config).parameter_names
        if any(fnmatchcase(name.removesuffix(".weight"), target) for target in selected)
    }
    assert selected_parameters
    assert all(".self_attn." in name and ".indexer." not in name for name in selected_parameters)


def test_remote_config_with_native_class_name():
    config = _small_config("deepseek_v2", {})
    remote_config_class = type("DeepseekV2Config", (PretrainedConfig,), {"model_type": "deepseek_v2"})
    remote_fields = config.to_dict()
    remote_fields.pop("head_dim")
    remote_config = remote_config_class(**remote_fields)

    assert (
        HfWeightMapping.from_config(remote_config).parameter_names
        == HfWeightMapping.from_config(config).parameter_names
    )
