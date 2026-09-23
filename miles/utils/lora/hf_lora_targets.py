"""LoRA projection groups in the HF model namespace; backend layouts are resolved separately."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.lora.utils import matches_lora_target

logger = logging.getLogger(__name__)

LORA_TARGET_GROUPS = ("attn", "mlp", "unembed")


@dataclass(frozen=True)
class HfLoraTargets:
    attention: tuple[str, ...]
    mlp: tuple[str, ...]
    unembed: tuple[str, ...]
    default_train_unembed: bool = False


@dataclass(frozen=True)
class _HfLoraModelSpec:
    build_groups: Callable[[dict], tuple[tuple[str, ...], tuple[str, ...]]]
    layer_prefix: str = "model.layers.*"
    unembed: str = "lm_head"
    unwrap_text_config: bool = False
    default_train_unembed: bool = False


def _prefix_paths(prefix, *paths):
    return tuple(f"{prefix}.{path}" for path in paths)


_QKVO_ATTENTION = _prefix_paths("self_attn", "q_proj", "k_proj", "v_proj", "o_proj")
_DENSE_MLP = _prefix_paths("mlp", "gate_proj", "up_proj", "down_proj")
_ROUTED_EXPERTS = _prefix_paths("mlp.experts.*", "gate_proj", "up_proj", "down_proj")
_SHARED_EXPERTS = _prefix_paths("mlp.shared_experts", "gate_proj", "up_proj", "down_proj")
_QWEN_SHARED_EXPERT = _prefix_paths("mlp.shared_expert", "gate_proj", "up_proj", "down_proj")
_PACKED_EXPERTS = _prefix_paths("mlp.experts", "gate_up_proj", "down_proj")
_GDN_NEXT = _prefix_paths("linear_attn", "in_proj_qkvz", "in_proj_ba", "out_proj")
_GDN_35 = _prefix_paths("linear_attn", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


def _dense_targets(config):
    return _QKVO_ATTENTION, _DENSE_MLP


def _gpt_oss_targets(config):
    return _QKVO_ATTENTION, _PACKED_EXPERTS


def _deepseek_mlp_targets(config):
    mlp = []
    if config["first_k_dense_replace"] > 0:
        mlp.extend(_DENSE_MLP)
    if config["first_k_dense_replace"] < config["num_hidden_layers"]:
        mlp.extend(_PACKED_EXPERTS)
        if config["n_shared_experts"]:
            mlp.extend(_SHARED_EXPERTS)
    return tuple(mlp)


def _mla_attention_targets(config):
    query = ("q_proj",) if config["q_lora_rank"] is None else ("q_a_proj", "q_b_proj")
    return _prefix_paths("self_attn", *query, "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def _deepseek_targets(config):
    return _mla_attention_targets(config), _deepseek_mlp_targets(config)


def _kimi_targets(config):
    # Kimi's custom HF implementation retains per-expert modules and interleaved dense layers.
    num_moe_layers = sum(
        bool(config["n_routed_experts"])
        and layer_id >= config["first_k_dense_replace"]
        and layer_id % config.get("moe_layer_freq", 1) == 0
        for layer_id in range(config["num_hidden_layers"])
    )
    mlp = []
    if num_moe_layers < config["num_hidden_layers"]:
        mlp.extend(_DENSE_MLP)
    if num_moe_layers:
        mlp.extend(_ROUTED_EXPERTS)
        if config["n_shared_experts"]:
            mlp.extend(_SHARED_EXPERTS)
    return _mla_attention_targets(config), tuple(mlp)


def _kimi_k3_targets(config):
    attention = _prefix_paths("self_attn", "o_proj", "q_a_proj", "kv_a_proj_with_mqa")
    num_moe_layers = sum(
        config["num_experts"] is not None
        and layer >= config["first_k_dense_replace"]
        and layer % config["moe_layer_freq"] == 0
        for layer in range(config["num_hidden_layers"])
    )
    mlp = list(_DENSE_MLP) if num_moe_layers < config["num_hidden_layers"] else []
    if num_moe_layers:
        mlp.extend(_prefix_paths("block_sparse_moe.experts.*", "w1", "w2", "w3"))
        if config["num_shared_experts"] is not None:
            mlp.extend(_prefix_paths("block_sparse_moe.shared_experts", "gate_proj", "up_proj", "down_proj"))
    return attention, tuple(mlp)


def _glm4_moe_targets(config):
    return _QKVO_ATTENTION, _deepseek_mlp_targets(config)


def _glm_dsa_targets(config):
    mlp = []
    if "dense" in config["mlp_layer_types"]:
        mlp.extend(_DENSE_MLP)
    if "sparse" in config["mlp_layer_types"]:
        mlp.extend(_PACKED_EXPERTS)
        if config["n_shared_experts"]:
            mlp.extend(_SHARED_EXPERTS)
    return _mla_attention_targets(config), tuple(mlp)


def _qwen_moe_mlp_targets(config, *, shared_expert=False):
    # Qwen3MoE serializes its num_experts alias as num_local_experts.
    num_experts = config["num_experts"] if "num_experts" in config else config["num_local_experts"]
    # These optional HF fields allow dense layers inside an otherwise MoE model.
    num_moe_layers = sum(
        bool(num_experts)
        and layer_id not in config.get("mlp_only_layers", [])
        and (layer_id + 1) % config.get("decoder_sparse_step", 1) == 0
        for layer_id in range(config["num_hidden_layers"])
    )
    mlp = []
    if num_moe_layers < config["num_hidden_layers"]:
        mlp.extend(_DENSE_MLP)
    if num_moe_layers:
        mlp.extend(_PACKED_EXPERTS)
        if shared_expert and config["shared_expert_intermediate_size"]:
            mlp.extend(_QWEN_SHARED_EXPERT)
    return tuple(mlp)


def _hybrid_attention_targets(config, linear_attention):
    attention = []
    if "full_attention" in config["layer_types"]:
        attention.extend(_QKVO_ATTENTION)
    if "linear_attention" in config["layer_types"]:
        attention.extend(linear_attention)
    return tuple(attention)


def _qwen3_moe_targets(config):
    return _QKVO_ATTENTION, _qwen_moe_mlp_targets(config)


def _qwen3_next_targets(config):
    return _hybrid_attention_targets(config, _GDN_NEXT), _qwen_moe_mlp_targets(config, shared_expert=True)


def _qwen3_5_targets(config):
    return _hybrid_attention_targets(config, _GDN_35), _DENSE_MLP


def _qwen3_5_moe_targets(config):
    mlp = _PACKED_EXPERTS
    if config["shared_expert_intermediate_size"]:
        mlp += _QWEN_SHARED_EXPERT
    return _hybrid_attention_targets(config, _GDN_35), mlp


def _inkling_targets(config):
    attention = _prefix_paths("self_attn", "q_proj", "k_proj", "v_proj", "r_proj", "o_proj")
    # Older Inkling configs encode leading dense layers instead of mlp_layer_types.
    layer_types = config.get("mlp_layer_types")
    if layer_types is None:
        layer_types = [
            "dense" if layer < config["dense_mlp_idx"] else "sparse" for layer in range(config["num_hidden_layers"])
        ]
    mlp = []
    if "dense" in layer_types:
        mlp.extend(_DENSE_MLP)
    if "sparse" in layer_types:
        mlp.extend(_PACKED_EXPERTS)
        if config["n_shared_experts"]:
            mlp.extend(_SHARED_EXPERTS)
    return attention, tuple(mlp)


_HF_LORA_MODELS = {
    "llama": _HfLoraModelSpec(_dense_targets),
    "qwen2": _HfLoraModelSpec(_dense_targets),
    "qwen3": _HfLoraModelSpec(_dense_targets),
    "qwen3_moe": _HfLoraModelSpec(_qwen3_moe_targets),
    "qwen3_next": _HfLoraModelSpec(_qwen3_next_targets),
    "qwen3_5_text": _HfLoraModelSpec(_qwen3_5_targets),
    "qwen3_5_moe_text": _HfLoraModelSpec(_qwen3_5_moe_targets),
    "qwen3_5": _HfLoraModelSpec(
        _qwen3_5_targets, layer_prefix="model.language_model.layers.*", unwrap_text_config=True
    ),
    "qwen3_5_moe": _HfLoraModelSpec(
        _qwen3_5_moe_targets, layer_prefix="model.language_model.layers.*", unwrap_text_config=True
    ),
    "gpt_oss": _HfLoraModelSpec(_gpt_oss_targets),
    "deepseek_v2": _HfLoraModelSpec(_deepseek_targets),
    "deepseek_v3": _HfLoraModelSpec(_deepseek_targets),
    "deepseek_v32": _HfLoraModelSpec(_deepseek_targets),
    "kimi_k2": _HfLoraModelSpec(_kimi_targets),
    "kimi_k25": _HfLoraModelSpec(
        _kimi_targets,
        layer_prefix="language_model.model.layers.*",
        unembed="language_model.lm_head",
        unwrap_text_config=True,
    ),
    "kimi_k3": _HfLoraModelSpec(
        _kimi_k3_targets,
        layer_prefix="language_model.model.layers.*",
        unembed="language_model.lm_head",
        unwrap_text_config=True,
    ),
    "glm4_moe": _HfLoraModelSpec(_glm4_moe_targets),
    "glm_moe_dsa": _HfLoraModelSpec(_glm_dsa_targets),
    "inkling_text": _HfLoraModelSpec(_inkling_targets, default_train_unembed=True),
    "inkling_model": _HfLoraModelSpec(_inkling_targets, default_train_unembed=True),
    "inkling_mm_model": _HfLoraModelSpec(
        _inkling_targets,
        layer_prefix="model.language_model.layers.*",
        unwrap_text_config=True,
        default_train_unembed=True,
    ),
}


def get_hf_lora_targets(hf_config: dict) -> HfLoraTargets:
    model_type = hf_config["model_type"]
    assert model_type in _HF_LORA_MODELS, f"HF LoRA target layout is not defined for model_type={model_type!r}"
    spec = _HF_LORA_MODELS[model_type]
    text_config = hf_config["text_config"] if spec.unwrap_text_config else hf_config
    attention, mlp = spec.build_groups(text_config)
    return HfLoraTargets(
        attention=_prefix_paths(spec.layer_prefix, *attention),
        mlp=_prefix_paths(spec.layer_prefix, *mlp),
        unembed=(spec.unembed,),
        default_train_unembed=spec.default_train_unembed,
    )


def resolve_hf_lora_targets(hf_config: dict, *, target_modules: list[str] | None = None) -> list[str]:
    if target_modules is None or target_modules == ["all-linear"]:
        layout = get_hf_lora_targets(hf_config)
        target_modules = ["attn", "mlp", *(["unembed"] if layout.default_train_unembed else [])]
    assert target_modules and "all-linear" not in target_modules, "Use all-linear alone or provide explicit targets"
    if not any(target in LORA_TARGET_GROUPS for target in target_modules):
        return list(target_modules)

    layout = get_hf_lora_targets(hf_config)
    groups = dict(zip(LORA_TARGET_GROUPS, (layout.attention, layout.mlp, layout.unembed), strict=True))
    targets = [module for target in target_modules for module in groups.get(target, (target,))]
    assert targets, "At least one trainable LoRA module group is required"
    return list(dict.fromkeys(targets))


def expand_packed_hf_lora_targets(targets: list[str], modules: list[str]) -> list[str]:
    added, replaced = [], set()
    for gate in targets:
        if gate.rsplit(".", 1)[-1] != "gate_proj":
            continue
        prefix = gate.removesuffix("gate_proj")
        up, packed = prefix + "up_proj", prefix + "gate_up_proj"
        if up not in targets or not any(matches_lora_target(module, packed) for module in modules):
            continue
        if not any(matches_lora_target(packed, target) for target in targets):
            added.append(packed)
        replaced.update(
            target for target in (gate, up) if not any(matches_lora_target(module, target) for module in modules)
        )
    if added:
        logger.warning("Expanding paired gate_proj/up_proj LoRA targets to packed HF targets: %s", added)
    return list(dict.fromkeys([target for target in targets if target not in replaced] + added))


def parse_lora_targets(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    targets = value.split(",") if isinstance(value, str) else value
    targets = [target.strip() for target in targets]
    assert targets and all(targets), "LoRA target lists must not contain empty entries"
    return list(dict.fromkeys(targets))


def exclude_hf_lora_targets(targets: list[str], exclusions: list[str]) -> list[str]:
    selected = [
        target for target in targets if not any(matches_lora_target(target, pattern) for pattern in exclusions)
    ]
    assert selected, "LoRA target selection is empty after --exclude-modules"
    return selected


def expand_hf_lora_targets(targets: list[str], layout: HfLoraTargets) -> list[str]:
    available = layout.attention + layout.mlp + layout.unembed
    for target in targets:
        assert any(
            matches_lora_target(module, target) for module in available
        ), f"LoRA target {target!r} is not an HF target of this model"
    return [module for module in available if any(matches_lora_target(module, target) for target in targets)]
