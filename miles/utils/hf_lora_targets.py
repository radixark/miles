from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HfLoraTargets:
    attention: tuple[str, ...]
    mlp: tuple[str, ...]
    unembed: tuple[str, ...] = ("lm_head",)
    default_exclude: tuple[str, ...] = ()
    default_train_unembed: bool = False


def _projections(prefix, *names):
    return tuple(f"{prefix}.{name}" for name in names)


_ATTENTION = _projections("self_attn", "q_proj", "k_proj", "v_proj", "o_proj")
_MLP = _projections("mlp", "gate_proj", "up_proj", "down_proj")
_EXPERTS = _projections("mlp.experts.*", "gate_proj", "up_proj", "down_proj")
_SHARED_EXPERTS = _projections("mlp.shared_experts", "gate_proj", "up_proj", "down_proj")
_SHARED_EXPERT = _projections("mlp.shared_expert", "gate_proj", "up_proj", "down_proj")
_PACKED_EXPERTS = _projections("mlp.experts", "gate_up_proj", "down_proj")
_MLA = _projections("self_attn", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
_INDEXER = _projections("self_attn.indexer", "wq_b", "wk", "weights_proj")
_GDN_NEXT = _projections("linear_attn", "in_proj_qkvz", "in_proj_ba", "out_proj")
_GDN_35 = _projections("linear_attn", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")

# Language-layer projections; backend fusion and checkpoint packing are separate conversions.
HF_LORA_TARGETS = {
    "llama": HfLoraTargets(_ATTENTION, _MLP),
    "qwen2": HfLoraTargets(_ATTENTION, _MLP),
    "qwen3": HfLoraTargets(_ATTENTION, _MLP),
    "qwen3_moe": HfLoraTargets(_ATTENTION, _EXPERTS),
    "qwen3_next": HfLoraTargets(_ATTENTION + _GDN_NEXT, _EXPERTS + _SHARED_EXPERT),
    "qwen3_5_text": HfLoraTargets(_ATTENTION + _GDN_35, _MLP),
    "qwen3_5_moe_text": HfLoraTargets(_ATTENTION + _GDN_35, _PACKED_EXPERTS + _SHARED_EXPERT),
    "gpt_oss": HfLoraTargets(_ATTENTION, _PACKED_EXPERTS),
    "deepseek_v2": HfLoraTargets(_MLA, _MLP + _EXPERTS + _SHARED_EXPERTS),
    "deepseek_v3": HfLoraTargets(_MLA, _MLP + _EXPERTS + _SHARED_EXPERTS),
    "deepseek_v32": HfLoraTargets(
        _MLA + _INDEXER, _MLP + _EXPERTS + _SHARED_EXPERTS, default_exclude=_INDEXER
    ),
    "kimi_k2": HfLoraTargets(_MLA, _MLP + _EXPERTS + _SHARED_EXPERTS),
    "glm4_moe": HfLoraTargets(_ATTENTION, _MLP + _EXPERTS + _SHARED_EXPERTS),
    "glm_moe_dsa": HfLoraTargets(
        _MLA + _INDEXER, _MLP + _EXPERTS + _SHARED_EXPERTS, default_exclude=_INDEXER
    ),
    # Inkling uses its HF adapter export schema, which differs from its base checkpoint packing.
    "inkling_model": HfLoraTargets(
        _projections("attn", "wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"),
        _projections("mlp", "gate_up_proj", "down_proj")
        + _projections("mlp.experts", "w1", "w3", "w2")
        + _projections("mlp.shared_experts", "w1", "w3", "w2"),
        default_train_unembed=True,
    ),
}

# Wrapper model_type -> text layout, language-layer prefix, output-head path.
_HF_LANGUAGE_MODELS = {
    "qwen3_5": ("qwen3_5_text", "model.language_model.layers.*", "lm_head"),
    "qwen3_5_moe": ("qwen3_5_moe_text", "model.language_model.layers.*", "lm_head"),
    "kimi_k25": ("kimi_k2", "language_model.model.layers.*", "language_model.lm_head"),
    "inkling_mm_model": ("inkling_model", "language_model.layers.*", "language_model.lm_head"),
}


def get_hf_lora_targets(hf_config: dict) -> HfLoraTargets:
    model_type = hf_config["model_type"]
    layer_prefix, unembed = "model.layers.*", "lm_head"
    if model_type in _HF_LANGUAGE_MODELS:
        model_type, layer_prefix, unembed = _HF_LANGUAGE_MODELS[model_type]
        hf_config = hf_config["text_config"]
    elif model_type == "inkling_model":
        layer_prefix, unembed = "language_model.layers.*", "language_model.lm_head"
    assert model_type in HF_LORA_TARGETS, f"HF LoRA target layout is not defined for model_type={model_type!r}"
    layout = HF_LORA_TARGETS[model_type]
    attention, mlp = layout.attention, layout.mlp

    if "self_attn.q_a_proj" in attention and hf_config["q_lora_rank"] is None:
        attention = ("self_attn.q_proj",) + tuple(p for p in attention if p not in _MLA[:2])
    if _SHARED_EXPERTS[0] in mlp:
        # Some DeepSeek configs interleave dense layers after the initial dense block.
        num_moe_layers = sum(
            bool(hf_config["n_routed_experts"])
            and i >= hf_config["first_k_dense_replace"]
            and i % hf_config.get("moe_layer_freq", 1) == 0
            for i in range(hf_config["num_hidden_layers"])
        )
        if not num_moe_layers:
            mlp = tuple(p for p in mlp if p not in _EXPERTS + _SHARED_EXPERTS)
        elif not hf_config["n_shared_experts"]:
            mlp = tuple(p for p in mlp if p not in _SHARED_EXPERTS)
        if num_moe_layers == hf_config["num_hidden_layers"]:
            mlp = tuple(p for p in mlp if p not in _MLP)
    if model_type in ("qwen3_moe", "qwen3_next"):
        # These optional HF fields allow dense layers inside an otherwise MoE model.
        num_moe_layers = sum(
            bool(hf_config["num_experts"])
            and i not in hf_config.get("mlp_only_layers", [])
            and (i + 1) % hf_config.get("decoder_sparse_step", 1) == 0
            for i in range(hf_config["num_hidden_layers"])
        )
        if not num_moe_layers:
            mlp = ()
        if num_moe_layers < hf_config["num_hidden_layers"]:
            mlp = _MLP + mlp
    if _SHARED_EXPERT[0] in mlp and not hf_config["shared_expert_intermediate_size"]:
        mlp = tuple(p for p in mlp if p not in _SHARED_EXPERT)
    if "linear_attn.out_proj" in attention:
        layer_types = set(hf_config["layer_types"])
        if "full_attention" not in layer_types:
            attention = tuple(p for p in attention if p not in _ATTENTION)
        if "linear_attention" not in layer_types:
            attention = tuple(p for p in attention if not p.startswith("linear_attn."))
    if model_type == "inkling_model":
        if not hf_config["n_shared_experts"]:
            mlp = tuple(p for p in mlp if not p.startswith("mlp.shared_experts."))
        # Despite its name, dense_mlp_idx is the number of leading dense layers.
        if hf_config["dense_mlp_idx"] <= 0:
            mlp = tuple(p for p in mlp if p not in ("mlp.gate_up_proj", "mlp.down_proj"))
        elif hf_config["dense_mlp_idx"] >= hf_config["num_hidden_layers"]:
            mlp = tuple(p for p in mlp if not p.startswith(("mlp.experts.", "mlp.shared_experts.")))

    return replace(
        layout,
        attention=_projections(layer_prefix, *attention),
        mlp=_projections(layer_prefix, *mlp),
        unembed=(unembed,),
        default_exclude=_projections(layer_prefix, *layout.default_exclude),
    )


def resolve_hf_lora_targets(
    hf_config: dict,
    *,
    target_modules: list[str] | None = None,
    train_attn: bool | None = None,
    train_mlp: bool | None = None,
    train_unembed: bool | None = None,
) -> list[str]:
    if target_modules is not None:
        assert target_modules, "Explicit LoRA targets must not be empty"
        return list(target_modules)

    layout = get_hf_lora_targets(hf_config)
    flags = (train_attn, train_mlp, train_unembed)
    use_defaults = all(flag is None for flag in flags)
    if use_defaults:
        flags = (True, True, layout.default_train_unembed)
    else:
        assert all(flag is not None for flag in flags), "Specify all three LoRA training group flags together"
    targets = []
    for enabled, group in zip(flags, (layout.attention, layout.mlp, layout.unembed), strict=True):
        if enabled:
            targets.extend(group)
    if use_defaults:
        targets = [target for target in targets if target not in layout.default_exclude]
    assert targets, "At least one trainable LoRA module group is required"
    return targets
