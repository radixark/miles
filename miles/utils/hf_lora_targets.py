from dataclasses import dataclass


@dataclass(frozen=True)
class HfLoraTargets:
    attention: tuple[str, ...]
    mlp: tuple[str, ...]
    unembed: tuple[str, ...] = ("lm_head",)


_DENSE_TARGETS = HfLoraTargets(
    attention=tuple(f"model.layers.*.self_attn.{name}_proj" for name in ("q", "k", "v", "o")),
    mlp=tuple(f"model.layers.*.mlp.{name}_proj" for name in ("gate", "up", "down")),
)

# HF checkpoint module paths, independent of runtime fusion, recipe defaults, and backend support.
HF_LORA_TARGETS = {
    "llama": _DENSE_TARGETS,
    "qwen2": _DENSE_TARGETS,
    "qwen3": _DENSE_TARGETS,
    "qwen3_moe": HfLoraTargets(
        attention=_DENSE_TARGETS.attention,
        mlp=tuple(f"model.layers.*.mlp.experts.*.{name}_proj" for name in ("gate", "up", "down")),
    ),
}


def resolve_hf_lora_targets(model_type: str, *, train_attn: bool, train_mlp: bool, train_unembed: bool) -> list[str]:
    assert model_type in HF_LORA_TARGETS, f"HF LoRA target layout is not defined for model_type={model_type!r}"
    layout = HF_LORA_TARGETS[model_type]
    targets = []
    for enabled, group in ((train_attn, layout.attention), (train_mlp, layout.mlp), (train_unembed, layout.unembed)):
        if enabled:
            targets.extend(group)
    assert targets, "At least one trainable LoRA module group is required"
    return targets
