"""HF ``model_type`` to native-LoRA architecture spec, and the arg-time target contract."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import matches_lora_target
from miles_plugins.lora.spec.attention import (
    GQAAttentionSpec,
    HybridGQAGDNAttentionSpec,
    InklingAttentionSpec,
    MLAAttentionSpec,
)
from miles_plugins.lora.spec.base import LoRAArchSpec
from miles_plugins.lora.spec.layout import AttentionSpecBase
from miles_plugins.lora.spec.lm_head import InklingLMHeadSpec
from miles_plugins.lora.spec.mlp import FusedGatedMLPSpec, InklingDenseMLPSpec
from miles_plugins.lora.spec.moe import InklingExpertsSpec

logger = logging.getLogger(__name__)


def _arch_spec(attention: AttentionSpecBase, *, allows_mixer_only_adapter_chunks: bool = False) -> LoRAArchSpec:
    return LoRAArchSpec(
        name=attention.name,
        model_family=attention.family,
        attention=attention,
        mlp=FusedGatedMLPSpec(),
        allows_mixer_only_adapter_chunks=allows_mixer_only_adapter_chunks,
    )


def _inkling_arch_spec() -> LoRAArchSpec:
    attention = InklingAttentionSpec()
    return LoRAArchSpec(
        name=attention.name,
        model_family=attention.family,
        attention=attention,
        mlp=InklingDenseMLPSpec(),
        experts=InklingExpertsSpec(),
        lm_head=InklingLMHeadSpec(),
        complete_layout=True,
    )


def _build_model_specs() -> dict[str, LoRAArchSpec]:
    gqa = _arch_spec(GQAAttentionSpec())
    mla = _arch_spec(MLAAttentionSpec())
    hybrid = _arch_spec(HybridGQAGDNAttentionSpec(), allows_mixer_only_adapter_chunks=True)
    inkling = _inkling_arch_spec()
    return {
        "llama": gqa,
        "qwen2": gqa,
        "qwen2_moe": gqa,
        "qwen3": gqa,
        "qwen3_moe": gqa,
        "mimo": gqa,
        "glm4": gqa,
        "glm4_moe": gqa,
        "qwen3_5": hybrid,
        "qwen3_5_moe": hybrid,
        "qwen3_6": hybrid,
        "qwen3_6_moe": hybrid,
        "qwen3_next": hybrid,
        "deepseek_v3": mla,
        "deepseek_v32": mla,
        "glm4_moe_lite": mla,
        "glm_moe_dsa": mla,
        "kimi_k2": mla,
        "kimi_k25": mla,
        "joyai_llm_flash": mla,
        "inkling_text": inkling,
        "inkling_model": inkling,
        "inkling_mm_model": inkling,
    }


MODEL_SPECS: dict[str, LoRAArchSpec] = _build_model_specs()


def resolve_config_spec(hf_config: dict) -> tuple[str, LoRAArchSpec]:
    """Resolve the spec from the outer or nested text ``model_type`` of an HF config."""
    text_config = hf_config.get("text_config") or {}
    candidates = [
        model_type for model_type in (hf_config.get("model_type"), text_config.get("model_type")) if model_type
    ]
    model_type = next((candidate for candidate in candidates if candidate in MODEL_SPECS), None)
    assert model_type is not None, (
        f"native LoRA has no spec registered for model_type {candidates}. Registered architectures: "
        f"{sorted(MODEL_SPECS)}. Verify the adapter math for this architecture and register it in "
        "miles_plugins.lora.registry.MODEL_SPECS, or use --megatron-to-hf-mode bridge."
    )
    return model_type, MODEL_SPECS[model_type]


def resolve_checkpoint_spec(hf_checkpoint: str) -> tuple[str, LoRAArchSpec]:
    with open(os.path.join(hf_checkpoint, "config.json")) as handle:
        return resolve_config_spec(json.load(handle))


def resolve_adapter_targets(hf_config: dict, hf_targets: list[str], *, hf_modules: list[str]) -> list[str] | str:
    """Validate the resolved HF targets against the native spec; return the targets SGLang serves.

    ``hf_modules`` lists the HF model's module names; when it is empty (custom HF
    implementations) the target selectors themselves are checked.
    """
    _model_type, spec = resolve_config_spec(hf_config)
    if spec.complete_layout:
        assert set(hf_targets) == set(resolve_hf_lora_targets(hf_config)), (
            f"native {spec.name} LoRA requires its complete adapter layout; omit --target-modules and "
            "--exclude-modules"
        )
        return "all-linear"

    selected = [module for module in hf_modules if any(matches_lora_target(module, t) for t in hf_targets)]
    unattachable = [module for module in selected or hf_targets if not spec.attaches(module)]
    assert not unattachable, (
        f"native LoRA ({spec.name} spec) does not implement adapters for {unattachable[:8]}"
        f"{' ...' if len(unattachable) > 8 else ''}. Select supported projections (e.g. "
        f"--target-modules {default_target_modules(hf_config)}) or use --megatron-to-hf-mode bridge."
    )
    return _expand_fused_families(hf_targets, spec.serving_fused_families())


def _expand_fused_families(targets: list[str], families: list[frozenset[str]]) -> list[str]:
    """Add every sibling of a selected fused-buffer member, which the serving export zero-fills."""
    expanded = list(targets)
    for target in targets:
        block, dot, leaf = target.rpartition(".")
        for family in families:
            if leaf in family:
                expanded.extend(f"{block}{dot}{member}" for member in sorted(family))
    return list(dict.fromkeys(expanded))


def default_target_modules(hf_config: dict) -> str:
    """Attention-only ``--target-modules`` for the checkpoint's native spec, in declaration order."""
    _model_type, spec = resolve_config_spec(hf_config)
    return spec.attention.canonical_targets_csv


@dataclass(frozen=True)
class PreflightReport:
    """Cheap, no-GPU audit of everything a native-LoRA run will need."""

    model_type: str
    spec_name: str
    mbridge_registered: bool
    model_args_script: str | None  # path when megatron_model_type was given and found

    @property
    def convertible(self) -> bool:
        return self.mbridge_registered

    def render(self) -> str:
        lines = [
            f"model_type={self.model_type} spec={self.spec_name}",
            "mbridge bridge: "
            + ("registered" if self.mbridge_registered else "MISSING (convert_hf_to_torch_dist will fail)"),
        ]
        if self.model_args_script is not None:
            lines.append(f"model-args script: {self.model_args_script}")
        return "\n".join(f"[lora-preflight] {line}" for line in lines)


def preflight_native_lora(
    hf_checkpoint: str,
    megatron_model_type: str | None = None,
    *,
    strict: bool = False,
) -> PreflightReport:
    """Audit native-LoRA support for a checkpoint before touching any GPU.

    Checks the plugin registry, the mbridge conversion bridge, and — when
    ``megatron_model_type`` is given — the ``scripts/models`` model args file
    the launch tooling will source. ``strict=True`` raises on any gap a run
    cannot survive; the default returns the report for the caller to log.
    """
    model_type, spec = resolve_checkpoint_spec(hf_checkpoint)

    try:
        import miles_plugins.mbridge  # noqa: F401  (registers Miles bridge subclasses)
        from mbridge.core.bridge import _MODEL_REGISTRY

        mbridge_registered = model_type in _MODEL_REGISTRY
    except ImportError:
        mbridge_registered = False

    model_args_script = None
    if megatron_model_type is not None:
        try:
            from miles.utils.external_utils.command_utils import repo_base_dir

            model_args_script = next(
                (
                    candidate
                    for suffix in (".py", ".sh")
                    if os.path.exists(
                        candidate := os.path.join(repo_base_dir, "scripts", "models", f"{megatron_model_type}{suffix}")
                    )
                ),
                None,
            )
        except ImportError:
            model_args_script = None
        if strict:
            assert model_args_script is not None, (
                f"[lora-preflight] scripts/models/{megatron_model_type}.py not found; raw-mode conversion "
                "and training source MODEL_ARGS from that file."
            )

    report = PreflightReport(
        model_type=model_type,
        spec_name=spec.name,
        mbridge_registered=mbridge_registered,
        model_args_script=model_args_script,
    )
    if strict:
        assert report.convertible, (
            f"[lora-preflight] no mbridge bridge is registered for model_type {model_type!r}: "
            "convert_hf_to_torch_dist cannot build the raw-mode torch_dist base. Add a bridge under "
            "miles_plugins/mbridge/ (see kimi_k25.py for the multimodal-shell pattern)."
        )
    return report
