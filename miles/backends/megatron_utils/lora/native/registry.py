"""HF ``model_type`` to native-LoRA architecture spec, and the arg-time target contract."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from miles.backends.megatron_utils.lora.native.naming import DeclaredNaming
from miles.backends.megatron_utils.lora.native.spec.attention import (
    GQAAttentionSpec,
    HybridGQAGDNAttentionSpec,
    InklingAttentionSpec,
    KimiK3AttentionSpec,
    MLAAttentionSpec,
)
from miles.backends.megatron_utils.lora.native.spec.base import FixedTargets, LoRAArchSpec
from miles.backends.megatron_utils.lora.native.spec.layout import AttentionSpecBase
from miles.backends.megatron_utils.lora.native.spec.lm_head import InklingLMHeadSpec
from miles.backends.megatron_utils.lora.native.spec.mlp import FusedGateUpMLPSpec, SplitGateUpMLPSpec
from miles.backends.megatron_utils.lora.native.spec.moe import GroupedExpertsSpec
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import lora_rollout_enabled, matches_lora_target

logger = logging.getLogger(__name__)

_SHARED_EXPERT_BLOCK = re.compile(r"(?:^|\.)shared_experts?$")


def _arch_spec(attention: AttentionSpecBase, *, mlp=None, has_linear_attention_layers: bool = False) -> LoRAArchSpec:
    return LoRAArchSpec(
        name=attention.name,
        attention=attention,
        mlp=mlp or SplitGateUpMLPSpec(),
        has_linear_attention_layers=has_linear_attention_layers,
    )


def _inkling_arch_spec() -> LoRAArchSpec:
    attention = InklingAttentionSpec()
    return LoRAArchSpec(
        name=attention.name,
        attention=attention,
        mlp=FusedGateUpMLPSpec(),
        experts=GroupedExpertsSpec(),
        lm_head=InklingLMHeadSpec(),
        # SGLang serves Inkling under its own module names, which its converter does not produce
        declared_naming=DeclaredNaming(
            "model.llm.layers.",
            {
                "self_attention.": "attn.",
                "mlp.": "mlp.",
                "mlp.experts.": "mlp.experts.",
                "mlp.shared_experts.": "mlp.shared_experts.",
            },
        ),
        # TML export names are not HF targets; SGLang auto-detects them.
        fixed_targets=FixedTargets(serving=lambda _targets: "all-linear", select_all=True),
    )


def _kimi_k3_arch_spec() -> LoRAArchSpec:
    attention = KimiK3AttentionSpec()
    return LoRAArchSpec(
        name=attention.name,
        attention=attention,
        mlp=SplitGateUpMLPSpec(),
        experts=GroupedExpertsSpec(),
        # the routed-expert down-proj may be omitted: its EP-shared w2_lora_B dominates adapter growth (#1559)
        fixed_targets=FixedTargets(
            serving=_shared_outer_serving_targets, optional=frozenset({"block_sparse_moe.experts.*.w2"})
        ),
    )


def _shared_outer_serving_targets(targets: list[str]) -> list[str]:
    """Shared-outer export stores the expert dimension in each tensor, not in its name."""
    return [target.replace(".experts.*.", ".experts.") for target in targets]


def _build_model_specs() -> dict[str, LoRAArchSpec]:
    gqa = _arch_spec(GQAAttentionSpec())
    # HF GLM-4 keeps gate/up fused in one gate_up_proj module
    glm4 = _arch_spec(GQAAttentionSpec(), mlp=FusedGateUpMLPSpec())
    mla = _arch_spec(MLAAttentionSpec())
    hybrid = _arch_spec(HybridGQAGDNAttentionSpec(), has_linear_attention_layers=True)
    inkling = _inkling_arch_spec()
    return {
        "llama": gqa,
        "qwen2": gqa,
        "qwen3": gqa,
        "qwen3_moe": gqa,
        "mimo": gqa,
        "glm4": glm4,
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
        "kimi_k3": _kimi_k3_arch_spec(),
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
        "miles.backends.megatron_utils.lora.native.registry.MODEL_SPECS, or use --megatron-to-hf-mode bridge."
    )
    return model_type, MODEL_SPECS[model_type]


def resolve_checkpoint_spec(hf_checkpoint: str) -> tuple[str, LoRAArchSpec]:
    with open(os.path.join(hf_checkpoint, "config.json")) as handle:
        return resolve_config_spec(json.load(handle))


def resolve_adapter_targets(
    hf_config: dict,
    hf_targets: list[str],
    *,
    hf_modules: list[str],
    lora_type: str,
    experts_shared_outer_loras: bool,
) -> list[str] | str:
    """Validate the resolved HF targets against the native spec; return the targets SGLang serves.

    ``hf_modules`` lists the HF model's module names; when it is empty (custom HF
    implementations) the target selectors themselves are checked.
    """
    assert lora_type == "lora", "native LoRA does not implement --lora-type canonical_lora; use bridge mode"
    _model_type, spec = resolve_config_spec(hf_config)
    assert (
        spec.experts is None or experts_shared_outer_loras
    ), f"native {spec.name} LoRA trains shared-outer expert adapters; pass --experts-shared-outer-loras"
    if spec.fixed_targets is not None:
        _assert_fixed_target_layout(spec, hf_config, hf_targets)
        return spec.fixed_targets.serving(hf_targets)

    selected = [module for module in hf_modules if any(matches_lora_target(module, t) for t in hf_targets)]
    unattachable = [module for module in selected or hf_targets if not spec.attaches(module)]
    assert not unattachable, (
        f"native LoRA ({spec.name} spec) does not implement adapters for {unattachable[:8]}"
        f"{' ...' if len(unattachable) > 8 else ''}. Select supported projections (e.g. "
        f"--target-modules {attention_target_modules_csv(hf_config)}) or use --megatron-to-hf-mode bridge."
    )
    _assert_shared_expert_up_proj_has_gate_proj(selected or hf_targets)
    return hf_targets


def validate_native_lora_args(args) -> None:
    """Reject run configurations native LoRA has not been validated with; needs Megatron's resolved args."""
    assert args.colocate or not lora_rollout_enabled(args), (
        "native LoRA syncs adapters only to colocated engines; use --colocate, --lora-train-only, "
        "or --megatron-to-hf-mode bridge"
    )
    assert not args.overlap_param_gather, (
        "native LoRA does not support --overlap-param-gather: sibling-module adapter attachment has not "
        "been validated against MCore's bucket prefetch ordering."
    )
    assert not args.moe_shared_expert_overlap, (
        "native LoRA does not support --moe-shared-expert-overlap: the dispatcher owns the shared-expert "
        "communication, so the adapter's gather/reduce no longer matches the module's parallel mode."
    )
    assert not args.overlap_grad_reduce, (
        "native LoRA does not support --overlap-grad-reduce: replicated adapter gradients need a "
        "tensor-parallel sum over the buffer MCore's in-flight data-parallel reduce-scatter writes."
    )
    _model_type, spec = resolve_checkpoint_spec(args.hf_checkpoint)
    assert (
        spec.experts is None or args.expert_tensor_parallel_size == 1
    ), f"native {spec.name} expert LoRA assumes --expert-tensor-parallel-size 1"


def _assert_fixed_target_layout(spec: LoRAArchSpec, hf_config: dict, hf_targets: list[str]) -> None:
    layout = resolve_hf_lora_targets(hf_config)
    optional = {t for t in layout if any(t.endswith(f".{suffix}") for suffix in spec.fixed_targets.optional)}
    missing = sorted(set(layout) - set(hf_targets) - optional)
    unexpected = sorted(set(hf_targets) - set(layout))
    assert not (missing or unexpected), (
        f"native {spec.name} LoRA requires its verified adapter layout (missing={missing}, "
        f"unexpected={unexpected}); omit --target-modules and --exclude-modules to use it"
    )


def _assert_shared_expert_up_proj_has_gate_proj(hf_targets: list[str]) -> None:
    """SGLang stacks a shared expert's gate/up adapter and zero-fills only a missing up_proj,
    so a shared expert's up_proj adapter needs its gate_proj selected too."""
    leaves_by_block: dict[str, set[str]] = {}
    for target in hf_targets:
        block, _, leaf = target.rpartition(".")
        leaves_by_block.setdefault(block, set()).add(leaf)
    lone_up_projs = sorted(
        f"{block}.up_proj" if block else "up_proj"
        for block, leaves in leaves_by_block.items()
        # a bare leaf selector may reach a shared expert
        if (not block or _SHARED_EXPERT_BLOCK.search(block)) and "up_proj" in leaves and "gate_proj" not in leaves
    )
    assert (
        not lone_up_projs
    ), f"native LoRA needs gate_proj selected with {lone_up_projs}: SGLang drops a shared expert's lone up_proj"


def attention_target_modules_csv(hf_config: dict) -> str:
    """Attention-only ``--target-modules`` for the checkpoint's native spec, in declaration order."""
    _model_type, spec = resolve_config_spec(hf_config)
    return spec.attention.canonical_targets_csv


@dataclass(frozen=True)
class PreflightReport:
    """Cheap, no-GPU audit of everything a native-LoRA run will need."""

    model_type: str
    spec_name: str
    mbridge_registered: bool
    model_args_script: str | None  # path when one was given and found

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
    model_args_script: str | None = None,
    *,
    strict: bool = False,
) -> PreflightReport:
    """Audit native-LoRA support for a checkpoint before touching any GPU.

    Checks the plugin registry, the mbridge conversion bridge, and — when the
    launch tooling passes ``model_args_script`` — that the model args file it
    will source exists. ``strict=True`` raises on any gap a run cannot survive;
    the default returns the report for the caller to log.
    """
    model_type, spec = resolve_checkpoint_spec(hf_checkpoint)

    try:
        import miles_plugins.mbridge  # noqa: F401  (registers Miles bridge subclasses)
        from mbridge.core.bridge import _MODEL_REGISTRY

        mbridge_registered = model_type in _MODEL_REGISTRY
    except ImportError:
        mbridge_registered = False

    if model_args_script is not None and not os.path.exists(model_args_script):
        assert not strict, (
            f"[lora-preflight] {model_args_script} not found; raw-mode conversion "
            "and training source MODEL_ARGS from that file."
        )
        model_args_script = None

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
