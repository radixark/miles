"""HF ``model_type`` to complete native-LoRA architecture-spec registry."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.spec.attention import (
    GQAAttentionSpec,
    HybridGQAGDNAttentionSpec,
    InklingAttentionSpec,
    MLAAttentionSpec,
)
from miles_plugins.lora.spec.base import AttentionFamily, LoRAArchSpec
from miles_plugins.lora.spec.layout import AttentionSpecBase
from miles_plugins.lora.spec.lm_head import InklingLMHeadSpec
from miles_plugins.lora.spec.mlp import FusedGatedMLPSpec, InklingDenseMLPSpec
from miles_plugins.lora.spec.moe import InklingMoESpec, SharedOuterExpertMoESpec

logger = logging.getLogger(__name__)


def _arch_spec(attention: AttentionSpecBase, *, allows_mixer_only_adapter_chunks: bool = False) -> LoRAArchSpec:
    """Assemble one architecture spec; MoE policy learns the MLP target names by injection."""
    mlp = FusedGatedMLPSpec()
    return LoRAArchSpec(
        name=attention.name,
        model_family=attention.family,
        attention=attention,
        mlp=mlp,
        moe=SharedOuterExpertMoESpec(mlp.supported_targets),
        allows_mixer_only_adapter_chunks=allows_mixer_only_adapter_chunks,
    )


def _inkling_arch_spec() -> LoRAArchSpec:
    attention = InklingAttentionSpec()
    return LoRAArchSpec(
        name=attention.name,
        model_family=attention.family,
        attention=attention,
        mlp=InklingDenseMLPSpec(),
        moe=InklingMoESpec(),
        lm_head=InklingLMHeadSpec(),
        sglang_lora_target_modules=("all",),
    )


def _build_model_specs() -> dict[str, LoRAArchSpec]:
    gqa = _arch_spec(GQAAttentionSpec())
    mla = _arch_spec(MLAAttentionSpec())
    hybrid = _arch_spec(HybridGQAGDNAttentionSpec(), allows_mixer_only_adapter_chunks=True)
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
        "inkling_mm_model": _inkling_arch_spec(),
    }


MODEL_SPECS: dict[str, LoRAArchSpec] = _build_model_specs()


def _model_type_candidates(hf_checkpoint: str | None) -> list[str]:
    """Return outer and nested text ``model_type`` values from HF config.json."""
    if not hf_checkpoint:
        return []
    path = os.path.join(hf_checkpoint, "config.json")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        config = json.load(handle)
    text_config = config.get("text_config") or {}
    return [model_type for model_type in (config.get("model_type"), text_config.get("model_type")) if model_type]


def _structural_spec(config) -> LoRAArchSpec:
    """Spec used only by bare numerical/unit harnesses without an HF checkpoint."""
    if bool(getattr(config, "multi_latent_attention", False)):
        return _arch_spec(MLAAttentionSpec())
    return _arch_spec(GQAAttentionSpec())


def _resolve_registered_spec(hf_checkpoint: str | None) -> tuple[str, LoRAArchSpec]:
    """Resolve a registered spec from checkpoint metadata without a built model.

    No structural fallback: serving/configuration callers run before model
    construction and must fail closed when checkpoint metadata is absent or
    unsupported.
    """
    candidates = _model_type_candidates(hf_checkpoint)
    if not candidates:
        if hf_checkpoint:
            config_path = os.path.join(hf_checkpoint, "config.json")
            if os.path.exists(config_path):
                raise AssertionError(
                    f"native LoRA could not resolve a model_type: {config_path} declares neither "
                    "'model_type' nor 'text_config.model_type'. Fix the checkpoint config or use a "
                    "model-specific --lora-provider-path."
                )
            raise AssertionError(
                f"native LoRA could not load model_type because {hf_checkpoint!r}/config.json is missing. "
                "Provide a valid --hf-checkpoint or use a model-specific --lora-provider-path."
            )
        raise AssertionError(
            "native LoRA requires --hf-checkpoint/config.json to resolve an architecture spec before "
            "model construction; provide a checkpoint or a model-specific --lora-provider-path."
        )

    model_type = next((candidate for candidate in candidates if candidate in MODEL_SPECS), None)
    assert model_type is not None, (
        f"native LoRA has no spec registered for model_type {candidates[0]!r} "
        f"(--hf-checkpoint {hf_checkpoint}). Registered architectures: {sorted(MODEL_SPECS)}. "
        "Verify the adapter math for this architecture and register it in "
        "miles_plugins.lora.registry.MODEL_SPECS, use --megatron-to-hf-mode bridge, or point "
        "--lora-provider-path at a model-specific provider."
    )
    return model_type, MODEL_SPECS[model_type]


def resolve_native_lora_config(args) -> LoRAConfig:
    """Return the architecture-normalized config before native model build.

    Rollout setup can consume ``.target_modules`` from this result so SGLang
    allocates buffers for the same effective projection set the native model
    will attach (notably MLA ``all-linear`` normalization).
    """
    _model_type, spec = _resolve_registered_spec(getattr(args, "hf_checkpoint", None))
    config = spec.normalize_config(LoRAConfig.from_args(args))
    spec.validate_targets(config.target_modules)
    return config


def resolve_model_spec(args, config) -> tuple[str | None, LoRAArchSpec]:
    """Resolve a complete architecture spec and verify it matches the built model.

    Production checkpoints fail closed when their model type is not registered.
    Bare test harnesses without config.json retain a warned structural fallback,
    but still receive a concrete spec that drives attachment.
    """
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        spec = _structural_spec(config)
        logger.warning(
            "[lora-native] no config.json under %r; using the %s architecture spec from the built "
            "model structure. Production checkpoints must register model_type explicitly.",
            hf_checkpoint,
            spec.name,
        )
        return None, spec

    model_type, spec = _resolve_registered_spec(hf_checkpoint)
    built = AttentionFamily.MLA if bool(getattr(config, "multi_latent_attention", False)) else AttentionFamily.GQA
    assert spec.model_family == built, (
        f"registry entry for model_type {model_type!r} says {spec.model_family} attention but the "
        f"built model uses {built}; the registry and the checkpoint disagree."
    )
    return model_type, spec


def default_target_modules(hf_checkpoint: str) -> str:
    """Canonical attention-only ``--target-modules`` CSV for this checkpoint's family.

    Derived from the attention spec's own layout declaration, in declaration
    order (routed/grouped-expert adapters are out of scope; shared-expert/dense
    MLP targets remain an explicit opt-in).
    """
    _model_type, spec = _resolve_registered_spec(hf_checkpoint)
    return spec.attention.canonical_targets_csv


def serving_fused_families() -> list[frozenset[str]]:
    """Projection families some fused SGLang buffer stores, across shipped layouts.

    Derived from the registered specs, so a new architecture's fused groups
    automatically reach the serving-side target expansion.
    """
    families: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for arch_spec in MODEL_SPECS.values():
        for spec in (arch_spec.attention, arch_spec.mlp):
            collect = getattr(spec, "serving_fused_families", None)
            if collect is None:
                continue
            for family in collect():
                if family not in seen:
                    seen.add(family)
                    families.append(family)
    return families


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
    model_type, spec = _resolve_registered_spec(hf_checkpoint)

    try:
        import miles_plugins.mbridge  # noqa: F401  (registers Miles bridge subclasses)
        from mbridge.core.bridge import _MODEL_REGISTRY

        mbridge_registered = model_type in _MODEL_REGISTRY
    except Exception:
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
        except Exception:
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
