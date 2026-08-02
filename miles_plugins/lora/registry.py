"""HF ``model_type`` to complete native-LoRA architecture-spec registry."""

from __future__ import annotations

import enum
import json
import logging
import os
from dataclasses import dataclass, field

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.spec.attention import GQAAttentionSpec, HybridGQAGDNAttentionSpec, MLAAttentionSpec
from miles_plugins.lora.spec.base import AttentionFamily, LoRAArchSpec
from miles_plugins.lora.spec.layout import AttentionSpecBase
from miles_plugins.lora.spec.mlp import FusedGatedMLPSpec
from miles_plugins.lora.spec.moe import SharedOuterExpertMoESpec

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


class SupportStatus(enum.Enum):
    """How much end-to-end evidence backs a registry entry.

    VALIDATED  — a full train -> export -> serve loop ran green for this
                 model_type (train/rollout logprob_abs_diff at the ~1e-2 noise
                 floor). Silent at attach.
    STRUCTURAL — the architecture layout is covered by a shipped spec, but no
                 end-to-end run is on record. One info line at attach.
    UNSTABLE   — the adapter side is verified but a known model-path defect
                 makes raw-mode training unreliable; ``reason`` is surfaced
                 verbatim as a warning at attach.
    """

    VALIDATED = "validated"
    STRUCTURAL = "structural"
    UNSTABLE = "unstable"


_GDN_RAW_BACKWARD_NOTE = (
    "adapter attachment is verified, but raw mode's own backward is known to diverge on this "
    "family once the base is frozen (grad_norm 1e7-1e10 with recompute, NaN without it, NaN at "
    "CP=2), while bridge mode stays stable on the identical batch. This is a model-path issue "
    "rather than a LoRA one; prefer --megatron-to-hf-mode bridge until it is fixed, and watch "
    "grad_norm from step 1 if you continue."
)


@dataclass(frozen=True)
class ModelEntry:
    """One registered model_type: its architecture spec plus its support status."""

    spec: LoRAArchSpec
    status: SupportStatus = SupportStatus.STRUCTURAL
    reason: str = field(default="", compare=False)

    def __post_init__(self):
        assert (
            self.status is not SupportStatus.UNSTABLE or self.reason
        ), "an UNSTABLE registry entry must record why (the reason is shown to users verbatim)"


def _build_model_specs() -> dict[str, ModelEntry]:
    """Every entry maps to a structurally covered spec; ``status`` records how much
    end-to-end evidence exists so warnings and coverage tests key off one field
    instead of comments."""
    gqa = _arch_spec(GQAAttentionSpec())
    mla = _arch_spec(MLAAttentionSpec())
    # A PP/VPP chunk may contain only GDN mixer layers. Native GDN adapters are
    # intentionally absent, while GQA layers in another chunk still carry LoRA.
    hybrid = _arch_spec(HybridGQAGDNAttentionSpec(), allows_mixer_only_adapter_chunks=True)
    return {
        "llama": ModelEntry(gqa),
        "qwen2": ModelEntry(gqa, SupportStatus.VALIDATED),
        "qwen2_moe": ModelEntry(gqa),
        "qwen3": ModelEntry(gqa, SupportStatus.VALIDATED),
        "qwen3_moe": ModelEntry(gqa, SupportStatus.VALIDATED),
        "mimo": ModelEntry(gqa),
        "glm4": ModelEntry(gqa),
        "glm4_moe": ModelEntry(gqa),
        "qwen3_5": ModelEntry(hybrid, SupportStatus.UNSTABLE, _GDN_RAW_BACKWARD_NOTE),
        "qwen3_5_moe": ModelEntry(hybrid, SupportStatus.VALIDATED),
        "qwen3_6": ModelEntry(hybrid, SupportStatus.UNSTABLE, _GDN_RAW_BACKWARD_NOTE),
        "qwen3_6_moe": ModelEntry(hybrid, SupportStatus.UNSTABLE, _GDN_RAW_BACKWARD_NOTE),
        "qwen3_next": ModelEntry(hybrid, SupportStatus.UNSTABLE, _GDN_RAW_BACKWARD_NOTE),
        "deepseek_v3": ModelEntry(mla),
        "deepseek_v32": ModelEntry(mla),
        # deepseek_v4 (DeepSeek-V4-Flash) stays unregistered: its wq_a/wq_b/wkv attention is not
        # mcore MLA, and docs/advanced/lora.md declares that layout out of scope for this provider.
        "glm4_moe_lite": ModelEntry(mla, SupportStatus.VALIDATED),
        "glm_moe_dsa": ModelEntry(mla, SupportStatus.VALIDATED),
        "kimi_k2": ModelEntry(mla),
        # kimi_k25 needs the dequantized BF16 base to carry no quantization_config
        # (convert_kimi_int4_to_bf16.py strips it), or SGLang serves it through the
        # CompressedTensors path with a context-free forward.
        "kimi_k25": ModelEntry(mla, SupportStatus.VALIDATED),
        "joyai_llm_flash": ModelEntry(mla),
    }


MODEL_SPECS: dict[str, ModelEntry] = _build_model_specs()


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


def _resolve_registered_entry(hf_checkpoint: str | None) -> tuple[str, ModelEntry]:
    """Resolve a registered entry from checkpoint metadata without a built model.

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
    _model_type, entry = _resolve_registered_entry(getattr(args, "hf_checkpoint", None))
    config = entry.spec.normalize_config(LoRAConfig.from_args(args))
    entry.spec.validate_targets(config.target_modules)
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

    model_type, entry = _resolve_registered_entry(hf_checkpoint)
    if entry.status is SupportStatus.UNSTABLE:
        logger.warning("[lora-native] %s: %s", model_type, entry.reason)
    elif entry.status is SupportStatus.STRUCTURAL:
        logger.info(
            "[lora-native] %s is structurally covered by the %s spec but has no end-to-end validation "
            "on record%s; watch train_rollout_logprob_abs_diff on the first rollouts.",
            model_type,
            entry.spec.name,
            f" ({entry.reason})" if entry.reason else "",
        )

    built = AttentionFamily.MLA if bool(getattr(config, "multi_latent_attention", False)) else AttentionFamily.GQA
    assert entry.spec.model_family == built, (
        f"registry entry for model_type {model_type!r} says {entry.spec.model_family} attention but the "
        f"built model uses {built}; the registry and the checkpoint disagree."
    )
    return model_type, entry.spec


# --------------------------------------------------------------------------
# Launcher-facing helpers: scripts should ask the plugin instead of
# re-declaring per-family facts (target sets, support gaps).
# --------------------------------------------------------------------------


def default_target_modules(hf_checkpoint: str) -> str:
    """Canonical attention-only ``--target-modules`` CSV for this checkpoint's family.

    Derived from the attention spec's own layout declaration, in declaration
    order (routed/grouped-expert adapters are out of scope; shared-expert/dense
    MLP targets remain an explicit opt-in).
    """
    _model_type, entry = _resolve_registered_entry(hf_checkpoint)
    return entry.spec.attention.canonical_targets_csv


def serving_fused_families() -> list[frozenset[str]]:
    """Projection families some fused SGLang buffer stores, across shipped layouts.

    Derived from the registered specs, so a new architecture's fused groups
    automatically reach the serving-side target expansion.
    """
    families: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for entry in MODEL_SPECS.values():
        for spec in (entry.spec.attention, entry.spec.mlp):
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
    status: SupportStatus
    reason: str
    mbridge_registered: bool
    model_args_script: str | None  # path when megatron_model_type was given and found

    @property
    def convertible(self) -> bool:
        return self.mbridge_registered

    def render(self) -> str:
        lines = [
            f"model_type={self.model_type} spec={self.spec_name} status={self.status.value}"
            + (f" ({self.reason})" if self.reason else ""),
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

    Checks the plugin registry (with status), the mbridge conversion bridge,
    and — when ``megatron_model_type`` is given — the ``scripts/models`` model
    args file the launch tooling will source. ``strict=True`` raises on any gap
    a run cannot survive; the default returns the report for the caller to log.
    """
    model_type, entry = _resolve_registered_entry(hf_checkpoint)

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

            candidate = os.path.join(repo_base_dir, "scripts", "models", f"{megatron_model_type}.sh")
            model_args_script = candidate if os.path.exists(candidate) else None
        except Exception:
            model_args_script = None
        if strict:
            assert model_args_script is not None, (
                f"[lora-preflight] scripts/models/{megatron_model_type}.sh not found; raw-mode conversion "
                "and training source MODEL_ARGS from that file."
            )

    report = PreflightReport(
        model_type=model_type,
        spec_name=entry.spec.name,
        status=entry.status,
        reason=entry.reason,
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
