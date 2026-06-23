"""HuggingFace-version compatibility patches for the experimental FSDP backend.

The FSDP backend trains the stock HuggingFace model, so it is sensitive to
transformers-version drift. These patches keep the training forward runnable and warn
on architectures whose forward diverges from the SGLang rollout. All are idempotent and
no-op when the underlying issue is absent.
"""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)


def apply_flash_attn_saux_guard() -> bool:
    """Guard ``s_aux`` (attention sink) against ``None`` in flash_attention_forward.

    transformers 5.6.0 does ``s_aux.to(query.dtype)`` unconditionally; for models without
    sinks (Qwen3, Qwen3.5) ``s_aux`` is None and the first training forward raises. Recompile
    the function with a None-guard and re-register it. Returns True if patched.
    """
    try:
        import transformers.integrations.flash_attention as fa
    except Exception:  # pragma: no cover
        return False
    try:
        src = inspect.getsource(fa.flash_attention_forward)
    except (OSError, TypeError):
        return False

    BUG = "s_aux=s_aux.to(query.dtype)"
    if "if s_aux is not None" in src or BUG not in src:
        return False  # already guarded, or an unrecognized layout

    new_src = textwrap.dedent(src).replace(BUG, "s_aux=(s_aux.to(query.dtype) if s_aux is not None else None)")
    ns = vars(fa)
    try:
        exec(compile(new_src, fa.__file__, "exec"), ns)  # noqa: S102 - controlled recompile
    except Exception as e:  # pragma: no cover
        logger.warning(f"[fsdp class_patches] s_aux guard compile failed: {e}")
        return False
    patched = ns["flash_attention_forward"]
    patched._saux_guarded = True
    fa.flash_attention_forward = patched

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as A

        for key in list(A.valid_keys()):
            try:
                cur = A[key]
            except Exception:
                continue
            if getattr(cur, "__name__", None) == "flash_attention_forward":
                try:
                    A[key] = patched
                except Exception:
                    try:
                        A.register(key, patched, exist_ok=True)
                    except Exception:
                        pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"[fsdp class_patches] s_aux guard re-register skipped: {e}")

    logger.info("[fsdp class_patches] applied flash-attention s_aux None-guard")
    return True


def check_train_infer_consistency(hf_config) -> None:
    """Warn when an arch's training forward diverges structurally from the SGLang rollout.

    DeepSeek-V3.2/DSA: the sparse-attention indexer is absent from HF training modeling, so
    its weights load UNEXPECTED and are never trained (train is dense, rollout is sparse).
    """
    model_type = str(getattr(hf_config, "model_type", "") or "")
    is_dsa = (
        "deepseek_v3" in model_type
        or bool(getattr(hf_config, "index_topk", None))
        or getattr(hf_config, "attn_module_list_cfg", None) is not None
    )
    if is_dsa:
        logger.warning(
            "[fsdp class_patches] DeepSeek sparse-attention (DSA) detected (model_type=%s): the HF "
            "training forward has no indexer, so it is dropped and train attention is DENSE while "
            "the rollout is SPARSE. RL on DSA via FSDP is not currently consistent.",
            model_type,
        )


def check_fp8_checkpoint(hf_config) -> None:
    """Fail fast on native-fp8 checkpoints (the actor has no inline dequant)."""
    qc = getattr(hf_config, "quantization_config", None)
    if not qc:
        return
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if str(method or "").lower() == "fp8":
        raise ValueError(
            "FSDP backend cannot train from an fp8-quantized checkpoint "
            "(quantization_config.quant_method='fp8'). Convert to bf16 first:\n"
            "  python tools/fp8_cast_bf16.py --input-fp8-hf-path <src> --output-bf16-hf-path <dst>\n"
            "then copy config/tokenizer (dropping quantization_config) into <dst> and point "
            "--hf-checkpoint at it."
        )


class ModelPatchHook:
    """A config-time HF-compat/model patch: an ``applies_to(hf_config)`` predicate + an
    ``apply(hf_config, args)`` action (``args`` is the actor Namespace, e.g. for true_on_policy_mode).

    Makes the per-arch dispatch a first-class registry instead of a hardcoded per-arch if-chain.
    New archs register a hook rather than editing ``apply_class_patches``. Self-gating checks
    (fp8 fail-fast, DSA warn) use ``applies_to=_has_config`` and gate internally.
    """

    def __init__(self, name, applies_to, apply):
        self.name = name
        self.applies_to = applies_to
        self.apply = apply


_MODEL_PATCH_HOOKS: list[ModelPatchHook] = []


def register_model_patch(hook: ModelPatchHook) -> None:
    _MODEL_PATCH_HOOKS.append(hook)


def _always(hf_config) -> bool:
    return True


def _has_config(hf_config) -> bool:
    return hf_config is not None


register_model_patch(ModelPatchHook("flash_attn_saux_guard", _always, lambda cfg, args: apply_flash_attn_saux_guard()))
register_model_patch(ModelPatchHook("fp8_checkpoint_guard", _has_config, lambda cfg, args: check_fp8_checkpoint(cfg)))
register_model_patch(
    ModelPatchHook("dsa_train_infer_warn", _has_config, lambda cfg, args: check_train_infer_consistency(cfg))
)
# Per-arch model patches (e.g. the qwen3_moe MoE-block patch) register themselves in their spec
# (adaptations/specs/), so this module keeps only the generic, always-applicable HF-compat patches.


def apply_class_patches(hf_config=None, args=None) -> None:
    """Apply all registered FSDP HF-compat/model patches. Safe to call once at actor init.

    Scope is the ModelPatchHook registry only. The actor driver invokes the other registries
    around this call: config-lifetime packing before construction, post-load packing and weight
    fixups after ``from_pretrained``, and the backend-level true_on_policy batch-invariant setup.
    """
    for hook in _MODEL_PATCH_HOOKS:
        if hook.applies_to(hf_config):
            hook.apply(hf_config, args)
