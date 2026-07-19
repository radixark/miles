"""HuggingFace-version compatibility patches for the experimental FSDP backend."""

import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)


def apply_flash_attn_saux_guard() -> bool:
    """Guard s_aux against None in transformers 5.6.0 flash_attention_forward (crashes sink-less models). Returns True if patched."""
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
    """Warn when an arch's training forward diverges structurally from the rollout (e.g. DeepSeek DSA)."""
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
    """A config-time patch: an applies_to(hf_config) predicate + an apply(hf_config, args) action."""

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


def _has_attention_sink(hf_config) -> bool:
    return hf_config is not None and getattr(hf_config, "model_type", "") == "gpt_oss"


def require_flash_for_attention_sink(hf_config, args) -> None:
    """Sink models (gpt-oss) only get the sink applied on the flash path; force flash to match SGLang's fa3."""
    if args is None:
        return
    impl = getattr(args, "attn_implementation", None)
    if impl and not str(impl).startswith("flash"):
        logger.warning(
            "[fsdp class_patches] attention-sink model needs flash attention to match the SGLang rollout; "
            f"overriding attn_implementation {impl!r} -> flash_attention_3 (eager/sdpa drop the sink)."
        )
        args.attn_implementation = "flash_attention_3"


register_model_patch(ModelPatchHook("flash_attn_saux_guard", _always, lambda cfg, args: apply_flash_attn_saux_guard()))
register_model_patch(ModelPatchHook("attn_sink_requires_flash", _has_attention_sink, require_flash_for_attention_sink))
register_model_patch(ModelPatchHook("fp8_checkpoint_guard", _has_config, lambda cfg, args: check_fp8_checkpoint(cfg)))
register_model_patch(
    ModelPatchHook("dsa_train_infer_warn", _has_config, lambda cfg, args: check_train_infer_consistency(cfg))
)
# Per-arch model patches register in their spec (adaptations/specs/); this module keeps only generic ones.


def apply_class_patches(hf_config=None, args=None) -> None:
    """Apply all registered ModelPatchHooks. Safe to call once at actor init."""
    for hook in _MODEL_PATCH_HOOKS:
        if hook.applies_to(hf_config):
            hook.apply(hf_config, args)
