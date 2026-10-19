"""Architecture-independent checks on the HF config before the FSDP backend builds a model.

The FSDP backend trains the stock HF model, so it is sensitive to checkpoints and architectures it cannot
train faithfully; these checks fail fast or warn before any weights load.
"""

import logging

logger = logging.getLogger(__name__)


def validate_hf_config(hf_config, *, verified: bool, rank: int) -> None:
    check_fp8_checkpoint(hf_config)
    check_train_infer_consistency(hf_config)
    if not verified and rank == 0:
        logger.warning(
            "[fsdp config_checks] model_type=%r has no recorded FSDP validation; "
            "it will load via the generic HF path — correctness is not guaranteed.",
            getattr(hf_config, "model_type", None),
        )


def check_train_infer_consistency(hf_config) -> None:
    """Warn when an arch's training forward diverges structurally from the rollout (e.g. DeepSeek DSA: the
    sparse-attention indexer is absent from HF training, so train is dense while the rollout is sparse)."""
    model_type = str(getattr(hf_config, "model_type", "") or "")
    is_dsa = (
        "deepseek_v3" in model_type
        or bool(getattr(hf_config, "index_topk", None))
        or getattr(hf_config, "attn_module_list_cfg", None) is not None
    )
    if is_dsa:
        logger.warning(
            "[fsdp config_checks] DeepSeek sparse-attention (DSA) detected (model_type=%s): the HF "
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
