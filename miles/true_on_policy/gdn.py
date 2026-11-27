"""Qwen GDN models under TOP v1 (Qwen3.5/3.6/3.8 dense, Qwen3.5/3.6 MoE, Qwen3.8-Flash-Next): checkpoint programs.

Serving's GDN decode follows its fixed-tile chunk prefill bit for bit (SGLang ``true_on_policy.gdn_chunked``), so the
trainer runs one plain chunk prefill per sample. Both sides pin that prefill's kernel config through one environment
variable, read when SGLang's FLA kernels are imported (``GDN_MODE_ENV``), which ``top_launch`` sets for engines and
trainer alike.
"""

import logging
import os
from dataclasses import dataclass
from functools import cache

from miles.true_on_policy.contracts import uses_true_on_policy_v1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QwenGDNModel:
    """A Qwen GDN checkpoint family under TOP v1."""

    moe: bool  # routed experts + gated shared expert (DeepEP/DeepGEMM) vs a dense SwiGLU MLP


# HF top-level checkpoint model_type -> its program. Qwen3.6 and Qwen3.8-27B checkpoints reuse these types.
QWEN_GDN_MODELS = {
    "qwen3_5": QwenGDNModel(moe=False),
    "qwen3_5_moe": QwenGDNModel(moe=True),
    # Qwen3.8-Flash-Next: the same GDN layers under hyper-connections, QSA full attention and the PLE layer.
    "qwen4_exp": QwenGDNModel(moe=True),
}


# The environment variable SGLang reads for the TOP GDN program; under "chunked" it pins the chunk prefill's kkt
# kernel config when its FLA kernels are imported, so the trainer process must carry it too.
GDN_MODE_ENV = "SGLANG_TOP_GDN_MODE"
GDN_CHUNKED = "chunked"


def require_gdn_program() -> None:
    """Refuse a GDN TOP process whose environment does not select the chunked GDN program."""
    program = os.environ.get(GDN_MODE_ENV)
    if program != GDN_CHUNKED:
        raise ValueError(
            f"[top] Qwen GDN TOP needs {GDN_MODE_ENV}={GDN_CHUNKED} in every engine and trainer process (it pins "
            f"serving's chunk prefill kernel config; miles.true_on_policy.launch.top_launch sets it); got {program!r}"
        )


def qwen_gdn_model(model_type: str | None) -> QwenGDNModel | None:
    return QWEN_GDN_MODELS.get(model_type)


@cache
def _checkpoint_model_type(path: str) -> str | None:
    from miles.utils.hf_utils.config import load_hf_config

    return getattr(
        load_hf_config(path), "model_type", None
    )  # config-access-exempt: third-party HF configurations may omit model_type


def is_top_gdn(args) -> bool:
    """TOP v1 on a Qwen GDN checkpoint."""
    return uses_true_on_policy_v1(args) and qwen_gdn_model(_checkpoint_model_type(args.hf_checkpoint)) is not None


def is_top_glm53(args) -> bool:
    """TOP v1 on a GLM-5.3-Flash (glm5_next) checkpoint, whose TOP block loads HF weights per expert."""
    return uses_true_on_policy_v1(args) and _checkpoint_model_type(args.hf_checkpoint) == "glm5_next"


def default_serving_context_length(args) -> None:
    """Without --sglang-context-length, give GDN TOP engines the run's declared maximum context.

    The decode KV split (sglang.srt.true_on_policy.attention.decode_kv_split) runs one piece per split of the page
    table's width, and under CUDA graphs that width is the engine's context length, not the batch's: at Qwen3.5's
    262k default every decoded row runs 64 pieces. Narrowing it changes no output bytes, and Miles' request builder
    already caps prompt + response at --max-seq-len, so no admitted request is refused.
    """
    if args.sglang_context_length is not None:
        return
    from miles.utils.hf_utils.config import load_hf_config

    config = load_hf_config(args.hf_checkpoint)
    model_max = getattr(
        getattr(config, "text_config", config), "max_position_embeddings", None
    )  # config-access-exempt: module variants differ in whether they define max_position_embeddings
    if args.max_seq_len is None:
        logger.warning(
            f"[top] GDN TOP decode under CUDA graphs runs a key piece per 4096 tokens of the engine's context length "
            f"({model_max} by default), whatever the batch's lengths; set --max-seq-len or "
            "--sglang-context-length to the run's maximum"
        )
        return
    bound = args.max_seq_len
    if model_max is not None and bound >= model_max:
        return
    args.sglang_context_length = bound
    logger.info(
        f"[top] GDN TOP engines run --context-length {bound} (the run's max context; model default {model_max})"
    )
