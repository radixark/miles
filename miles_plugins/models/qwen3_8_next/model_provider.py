"""Custom provider: miles' default model + forward hooks publishing the PLE
n-gram side channel (input token ids never reach a transformer layer).
"""

import copy
import logging

from megatron.core.models.gpt import GPTModel

from miles_plugins.models.qwen3_8_next.ops.ple import (
    Qwen38NextFrozenNGramEmbedding,
    build_ngram_contexts_packed,
    clear_ple_batch,
    publish_ple_batch,
)

logger = logging.getLogger(__name__)


def _find_ple_embedding(model):
    for _, module in model.named_modules():
        if isinstance(module, Qwen38NextFrozenNGramEmbedding):
            return module
    return None


def _install_ple_context_hooks(model: GPTModel) -> None:
    ple_embedding = _find_ple_embedding(model)
    if ple_embedding is None:
        return

    def pre_hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return
        packed = kwargs.get("packed_seq_params")
        cu_seqlens = getattr(packed, "cu_seqlens_q", None) if packed is not None else None
        flat = input_ids.reshape(-1)
        contexts = build_ngram_contexts_packed(flat, cu_seqlens, ple_embedding.ngram_size, ple_embedding.eos_token_id)
        ngram_ids = ple_embedding.compute_ngram_ids(contexts)
        publish_ple_batch(ngram_ids, cu_seqlens)

    def post_hook(_module, _args, _output):
        clear_ple_batch()

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook)
    logger.info("PLE n-gram context hooks installed on the stage hosting the PLE layer")


def get_qwen3_8_next_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = copy.copy(get_args())
    args.custom_model_provider_path = None
    base_provider = get_model_provider_func(args)

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    _install_ple_context_hooks(model)
    return model


def get_qwen3_8_next_vlm_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    from megatron.training import get_args

    from miles_plugins.models.qwen3_8_next.vision import wire_qwen3_8_next_visual

    model = get_qwen3_8_next_model_provider(
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    wire_qwen3_8_next_visual(model, get_args().hf_checkpoint)
    return model
