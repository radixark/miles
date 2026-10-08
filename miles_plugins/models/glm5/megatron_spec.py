"""Native Megatron DSA spec provider for the raw model-provider path."""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from miles.kernels.attention.dsa.topk import flashinfer_tie_break_value


def get_dsa_spec(args, config, vp_stage):
    # Resolve inside the training worker so --train-env-vars and Ray's runtime
    # environment select exactly the same tie policy as the Miles indexer.
    if config.dsa_indexer_topk_backend == "flashinfer":
        from sglang.srt.environ import envs

        config.dsa_indexer_topk_deterministic = envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get()
        config.dsa_indexer_topk_tie_break = flashinfer_tie_break_value()
    return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
