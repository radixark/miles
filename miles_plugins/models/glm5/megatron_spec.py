"""Native Megatron DSA spec provider for the raw model-provider path."""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from miles_plugins.models.dsa_topk import get_flashinfer_dsa_topk_options


def get_dsa_spec(args, config, vp_stage):
    if not hasattr(config, "dsa_indexer_topk_backend"):
        raise RuntimeError("--dsa-impl megatron requires Megatron's configurable DSA top-k backend support")
    # Resolve inside the training worker so --train-env-vars and Ray's runtime
    # environment select exactly the same tie policy as the Miles indexer.
    if config.dsa_indexer_topk_backend == "flashinfer":
        options = get_flashinfer_dsa_topk_options()
        config.dsa_indexer_topk_deterministic = options["deterministic"]
        config.dsa_indexer_topk_tie_break = options["tie_break"]
    return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
