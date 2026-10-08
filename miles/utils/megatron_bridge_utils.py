from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_mtp_args(provider, args) -> None:
    """Give the model the MTP layers args.mtp_num_layers names, detached.

    0 builds none. N builds N: a model that defines a different number is refused, and one that
    defines none gains none. Unset keeps the model's own, which for a Bridge provider are the HF
    config's, as checkpoint conversion needs. A trainer's arguments always name a number, N only for
    MTP training (compute_trainer_args).

    Megatron adds the MTP loss to every training forward of a model with MTP layers, rolling the
    labels from input_ids when labels is None. Detached, that loss trains only the MTP layers, never
    the shared trunk, embedding or output layer.

    ``provider`` is a Bridge provider or a TransformerConfig.
    """
    named = args.mtp_num_layers
    if named == 0:
        provider.mtp_num_layers = None
        # A hybrid provider's finalize() rebuilds its MTP layers from this pattern.
        if getattr(provider, "mtp_hybrid_override_pattern", None):
            provider.mtp_hybrid_override_pattern = None
    elif named is not None:
        inherited = provider.mtp_num_layers
        assert inherited in (None, named), f"--mtp-num-layers {named}, but the model has {inherited} MTP layers"
    provider.mtp_detach_heads = True


def apply_dsa_backend_args(provider, args) -> None:
    """Map --dsa-attention-backend onto the provider's dsa_attention_backend (bridge) or dsa_kernel_backend (main)."""
    backend = getattr(args, "dsa_attention_backend", "megatron")
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):
        explicit = getattr(args, "dsa_kernel_backend", None)
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
