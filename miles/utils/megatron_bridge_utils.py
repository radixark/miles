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


def apply_dsa_backend_args(provider, args) -> None:
    """Map --dsa-attention-backend onto the provider's dsa_attention_backend (bridge) or dsa_kernel_backend (main)."""
    backend = getattr(args, "dsa_attention_backend", "megatron")
    if backend == "loom":
        # The generated deterministic kernels are wired through the plugin specs (get_glm5_spec /
        # get_dsv4_spec), which read args.dsa_attention_backend themselves; the bridge providers only
        # know Megatron-core's DSA and the TileLang module.
        raise ValueError(
            "--dsa-attention-backend loom is only available with the miles_plugins model specs "
            "(--spec miles_plugins.models.glm5.glm5 get_glm5_spec / miles_plugins.models.deepseek_v4.deepseek_v4 "
            "get_dsv4_spec), not under --megatron-to-hf-mode bridge."
        )
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):
        explicit = getattr(args, "dsa_kernel_backend", None)
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
