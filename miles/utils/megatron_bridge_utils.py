from contextlib import contextmanager

from miles.backends.training_utils.model_companion import ModelCompanionInstallationUtils

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(
        model_config, "share_embeddings_and_output_weights"
    ):  # config-access-exempt: Bridge module capabilities vary by architecture
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(
                module, "_maintain_float32_expert_bias"
            ):  # config-access-exempt: Bridge module capabilities vary by architecture
                module._maintain_float32_expert_bias()

    try:
        with ModelCompanionInstallationUtils.hide(model):
            yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_dsa_backend_args(provider, args) -> None:
    """Map --dsa-attention-backend onto the provider's dsa_attention_backend (bridge) or dsa_kernel_backend (main)."""
    backend = args.dsa_attention_backend
    if hasattr(
        provider, "dsa_attention_backend"
    ):  # config-access-exempt: third-party providers differ in dsa_attention_backend support
        provider.dsa_attention_backend = backend
    elif hasattr(
        provider, "dsa_kernel_backend"
    ):  # config-access-exempt: third-party providers differ in dsa_kernel_backend support
        explicit = getattr(
            args.backend, "dsa_kernel_backend", None
        )  # config-access-exempt: older Megatron parsers omit the DSA kernel backend switch
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
