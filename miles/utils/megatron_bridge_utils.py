from contextlib import contextmanager

from miles.backends.training_utils.model_companion import ModelCompanionInstallationUtils

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def load_auto_bridge(hf_checkpoint: str):
    from megatron.bridge import AutoBridge

    import miles_plugins.megatron_bridge  # noqa: F401

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


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


def apply_pipeline_layout(provider, args) -> None:
    # arg name != provider field; arg default None, so propagate only when the user set it
    if args.backend.decoder_first_pipeline_num_layers is not None:
        provider.num_layers_in_first_pipeline_stage = args.backend.decoder_first_pipeline_num_layers
    if args.backend.decoder_last_pipeline_num_layers is not None:
        provider.num_layers_in_last_pipeline_stage = args.backend.decoder_last_pipeline_num_layers
    if args.backend.pipeline_model_parallel_layout is not None:
        provider.pipeline_model_parallel_layout = args.backend.pipeline_model_parallel_layout


def apply_dsa_backend_args(provider, args) -> None:
    """Resolve DSV4 kernel selection separately from the GLM DSA backend option."""
    explicit = getattr(
        args.backend, "dsa_kernel_backend", None
    )  # config-access-exempt: older Megatron parsers omit the DSA kernel backend switch
    if (
        getattr(provider, "experimental_attention_variant", None) == "dsv4_hybrid"
    ):  # config-access-exempt: Megatron providers without experimental attention omit this field
        provider.dsa_kernel_backend = explicit or "cudnn"
        return

    backend = args.dsa_attention_backend
    if hasattr(
        provider, "dsa_attention_backend"
    ):  # config-access-exempt: third-party providers differ in dsa_attention_backend support
        provider.dsa_attention_backend = backend
    elif hasattr(
        provider, "dsa_kernel_backend"
    ):  # config-access-exempt: third-party providers differ in dsa_kernel_backend support
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
