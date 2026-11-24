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


# provider field <- args field, as core_transformer_config_from_args maps them; only the *_param names differ
BRIDGE_PRECISION_FIELDS = {
    "fp8": "fp8",
    "fp8_recipe": "fp8_recipe",
    "fp8_param": "fp8_param_gather",
    "fp8_quantizer_factory": "fp8_quantizer_factory",
    "fp8_margin": "fp8_margin",
    "fp8_interval": "fp8_interval",
    "fp8_amax_history_len": "fp8_amax_history_len",
    "fp8_amax_compute_algo": "fp8_amax_compute_algo",
    "fp8_wgrad": "fp8_wgrad",
    "fp8_output_proj": "fp8_output_proj",
    "fp4": "fp4",
    "fp4_recipe": "fp4_recipe",
    "fp4_param": "fp4_param_gather",
    "fp4_quantizer_factory": "fp4_quantizer_factory",
    "first_last_layers_bf16": "first_last_layers_bf16",
    "num_layers_at_start_in_bf16": "num_layers_at_start_in_bf16",
    "num_layers_at_end_in_bf16": "num_layers_at_end_in_bf16",
}


def apply_precision_args(provider, args) -> None:
    """Set the low-precision recipe from args; a field left at the provider default silently trains in BF16."""
    for provider_field, args_field in BRIDGE_PRECISION_FIELDS.items():
        setattr(
            provider, provider_field, getattr(args.backend, args_field)
        )  # config-access-exempt: names come from BRIDGE_PRECISION_FIELDS
    provider.quant_recipe = None
    if args.backend.te_precision_config_file:
        from megatron.core.quantization.utils import load_quantization_recipe  # this module imports without Megatron

        provider.quant_recipe = load_quantization_recipe(args.backend.te_precision_config_file)


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
