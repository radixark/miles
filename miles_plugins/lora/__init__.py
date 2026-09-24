"""Miles-native LoRA: adapters attached directly to the mcore model for ``--megatron-to-hf-mode raw``.

Megatron is imported inside functions so argument validation can import this package.
"""

from miles_plugins.lora.lora import (
    apply_native_lora,
    export_lora_hf_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)
from miles_plugins.lora.registry import preflight_native_lora, resolve_adapter_targets
from miles_plugins.lora.sglang_adapter import export_lora_sglang_named

__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "export_lora_sglang_named",
    "load_lora_adapter_hf",
    "preflight_native_lora",
    "resolve_adapter_targets",
    "wrap_model_provider_with_lora",
]
