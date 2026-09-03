"""Miles-native LoRA plugin implementing the provider protocol.

``--lora-provider-path miles_plugins.lora`` (the default) resolves here; the
older ``miles_plugins.lora.lora`` path still works for explicit pins.

Constraint:

- Import ``miles``/``megatron`` inside functions, never at module level.
"""

from miles_plugins.lora.lora import (
    apply_native_lora,
    export_lora_hf_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)
from miles_plugins.lora.registry import default_target_modules, preflight_native_lora
from miles_plugins.lora.sglang_adapter import export_lora_sglang_named

__all__ = [
    "apply_native_lora",
    "default_target_modules",
    "export_lora_hf_named",
    "export_lora_sglang_named",
    "load_lora_adapter_hf",
    "preflight_native_lora",
    "wrap_model_provider_with_lora",
]
