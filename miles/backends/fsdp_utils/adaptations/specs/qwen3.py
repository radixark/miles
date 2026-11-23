"""Qwen3 dense."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter


class Qwen3Adapter(ArchAdapter):
    model_types = frozenset({"qwen3"})
    verified = True
    passes_attention_packing_kwargs = True
