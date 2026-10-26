"""Qwen3-VL runs on the stock HF path; its adapter only records the FSDP validation."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter


class Qwen3VLAdapter(ArchAdapter):
    model_types = frozenset({"qwen3_vl"})
    verified = True
