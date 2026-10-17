from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Qwen3VLAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"qwen3_vl"})
    verified = True


class Qwen3VLMoeAdapter(Qwen3VLAdapter):
    model_types = frozenset({"qwen3_vl_moe"})
    verified = False
