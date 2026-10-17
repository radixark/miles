from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Ernie45VLMoeAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"ernie4_5_vl_moe"})
