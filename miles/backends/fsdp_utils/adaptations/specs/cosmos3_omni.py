from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Cosmos3OmniAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"cosmos3_omni"})
