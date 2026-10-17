from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Glm4vAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"glm4v", "glm4v_moe", "glm46v", "glmga", "glm_ocr"})
