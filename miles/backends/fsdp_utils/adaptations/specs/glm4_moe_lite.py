"""glm4_moe_lite (GLM-4.7-Flash): its batched expert layout matches qwen3_moe, so it reuses the same
HF-native unfuse at weight sync, plus the rollout-routing-replay hook on its group-limited router."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter
from miles.backends.fsdp_utils.adaptations.weight_bridge import is_batched_experts_param, unfuse_batched_experts
from miles.backends.fsdp_utils.models.replay_routers import install_glm4_moe_lite_router_replay


class Glm4MoeLiteAdapter(ArchAdapter):
    model_types = frozenset({"glm4_moe_lite"})
    verified = True
    routing_replay = RoutingReplayAdapter(
        name="glm4_moe_lite", module_cls_name="Glm4MoeLiteMoE", install=install_glm4_moe_lite_router_replay
    )

    def param_transform(self, name, param):
        return unfuse_batched_experts if is_batched_experts_param(name, param) else None
