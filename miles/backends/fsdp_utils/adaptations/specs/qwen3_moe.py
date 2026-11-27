"""qwen3_moe: unfuse transformers>=5.6 batched experts into the per-expert names SGLang expects (via
HF's own reverse conversion) and the rollout-routing-replay hook."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter
from miles.backends.fsdp_utils.adaptations.weight_bridge import is_batched_experts_param, unfuse_batched_experts
from miles.backends.fsdp_utils.models.replay_routers import install_qwen3_router_replay


class Qwen3MoeAdapter(ArchAdapter):
    model_types = frozenset({"qwen3_moe"})
    verified = True
    passes_attention_packing_kwargs = True
    routing_replay = RoutingReplayAdapter(
        name="qwen3_moe", module_cls_name="Qwen3MoeTopKRouter", install=install_qwen3_router_replay
    )

    def param_transform(self, name, param):
        return unfuse_batched_experts if is_batched_experts_param(name, param) else None
