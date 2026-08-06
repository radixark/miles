"""glm4_moe_lite (GLM-4.7-Flash) adaptations: the train-to-rollout weight transform (its batched
expert layout matches qwen3_moe, so it reuses the same HF-native unfuse) and the R3 routing-replay
hook on its group-limited router."""

from ..routing_replay import RoutingReplayAdapter, register_routing_replay_adapter
from ..weight_bridge import _batched_experts_matches, _hf_unfuse_experts_expand, register_param_transform

register_param_transform("glm4_moe_lite", _batched_experts_matches, _hf_unfuse_experts_expand)


def _is_glm4_moe_lite(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "glm4_moe_lite"


def _install_routing_replay(module) -> None:
    from ...models.routing_replay_glm4_moe_lite import install_glm4_moe_lite_routing_replay

    install_glm4_moe_lite_routing_replay(module)


# Discovery matches Glm4MoeLiteMoE, so the leading dense layers (first_k_dense_replace) whose
# mlp is a plain Glm4MoeLiteMLP are skipped without needing a config lookup.
register_routing_replay_adapter(
    RoutingReplayAdapter(
        name="glm4_moe_lite",
        applies_to=_is_glm4_moe_lite,
        module_cls_name="Glm4MoeLiteMoE",
        install=_install_routing_replay,
    )
)
