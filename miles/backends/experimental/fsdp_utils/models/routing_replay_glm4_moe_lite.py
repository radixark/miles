"""R3 topk hook for glm4_moe_lite (GLM-4.7-Flash).

GLM selects experts in ``Glm4MoeLiteMoE.route_tokens_to_experts`` rather than in the gate,
behind group-limited routing. Only the final expert-selection topk is replaced; the
intra-group ``topk(2)`` and the group-selection topk stay untouched, since replaying those
would substitute expert ids into a group index space.

The weights are gathered from the sigmoid'd router logits afterwards, so replaying only the
indices keeps routing differentiable in exactly the way Megatron's ``_get_replay_result``
does with ``scores.gather(1, top_indices)``.
"""

import types

import torch
import torch.nn as nn

from miles.utils.replay_base import routing_replay_manager


def _replay_route_tokens_to_experts(self, router_logits):
    router_logits = router_logits.sigmoid()
    router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
    group_scores = (
        router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
        .reshape(-1, self.n_routed_experts)
    )
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = self._miles_replay_topk(scores_for_choice, self.top_k)
    topk_weights = router_logits.gather(1, topk_indices)
    if self.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        # Out-of-place: topk_weights is a gather output autograd still needs.
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights


def install_glm4_moe_lite_routing_replay(block: nn.Module) -> None:
    """Rebind ``block.route_tokens_to_experts`` so expert selection goes through the manager."""
    block._miles_replay_topk = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
    )
    block.route_tokens_to_experts = types.MethodType(_replay_route_tokens_to_experts, block)
