"""R3 topk hook for the Qwen3 MoE routers.

``Qwen3MoeTopKRouter.forward`` and ``Qwen3_5MoeTopKRouter.forward`` (transformers 5.12.1) are
the same computation; the only difference is that qwen3_moe gates renormalization on
``norm_topk_prob`` while qwen3_5 always renormalizes. One rewritten forward serves both.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.utils.replay_base import routing_replay_manager


def _replay_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = self._miles_replay_topk(router_probs, self.top_k)
    # qwen3_5's router has no norm_topk_prob attribute and always renormalizes.
    if getattr(self, "norm_topk_prob", True):
        # Out-of-place: under replay these values come from a gather, and dividing in place
        # would mutate a tensor autograd still needs for the backward pass.
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


def install_qwen3_topk_router_replay(router: nn.Module) -> None:
    """Rebind ``router.forward`` so expert selection goes through the replay manager."""
    router._miles_replay_topk = routing_replay_manager.get_topk_fn(
        lambda scores, k: torch.topk(scores, k, dim=-1), return_probs=True
    )
    router.forward = types.MethodType(_replay_forward, router)
