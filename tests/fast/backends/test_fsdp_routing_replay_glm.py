from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.experimental.fsdp_utils.models.routing_replay_glm4_moe_lite import (
    install_glm4_moe_lite_routing_replay,
)
from miles.utils.replay_base import routing_replay_manager


class _Gate(nn.Module):
    def __init__(self, n_routed_experts):
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))


class _Glm4MoeLiteMoE(nn.Module):
    """Mirrors transformers 5.12.1 Glm4MoeLiteMoE.route_tokens_to_experts."""

    def __init__(self, n_routed_experts=8, n_group=2, topk_group=1, top_k=2):
        super().__init__()
        self.gate = _Gate(n_routed_experts)
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.5
        self.top_k = top_k

    def route_tokens_to_experts(self, router_logits):
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
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class _CpuReplay:
    """Device-neutral stand-in for ``Replay``; see test_fsdp_routing_replay_qwen3."""

    def __init__(self, stream_idx=0):
        self.stream_idx = stream_idx
        self.recorded = []
        self.forward_index = 0
        self.backward_index = 0

    def record(self, top_indices):
        self.recorded.append(top_indices.detach().clone())

    def pop_forward(self):
        top_indices = self.recorded[self.forward_index]
        self.forward_index += 1
        return top_indices

    def pop_backward(self):
        top_indices = self.recorded[self.backward_index]
        self.backward_index += 1
        return top_indices

    def clear_forward(self):
        self.forward_index = 0


@pytest.fixture(autouse=True)
def _reset_manager():
    routing_replay_manager.enabled = True
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = "fallthrough"
    yield
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = "fallthrough"


def _wire(block):
    install_glm4_moe_lite_routing_replay(block)
    replay = _CpuReplay()
    routing_replay_manager.replays.append(replay)
    routing_replay_manager.set_current(replay)
    return replay


def test_fallthrough_matches_stock_routing():
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    logits = torch.randn(5, 8)
    exp_idx, exp_w = block.route_tokens_to_experts(logits)

    _wire(block)
    routing_replay_manager.stage = "fallthrough"
    got_idx, got_w = block.route_tokens_to_experts(logits)

    assert torch.equal(exp_idx, got_idx)
    assert torch.allclose(exp_w, got_w)


def test_replay_forward_overrides_expert_selection():
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    _wire(block)

    routing_replay_manager.stage = "record"
    recorded, _ = block.route_tokens_to_experts(torch.randn(5, 8))

    routing_replay_manager.clear_all_forward()
    routing_replay_manager.stage = "replay_forward"
    replayed, weights = block.route_tokens_to_experts(torch.randn(5, 8))

    assert torch.equal(replayed.cpu(), recorded.cpu())
    assert weights.shape == recorded.shape
    assert torch.isfinite(weights).all()


def test_replayed_weights_gather_from_the_sigmoid_logits():
    # The weight for a replayed expert must be sigmoid(logit) at that expert, scaled -- not a
    # value read out of the masked group-routing scores, which are -inf outside the chosen groups.
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    block.norm_topk_prob = False
    _wire(block)

    routing_replay_manager.stage = "record"
    block.route_tokens_to_experts(torch.randn(5, 8))
    routing_replay_manager.clear_all_forward()

    logits = torch.randn(5, 8)
    routing_replay_manager.stage = "replay_forward"
    idx, weights = block.route_tokens_to_experts(logits)

    expected = logits.sigmoid().gather(1, idx) * block.routed_scaling_factor
    assert torch.allclose(weights, expected, atol=1e-6)


def test_replay_can_select_experts_outside_the_recomputed_groups():
    # The point of R3: the training-side group mask may exclude the rollout's experts, and the
    # replayed indices must win anyway with finite weights.
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    block.norm_topk_prob = False
    replay = _wire(block)

    forced = torch.tensor([[0, 7], [7, 0], [1, 6]])
    replay.recorded.append(forced)
    routing_replay_manager.stage = "replay_forward"

    logits = torch.randn(3, 8)
    idx, weights = block.route_tokens_to_experts(logits)

    assert torch.equal(idx, forced)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights, logits.sigmoid().gather(1, forced) * block.routed_scaling_factor, atol=1e-6)


def test_gradients_flow_to_the_router_logits_under_replay():
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    _wire(block)

    routing_replay_manager.stage = "record"
    block.route_tokens_to_experts(torch.randn(4, 8))
    routing_replay_manager.clear_all_forward()

    logits = torch.randn(4, 8, requires_grad=True)
    routing_replay_manager.stage = "replay_forward"
    _, weights = block.route_tokens_to_experts(logits)
    weights.sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_spec_registers_the_glm_adapter():
    import miles.backends.experimental.fsdp_utils.adaptations.specs  # noqa: F401
    from miles.backends.experimental.fsdp_utils.adaptations.routing_replay import (
        resolve_routing_replay_adapter,
    )

    adapter = resolve_routing_replay_adapter(SimpleNamespace(model_type="glm4_moe_lite"))
    assert adapter is not None
    assert adapter.module_cls_name == "Glm4MoeLiteMoE"
