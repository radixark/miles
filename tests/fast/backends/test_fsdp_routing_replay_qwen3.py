from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.fast.fixtures.replay_fixtures import reset_routing_replay_manager, wire_replay

from miles.backends.experimental.fsdp_utils.models.replay_routers import install_qwen3_router_replay
from miles.utils.replay_base import routing_replay_manager


class _Qwen3MoeTopKRouter(nn.Module):
    """Mirrors transformers 5.12.1 Qwen3MoeTopKRouter.forward."""

    def __init__(self, num_experts=8, hidden_dim=4, top_k=2, norm_topk_prob=True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


@pytest.fixture(autouse=True)
def _reset_manager():
    reset_routing_replay_manager(enabled=True)
    yield
    reset_routing_replay_manager(enabled=False)


def _wire(router):
    return wire_replay(router, install_qwen3_router_replay)


def test_fallthrough_matches_stock_forward():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    x = torch.randn(6, 4)
    expected = router(x)

    _wire(router)
    routing_replay_manager.stage = "fallthrough"
    got = router(x)

    for a, b in zip(expected, got, strict=True):
        assert torch.allclose(a, b)


def test_replay_forward_returns_the_recorded_indices():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire(router)
    x = torch.randn(6, 4)

    routing_replay_manager.stage = "record"
    _, _, recorded = router(x)

    routing_replay_manager.clear_all_forward()
    routing_replay_manager.stage = "replay_forward"
    _, weights, replayed = router(torch.randn(6, 4))

    assert torch.equal(replayed.cpu(), recorded.cpu())
    assert weights.shape == recorded.shape


def test_forward_and_backward_cursors_are_independent():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire(router)

    routing_replay_manager.stage = "record"
    _, _, first = router(torch.randn(3, 4))
    router(torch.randn(3, 4))
    routing_replay_manager.clear_all_forward()

    routing_replay_manager.stage = "replay_forward"
    _, _, fwd0 = router(torch.randn(3, 4))
    routing_replay_manager.stage = "replay_backward"
    _, _, bwd0 = router(torch.randn(3, 4))

    assert torch.equal(fwd0.cpu(), first.cpu())
    assert torch.equal(bwd0.cpu(), first.cpu())


def test_weights_stay_differentiable_under_replay():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire(router)

    routing_replay_manager.stage = "record"
    router(torch.randn(5, 4))
    routing_replay_manager.clear_all_forward()

    routing_replay_manager.stage = "replay_forward"
    _, weights, _ = router(torch.randn(5, 4))
    weights.sum().backward()

    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()


def test_qwen3_5_variant_without_norm_topk_prob_attribute():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    del router.norm_topk_prob
    _wire(router)

    routing_replay_manager.stage = "record"
    _, weights, _ = router(torch.randn(4, 4))

    assert torch.allclose(weights.float().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_specs_register_adapters_for_the_real_model_types():
    from types import SimpleNamespace

    import miles.backends.experimental.fsdp_utils.adaptations.specs  # noqa: F401
    from miles.backends.experimental.fsdp_utils.adaptations.routing_replay import resolve_routing_replay_adapter

    qwen3_moe = resolve_routing_replay_adapter(SimpleNamespace(model_type="qwen3_moe"))
    assert qwen3_moe is not None
    assert qwen3_moe.module_cls_name == "Qwen3MoeTopKRouter"

    qwen3_5 = resolve_routing_replay_adapter(SimpleNamespace(model_type="qwen3_5_moe_text"))
    assert qwen3_5 is not None
    assert qwen3_5.module_cls_name == "Qwen3_5MoeTopKRouter"

    assert resolve_routing_replay_adapter(SimpleNamespace(model_type="qwen3_5_text")) is None


def test_replayed_weights_are_the_probs_at_the_replayed_experts():
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter(norm_topk_prob=False)
    _wire(router)

    routing_replay_manager.stage = "record"
    router(torch.randn(4, 4))
    routing_replay_manager.clear_all_forward()

    x = torch.randn(4, 4)
    routing_replay_manager.stage = "replay_forward"
    logits, weights, idx = router(x)

    expected = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1).gather(1, idx)
    assert torch.allclose(weights.float(), expected, atol=1e-6)
