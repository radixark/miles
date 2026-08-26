from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch

from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)

    def pop_forward(self):
        return self.top_indices.pop(0)

    def pop_backward(self):
        return self.pop_forward()


def _topk(scores, topk):
    return torch.topk(scores, topk, dim=1).indices.to(torch.int32)


def _make_replay_manager(top_indices):
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def _make_routing_replay_manager(top_indices):
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_get_topk_fn_fills_all_invalid_rows_with_arange():
    # an all-(-1) row is a masked/pad token; fill it with arange to avoid reading
    # invalid (-1) positions downstream
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_replay_manager(torch.tensor([[-1, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)


def test_routing_get_topk_fn_fills_partial_padding():
    # Megatron's MoE router validates every expert ID before applying the
    # padding mask, so router replay must sanitize individual invalid slots.
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    manager = _make_routing_replay_manager(torch.tensor([[2, -1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    result = topk_fn(scores, 3)
    torch.testing.assert_close(result, torch.tensor([[2, 0, 1]], dtype=torch.int64))
    assert result.dtype == torch.long


def test_routing_get_topk_fn_gathers_probs_after_sanitizing_without_mutating_replay():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_routing_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=True)
    probs, top_indices = topk_fn(scores, 3)

    torch.testing.assert_close(top_indices, torch.tensor([[2, 0, 1]], dtype=torch.int64))
    torch.testing.assert_close(probs, torch.tensor([[0.3, 0.1, 0.2]]))
    # The replay tensor can be reused by the backward pass and must retain its
    # sentinel padding rather than inheriting the forward pass's safe IDs.
    torch.testing.assert_close(replayed_top_indices, torch.tensor([[2, -1, -1]], dtype=torch.int32))
