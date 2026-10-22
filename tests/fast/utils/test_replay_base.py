from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import torch

from miles.utils import replay_base
from miles.utils.replay_base import BaseReplayManager, RoutingReplayManager


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)
        self.staged_forward = None

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


def _make_routing_replay_manager(top_indices):
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_FakeReplay(top_indices))
    return manager


def test_routing_rows_without_replayed_ids_take_the_routers_own_topk_on_tpu(monkeypatch):
    """Uncaptured rows (-1) and ids outside the experts are routed by the router, not pinned to experts 0..topk-1."""
    monkeypatch.setattr(replay_base.platforms, "IS_TPU", True)
    scores = torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0]] * 3)
    replayed = torch.tensor([[-1, -1], [0, 7], [2, 0]], dtype=torch.int32)
    manager = _make_routing_replay_manager(replayed)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[1, 3], [1, 3], [2, 0]], dtype=torch.int32))


def test_routing_padding_rows_keep_the_arange_fill_off_tpu():
    scores = torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0]])
    manager = _make_routing_replay_manager(torch.tensor([[-1, -1]], dtype=torch.int32))

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[0, 1]], dtype=torch.int32))


def test_routing_replay_returns_the_probs_of_the_filled_rows_on_tpu(monkeypatch):
    monkeypatch.setattr(replay_base.platforms, "IS_TPU", True)

    def topk_with_probs(scores, topk):
        values, indices = torch.topk(scores, topk, dim=1)
        return values, indices.to(torch.int32)

    scores = torch.tensor([[0.0, 4.0, 1.0, 3.0, 2.0]] * 2)
    manager = _make_routing_replay_manager(torch.tensor([[-1, -1], [2, 0]], dtype=torch.int32))

    probs, indices = manager.get_topk_fn(topk_with_probs, return_probs=True)(scores, 2)

    torch.testing.assert_close(indices, torch.tensor([[1, 3], [2, 0]], dtype=torch.int32))
    torch.testing.assert_close(probs, torch.tensor([[4.0, 3.0], [1.0, 0.0]]))


def test_a_staged_forward_replays_without_popping(monkeypatch):
    """torchtitan pops ids before the compiled blocks run; the blocks must read the staged ids, not pop again."""
    monkeypatch.setattr(replay_base.platforms, "IS_TPU", True)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    replay = manager.create_replay()
    replay.record(torch.tensor([[3, 1]], dtype=torch.int32))
    replay.record(torch.tensor([[4, 2]], dtype=torch.int32))
    manager.set_current(replay)

    replay.staged_forward = replay.pop_forward()
    topk_fn = manager.get_topk_fn(_topk, return_probs=False)
    scores = torch.zeros(1, 5)

    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[3, 1]], dtype=torch.int32))
    torch.testing.assert_close(topk_fn(scores, 2), torch.tensor([[3, 1]], dtype=torch.int32))
    assert replay.forward_index == 1


def test_tpu_rollout_ids_stay_on_the_host_until_popped(monkeypatch):
    """Host ids are padded per microbatch; on the device each new length would compile a new program."""
    monkeypatch.setattr(replay_base.platforms, "IS_TPU", True)
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("meta"))
    replay = RoutingReplayManager().create_replay()
    rollout_ids = torch.tensor([[3, 1]], dtype=torch.int32)
    replay.record(rollout_ids)
    replay.record(torch.empty(1, 2, dtype=torch.int32, device="meta"))  # recorded inside a forward

    assert replay.top_indices_list[0].device.type == "cpu"
    assert replay.top_indices_list[0].data_ptr() != rollout_ids.data_ptr()
    assert replay.top_indices_list[1].device.type == "meta"
    assert replay.pop_forward().device.type == "meta"
