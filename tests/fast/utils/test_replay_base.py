from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import torch

from miles.utils import replay_base
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


def test_check_stats_count_all_rows_and_pop_resets(monkeypatch):
    manager = RoutingReplayManager()
    scores = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    replayed_top_indices = torch.tensor([[0], [-1]])
    monkeypatch.setenv("MILES_TEST_R3_THRESHOLD", "1.0")

    manager.check_replay_result(_topk, scores, 1, replayed_top_indices)

    assert manager.pop_check_stats() == (1, 2)
    assert manager.pop_check_stats() == (0, 0)


def test_reduce_check_stats_sums_group(monkeypatch):
    group = object()
    captured = {}

    class FakePGUtil:
        def all_reduce(self, values, actual_group, op):
            captured.update(group=actual_group, op=op)
            values += torch.tensor([3, 7])

    def fake_create(actual_group):
        captured["created_for"] = actual_group
        return FakePGUtil()

    monkeypatch.setattr(replay_base.GeneralPGUtil, "create", fake_create)

    assert replay_base.reduce_check_stats((2, 5), group) == (5, 12)
    assert captured == {
        "created_for": group,
        "group": group,
        "op": torch.distributed.ReduceOp.SUM,
    }
