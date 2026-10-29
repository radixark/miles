from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import torch

from miles.utils.replay_base import BaseReplayManager


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


def test_fill_through_fills_each_micro_batch_once_up_to_the_cursor():
    manager = BaseReplayManager()
    filled = []
    manager.attach(filled.append)

    manager.fill_through(1)
    manager.fill_through(0)
    manager.fill_through(2)

    assert filled == [0, 1, 2]


def test_attach_and_clear_all_drop_the_previous_source_and_entries():
    manager = BaseReplayManager()
    replay = manager.create_replay()
    replay.top_indices_list = [torch.zeros(1)]
    first, second = [], []

    manager.attach(first.append)
    assert replay.top_indices_list == []
    manager.fill_through(0)
    manager.attach(second.append)
    manager.fill_through(1)
    manager.clear_all()
    manager.fill_through(2)

    assert first == [0]
    assert second == [0, 1]
