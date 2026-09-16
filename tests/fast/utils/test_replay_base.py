from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

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


def test_get_topk_fn_strides_across_noncontiguous_invalid_rows():
    # BSHD padding can recur at the same position in each sample. Stride by
    # invalid-row ordinal so those rows do not alias to the same expert block.
    scores = torch.arange(8, dtype=torch.float32).repeat(8, 1)
    replayed_top_indices = torch.tensor(
        [[6, 7], [6, 7], [6, 7], [-1, -1], [6, 7], [6, 7], [6, 7], [-1, -1]],
        dtype=torch.int32,
    )
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    expected = torch.tensor(
        [[6, 7], [6, 7], [6, 7], [0, 1], [6, 7], [6, 7], [6, 7], [2, 3]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(topk_fn(scores, 2), expected)


def test_get_topk_fn_preserves_partial_padding():
    # a row with some valid picks keeps its -1 padding (only all-(-1) rows are filled)
    scores = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    replayed_top_indices = torch.tensor([[2, -1, -1]], dtype=torch.int32)
    manager = _make_replay_manager(replayed_top_indices)

    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(scores, 3), replayed_top_indices)
