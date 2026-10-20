from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import pytest
import torch

from miles.utils.replay_base import BaseReplayManager, Replay


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


@pytest.mark.parametrize(
    "stored,expected", [(torch.int16, torch.int32), (torch.int32, torch.int32), (torch.int64, torch.int64)]
)
def test_replay_pop_hands_gather_an_int32_or_int64_index(monkeypatch, stored, expected):
    """Rollout ids are stored int16 on the host; torch.gather rejects int16 indices."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    replay = Replay()
    replay.top_indices_list = [torch.tensor([[3, -1]], dtype=stored)] * 2
    for popped in (replay.pop_forward(), replay.pop_backward()):
        assert popped.dtype == expected
        assert popped.tolist() == [[3, -1]]
