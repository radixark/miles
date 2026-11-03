from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import threading

import pytest
import torch

from miles.utils.replay_base import BaseReplayManager, Replay


class _FakeReplay:
    def __init__(self, *top_indices):
        self.top_indices = list(top_indices)
        self.forward_index = self.backward_index = 0

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


class _QueueReplay:
    """A replay whose entries a fill appends, popped by cursor as `Replay` does."""

    def __init__(self):
        self.top_indices_list = []
        self.forward_index = 0

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices


def test_fills_run_in_order_on_the_worker_and_wait_filled_waits_up_to_the_cursor():
    manager = BaseReplayManager()
    filled, release = [], threading.Event()

    def fill(index):
        if index == 1:
            release.wait(timeout=5)
        filled.append((index, threading.current_thread().name))

    manager.attach(fill, 3)
    manager.wait_filled(0)
    assert [index for index, _ in filled] == [0]
    release.set()
    manager.wait_filled(2)
    manager.wait_filled(1)

    assert [index for index, _ in filled] == [0, 1, 2]
    assert {name.startswith("replay-") for _, name in filled} == {True}


def test_finish_runs_once_on_the_waiting_thread_after_the_last_fill():
    manager = BaseReplayManager()
    finished = []
    manager.attach(lambda index: None, 2, lambda: finished.append(threading.current_thread()))

    manager.wait_filled(0)
    assert finished == []
    manager.wait_filled(1)
    manager.wait_filled(1)

    assert finished == [threading.current_thread()]


def test_a_failed_fill_raises_at_its_pop_and_the_later_fills_are_skipped():
    manager = BaseReplayManager()
    filled = []

    def fill(index):
        if index == 1:
            raise RuntimeError("store read failed")
        filled.append(index)

    manager.attach(fill, 3)
    manager.wait_filled(0)
    with pytest.raises(RuntimeError, match="store read failed"):
        manager.wait_filled(2)
    with pytest.raises(RuntimeError, match="store read failed"):
        manager.wait_filled(1)
    manager.clear_all()

    assert filled == [0]


def test_clear_all_drains_the_running_fill_and_drops_the_queued_ones_and_the_entries():
    manager = BaseReplayManager()
    replay = manager.create_replay()
    filled, release = [], threading.Event()

    def fill(index):
        release.wait(timeout=5)
        replay.top_indices_list.append(torch.zeros(1))
        filled.append(index)

    manager.attach(fill, 2)
    threading.Timer(0.05, release.set).start()
    manager.clear_all()

    assert filled == [0]
    assert replay.top_indices_list == []
    manager.wait_filled(5)


def test_attach_drops_the_previous_step_before_filling_the_next():
    manager = BaseReplayManager()
    replay = manager.create_replay()
    replay.top_indices_list = [torch.zeros(1)]
    first, second = [], []

    manager.attach(first.append, 1)
    assert replay.top_indices_list == []
    manager.wait_filled(0)
    manager.attach(second.append, 2)
    manager.wait_filled(1)

    assert first == [0]
    assert second == [0, 1]


def test_replay_forward_pops_only_after_its_micro_batch_is_filled():
    manager = BaseReplayManager()
    manager.enable_check_replay_result = False
    manager.enabled = True
    manager.stage = "replay_forward"
    replay = _QueueReplay()
    manager.set_current(replay)
    release = threading.Event()
    expected = torch.tensor([[1, 0]], dtype=torch.int32)

    def fill(index):
        release.wait(timeout=5)
        replay.top_indices_list.append(expected)

    manager.attach(fill, 1)
    threading.Timer(0.05, release.set).start()
    topk_fn = manager.get_topk_fn(_topk, return_probs=False)

    torch.testing.assert_close(topk_fn(torch.tensor([[0.1, 0.9]]), 2), expected)


@pytest.fixture
def cpu_device(monkeypatch):
    import miles.utils.replay_base as replay_base

    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))


def test_pop_backward_releases_its_entry_and_pop_forward_keeps_it(cpu_device):
    replay = Replay()
    first, second = torch.tensor([[0]], dtype=torch.int32), torch.tensor([[1]], dtype=torch.int32)
    replay.top_indices_list = [first, second]

    torch.testing.assert_close(replay.pop_forward(), first)
    torch.testing.assert_close(replay.pop_backward(), first)
    assert replay.top_indices_list == [None, second]
    torch.testing.assert_close(replay.pop_forward(), second)
    torch.testing.assert_close(replay.pop_backward(), second)

    assert replay.top_indices_list == [None, None]
    replay.forward_index = replay.backward_index = 0
    with pytest.raises(IndexError, match="pop_forward: entry 0 was already released"):
        replay.pop_forward()
    with pytest.raises(IndexError, match="pop_backward: entry 0 was already released"):
        replay.pop_backward()


def test_entries_without_a_backward_pop_live_until_clear_all(cpu_device):
    manager = BaseReplayManager()
    replay = manager.create_replay()
    entries = [torch.tensor([[0]], dtype=torch.int32), torch.tensor([[1]], dtype=torch.int32)]
    replay.top_indices_list = list(entries)

    replay.pop_forward()
    replay.pop_forward()
    manager.clear_all_forward()
    replay.pop_forward()
    assert replay.top_indices_list == entries

    manager.clear_all()

    assert replay.top_indices_list == []
    assert (replay.forward_index, replay.backward_index) == (0, 0)
