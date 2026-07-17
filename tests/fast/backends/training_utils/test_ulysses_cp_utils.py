import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(*, cp_size: int, cp_rank: int = 0, cp_comm_type: str | None = None) -> ParallelState:
    return ParallelState(
        intra_dp=GroupInfo(rank=0, size=1, group=None),
        intra_dp_cp=GroupInfo(rank=0, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=0, size=1, group=None),
        pp=GroupInfo(rank=0, size=1, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=GroupInfo(rank=0, size=1, group=None),
        cp_comm_type=cp_comm_type,
    )


def test_parallel_state_detects_ulysses_cp():
    assert _parallel_state(cp_size=4, cp_comm_type="a2a").is_ulysses_cp
    assert _parallel_state(cp_size=4, cp_comm_type=["a2a"]).is_ulysses_cp
    assert not _parallel_state(cp_size=1, cp_comm_type="a2a").is_ulysses_cp
    assert not _parallel_state(cp_size=4, cp_comm_type="p2p").is_ulysses_cp


def test_slice_with_cp_uses_sequence_shards_for_sglang_ulysses(monkeypatch):
    monkeypatch.setattr(
        cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=4, cp_rank=2, cp_comm_type="a2a")
    )

    tokens = torch.arange(11)

    sliced = cp_utils.slice_with_cp(tokens, 0)

    torch.testing.assert_close(sliced, torch.tensor([4, 5, 10, 0]))


def test_slice_log_prob_with_cp_uses_sequence_shards_for_sglang_ulysses(monkeypatch):
    monkeypatch.setattr(
        cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=4, cp_rank=2, cp_comm_type="a2a")
    )

    log_probs = torch.arange(7, dtype=torch.float32)

    torch.testing.assert_close(
        cp_utils.slice_log_prob_with_cp(log_probs, total_length=11, response_length=7),
        torch.tensor([1.0, 2.0]),
    )


def test_regular_cp_still_uses_zigzag_response_slice(monkeypatch):
    monkeypatch.setattr(
        cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=2, cp_rank=1, cp_comm_type="p2p")
    )

    log_probs = torch.arange(7, dtype=torch.float32)

    sliced = cp_utils.slice_log_prob_with_cp(log_probs, total_length=11, response_length=7)

    torch.testing.assert_close(sliced, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_local_response_masks_use_sequence_shards_for_sglang_ulysses(monkeypatch):
    monkeypatch.setattr(
        cp_utils, "get_parallel_state", lambda: _parallel_state(cp_size=4, cp_rank=2, cp_comm_type="a2a")
    )

    masks = [torch.ones(7)]

    local_masks = cp_utils.get_local_response_loss_masks([11], [7], masks)

    torch.testing.assert_close(local_masks[0], torch.ones(2))
