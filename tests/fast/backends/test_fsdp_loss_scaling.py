"""CPU tests for the FSDP per-token loss pre-scan (``fsdp_utils/loss_scaling``).

Covers:
- the scan walks the training schedule — contiguous and explicit-index
  iterators — and never disturbs the training iterator's offset;
- per-step counts follow the shared wrapper's ``num_tokens`` protocol,
  counting a fully-masked sample as one token;
- scales are ``dp_size / global_num_tokens`` once the SUM all-reduce has
  shared the per-rank counts.
"""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.loss_scaling import get_per_token_loss_scales
from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.parallel import GroupInfo, set_parallel_state
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state


def _make_rollout_data(mask_rows: list[list[int]]) -> dict:
    return {"loss_masks": [torch.tensor(row, dtype=torch.int) for row in mask_rows]}


def _set_dp_size(dp_size: int) -> None:
    state = make_parallel_state()
    state.intra_dp = GroupInfo(rank=0, size=dp_size, group=None)
    set_parallel_state(state)


def _no_all_reduce(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: None)


def _peer_all_reduce(monkeypatch, peer_counts: torch.Tensor) -> None:
    def _all_reduce(tensor, op=None, group=None):
        tensor += peer_counts

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)


def _eight_samples() -> list[list[int]]:
    # two optimizer steps of two 2-sample micro-batches: 4+6+3+5 tokens, then 2+7+1+3
    return [[1] * n for n in (4, 6, 3, 5, 2, 7, 1, 3)]


class TestScanSchedule:
    def test_contiguous_schedule_splits_counts_by_optimizer_step(self, monkeypatch):
        _set_dp_size(1)
        _no_all_reduce(monkeypatch)
        rollout_data = _make_rollout_data(_eight_samples())

        scales = get_per_token_loss_scales(DataIterator(rollout_data, micro_batch_size=2), [2, 2])

        assert torch.allclose(torch.stack(scales), torch.tensor([1 / 18, 1 / 13]))

    def test_explicit_index_schedule_follows_micro_batch_indices(self, monkeypatch):
        _set_dp_size(1)
        _no_all_reduce(monkeypatch)
        rollout_data = _make_rollout_data([[1] * 4, [1] * 6, [1] * 3, [1] * 5])
        iterator = DataIterator(rollout_data, micro_batch_indices=[[0, 2], [1, 3]])

        scales = get_per_token_loss_scales(iterator, [1, 1])

        assert torch.allclose(torch.stack(scales), torch.tensor([1 / 7, 1 / 11]))

    def test_scan_starts_at_offset_zero_regardless_of_iterator_position(self, monkeypatch):
        _set_dp_size(1)
        _no_all_reduce(monkeypatch)
        rollout_data = _make_rollout_data(_eight_samples())
        mid_rollout = DataIterator(rollout_data, micro_batch_size=2)
        mid_rollout.get_next(["loss_masks"])
        mid_rollout.get_next(["loss_masks"])

        late_scales = get_per_token_loss_scales(mid_rollout, [2, 2])

        fresh_scales = get_per_token_loss_scales(DataIterator(rollout_data, micro_batch_size=2), [2, 2])
        assert torch.allclose(torch.stack(late_scales), torch.stack(fresh_scales))

    def test_scan_leaves_training_iterator_offset_untouched(self, monkeypatch):
        _set_dp_size(1)
        _no_all_reduce(monkeypatch)
        rollout_data = _make_rollout_data(_eight_samples())
        iterator = DataIterator(rollout_data, micro_batch_size=2)
        iterator.get_next(["loss_masks"])
        offset_before = iterator.offset

        get_per_token_loss_scales(iterator, [2, 2])

        assert iterator.offset == offset_before

    def test_zero_mask_sample_counts_as_one_token(self, monkeypatch):
        _set_dp_size(1)
        _no_all_reduce(monkeypatch)
        rollout_data = _make_rollout_data([[0, 0, 0], [1, 1]])

        scales = get_per_token_loss_scales(DataIterator(rollout_data, micro_batch_size=2), [1])

        assert torch.allclose(torch.stack(scales), torch.tensor([1 / 3]))


class TestGlobalScale:
    def test_scales_divide_by_dp_wide_global_tokens(self, monkeypatch):
        _set_dp_size(2)
        # a peer rank contributes 14 tokens to step 0 and 2 to step 1
        _peer_all_reduce(monkeypatch, torch.tensor([14, 2]))
        rollout_data = _make_rollout_data(_eight_samples())

        scales = get_per_token_loss_scales(DataIterator(rollout_data, micro_batch_size=2), [2, 2])

        # global counts are 32 and 15, so each micro-batch scales by dp_size / global
        assert torch.allclose(torch.stack(scales), torch.tensor([2 / 32, 2 / 15]))

    def test_empty_schedule_returns_no_scales_without_collectives(self, monkeypatch):
        _set_dp_size(2)
        called = []
        monkeypatch.setattr(
            torch.distributed, "all_reduce", lambda tensor, op=None, group=None: called.append(tensor)
        )
        rollout_data = _make_rollout_data([[1] * 4])

        assert get_per_token_loss_scales(DataIterator(rollout_data, micro_batch_size=2), []) == []
        assert called == []
