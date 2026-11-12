import pytest
import torch

from miles_plugins.models import cp_utils
from miles_plugins.models.cp_utils import _receive_permutation, _relayout_indices, _relayout_split_sizes


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("multipliers", [(1,), (2, 6, 1), (1, 10, 2), (8, 2, 10, 3), (2,) * 32])
def test_counts_and_sort_free_permutation_match_native_maps(cp_size, multipliers):
    boundaries = [0]
    for multiplier in multipliers:
        boundaries.append(boundaries[-1] + multiplier * cp_size)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    local_length = boundaries[-1] // cp_size
    for rank in range(cp_size):
        to_packed, to_zigzag = _relayout_indices(cu_seqlens, rank, cp_size, boundaries[-1])
        send_sizes, recv_sizes = _relayout_split_sizes(tuple(boundaries), rank, cp_size, local_length)
        assert send_sizes == torch.bincount(to_zigzag // local_length, minlength=cp_size).tolist()
        assert recv_sizes == torch.bincount(to_packed // local_length, minlength=cp_size).tolist()
        torch.testing.assert_close(to_zigzag, to_zigzag.sort().values)
        permutation = _receive_permutation(to_packed, cp_size)
        torch.testing.assert_close(permutation.argsort(), to_packed.argsort())
        for peer in range(cp_size):
            rows = to_packed[to_packed // local_length == peer] % local_length
            if rows.numel():
                torch.testing.assert_close(rows, torch.arange(rows[0], rows[0] + rows.numel()))


@pytest.mark.parametrize("transform", [cp_utils.zigzag_to_packed_shard, cp_utils.packed_shard_to_zigzag])
def test_auto_preserves_legacy_all_gather_without_host_metadata(monkeypatch, transform):
    hidden = torch.zeros(4, 1, 2)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    calls = []
    monkeypatch.setattr(cp_utils._Relayout, "apply", lambda *args: calls.append(args) or hidden)
    assert transform(hidden, cu_seqlens, object(), 0, 2) is hidden
    assert len(calls) == 1
    with pytest.raises(ValueError, match="cu_seqlens_host"):
        transform(hidden, cu_seqlens, object(), 0, 2, communication="all_to_all")
    with pytest.raises(ValueError, match="Unsupported"):
        transform(hidden, cu_seqlens, object(), 0, 2, communication="invalid")


@pytest.mark.parametrize("transform", [cp_utils.zigzag_to_packed_shard, cp_utils.packed_shard_to_zigzag])
def test_auto_uses_a2a_with_native_host_metadata(monkeypatch, transform):
    hidden = torch.zeros(4, 1, 2)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    calls = []
    monkeypatch.setattr(cp_utils._AllToAllRelayout, "apply", lambda *args: calls.append(args) or hidden)
    assert transform(hidden, cu_seqlens, object(), 0, 2, cu_seqlens_host=(0, 8)) is hidden
    assert len(calls) == 1
    assert calls[0][2] == calls[0][3] == [2, 2]
    assert calls[0][-1] == (transform is cp_utils.zigzag_to_packed_shard)
