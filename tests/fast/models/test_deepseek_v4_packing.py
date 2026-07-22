from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.training_utils.data import (
    DataIterator,
    _get_thd_allgather_pad_multiple,
    get_thd_padded_total_lengths,
)
from miles.backends.training_utils.loss_hub import logit_processors
from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.cp_utils import (
    get_compress_cu_seqlens_for_packed,
    get_compress_query_ranges_for_packed,
    get_compress_topk_idxs_packed,
    get_seq_ids_and_offsets_from_cu_seqlens,
    get_window_topk_idxs_packed,
    is_packed_thd_contiguous_cp,
)


def test_packed_window_and_compressed_indices_do_not_cross_samples():
    cu_seqlens = torch.tensor([0, 128, 384], dtype=torch.long)
    q_positions = torch.arange(96, 192)

    window = get_window_topk_idxs_packed(q_positions, cu_seqlens, window_size=128, bsz=1)[0]
    seq_ids, _, seq_starts, _ = get_seq_ids_and_offsets_from_cu_seqlens(cu_seqlens, q_positions)
    valid_window = window >= 0
    assert torch.all(window[valid_window] >= seq_starts.unsqueeze(1).expand_as(window)[valid_window])
    assert torch.all(window[valid_window] <= q_positions.unsqueeze(1).expand_as(window)[valid_window])

    first_query_of_second_sample = (q_positions == 128).nonzero(as_tuple=True)[0].item()
    assert window[first_query_of_second_sample, 0].item() == 128
    assert torch.all(window[first_query_of_second_sample, 1:] == -1)
    assert seq_ids[first_query_of_second_sample].item() == 1

    compressed = get_compress_topk_idxs_packed(q_positions, cu_seqlens, ratio=128, bsz=1)[0]
    assert compressed[(q_positions == 127).nonzero(as_tuple=True)[0].item(), 0].item() == 384
    assert torch.all(compressed[first_query_of_second_sample] == -1)

    starts, ends = get_compress_query_ranges_for_packed(q_positions, cu_seqlens, ratio=4)
    assert starts[first_query_of_second_sample].item() == 32
    assert ends[first_query_of_second_sample].item() == 32
    assert torch.all(ends >= starts)


def test_compressed_boundaries_require_per_sample_alignment():
    assert torch.equal(
        get_compress_cu_seqlens_for_packed(torch.tensor([0, 128, 384]), ratio=128),
        torch.tensor([0, 1, 3]),
    )
    with pytest.raises(AssertionError, match="divisible"):
        get_compress_cu_seqlens_for_packed(torch.tensor([0, 127, 384]), ratio=128)

    packed = SimpleNamespace(qkv_format="thd", miles_allgather_cp=True)
    zigzag = SimpleNamespace(qkv_format="thd", miles_allgather_cp=False)
    assert is_packed_thd_contiguous_cp(packed, cp_size=4)
    assert not is_packed_thd_contiguous_cp(zigzag, cp_size=4)
    assert is_packed_thd_contiguous_cp(zigzag, cp_size=1)


def test_c4_overlap_resets_at_each_packed_sample():
    compressor = DeepSeekV4Compressor.__new__(DeepSeekV4Compressor)
    nn.Module.__init__(compressor)
    compressor.compress_ratio = 4
    compressor.head_dim = 1
    compressor.cp_size = 1

    groups = torch.arange(1, 1 + 4 * 4 * 2, dtype=torch.float32).view(1, 4, 4, 2)
    packed = compressor.overlap_transform_packed(groups, torch.tensor([0, 8, 16]), value=0)
    unbounded = compressor.overlap_transform_raw(groups, value=0)

    assert torch.all(packed[:, 2, :4] == 0)
    assert torch.any(unbounded[:, 2, :4] != 0)
    assert torch.equal(packed[:, 2:, 4:], unbounded[:, 2:, 4:])


def test_thd_padding_and_response_logits_keep_original_alignment(monkeypatch):
    args = SimpleNamespace(
        model_name="deepseekv4",
        compress_ratios=[0, 4, 128],
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
    )
    assert get_thd_padded_total_lengths(args, [3, 129]) == [128, 256]
    assert _get_thd_allgather_pad_multiple(cp_size=4, pad_size=128, sample_pad_multiple=128) == 1024

    rollout_data = {"tokens": [10, 20]}
    iterator = DataIterator(rollout_data, 1, args=args)
    assert iterator.get_next(["tokens"])["tokens"] == [10]

    monkeypatch.setattr(
        logit_processors,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1, rank=0)),
    )
    logits = torch.arange(8 * 2, dtype=torch.float32).view(1, 8, 2)
    chunks = list(
        logit_processors.get_responses(
            logits,
            args=args,
            unconcat_tokens=[torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
            total_lengths=[3, 2],
            response_lengths=[2, 1],
            padded_total_lengths=[4, 4],
        )
    )

    assert torch.equal(chunks[0][0], logits[0, 0:2])
    assert torch.equal(chunks[0][1], torch.tensor([11, 12]))
    assert torch.equal(chunks[1][0], logits[0, 4:5])
    assert torch.equal(chunks[1][1], torch.tensor([21]))


def test_indexer_workspace_padding_keeps_logical_lengths():
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    get_lengths = tilelang_indexer_fwd._get_indexer_padded_lengths
    assert get_lengths(seq_len=257, seq_len_kv=513, heads=64) == (258, 1024)
    assert get_lengths(seq_len=256, seq_len_kv=256, heads=64) == (256, 512)
    assert get_lengths(seq_len=0, seq_len_kv=0, heads=64) == (0, 0)


def test_indexer_workspace_padding_restores_logical_shape(monkeypatch):
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    captured = {}

    def fake_indexer_factory(*, heads, index_dim):
        captured["heads"] = heads
        captured["index_dim"] = index_dim

        def fake_indexer(q, kv, logits, weights, cu_ks, cu_ke):
            captured["q"] = q.clone()
            captured["kv"] = kv.clone()
            captured["weights"] = weights.clone()
            captured["cu_ks"] = cu_ks.clone()
            captured["cu_ke"] = cu_ke.clone()
            logits.zero_()

        return fake_indexer

    def fake_clean_factory():
        def fake_clean(logits, cu_ks, cu_ke):
            captured["clean_shape"] = tuple(logits.shape)

        return fake_clean

    monkeypatch.setattr(tilelang_indexer_fwd, "tl_indexer_fwd_impl", fake_indexer_factory)
    monkeypatch.setattr(tilelang_indexer_fwd, "clean_logits_", fake_clean_factory)

    q = torch.ones(3, 64, 2, dtype=torch.bfloat16)
    kv = torch.ones(257, 2, dtype=torch.bfloat16)
    weights = torch.ones(3, 64, dtype=torch.float32)
    cu_ks = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_ke = torch.tensor([1, 2, 257], dtype=torch.int32)

    logits = tilelang_indexer_fwd.indexer_fwd_interface(q, kv, weights, cu_ks, cu_ke)

    assert logits.shape == (3, 257)
    assert captured["q"].shape == (4 * 64, 2)
    assert captured["kv"].shape == (768, 2)
    assert captured["weights"].shape == (4, 64)
    assert captured["clean_shape"] == (4, 768)
    assert captured["cu_ks"][-1].item() == 257
    assert captured["cu_ke"][-1].item() == 257
    assert torch.count_nonzero(captured["kv"][257:]) == 0
