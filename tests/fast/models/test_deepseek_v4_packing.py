from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.training_utils.data import (
    DataIterator,
    _dynamic_batch_schedule_fingerprint,
    _get_thd_allgather_pad_multiple,
    _get_capped_thd_partitions,
    _summarize_thd_packing,
    _sync_thd_dynamic_batch_schedule,
    _validate_dynamic_batch_schedule,
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


def test_thd_dynamic_packing_stats_measure_real_and_padded_tokens():
    stats = _summarize_thd_packing(
        [60000, 61000, 30000, 30000],
        [[0, 1], [2, 3]],
        sample_pad_multiple=128,
        global_pad_multiple=4096,
        seq_length=131072,
    )

    assert stats["samples"] == 4
    assert stats["packs"] == 2
    assert stats["samples_per_pack"] == 2
    assert stats["actual_tokens"] == 181000
    assert stats["padded_tokens"] == 184320
    assert stats["no_pack_padded_tokens"] == 188416
    assert stats["packing_efficiency"] == pytest.approx(181000 / 184320)
    assert stats["seq_capacity_fill"] == pytest.approx(181000 / (2 * 131072))
    assert stats["padding_reduction_vs_mbs1"] == pytest.approx(1 - 184320 / 188416)
    assert stats["max_samples_per_pack"] == 2
    assert stats["max_pack_actual_tokens"] == 121000
    assert stats["max_pack_padded_tokens"] == 122880
    assert stats["oversized_packs"] == 0


def test_thd_dynamic_packing_stats_flag_pack_over_seq_length():
    stats = _summarize_thd_packing(
        [70000, 70000],
        [[0, 1]],
        sample_pad_multiple=128,
        global_pad_multiple=4096,
        seq_length=131072,
    )

    assert stats["max_pack_padded_tokens"] == 143360
    assert stats["oversized_packs"] == 1


def test_thd_capped_fallback_accounts_for_dsv4_alignment_and_global_padding():
    partitions = _get_capped_thd_partitions(
        [70000, 70000, 30000],
        2,
        sample_pad_multiple=128,
        global_pad_multiple=2048,
        max_padded_tokens=131072,
    )

    stats = _summarize_thd_packing(
        [70000, 70000, 30000],
        partitions,
        sample_pad_multiple=128,
        global_pad_multiple=2048,
        seq_length=131072,
    )
    assert sorted(sum(partitions, [])) == [0, 1, 2]
    assert stats["max_pack_padded_tokens"] <= 131072
    assert stats["oversized_packs"] == 0


def test_thd_capped_fallback_rejects_an_unplaceable_single_sample():
    with pytest.raises(ValueError, match="sample into a THD sequence"):
        _get_capped_thd_partitions(
            [131073],
            1,
            sample_pad_multiple=128,
            global_pad_multiple=2048,
            max_padded_tokens=131072,
        )


def test_dynamic_batch_schedule_validation_rejects_duplicate_or_missing_samples():
    with pytest.raises(ValueError, match="every local sample exactly once"):
        _validate_dynamic_batch_schedule([2], [[0, 1], [1, 2]], num_local_samples=3)


def test_thd_dynamic_batch_schedule_sync_uses_canonical_model_replica_schedule(monkeypatch):
    local_indices = [[0, 2], [1]]
    canonical_indices = [[0], [1, 2]]
    canonical_payload = {
        "num_microbatches": [2],
        "micro_batch_indices": canonical_indices,
        "num_local_samples": 3,
    }
    groups = [object(), object(), object()]
    parallel = SimpleNamespace(
        pp=SimpleNamespace(rank=1, size=2, group=groups[0]),
        cp=SimpleNamespace(rank=1, size=2, group=groups[1]),
        tp=SimpleNamespace(rank=1, size=2, group=groups[2]),
    )
    calls = []

    monkeypatch.setattr("miles.backends.training_utils.data.dist.is_initialized", lambda: True)
    monkeypatch.setattr("miles.backends.training_utils.data.dist.get_rank", lambda: 1)
    monkeypatch.setattr(
        "miles.backends.training_utils.data.dist.get_global_rank",
        lambda group, rank: 100 + groups.index(group),
    )

    def fake_broadcast_object_list(object_list, *, src, group):
        calls.append((src, group))
        object_list[0] = canonical_payload

    monkeypatch.setattr(
        "miles.backends.training_utils.data.dist.broadcast_object_list",
        fake_broadcast_object_list,
    )

    num_microbatches, indices = _sync_thd_dynamic_batch_schedule(parallel, [2], local_indices, 3)

    assert num_microbatches == [2]
    assert indices == canonical_indices
    assert calls == [(100, groups[0]), (101, groups[1]), (102, groups[2])]
    assert _dynamic_batch_schedule_fingerprint([2], local_indices) != _dynamic_batch_schedule_fingerprint([2], canonical_indices)


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


def test_indexer_query_chunk_size_bounds_fp32_workspace():
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    max_workspace_bytes = 8192
    chunk_size = tilelang_indexer_fwd._get_indexer_query_chunk_size(
        seq_len=17,
        seq_len_kv=257,
        heads=64,
        max_workspace_bytes=max_workspace_bytes,
    )
    _, padded_seq_len_kv = tilelang_indexer_fwd._get_indexer_padded_lengths(1, 257, 64)

    assert chunk_size == 2
    assert chunk_size % 2 == 0
    assert chunk_size * padded_seq_len_kv * torch.float32.itemsize <= max_workspace_bytes


def test_indexer_query_chunk_size_rejects_workspace_smaller_than_one_tile():
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    _, padded_seq_len_kv = tilelang_indexer_fwd._get_indexer_padded_lengths(1, 257, 64)
    minimum_workspace_bytes = 2 * padded_seq_len_kv * torch.float32.itemsize

    with pytest.raises(ValueError, match="smaller than one TileLang query tile"):
        tilelang_indexer_fwd._get_indexer_query_chunk_size(
            seq_len=17,
            seq_len_kv=257,
            heads=64,
            max_workspace_bytes=minimum_workspace_bytes - 1,
        )


def test_batched_indexer_topk_chunks_queries_without_full_logits(monkeypatch):
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    seen_chunk_sizes = []

    def fake_indexer(q, kv, weights, cu_ks, cu_ke):
        seen_chunk_sizes.append(q.shape[0])
        scores = torch.arange(kv.shape[0], dtype=torch.float32).expand(q.shape[0], -1).clone()
        positions = torch.arange(kv.shape[0]).unsqueeze(0)
        valid = (positions >= cu_ks.unsqueeze(1)) & (positions < cu_ke.unsqueeze(1))
        return scores.masked_fill(~valid, float("-inf"))

    def torch_topk_indices(logits, topk):
        scores, indices = torch.topk(logits, topk, dim=-1)
        return indices.to(torch.int32).masked_fill(scores == -torch.inf, -1)

    monkeypatch.setattr(tilelang_indexer_fwd, "indexer_fwd_interface", fake_indexer)
    q = torch.zeros(5, 2, 64, 1, dtype=torch.bfloat16)
    k = torch.zeros(5, 2, 1, dtype=torch.bfloat16)
    weights = torch.ones(5, 2, 64, dtype=torch.float32)
    cu_ks = torch.zeros(5, dtype=torch.int32)
    cu_ke = torch.full((5,), 5, dtype=torch.int32)

    indices = tilelang_indexer_fwd.batched_indexer_topk(
        q,
        k,
        weights,
        cu_ks,
        cu_ke,
        topk=2,
        topk_fn=torch_topk_indices,
        max_workspace_bytes=4096,
    )

    assert indices.shape == (2, 5, 2)
    assert torch.equal(indices, torch.tensor([[[4, 3]] * 5] * 2, dtype=torch.int32))
    assert seen_chunk_sizes == [2, 2, 1, 2, 2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA TileLang kernel")
def test_batched_indexer_topk_gpu_chunks_match_full_logits():
    tilelang_indexer_fwd = pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd"
    )
    from miles_plugins.models.dsa_topk import torch_dsa_topk

    torch.manual_seed(1234)
    device = torch.device("cuda")
    q = torch.randn(16, 1, 64, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(8, 1, 128, device=device, dtype=torch.bfloat16)
    weights = torch.randn(16, 1, 64, device=device, dtype=torch.float32)
    cu_ks = torch.zeros(16, device=device, dtype=torch.int32)
    cu_ke = torch.full((16,), 8, device=device, dtype=torch.int32)

    full_logits = tilelang_indexer_fwd.batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)
    expected = torch_dsa_topk(full_logits, 4)
    actual = tilelang_indexer_fwd.batched_indexer_topk(
        q,
        k,
        weights,
        cu_ks,
        cu_ke,
        topk=4,
        topk_fn=torch_dsa_topk,
        max_workspace_bytes=4096,
    )

    assert torch.equal(actual, expected)
