from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zstandard

from miles.utils.delta_preparation import CpuDeltaPreparer, PackedDeltaLayout
from miles.utils.disk_delta import checksum


@pytest.fixture(scope="module")
def compiled_stage():
    layout = PackedDeltaLayout((("empty", 0), ("tail", 65), ("unchanged", 128), ("byte_pairs", 65536)), block_bytes=64)
    stage = CpuDeltaPreparer(layout)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(stage.warmup).result()
    return stage


def _inputs(layout, *, changed):
    old, new = layout.allocate(), layout.allocate()
    old_views, new_views = dict(layout.views(old)), dict(layout.views(new))
    for name, previous in old_views.items():
        values = torch.arange(previous.numel(), dtype=torch.int64).to(torch.uint8)
        previous.copy_(values)
        new_views[name].copy_(values)
    if changed:
        new_views["tail"][-1] ^= 0xFF
        new_views["byte_pairs"].copy_(torch.arange(256, dtype=torch.uint8).repeat_interleave(256))
    return old, new


@pytest.mark.parametrize("changed", [False, True])
def test_compiled_bucket_owns_outputs_and_preserves_per_name_wire(compiled_stage, changed):
    layout = compiled_stage.layout
    old, staging = _inputs(layout, changed=changed)
    original, incoming = old.clone(), staging.clone()
    expected = {
        name: int(torch.count_nonzero(dict(layout.views(old))[name] != tensor).item())
        for name, tensor in layout.views(staging)
    }

    prepared = compiled_stage.prepare(staging, old)
    assert prepared.changed_counts == tuple(expected.values())
    assert prepared.changed_bytes == sum(expected.values())
    assert prepared.copied_unchanged_bytes == (128 if changed else 0)
    assert prepared.copied_padding_bytes == (63 if changed else 0)
    if changed:
        assert prepared.snapshot.data_ptr() not in (staging.data_ptr(), old.data_ptr())
        assert prepared.xor.data_ptr() not in (staging.data_ptr(), old.data_ptr())
    else:
        assert prepared.snapshot is old and prepared.xor is None

    staging.fill_(0x42)  # The caller can recycle the pinned lease immediately.
    torch.testing.assert_close(old, original)
    torch.testing.assert_close(prepared.snapshot, incoming)
    decoder = zstandard.ZstdDecompressor()
    actual_names = []
    for name, snapshot, diff, count in prepared.changed_views():
        actual_names.append(name)
        assert count == expected[name]
        compressed = zstandard.ZstdCompressor(level=1).compress(diff.numpy())
        decoded = np.frombuffer(decoder.decompress(compressed), dtype=np.uint8)
        previous = dict(layout.views(original))[name].numpy()
        np.testing.assert_array_equal(previous ^ decoded, snapshot.numpy())
        assert checksum("adler32", snapshot.numpy()) == checksum("adler32", dict(layout.views(incoming))[name].numpy())
    assert actual_names == [name for name, count in expected.items() if count]


def test_compiled_bucket_reuses_no_change_snapshot_after_a_successful_update(compiled_stage):
    old, staging = _inputs(compiled_stage.layout, changed=True)
    first = compiled_stage.prepare(staging, old)
    second = compiled_stage.prepare(staging, first.snapshot)
    assert second.snapshot is first.snapshot and second.xor is None
    assert not second.changed_bytes and not second.copied_unchanged_bytes


def test_one_compiled_stage_can_prepare_distinct_concurrent_buckets(compiled_stage):
    def run(changed):
        old, staging = _inputs(compiled_stage.layout, changed=changed)
        prepared = compiled_stage.prepare(staging, old)
        expected = staging.clone()
        staging.fill_(0)
        torch.testing.assert_close(prepared.snapshot, expected)
        return prepared.changed_bytes

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, [False, True, False, True])) == [0, 65281, 0, 65281]


def test_compact_scalar_layout_does_not_pad_scalars_to_large_weight_rows():
    layout = PackedDeltaLayout((("scale0", 4), ("scale1", 4), ("scale2", 4)), block_bytes=4)
    old, staging = layout.allocate(), layout.allocate()
    dict(layout.views(staging))["scale1"][3] = 255
    result = CpuDeltaPreparer(layout, use_compile=False).prepare(staging, old)
    assert layout.padded_bytes == 12
    assert result.changed_counts == (0, 1, 0)
    assert result.copied_unchanged_bytes == 8 and result.copied_padding_bytes == 0


def test_reassigning_a_staging_lease_resets_only_padding():
    layout = PackedDeltaLayout((("full_row", 64), ("tail", 1)), block_bytes=64)
    staging = torch.full((layout.padded_bytes,), 255, dtype=torch.uint8)
    assert layout.reset_padding(staging) == 63
    old = layout.allocate()
    for _, tensor in layout.views(old):
        tensor.fill_(255)
    torch.testing.assert_close(staging, old)
    prepared = CpuDeltaPreparer(layout, use_compile=False).prepare(staging, old)
    assert prepared.snapshot is old and prepared.changed_counts == (0, 0)


def test_counts_reject_gross_padding_corruption():
    layout = PackedDeltaLayout((("tail", 1),), block_bytes=64)
    old = layout.allocate()
    with pytest.raises(ValueError, match="check staging padding/layout"):
        CpuDeltaPreparer(layout, use_compile=False).prepare(torch.ones_like(old), old)


@pytest.mark.parametrize(
    "entries,block_bytes", [((("a", 67), ("b", 129), ("c", 257), ("d", 513)), 64), ((("a", 4),), 4)]
)
def test_actor_inference_allocations_match_warmed_worker_dispatch(entries, block_bytes):
    from torch._dynamo.utils import counters

    with torch.inference_mode():
        layout = PackedDeltaLayout(entries, block_bytes=block_bytes)
        stage = CpuDeltaPreparer(layout)
        old, staging = layout.allocate(), layout.allocate()
        dict(layout.views(staging))[entries[-1][0]][0] = 255
    assert not torch.is_inference(old) and not torch.is_inference(staging)
    assert not torch.is_inference(layout._block_ends)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.submit(stage.warmup).result()
        graphs = counters["stats"]["unique_graphs"]
        results = [future.result() for future in [pool.submit(stage.prepare, staging, old) for _ in range(4)]]
    assert counters["stats"]["unique_graphs"] == graphs
    assert all(result.changed_counts == (0,) * (len(entries) - 1) + (1,) for result in results)


def test_warmup_uses_actual_layout_with_caller_snapshot_and_shared_scratch():
    from torch._dynamo.utils import counters

    layout = PackedDeltaLayout((("a", 67), ("b", 129), ("c", 257), ("d", 513)), block_bytes=64)
    stage = CpuDeltaPreparer(layout)
    old = layout.allocate()
    scratch = torch.full((layout.padded_bytes + 64,), 0x42, dtype=torch.uint8)
    staging = scratch.narrow(0, 32, layout.padded_bytes)
    staging.zero_()
    dict(layout.views(staging))["c"][0] = 1
    original_old, original_scratch = old.clone(), scratch.clone()

    with ThreadPoolExecutor(max_workers=4) as pool:
        with (
            patch.object(PackedDeltaLayout, "allocate", side_effect=AssertionError("unexpected bucket allocation")),
            patch.object(stage, "_materialize", wraps=stage._materialize) as materialize,
        ):
            pool.submit(stage.warmup, old_snapshot=old, staging=staging).result()
        materialize.assert_called_once_with(staging, old)
        graphs = counters["stats"]["unique_graphs"]
        results = [future.result() for future in [pool.submit(stage.prepare, staging, old) for _ in range(4)]]

    assert counters["stats"]["unique_graphs"] == graphs
    assert all(result.changed_counts == (0, 0, 1, 0) for result in results)
    torch.testing.assert_close(old, original_old)
    torch.testing.assert_close(scratch, original_scratch)


def test_default_block_counts_remain_exact_above_float32_integer_range():
    size = (1 << 24) + 17
    layout = PackedDeltaLayout((("weight", size),))
    stage = CpuDeltaPreparer(layout)
    old, staging = layout.allocate(), layout.allocate()
    dict(layout.views(staging))["weight"].fill_(1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(stage.warmup).result()
        result = pool.submit(stage.prepare, staging, old).result()
    assert result.changed_counts == (size,)
    assert result.copied_padding_bytes == layout.padded_bytes - size


@pytest.mark.parametrize("entries", [(), (("empty0", 0), ("empty1", 0))])
def test_empty_bucket_preserves_snapshot_without_materialization(entries):
    layout = PackedDeltaLayout(entries)
    stage = CpuDeltaPreparer(layout, use_compile=False)
    old = layout.allocate()
    with patch.object(stage, "_materialize", side_effect=AssertionError("unexpected allocation")):
        result = stage.prepare(layout.allocate(), old)
    assert result.snapshot is old and result.xor is None and not result.changed_bytes


def test_failed_preparation_and_native_handoff_leave_active_base_unchanged(compiled_stage):
    old, staging = _inputs(compiled_stage.layout, changed=True)
    original = old.clone()
    with patch.object(compiled_stage, "_materialize", side_effect=RuntimeError("allocation failed")):
        with pytest.raises(RuntimeError, match="allocation failed"):
            compiled_stage.prepare(staging, old)
    torch.testing.assert_close(old, original)
    prepared = compiled_stage.prepare(staging, old)
    with patch("zstandard.ZstdCompressor", side_effect=RuntimeError("compression failed")):
        with pytest.raises(RuntimeError, match="compression failed"):
            for _, _, diff, _ in prepared.changed_views():
                zstandard.ZstdCompressor(level=1).compress(diff.numpy())
    torch.testing.assert_close(old, original)


def test_layout_guards_exact_accounting_and_staging_contract():
    with pytest.raises(ValueError, match="positive int32"):
        PackedDeltaLayout((("weight", 1),), block_bytes=1 << 31)
    with pytest.raises(ValueError, match="int64 byte accounting"):
        PackedDeltaLayout((("weight", 1 << 63),))
    with pytest.raises(ValueError, match="unique"):
        PackedDeltaLayout((("weight", 1), ("weight", 2)))
    layout = PackedDeltaLayout((("weight", 1),))
    stage = CpuDeltaPreparer(layout, use_compile=False)
    old = layout.allocate()
    with pytest.raises(ValueError, match="independent storage"):
        stage.prepare(old, old)
    with pytest.raises(ValueError, match="contiguous CPU uint8"):
        stage.prepare(torch.zeros(layout.padded_bytes, dtype=torch.int32), old)
    with pytest.raises(ValueError, match="independent storage"):
        stage.warmup(old_snapshot=old, staging=old)
    with pytest.raises(ValueError, match="contiguous CPU uint8"):
        stage.warmup(old_snapshot=old, staging=torch.zeros(layout.padded_bytes, dtype=torch.int32))
