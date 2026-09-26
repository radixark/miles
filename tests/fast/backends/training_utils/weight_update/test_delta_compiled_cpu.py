from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import zstandard

from miles.backends.training_utils.weight_update.packed_delta import PackedDeltaEncoder, _split_bucket
from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.utils.delta_preparation import CpuDeltaPreparer
from miles.utils.disk_delta import checksum

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"
_PACKED_MODULE = "miles.backends.training_utils.weight_update.packed_delta"


@pytest.fixture
def make_encoder(monkeypatch):
    # Compiled graph correctness/concurrency is covered by test_delta_preparation;
    # these small tests exercise encoder ownership and lifecycle deterministically.
    monkeypatch.setattr(f"{_PACKED_MODULE}.NUM_WORKERS", 2)
    monkeypatch.setattr(f"{_PACKED_MODULE}._STAGING_BYTES", 32768)
    monkeypatch.setattr(f"{_PACKED_MODULE}._INFLIGHT_BYTES", 8192)
    monkeypatch.setattr(
        f"{_PACKED_MODULE}.CpuDeltaPreparer", lambda layout: CpuDeltaPreparer(layout, use_compile=False)
    )

    def make(buckets):
        snapshots = {
            name: tensor.reshape(-1).view(torch.uint8).numpy().copy() for bucket in buckets for name, tensor in bucket
        }
        encoder = PackedDeltaEncoder(snapshots, "adler32")
        for bucket in buckets:
            encoder.capture(bucket)
        encoder.initialize()
        return encoder

    return make


def test_packed_update_promotes_all_views_and_reuses_no_change_snapshot(make_encoder):
    bucket = [
        ("changed", torch.arange(67, dtype=torch.uint8)),
        ("unchanged", torch.ones(65, dtype=torch.uint8)),
        ("expert.weight_scale_2", torch.tensor(1.25)),
    ]
    encoder = make_encoder([bucket])
    original = {name: value.copy() for name, value in encoder.snapshots.items()}
    previous_weights = encoder._buckets[("changed", "unchanged")].snapshot
    previous_scalars = encoder._buckets[("expert.weight_scale_2",)].snapshot
    bucket[0][1][-1] ^= 255
    encoder.begin()
    encoder.submit(bucket)
    assert encoder.finish() is None
    assert encoder._pool is None and encoder._inflight_bytes == 0
    assert encoder.changed_bytes == 1 and encoder.total_bytes == 67 + 65 + 4
    assert set(encoder.deltas) == {"changed"}
    decoded = np.frombuffer(zstandard.ZstdDecompressor().decompress(encoder.deltas["changed"]), dtype=np.uint8)
    np.testing.assert_array_equal(original["changed"] ^ decoded, encoder.snapshots["changed"])
    assert encoder.checksums["changed"] == checksum("adler32", encoder.snapshots["changed"])
    weights = encoder._buckets[("changed", "unchanged")].snapshot
    assert weights is not previous_weights
    assert encoder._buckets[("expert.weight_scale_2",)].snapshot is previous_scalars
    assert encoder.snapshots["unchanged"].__array_interface__["data"][0] == weights.data_ptr() + 4096
    assert encoder.copied_unchanged_bytes == 65
    assert encoder.copied_padding_bytes == 8192 - 132
    # Poison all returned leases: neither the committed slab nor its views alias them.
    leases = [encoder._free.get_nowait() for _ in range(encoder._free.qsize())]
    for lease in leases:
        lease.buffer.fill_(42)
        encoder._free.put(lease)
    for name, tensor in bucket:
        np.testing.assert_array_equal(encoder.snapshots[name], tensor.reshape(-1).view(torch.uint8).numpy())
    encoder.begin()
    encoder.submit(bucket)
    assert encoder.finish() is None
    assert not encoder.deltas and not encoder.changed_bytes
    assert encoder._buckets[("changed", "unchanged")].snapshot is weights


@pytest.mark.parametrize("failure", ["shape", "omission", "duplicate", "compression"])
def test_failure_drains_workers_and_does_not_promote_failed_bucket(make_encoder, failure):
    bucket = [("weight", torch.zeros(65, dtype=torch.uint8))]
    encoder = make_encoder([bucket])
    previous = encoder._buckets[("weight",)].snapshot
    slots = encoder._free.qsize()
    encoder.begin()
    with (
        patch(f"{_PACKED_MODULE}.zstandard.ZstdCompressor", side_effect=RuntimeError("compress failed"))
        if failure == "compression"
        else patch(f"{_PACKED_MODULE}.logger")
    ):
        if failure == "shape":
            encoder.submit([("weight", torch.zeros(66, dtype=torch.uint8))])
        elif failure != "omission":
            incoming = [("weight", torch.ones(65, dtype=torch.uint8))]
            encoder.submit(incoming if failure == "compression" else bucket)
            if failure == "duplicate":
                encoder.submit(bucket)
        error = encoder.finish()
    assert error is not None
    assert encoder._pool is None and not encoder._inflight and encoder._inflight_bytes == 0
    assert encoder._free.qsize() == slots
    assert encoder._buckets[("weight",)].snapshot is previous
    assert not np.any(encoder.snapshots["weight"])


def test_bounded_inflight_drains_before_exceeding_budget(make_encoder):
    buckets = [[(f"weight{index}", torch.zeros(5000, dtype=torch.uint8))] for index in range(3)]
    encoder = make_encoder(buckets)
    encoder.begin()
    for bucket in buckets:
        encoder.submit(bucket)
        assert encoder._inflight_bytes <= 8192
    assert encoder.finish() is None
    assert encoder.total_bytes == 15000


def _args(tmp_path, encoding="xor"):
    return Namespace(
        hf_checkpoint="/fake/hf",
        update_weight_disk_dir=str(tmp_path / "delta"),
        update_weight_delta_encoding=encoding,
        update_weight_delta_checksum="adler32",
        update_weight_delta_cpu_backend="torch-compile",
        custom_update_weight_post_write_path=None,
    )


def test_compilation_failure_uses_existing_baseline_error_coordination(tmp_path):
    protocol = UpdateWeightFromDiskDelta(_args(tmp_path))
    protocol.is_sender = True
    observed = []

    def buckets(*, materialize):
        for index in range(2):
            observed.append(index)
            yield [(f"weight{index}", torch.zeros(1, dtype=torch.uint8))]

    def fail_compile():
        assert observed == [0, 1]
        raise RuntimeError("compiler unavailable")

    with (
        patch(f"{_MODULE}.dist") as distributed,
        patch(f"{_MODULE}.get_gloo_group", return_value=None),
        patch(f"{_MODULE}.make_tensor_reader", return_value=lambda *args, **kwargs: np.zeros(1, dtype=np.uint8)),
        patch.object(protocol._packed, "initialize", side_effect=fail_compile),
    ):
        distributed.get_rank.return_value = 1
        distributed.get_world_size.return_value = 2
        distributed.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(1, value)
        with pytest.raises(RuntimeError, match="baseline validation failed on rank 1.*compiler unavailable"):
            protocol._capture_baseline(buckets)


def test_non_sender_finishes_empty_encoder_before_error_coordination(tmp_path):
    protocol = UpdateWeightFromDiskDelta(_args(tmp_path))
    protocol.is_sender = False
    protocol._begin_encode(1)
    with patch.object(protocol, "_raise_packed_error") as coordinate:
        protocol.after_base_weights()
    coordinate.assert_called_once_with(None)
    assert protocol._packed._pool is None
    assert protocol.changed_bytes == protocol.total_bytes == 0


def test_compiled_backend_rejects_overwrite_encoding(tmp_path):
    with pytest.raises(ValueError, match="requires XOR encoding"):
        UpdateWeightFromDiskDelta(_args(tmp_path, encoding="overwrite"))


def test_grouping_preserves_order_and_keeps_scalars_out_of_weight_groups(monkeypatch):
    monkeypatch.setattr(f"{_PACKED_MODULE}._GROUP_BYTES", 8192)
    tensors = [
        ("a", torch.zeros(4097, dtype=torch.uint8)),
        ("s.weight_scale_2", torch.tensor(1.0)),
        ("b", torch.zeros(8193, dtype=torch.uint8)),
        ("c", torch.zeros(1, dtype=torch.uint8)),
        ("d", torch.zeros(4096, dtype=torch.uint8)),
    ]
    groups = list(_split_bucket(tensors))
    assert [(tuple(name for name, _ in group), block) for group, block in groups] == [
        (("a",), 4096),
        (("b",), 4096),
        (("c", "d"), 4096),
        (("s.weight_scale_2",), 4),
    ]


def test_capture_rejects_partially_overlapping_names(make_encoder):
    encoder = make_encoder([[("a", torch.zeros(1, dtype=torch.uint8))]])
    encoder.snapshots["b"] = np.zeros(1, dtype=np.uint8)
    with pytest.raises(ValueError, match="names repeat"):
        encoder.capture([("a", torch.zeros(1, dtype=torch.uint8)), ("b", torch.zeros(1, dtype=torch.uint8))])


def test_compact_scalars_use_one_copy_and_keep_per_name_wire(make_encoder):
    bucket = [("a.weight_scale_2", torch.tensor(1.0)), ("b.weight_scale_2", torch.tensor(2.0))]
    encoder = make_encoder([bucket])
    before = {name: value.copy() for name, value in encoder.snapshots.items()}
    bucket[1][1].add_(1)
    encoder.begin()
    with patch(f"{_PACKED_MODULE}.torch.stack", wraps=torch.stack) as stack:
        encoder.submit(bucket)
        assert encoder.finish() is None
    stack.assert_called_once()
    assert set(encoder.deltas) == {"b.weight_scale_2"}
    decoded = np.frombuffer(
        zstandard.ZstdDecompressor().decompress(encoder.deltas["b.weight_scale_2"]), dtype=np.uint8
    )
    np.testing.assert_array_equal(before["b.weight_scale_2"] ^ decoded, encoder.snapshots["b.weight_scale_2"])
    np.testing.assert_array_equal(before["a.weight_scale_2"], encoder.snapshots["a.weight_scale_2"])
