from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import safetensors.numpy
import safetensors.torch
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


def _protocol(tmp_path, *, encoding="xor"):
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            num_experts=4,
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding=encoding,
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )
    protocol.is_sender = False
    return protocol


def test_dense_and_iterators_without_owner_hooks_keep_existing_path(tmp_path):
    protocol = _protocol(tmp_path)
    protocol.bind_iterator(SimpleNamespace())
    assert not protocol._local_experts
    protocol.args.num_experts = None
    consumer = []
    protocol.bind_iterator(SimpleNamespace(set_local_expert_consumer=consumer.append))
    assert not consumer and not protocol._local_experts


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_non_sender_owns_canonical_snapshots_and_publishes_cpu_deltas(tmp_path, encoding):
    """BF16, FP8 and packed layouts share the ordinary CPU protocol on a non-sender owner."""
    canonical = {
        "bf16.weight": torch.zeros(8, dtype=torch.bfloat16),
        "fp8.weight": torch.zeros(8, dtype=torch.float8_e4m3fn),
        "packed.weight": torch.zeros(8, dtype=torch.uint8),
        "a.weight_scale_2": torch.ones(()),
        "b.weight_scale_2": torch.ones(()),
        "shared.weight": torch.zeros(8, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(canonical, tmp_path / "model.safetensors")
    owned = {name: tensor for name, tensor in canonical.items() if name != "shared.weight"}
    protocol = _protocol(tmp_path, encoding=encoding)
    consumer = []
    protocol.bind_iterator(SimpleNamespace(set_local_expert_consumer=consumer.append))
    assert len(consumer) == 1

    def buckets(*, materialize):
        assert not materialize
        # Deliberately differ from the checkpoint: baseline must use canonical bytes.
        consumer[0]([(name, torch.ones_like(tensor)) for name, tensor in owned.items()])
        return iter(())

    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        dist.get_rank.return_value = 1
        assert not protocol.begin_sync(1, buckets)
    assert set(protocol._snapshot) == set(owned)
    read = make_tensor_reader(str(tmp_path))
    received = {name: read(name).copy() for name in owned}
    for name in owned:
        np.testing.assert_array_equal(protocol._snapshot[name], received[name])

    for version, value in enumerate((1, 2, 2), start=1):
        with patch(f"{_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")):
            assert protocol.begin_sync(version, buckets)
        emitted = {name: torch.full_like(tensor, value) for name, tensor in owned.items()}
        consumer[0](list(emitted.items()))
        with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group", return_value=None):
            protocol.after_base_weights()
            dist.get_rank.return_value = 1
            dist.get_world_size.return_value = 2

            def gather(output, local, **kwargs):
                output[:] = [0 if isinstance(local, int) else {}, local]

            dist.all_gather_object.side_effect = gather
            protocol._write_delta_files(version)

        for name, compressed in protocol._delta.items():
            delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(compressed), dtype=np.uint8)
            if encoding == "xor":
                received[name] ^= delta
            else:
                count = int(delta[:4].view("<u4")[0])
                positions = delta[4 : 4 + count * 4].view("<u4")
                received[name][positions] = delta[4 + count * 4 :]
            assert checksum("adler32", received[name]) == protocol._checksums[name]
        for name, tensor in emitted.items():
            expected = tensor.reshape(-1).view(torch.uint8).numpy()
            np.testing.assert_array_equal(received[name], expected)
            np.testing.assert_array_equal(protocol._snapshot[name], expected)
        assert protocol.total_bytes == sum(value.nbytes for value in received.values())
        files = list((tmp_path / "deltas" / f"weight_v{version:06d}").glob("*.safetensors"))
        if version < 3:
            assert len(files) == 1
            assert set(safetensors.numpy.load_file(files[0])) == set(protocol._delta)
        else:
            assert not files and not protocol._delta and protocol.changed_bytes == 0


@pytest.mark.parametrize("phase", ["baseline", "update"])
def test_owner_failure_drains_iteration_before_collective_error(tmp_path, phase):
    safetensors.torch.save_file({"weight": torch.zeros(4)}, tmp_path / "model.safetensors")
    protocol = _protocol(tmp_path)
    consumer = []
    protocol.bind_iterator(SimpleNamespace(set_local_expert_consumer=consumer.append))
    observed = []

    def buckets(*, materialize):
        for index in range(3):
            consumer[0]([("weight", torch.ones(4))])
            observed.append(index)
        return iter(())

    def fail_read(*args, **kwargs):
        raise OSError("owner checkpoint read failed")

    def fail_worker(*args, **kwargs):
        raise RuntimeError("owner compressor failed")

    with (
        patch(f"{_MODULE}.dist") as dist,
        patch(f"{_MODULE}.get_gloo_group", return_value=None),
        patch(f"{_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")),
    ):
        dist.get_rank.return_value = 1
        dist.get_world_size.return_value = 2

        def gather_errors(output, message, **kwargs):
            assert observed == [0, 1, 2]
            assert protocol._pool is None
            output[:] = [None, message]

        dist.all_gather_object.side_effect = gather_errors
        with pytest.raises(RuntimeError, match=f"{phase} validation failed on rank 1"):
            if phase == "baseline":
                with patch(f"{_MODULE}.make_tensor_reader", return_value=fail_read):
                    protocol.begin_sync(1, buckets)
            else:
                protocol._baseline_captured = True
                protocol._snapshot = {"weight": make_tensor_reader(str(tmp_path))("weight")}
                protocol.begin_sync(1, buckets)
                with patch.object(protocol, "_diff_and_compress_batch", side_effect=fail_worker):
                    buckets(materialize=False)
                    protocol.after_base_weights()
    assert not list((tmp_path / "deltas").rglob("*.safetensors"))


@pytest.mark.parametrize("failure", ["duplicate", "missing"])
def test_changed_owner_inventory_cannot_publish(tmp_path, failure):
    canonical = {name: torch.zeros(4) for name in ("first", "second")}
    safetensors.torch.save_file(canonical, tmp_path / "model.safetensors")
    protocol = _protocol(tmp_path)
    consumer = []
    protocol.bind_iterator(SimpleNamespace(set_local_expert_consumer=consumer.append))
    protocol._baseline_captured = True
    read = make_tensor_reader(str(tmp_path))
    protocol._snapshot = {name: read(name) for name in canonical}
    with patch(f"{_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")):
        protocol.begin_sync(1, None)
    consumer[0]([("first", torch.ones(4))])
    if failure == "duplicate":
        consumer[0]([("first", torch.ones(4)), ("second", torch.ones(4))])
    with patch(f"{_MODULE}.dist") as dist, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        dist.get_world_size.return_value = 2
        dist.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(
            slice(None), [None, message]
        )
        with pytest.raises(RuntimeError, match="update validation failed on rank 1"):
            protocol.after_base_weights()
    assert protocol._pool is None
    assert not list((tmp_path / "deltas").rglob("*.safetensors"))


@pytest.mark.parametrize("largest_bytes, expected_buffers", [(0, 0), (3 << 30, 2), (9 << 30, 1)])
def test_owner_staging_uses_a_local_actor_share(tmp_path, largest_bytes, expected_buffers):
    protocol = _protocol(tmp_path)
    protocol.args.actor_num_gpus_per_node = 4
    protocol.bind_iterator(SimpleNamespace(set_local_expert_consumer=lambda consumer: None))
    protocol._snapshot = {"large": SimpleNamespace(nbytes=largest_bytes)} if largest_bytes else {}
    with patch(f"{_MODULE}.torch.empty") as allocate:
        protocol._begin_encode(1)
    try:
        assert protocol._pool._max_workers == max(1, NUM_WORKERS // 4)
        assert allocate.call_count == expected_buffers
        assert protocol._free_q.qsize() == expected_buffers
    finally:
        protocol._pool.shutdown()
