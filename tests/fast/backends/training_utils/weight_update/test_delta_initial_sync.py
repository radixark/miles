"""The shared initial-sync option also applies to the existing disk-delta transport."""

from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import safetensors.numpy
import torch
import zstandard
from tests.fast.backends.training_utils.weight_update.test_disk_delta_engine_calls import _RecordingApiClient

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


@pytest.mark.parametrize("initial_sync", [False, True])
def test_initial_sync_encodes_loaded_trainer_from_hf_baseline(tmp_path, initial_sync):
    """The opt-in first call publishes the same catch-up delta as the next ordinary call."""
    old = np.array([1, 2, 3, 4], dtype=np.uint8)
    new = np.array([5, 6, 7, 8], dtype=np.uint8)
    safetensors.numpy.save_file({"w": old}, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
            check_weight_update_equal=False,
            update_weight_delta_initial_sync=initial_sync,
        )
    )
    calls = []
    protocol.rollout_engines = [_RecordingApiClient(calls)]
    protocol.is_sender = True

    def buckets(materialize):
        assert materialize
        yield [("w", torch.from_numpy(new))]

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = lambda values, value, **kwargs: values.__setitem__(0, value)
        assert protocol.begin_sync(1, buckets) is initial_sync
        np.testing.assert_array_equal(protocol._snapshot["w"], old)
        if not initial_sync:
            assert protocol._pool is None
            assert protocol.begin_sync(1, buckets) is True
        protocol._use_pinned = False  # CPU test; production's CUDA staging is covered separately.
        for bucket in buckets(materialize=True):
            protocol.send_bucket(bucket)
        protocol.after_base_weights()
        protocol._write_delta_files(1)
    delta = safetensors.numpy.load_file(next((tmp_path / "delta" / "weight_v000001").glob("*.safetensors")))
    decoded = np.frombuffer(zstandard.ZstdDecompressor().decompress(delta["w"].tobytes()), dtype=np.uint8)
    np.testing.assert_array_equal(old ^ decoded, new)
    np.testing.assert_array_equal(protocol._snapshot["w"], new)
