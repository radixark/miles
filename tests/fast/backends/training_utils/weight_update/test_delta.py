from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.torch
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.disk_delta import checksum, make_tensor_reader, overwrite_encode

_DELTA_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


class _RejectingApiClient:
    def __init__(self, calls: list[tuple[str, dict]], failing_method: str) -> None:
        self._calls = calls
        self._failing_method = failing_method

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            if name == self._failing_method:
                return {"success": False, "error_message": "engine rejected the weights"}
            return {"success": True}

        return method


class TestPostWriteHookConstruction:
    def test_configured_post_write_hook_is_loaded_from_function_registry(self, tmp_path: Path) -> None:
        """A configured post-write path becomes the hook, resolved through the shared function registry."""
        hook = object()
        args = Namespace(
            update_weight_disk_dir=str(tmp_path / "delta"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="xxh3",
            custom_update_weight_post_write_path="miles_plugins.example:upload_delta",
        )

        with patch("miles.utils.function_registry.load_function", return_value=hook) as load_function:
            protocol = UpdateWeightFromDiskDelta(args)

        load_function.assert_called_once_with("miles_plugins.example:upload_delta")
        assert protocol._post_write_hook is hook


class TestCanonicalCheckpointLayout:
    @staticmethod
    def _protocol(checkpoint: Path) -> UpdateWeightFromDiskDelta:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(hf_checkpoint=str(checkpoint))
        return protocol

    def test_rejects_a_missing_quantization_step(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"expert.weight": torch.zeros((2, 3), dtype=torch.uint8)},
            tmp_path / "model.safetensors",
        )

        with pytest.raises(ValueError, match="must be produced by the model's weight converter"):
            self._protocol(tmp_path)._match_checkpoint_layout(
                "expert.weight", torch.ones((2, 3), dtype=torch.bfloat16)
            )

    def test_rejects_shape_and_name_mismatches(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"weight": torch.zeros((2, 3), dtype=torch.bfloat16)},
            tmp_path / "model.safetensors",
        )
        protocol = self._protocol(tmp_path)

        with pytest.raises(ValueError, match="has shape"):
            protocol._match_checkpoint_layout("weight", torch.ones((3, 2), dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="absent from the canonical checkpoint"):
            protocol._match_checkpoint_layout("missing", torch.ones((2, 3), dtype=torch.bfloat16))

    def test_rejects_a_float_cast_into_fp8_storage(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"weight_scale": torch.zeros((2, 1), dtype=torch.float8_e4m3fn)},
            tmp_path / "model.safetensors",
        )

        with pytest.raises(ValueError, match="must be produced by the model's weight converter"):
            self._protocol(tmp_path)._match_checkpoint_layout("weight_scale", torch.ones((2, 1), dtype=torch.float32))


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_nvfp4_bytes_roundtrip_across_syncs(tmp_path: Path, encoding: str) -> None:
    """Packed weights, FP8 scales, scalar scales and float casts preserve the checkpoint ABI."""
    tensors = {
        "expert.weight": torch.zeros((2, 4), dtype=torch.uint8),
        "expert.weight_scale": torch.zeros((2, 1), dtype=torch.float8_e4m3fn),
        "expert.weight_scale_2": torch.zeros((), dtype=torch.float32),
        "router": torch.zeros((2, 3), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding=encoding,
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )
    protocol.is_sender = True
    read = make_tensor_reader(str(tmp_path))
    protocol._snapshot = {name: read(name) for name in tensors}
    received = {name: value.copy() for name, value in protocol._snapshot.items()}

    for version, value in enumerate((1.001, 2.125, 2.125), start=1):
        emitted = {
            name: torch.full_like(tensor, value, dtype=torch.float32 if name == "router" else tensor.dtype)
            for name, tensor in tensors.items()
        }
        with patch(f"{_DELTA_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")):
            protocol._begin_encode(version)
        protocol.send_bucket(list(emitted.items()))
        with (
            patch(f"{_DELTA_MODULE}.dist") as distributed,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
        ):
            protocol.after_base_weights()
            distributed.all_reduce.assert_called_once()
            distributed.all_gather_object.assert_not_called()

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
            expected = tensor.to(tensors[name].dtype).reshape(-1).view(torch.uint8).numpy()
            np.testing.assert_array_equal(received[name], expected)
            np.testing.assert_array_equal(protocol._snapshot[name], expected)
        assert protocol.total_bytes == sum(value.nbytes for value in received.values())
        if version == 3:
            assert protocol.changed_bytes == 0
            assert protocol._delta == {}
        else:
            assert protocol.changed_bytes > 0


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
@pytest.mark.parametrize("changed", [False, True])
def test_pinned_diff_preserves_snapshot_after_buffer_reuse(encoding: str, changed: bool) -> None:
    old = np.array([1, 2, 3, 4], dtype=np.uint8)
    old.setflags(write=False)
    incoming = old.copy()
    if changed:
        incoming[1] = 7
    buf = torch.from_numpy(np.pad(incoming, (0, 2)))
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._snapshot = {"weight": old}
    protocol.delta_encoding = encoding
    protocol.checksum_algorithm = "adler32"
    protocol._free_q = MagicMock()
    # A producer may overwrite the entire buffer as soon as the worker returns it.
    protocol._free_q.put.side_effect = lambda returned: returned.fill_(255)

    with patch(f"{_DELTA_MODULE}.overwrite_encode", wraps=overwrite_encode) as encode:
        name, new, compressed, digest, count = protocol._diff_and_compress("weight", buf, incoming.nbytes, True)
        if not changed:
            encode.assert_not_called()

    protocol._free_q.put.assert_called_once_with(buf)
    assert name == "weight"
    assert count == int(changed)
    np.testing.assert_array_equal(old, [1, 2, 3, 4])
    np.testing.assert_array_equal(new, incoming)
    if changed:
        expected = incoming ^ old if encoding == "xor" else overwrite_encode(incoming, incoming != old)
        actual = np.frombuffer(zstandard.ZstdDecompressor().decompress(compressed), dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
        assert digest == checksum("adler32", incoming)
    else:
        assert new is old
        assert compressed is None and digest is None


@pytest.mark.parametrize("failure", ["np.count_nonzero", "np.copyto", "zstandard.ZstdCompressor"])
def test_pinned_diff_returns_buffer_on_worker_failure(failure: str) -> None:
    old = np.zeros(4, dtype=np.uint8)
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._snapshot = {"weight": old}
    protocol.delta_encoding = "xor"
    protocol.checksum_algorithm = "adler32"
    protocol._free_q = MagicMock()
    buf = torch.ones(4, dtype=torch.uint8)

    with patch(f"{_DELTA_MODULE}.{failure}", side_effect=RuntimeError("worker failed")):
        with pytest.raises(RuntimeError, match="worker failed"):
            protocol._diff_and_compress("weight", buf, old.nbytes, True)

    protocol._free_q.put.assert_called_once_with(buf)
    np.testing.assert_array_equal(protocol._snapshot["weight"], np.zeros(4, dtype=np.uint8))


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_scalar_batch_preserves_mixed_bucket_bytes_after_buffer_reuse(tmp_path: Path, encoding: str) -> None:
    tensors = {
        "packed": torch.zeros(16, dtype=torch.uint8),
        "a.weight_scale_2": torch.zeros((), dtype=torch.float32),
        "bias": torch.zeros(2, dtype=torch.float32),
        "b.weight_scale_2": torch.zeros((), dtype=torch.float32),
        "c.weight_scale_2": torch.zeros((), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding=encoding,
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )
    protocol.is_sender = True
    read = make_tensor_reader(str(tmp_path))
    protocol._snapshot = {name: read(name) for name in tensors}
    received = {name: value.copy() for name, value in protocol._snapshot.items()}
    emitted = dict(tensors)
    emitted["packed"] = torch.ones_like(tensors["packed"])
    emitted["b.weight_scale_2"] = torch.tensor(1.25)
    emitted["c.weight_scale_2"] = torch.tensor(2.5)
    expected = {name: tensor.reshape(-1).view(torch.uint8).numpy().copy() for name, tensor in emitted.items()}
    original_empty = torch.empty

    def cpu_buffer(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    for version in (1, 2):
        with patch(f"{_DELTA_MODULE}.torch.empty", side_effect=cpu_buffer):
            protocol._begin_encode(version)
        original_put = protocol._free_q.put

        def reuse_buffer(buffer, put_buffer=original_put):
            buffer.fill_(255)
            put_buffer(buffer)

        with (
            patch.object(protocol._pool, "submit", wraps=protocol._pool.submit) as submit,
            patch.object(protocol._free_q, "put", side_effect=reuse_buffer) as put,
            patch(f"{_DELTA_MODULE}.torch.cuda.current_stream") as stream,
            patch(f"{_DELTA_MODULE}.dist"),
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
        ):
            protocol.send_bucket(list(emitted.items()))
            protocol.after_base_weights()
        assert [call.args[1] for call in submit.call_args_list] == [
            ["packed"],
            ["a.weight_scale_2", "b.weight_scale_2", "c.weight_scale_2"],
            ["bias"],
        ]
        assert stream.return_value.synchronize.call_count == put.call_count == 3
        changed_bytes = sum(int(np.count_nonzero(received[name] != value)) for name, value in expected.items())
        assert protocol.changed_bytes == changed_bytes
        assert protocol.total_bytes == sum(value.nbytes for value in expected.values())
        for name, compressed in protocol._delta.items():
            delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(compressed), dtype=np.uint8)
            if encoding == "xor":
                received[name] ^= delta
            else:
                count = int(delta[:4].view("<u4")[0])
                positions = delta[4 : 4 + count * 4].view("<u4")
                received[name][positions] = delta[4 + count * 4 :]
            assert checksum("adler32", received[name]) == protocol._checksums[name]
        for name, value in expected.items():
            np.testing.assert_array_equal(received[name], value)
            np.testing.assert_array_equal(protocol._snapshot[name], value)
        if version == 2:
            assert protocol._delta == {}


@pytest.mark.parametrize("failure", ["np.copyto", "zstandard.ZstdCompressor"])
def test_scalar_batch_returns_buffer_on_failure(failure: str) -> None:
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._snapshot = {name: np.zeros(4, dtype=np.uint8) for name in ("a", "b")}
    protocol.delta_encoding = "xor"
    protocol.checksum_algorithm = "adler32"
    protocol._free_q = MagicMock()
    buf = torch.ones(8, dtype=torch.uint8)

    with patch(f"{_DELTA_MODULE}.{failure}", side_effect=RuntimeError("worker failed")):
        with pytest.raises(RuntimeError, match="worker failed"):
            protocol._diff_and_compress_batch(["a", "b"], buf, 8, True)

    protocol._free_q.put.assert_called_once_with(buf)
    assert all(not array.any() for array in protocol._snapshot.values())


@pytest.mark.parametrize("is_sender", [True, False])
def test_update_validation_failure_drains_stream_and_prevents_publication(tmp_path: Path, is_sender: bool) -> None:
    """Both the failing sender and a non-sender finish the iterator and raise before finalize."""
    tensor = torch.zeros((2, 3), dtype=torch.bfloat16)
    safetensors.torch.save_file({"weight": tensor}, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )
    protocol.is_sender = is_sender
    protocol._baseline_captured = True
    protocol._snapshot = {"weight": make_tensor_reader(str(tmp_path))("weight")} if is_sender else {}
    observed = []
    buckets = [
        [("weight", torch.ones_like(tensor))],
        [("weight", torch.ones((3, 2), dtype=torch.bfloat16))],
        [("missing", tensor)],
        [("weight", tensor)],
    ]

    def iter_weights(*args, **kwargs):
        for index, bucket in enumerate(buckets):
            observed.append(index)
            yield bucket if kwargs["materialize"] else []

    updater = WeightUpdater.__new__(WeightUpdater)
    updater.protocol = protocol
    updater.weight_version = 0
    updater.is_lora = False
    updater.weights_getter = lambda: {}
    updater._hf_weight_iterator = MagicMock()
    updater._hf_weight_iterator.iter_hf_weights.side_effect = iter_weights
    error_message = "ValueError: Checkpoint tensor 'weight' has shape (2, 3); trainer emitted (3, 2)"

    def reduce_errors(failed, **kwargs):
        assert failed.item() == int(is_sender)
        failed.fill_(1)

    def gather_errors(output, message, **kwargs):
        assert protocol._pool is None
        assert not protocol._inflight
        assert observed == [0, 1, 2, 3]
        assert message == (error_message if is_sender else None)
        output[:] = [error_message, None]

    with (
        patch(f"{_DELTA_MODULE}.dist") as distributed,
        patch("miles.backends.training_utils.weight_update.updater.dist", distributed),
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
        patch(
            "miles.backends.training_utils.weight_update.updater.get_gloo_group",
            return_value=None,
        ),
        patch(
            f"{_DELTA_MODULE}.torch.empty",
            side_effect=RuntimeError("CPU test has no pinned memory"),
        ),
        patch.object(protocol, "finalize") as finalize,
    ):
        distributed.get_rank.return_value = 0 if is_sender else 1
        distributed.get_world_size.return_value = 2
        distributed.all_reduce.side_effect = reduce_errors
        distributed.all_gather_object.side_effect = gather_errors
        with pytest.raises(RuntimeError, match="Disk-delta update validation failed on rank 0") as error:
            updater.update_weights()

    finalize.assert_not_called()
    if is_sender:
        assert isinstance(error.value.__cause__, ValueError)
        assert protocol.total_bytes == tensor.numel() * tensor.element_size()
    else:
        assert error.value.__cause__ is None
    assert not list((tmp_path / "deltas").rglob("*.safetensors"))
    assert not list((tmp_path / "deltas").rglob("*.json"))


class TestReloadEnginesFailureTransitions:
    @staticmethod
    def _make_protocol(calls: list[tuple[str, dict]], failing_method: str) -> UpdateWeightFromDiskDelta:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(
            update_weight_local_checkpoint_dir="/local/ckpt",
            update_weight_disk_dir="/shared/delta",
            pause_generation_mode="retract",
            check_weight_update_equal=False,
        )
        protocol.rollout_engines = [_RejectingApiClient(calls, failing_method)]
        protocol._post_write_hook = None
        protocol._version_dir = "/shared/delta/v7"
        return protocol

    @pytest.mark.parametrize(
        ("failing_method", "expected_calls"),
        [
            ("pull_weights", ["pull_weights"]),
            (
                "update_weights_from_disk",
                ["pull_weights", "pause_generation", "flush_cache", "update_weights_from_disk"],
            ),
        ],
    )
    def test_reload_engine_failure_stops_before_the_next_lifecycle_phase(
        self, failing_method: str, expected_calls: list[str]
    ) -> None:
        """A rejected pull never pauses the engine, and a rejected disk reload never resumes it."""
        calls: list[tuple[str, dict]] = []
        protocol = self._make_protocol(calls, failing_method)

        with (
            patch(f"{_DELTA_MODULE}.dist") as dist_mock,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
        ):
            dist_mock.get_rank.return_value = 0
            with pytest.raises(RuntimeError, match="engine rejected the weights"):
                protocol._reload_engines(7)

        assert [name for name, _kwargs in calls] == expected_calls
