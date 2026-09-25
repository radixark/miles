from argparse import Namespace
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.torch
import torch
import zstandard

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils.disk_delta import checksum, make_tensor_reader

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

    def test_casts_only_between_plain_float_storage_dtypes(self, tmp_path: Path) -> None:
        safetensors.torch.save_file(
            {"router": torch.zeros((2, 3), dtype=torch.bfloat16)},
            tmp_path / "model.safetensors",
        )

        emitted = torch.ones((2, 3), dtype=torch.float32)
        matched = self._protocol(tmp_path)._match_checkpoint_layout("router", emitted)

        assert matched.dtype is torch.bfloat16
        torch.testing.assert_close(matched.float(), emitted)

    def test_preserves_an_exact_nvfp4_layout(self, tmp_path: Path) -> None:
        tensors = {
            "expert.weight": torch.zeros((2, 3), dtype=torch.uint8),
            "expert.weight_scale": torch.zeros((2, 1), dtype=torch.float8_e4m3fn),
            "expert.weight_scale_2": torch.zeros((), dtype=torch.float32),
        }
        safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
        protocol = self._protocol(tmp_path)

        for name, emitted in tensors.items():
            assert protocol._match_checkpoint_layout(name, emitted) is emitted

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
            distributed.get_world_size.return_value = 1
            distributed.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
            protocol.after_base_weights()

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


def test_send_bucket_encodes_a_scalar_tensor(tmp_path: Path) -> None:
    safetensors.torch.save_file(
        {"weight_scale": torch.ones((), dtype=torch.float32)},
        tmp_path / "model.safetensors",
    )
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace(hf_checkpoint=str(tmp_path))
    protocol._use_pinned = False
    protocol._pool = MagicMock()
    protocol._inflight = deque()
    protocol._encode_error = None
    protocol.total_bytes = 0

    protocol.send_bucket([("weight_scale", torch.ones((), dtype=torch.float32))])

    _, name, payload, nbytes, pinned = protocol._pool.submit.call_args.args
    assert name == "weight_scale"
    assert payload.shape == (torch.float32.itemsize,)
    assert nbytes == torch.float32.itemsize
    assert not pinned


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
