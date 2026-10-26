from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

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
        protocol._snapshot_file_version = 7
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
                protocol._reload_engines(weight_version=42)

        assert [name for name, _kwargs in calls] == expected_calls


class _DeltaReceiver:
    def __init__(self, baseline: bytes) -> None:
        self.weights = bytearray(baseline)
        self.transport_version = 0
        self.served_versions: list[str] = []

    async def pull_weights(self, *, target_version: int, local_checkpoint_dir: str, source_dir: str) -> dict:
        import json

        import safetensors.numpy
        import zstandard

        if target_version == 0:
            return {"success": True}
        assert target_version == self.transport_version + 1
        directory = Path(source_dir) / f"weight_v{target_version:06d}"
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        assert int(index["metadata"]["base_version"]) == self.transport_version
        for name, filename in index["weight_map"].items():
            compressed = safetensors.numpy.load_file(directory / filename)[name]
            delta = zstandard.ZstdDecompressor().decompress(compressed.tobytes())
            self.weights = bytearray(a ^ b for a, b in zip(self.weights, delta, strict=True))
        self.transport_version = target_version
        return {"success": True}

    async def pause_generation(self, *, mode: str) -> None:
        pass

    async def update_weights_from_disk(self, *, model_path: str, weight_version: str) -> dict:
        self.served_versions.append(weight_version)
        return {"success": True}

    async def continue_generation(self) -> None:
        pass


class TestDeltaPublicationLifecycle:
    def test_repeat_jump_and_rewind_versions_preserve_the_delta_chain(self, tmp_path: Path) -> None:
        """Real finalize advances transport state while the receiver reconstructs each published model."""
        import safetensors.torch
        import torch

        baseline = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        checkpoint = tmp_path / "hf"
        checkpoint.mkdir()
        safetensors.torch.save_file({"weight": baseline}, checkpoint / "model.safetensors")
        protocol = UpdateWeightFromDiskDelta(
            Namespace(
                hf_checkpoint=str(checkpoint),
                update_weight_disk_dir=str(tmp_path / "deltas"),
                update_weight_local_checkpoint_dir=str(tmp_path / "receiver"),
                update_weight_delta_encoding="xor",
                update_weight_delta_checksum="adler32",
                custom_update_weight_post_write_path=None,
                pause_generation_mode="in_place",
            )
        )
        receiver = _DeltaReceiver(bytes(baseline.tolist()))
        protocol.rollout_engines = [receiver]
        protocol.is_sender = True
        protocol._baseline_captured = True
        protocol._snapshot = {"weight": baseline.numpy().copy()}

        with (
            patch(f"{_DELTA_MODULE}.dist") as distributed,
            patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
            patch.object(torch, "empty", side_effect=RuntimeError("CPU test has no pinned memory")),
            patch.object(torch.cuda, "current_device", return_value="cpu"),
        ):
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 1
            distributed.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
            for weight_version in [8, 8, 15, 2]:
                weights = torch.tensor([weight_version, 1, 2, 3], dtype=torch.uint8)
                assert protocol.begin_sync(
                    weight_version=weight_version,
                    iter_buckets=lambda weights=weights, **kwargs: [[("weight", weights)]],
                )
                protocol.send_bucket([("weight", weights)])
                protocol.after_base_weights()
                protocol.finalize(weight_version=weight_version)
                assert receiver.weights == bytes(weights.tolist())

        assert receiver.transport_version == 4
        assert receiver.served_versions == ["8", "8", "15", "2"]
