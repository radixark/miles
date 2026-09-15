from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
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


class TestConnectionOwnership:
    @staticmethod
    def _connect(protocol: UpdateWeightFromDiskDelta, engines) -> None:
        with patch(f"{_DELTA_MODULE}.get_data_replica_rank_and_size", return_value=(0, 1)):
            protocol.connect(
                rollout_engines=engines,
                engine_gpu_counts=None,
                engine_gpu_offsets=None,
                parallel_state=MagicMock(),
                placement=WeightUpdatePlacement(gather_pp=False),
                selector="all",
            )

    def test_artifact_writer_needs_no_engine_local_checkpoint_or_hook(self) -> None:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(update_weight_local_checkpoint_dir=None)
        protocol._post_write_hook = None

        self._connect(protocol, [])

        assert protocol.rollout_engines == []

    def test_connected_engines_still_require_a_local_checkpoint(self) -> None:
        protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
        protocol.args = Namespace(update_weight_local_checkpoint_dir=None)
        protocol._post_write_hook = MagicMock()

        with pytest.raises(ValueError, match="local-checkpoint-dir"):
            self._connect(protocol, [MagicMock()])


def test_artifact_only_publish_calls_the_hook_without_engine_requests() -> None:
    hook = MagicMock()
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol.args = Namespace()
    protocol._post_write_hook = hook
    protocol._version_dir = "/shared/delta/weight_v000007"
    protocol.rollout_engines = []

    with (
        patch(f"{_DELTA_MODULE}.dist") as dist_mock,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=MagicMock()),
    ):
        protocol._publish_and_reload_engines(7)

    hook.assert_called_once_with(protocol.args, protocol._version_dir, [])
    dist_mock.barrier.assert_called_once()


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
                protocol._publish_and_reload_engines(7)

        assert [name for name, _kwargs in calls] == expected_calls
