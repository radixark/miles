from argparse import Namespace
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.utils.disk_delta import checksum, get_tensor_locations

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
    def __init__(self, baseline: dict[str, bytes]) -> None:
        self.weights = {name: bytearray(value) for name, value in baseline.items()}
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
            if index["metadata"]["delta_encoding"] == "xor":
                self.weights[name] = bytearray(a ^ b for a, b in zip(self.weights[name], delta, strict=True))
            else:
                count = int(np.frombuffer(delta, dtype="<u4", count=1)[0])
                positions = np.frombuffer(delta, dtype="<u4", count=count, offset=4)
                for position, value in zip(positions, delta[4 + 4 * count :], strict=True):
                    self.weights[name][position] = value
            with safetensors.safe_open(directory / filename, framework="np") as shard:
                assert checksum(index["metadata"]["checksum_format"], self.weights[name]) == shard.metadata()[name]
        self.transport_version = target_version
        return {"success": True}

    async def get_weight_version(self) -> str:
        return "0"

    async def pause_generation(self, *, mode: str) -> None:
        pass

    async def update_weights_from_disk(self, *, model_path: str, weight_version: str) -> dict:
        self.served_versions.append(weight_version)
        return {"success": True}

    async def continue_generation(self) -> None:
        pass


@contextmanager
def _single_rank():
    with (
        patch(f"{_DELTA_MODULE}.dist") as distributed,
        patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=None),
        patch.object(torch, "empty", side_effect=RuntimeError("CPU test has no pinned memory")),
        patch.object(torch.cuda, "current_device", return_value="cpu"),
    ):
        distributed.get_rank.return_value = 0
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, value, **kwargs: output.__setitem__(0, value)
        yield


class TestDeltaPublicationLifecycle:
    @pytest.mark.parametrize("encoding", ["xor", "overwrite"])
    def test_repeat_jump_and_rewind_versions_preserve_the_delta_chain(self, tmp_path: Path, encoding: str) -> None:
        """Mixed tensor layouts reconstruct byte-for-byte while transport and model versions advance independently."""
        baseline = {
            "fp16": torch.zeros(2, 3, dtype=torch.float16),
            "bf16": torch.zeros(2, 3, dtype=torch.bfloat16),
            "fp32": torch.zeros(2, 3, dtype=torch.float32),
            "fp64": torch.zeros(2, 3, dtype=torch.float64),
            "packed": torch.zeros(2, 3, dtype=torch.uint8),
            "fp8": torch.zeros(2, 3, dtype=torch.float8_e4m3fn),
            "scalar": torch.zeros((), dtype=torch.float32),
        }
        checkpoint = tmp_path / "hf"
        checkpoint.mkdir()
        safetensors.torch.save_file(baseline, checkpoint / "model.safetensors")
        protocol = UpdateWeightFromDiskDelta(
            Namespace(
                hf_checkpoint=str(checkpoint),
                update_weight_disk_dir=str(tmp_path / "deltas"),
                update_weight_local_checkpoint_dir=str(tmp_path / "receiver"),
                update_weight_delta_encoding=encoding,
                update_weight_delta_checksum="adler32",
                custom_update_weight_post_write_path=None,
                pause_generation_mode="in_place",
                check_weight_update_equal=False,
            )
        )
        receiver = _DeltaReceiver(
            {name: tensor.reshape(-1).view(torch.uint8).numpy().tobytes() for name, tensor in baseline.items()}
        )
        protocol.rollout_engines = [receiver]
        protocol.is_sender = True

        with _single_rank():
            loaded = [(name, torch.ones_like(tensor)) for name, tensor in baseline.items()]
            assert protocol.begin_sync(weight_version=0, iter_buckets=lambda **kwargs: [loaded])
            assert {name: value.tobytes() for name, value in protocol._snapshot.items()} == receiver.weights
            # The baseline sync also publishes the loaded trainer weights.
            protocol.send_bucket(loaded)
            protocol.after_base_weights()
            protocol.finalize(weight_version=0)
            for name, tensor in loaded:
                assert receiver.weights[name] == tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            for weight_version in [8, 8, 15, 2]:
                weights = {
                    name: torch.full_like(
                        tensor,
                        weight_version + 0.001,
                        dtype=torch.float32 if name in ("fp16", "bf16", "fp64") else tensor.dtype,
                    )
                    for name, tensor in baseline.items()
                }
                assert protocol.begin_sync(
                    weight_version=weight_version,
                    iter_buckets=lambda weights=weights, **kwargs: [list(weights.items())],
                )
                protocol.send_bucket(list(weights.items()))
                protocol.after_base_weights()
                protocol.finalize(weight_version=weight_version)
                for name, tensor in weights.items():
                    expected = tensor.to(dtype=baseline[name].dtype).reshape(-1).view(torch.uint8)
                    assert receiver.weights[name] == expected.numpy().tobytes()

        assert receiver.transport_version == 5
        assert receiver.served_versions == ["0", "8", "8", "15", "2"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ([("a", torch.ones(2, 3))], r"omitted 1 baseline tensors from this update: \['b'\]"),
        ([("a", torch.ones(2, 3)), ("b", torch.ones(2, 3)), ("a", torch.ones(2, 3))], "'a' twice in one update"),
    ],
)
def test_update_must_emit_each_baseline_tensor_exactly_once(tmp_path: Path, update, message) -> None:
    safetensors.torch.save_file({"a": torch.zeros(2, 3), "b": torch.zeros(2, 3)}, tmp_path / "model.safetensors")
    protocol = UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
            check_weight_update_equal=False,
        )
    )
    protocol.rollout_engines = []
    protocol.is_sender = True

    with _single_rank(), patch.object(protocol, "finalize") as finalize:
        assert protocol.begin_sync(
            weight_version=0, iter_buckets=lambda **kwargs: [[("a", torch.zeros(2, 3)), ("b", torch.zeros(2, 3))]]
        )
        protocol.send_bucket(update)
        with pytest.raises(RuntimeError, match=f"update validation failed on rank 0:.*{message}"):
            protocol.after_base_weights()
    finalize.assert_not_called()
    assert protocol._pool is None


@pytest.mark.parametrize(
    ("name", "tensor", "message"),
    [
        ("missing", torch.ones(2, 3), "absent from the canonical checkpoint"),
        ("packed", torch.ones(2, 3, dtype=torch.bfloat16), "Packed and quantized layouts must match"),
        ("fp8", torch.ones(2, 3), "Packed and quantized layouts must match"),
    ],
)
def test_rejects_missing_names_and_implicit_quantization(tmp_path, name, tensor, message):
    safetensors.torch.save_file(
        {"packed": torch.zeros(2, 3, dtype=torch.uint8), "fp8": torch.zeros(2, 3, dtype=torch.float8_e4m3fn)},
        tmp_path / "model.safetensors",
    )
    protocol = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    protocol._tensor_locations = get_tensor_locations(str(tmp_path))
    with pytest.raises(ValueError, match=message):
        protocol._checkpoint_dtype(name, tensor)


def _validation_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        for phase, name, tensor, message in (
            ("baseline", "weight", torch.zeros(3, 2), "shape="),
            ("update", "frozen", torch.zeros(2, 3), "absent from the baseline snapshot"),
        ):
            protocol = UpdateWeightFromDiskDelta(
                Namespace(
                    hf_checkpoint=directory,
                    update_weight_disk_dir=str(root / "deltas"),
                    update_weight_delta_encoding="xor",
                    update_weight_delta_checksum="adler32",
                    custom_update_weight_post_write_path=None,
                    check_weight_update_equal=False,
                )
            )
            protocol.is_sender = rank == 0
            protocol.rollout_engines = []
            visited = []

            def buckets(*, materialize, name=name, tensor=tensor, visited=visited):
                for index, item in enumerate(
                    [("weight", torch.ones(2, 3)), (name, tensor), ("weight", torch.ones(2, 3))]
                ):
                    dist.all_reduce(torch.ones(1))
                    visited.append(index)
                    yield [item] if materialize else []

            with (
                patch(f"{_DELTA_MODULE}.get_gloo_group", return_value=dist.group.WORLD),
                patch.object(protocol, "finalize") as finalize,
            ):
                if phase == "update":
                    # The baseline sync also publishes; drain it before the update under test.
                    assert protocol.begin_sync(
                        weight_version=0, iter_buckets=lambda **kwargs: [[("weight", torch.zeros(2, 3))]]
                    )
                    if protocol.is_sender:
                        protocol.send_bucket([("weight", torch.zeros(2, 3))])
                    protocol.after_base_weights()
                with pytest.raises(RuntimeError, match=f"{phase} validation failed on rank 0:.*{message}"):
                    if protocol.begin_sync(weight_version=1, iter_buckets=buckets):
                        for bucket in buckets(materialize=protocol.is_sender):
                            if protocol.is_sender:
                                protocol.send_bucket(bucket)
                        protocol.after_base_weights()
                        protocol.finalize(weight_version=1)
                finalize.assert_not_called()
            assert visited == [0, 1, 2]
            assert protocol._pool is None
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_invalid_layout_fails_all_ranks_after_draining_the_iterator(tmp_path: Path) -> None:
    safetensors.torch.save_file(
        {"weight": torch.zeros(2, 3, dtype=torch.bfloat16), "frozen": torch.zeros(2, 3)},
        tmp_path / "model.safetensors",
    )
    mp.spawn(_validation_worker, args=(str(tmp_path),), nprocs=2, join=True)
