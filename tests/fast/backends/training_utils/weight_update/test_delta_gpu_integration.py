from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta
from miles.backends.training_utils.weight_update.protocols.nvfp4_gpu import Nvfp4GpuDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


def _protocol(tmp_path: Path) -> UpdateWeightFromDiskDelta:
    return UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_gpu=True,
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )


def test_gpu_requires_iterator_owner_hooks(tmp_path):
    protocol = _protocol(tmp_path)
    with pytest.raises(ValueError, match="direct Megatron"):
        protocol.bind_iterator(SimpleNamespace())
    iterator = SimpleNamespace(quantization_config={"quant_method": "nvfp4"}, set_local_expert_transform=MagicMock())
    protocol.bind_iterator(iterator)
    assert protocol._gpu_delta_quantization_config == iterator.quantization_config
    iterator.set_local_expert_transform.assert_called_once_with(
        prefetch=protocol._prefetch_expert, transform=protocol._process_expert
    )


def test_gpu_capture_excludes_handled_tensors_from_cpu_snapshot(tmp_path):
    tensors = {"router": torch.ones(2), "expert.weight": torch.zeros(4, dtype=torch.uint8)}
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    protocol = _protocol(tmp_path)
    protocol.is_sender = True
    protocol._gpu_delta = MagicMock()
    # Owner processing removed expert.weight before the ordinary iterator yielded.
    iterator = MagicMock(return_value=iter([[("router", tensors["router"])]]))
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_rank.return_value = 1
        protocol._capture_baseline(iterator)

    assert set(protocol._snapshot) == {"router"}
    protocol._gpu_delta.finish.assert_called_once_with()
    protocol._gpu_delta.commit.assert_called_once_with()


@pytest.mark.parametrize("is_sender", [False, True])
def test_gpu_owner_results_are_merged_on_every_rank(tmp_path, is_sender):
    protocol = _protocol(tmp_path)
    protocol.is_sender = is_sender
    protocol._gpu_delta = MagicMock()
    payload = np.array([1, 2, 3], dtype=np.uint8)
    protocol._gpu_delta.finish.return_value = SimpleNamespace(
        delta={"expert.weight": payload}, checksums={"expert.weight": "00000001"}, changed_bytes=2, total_bytes=8
    )
    with patch(f"{_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")):
        protocol._begin_encode(1)
    protocol._delta["ordinary"] = payload
    protocol._checksums["ordinary"] = "00000002"
    protocol.changed_bytes, protocol.total_bytes = 1, 4
    with patch(f"{_MODULE}.dist"), patch(f"{_MODULE}.get_gloo_group", return_value=None):
        protocol.after_base_weights()

    assert set(protocol._delta) == {"ordinary", "expert.weight"}
    assert set(protocol._checksums) == set(protocol._delta)
    assert (protocol.changed_bytes, protocol.total_bytes) == (3, 12)
    assert protocol._pool is None
    assert protocol._snapshot == {}


def test_gpu_failure_is_reported_after_cpu_pool_is_drained(tmp_path):
    protocol = _protocol(tmp_path)
    protocol.is_sender = False
    protocol._gpu_delta = MagicMock()
    protocol._gpu_delta.finish.side_effect = RuntimeError("GPU read failed")
    with patch(f"{_MODULE}.torch.empty", side_effect=RuntimeError("CPU test has no pinned memory")):
        protocol._begin_encode(1)
    observed = []
    protocol._inflight.append(protocol._pool.submit(lambda: observed.append("ordinary drained") or []))

    def gather_errors(output, message, **kwargs):
        assert observed == ["ordinary drained"] and protocol._pool is None
        output[:] = [None, message]

    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 2
        distributed.all_gather_object.side_effect = gather_errors
        with pytest.raises(RuntimeError, match="update validation failed on rank 1: RuntimeError: GPU read failed"):
            protocol.after_base_weights()

    protocol._gpu_delta.commit.assert_not_called()
    assert not list((tmp_path / "deltas").rglob("*.json"))


@pytest.mark.parametrize("commit_fails", [False, True])
def test_gpu_commit_follows_publication_and_precedes_receiver_reload(tmp_path, commit_fails):
    protocol = _protocol(tmp_path)
    protocol._gpu_delta = MagicMock()
    calls = []
    protocol._write_delta_files = lambda version: calls.append("publish")
    protocol._reload_engines = lambda version: calls.append("reload")
    protocol._record_metrics = lambda version: calls.append("metrics")

    def commit():
        calls.append("commit")
        if commit_fails:
            raise RuntimeError("baseline commit failed")

    protocol._gpu_delta.commit.side_effect = commit
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        if commit_fails:
            with pytest.raises(RuntimeError, match="GPU baseline commit validation failed"):
                protocol.finalize(1)
        else:
            protocol.finalize(1)

    assert calls == (["publish", "commit"] if commit_fails else ["publish", "commit", "reload", "metrics"])


@pytest.mark.parametrize("failing_rank", [None, 0, 1])
def test_gpu_publication_hook_completes_collectively_before_commit(tmp_path, failing_rank):
    protocol = _protocol(tmp_path)
    protocol._gpu_delta = MagicMock()
    protocol._version_dir = str(tmp_path / "deltas" / "weight_v000001")
    protocol.rollout_engines = []
    protocol.args.pause_generation_mode = "in_place"
    calls = []
    protocol._write_delta_files = lambda version: calls.append("files")
    protocol._record_metrics = MagicMock()
    protocol._gpu_delta.commit.side_effect = lambda: calls.append("commit")
    reload_engines = protocol._reload_engines

    def hook(args, path, engines):
        assert path == protocol._version_dir and engines == []
        calls.append("hook")
        if failing_rank == 0:
            raise OSError("upload failed")

    def reload(version):
        calls.append("reload")
        reload_engines(version)

    protocol._post_write_hook = MagicMock(side_effect=hook)
    protocol._reload_engines = MagicMock(side_effect=reload)
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_rank.return_value = 0
        distributed.get_world_size.return_value = 2
        if failing_rank is not None:
            distributed.all_reduce.side_effect = lambda failed, **kwargs: failed.fill_(1)
            distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(
                slice(None), ["OSError: upload failed" if rank == failing_rank else None for rank in range(2)]
            )
            with pytest.raises(RuntimeError, match=f"GPU publication validation failed on rank {failing_rank}"):
                protocol.finalize(1)
        else:
            protocol.finalize(1)
    protocol._post_write_hook.assert_called_once()
    assert calls == (["files", "hook"] if failing_rank is not None else ["files", "hook", "commit", "reload"])


def test_gpu_begin_failure_is_collective_before_iteration(tmp_path):
    protocol = _protocol(tmp_path)
    protocol._gpu_delta = MagicMock()
    protocol._gpu_delta.begin.side_effect = RuntimeError("GPU codec unavailable")
    iterator = MagicMock()
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        with pytest.raises(RuntimeError, match="GPU setup validation failed.*GPU codec unavailable"):
            protocol.begin_sync(1, iterator)
    iterator.assert_not_called()
    protocol._gpu_delta.begin.assert_called_once_with(capture_baseline=True, weight_version=0)


@pytest.mark.parametrize("failing_suffix", [None, ".safetensors", ".json"])
def test_gpu_non_sender_publishes_or_fails_before_commit(tmp_path, failing_suffix):
    protocol = _protocol(tmp_path)
    protocol.is_sender = False
    protocol._gpu_delta = MagicMock()
    protocol._version_dir = str(tmp_path / "deltas" / "weight_v000001")
    protocol._delta = {"expert.weight": np.array([1, 2, 3], dtype=np.uint8)}
    protocol._checksums = {"expert.weight": "00000001"}
    protocol._reload_engines = MagicMock()
    protocol._record_metrics = MagicMock()

    from miles.backends.training_utils.weight_update.protocols.delta import _atomic_write

    def write(path, data):
        if failing_suffix is not None and path.endswith(failing_suffix):
            raise OSError("publication failed")
        _atomic_write(path, data)

    with (
        patch(f"{_MODULE}.dist") as distributed,
        patch(f"{_MODULE}.get_gloo_group", return_value=None),
        patch(f"{_MODULE}._atomic_write", side_effect=write),
    ):
        distributed.get_rank.return_value = 0
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        if failing_suffix is not None:
            with pytest.raises(RuntimeError, match="GPU publication validation failed.*publication failed"):
                protocol.finalize(1)
        else:
            protocol.finalize(1)

    if failing_suffix is not None:
        protocol._gpu_delta.commit.assert_not_called()
        protocol._reload_engines.assert_not_called()
    else:
        assert (Path(protocol._version_dir) / "model.safetensors.index.json").is_file()
        assert protocol.wire_bytes > 0
        protocol._gpu_delta.commit.assert_called_once_with()
        protocol._reload_engines.assert_called_once_with(1)


@pytest.mark.parametrize("failure_stage", ["files", "hook"])
def test_gpu_failed_publication_forbids_reusing_prepared_cpu_baseline(tmp_path, failure_stage):
    protocol = _protocol(tmp_path)
    manager = Nvfp4GpuDelta.__new__(Nvfp4GpuDelta)
    manager.error, manager._active, manager._version = None, False, 0
    manager._units, manager._pending, manager._prefetched = {}, [], {}
    manager._slots = []
    # No owned tensors are needed to exercise the real prepare/commit state machine.
    manager.begin(capture_baseline=False, weight_version=1)
    manager.finish()
    protocol._gpu_delta, protocol._baseline_captured = manager, True
    protocol._write_delta_files = MagicMock()
    protocol._version_dir, protocol.rollout_engines = str(tmp_path / "version"), []
    protocol._reload_engines = MagicMock()
    if failure_stage == "files":
        protocol._write_delta_files.side_effect = OSError("publication failed")
    else:
        protocol._post_write_hook = MagicMock(side_effect=OSError("publication failed"))
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        with pytest.raises((OSError, RuntimeError), match="publication failed"):
            protocol.finalize(1)
    assert manager._active and manager._version == 0
    protocol._reload_engines.assert_not_called()

    iterator = MagicMock()
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        with pytest.raises(RuntimeError, match="GPU setup validation failed.*publication is uncommitted"):
            protocol.begin_sync(2, iterator)
    iterator.assert_not_called()
