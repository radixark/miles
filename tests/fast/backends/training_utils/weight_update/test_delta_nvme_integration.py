from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.training_utils.weight_update.protocols.delta"


def _protocol(tmp_path: Path) -> UpdateWeightFromDiskDelta:
    return UpdateWeightFromDiskDelta(
        Namespace(
            hf_checkpoint=str(tmp_path),
            update_weight_disk_dir=str(tmp_path / "deltas"),
            update_weight_delta_nvme_dir=str(tmp_path / "baselines"),
            update_weight_delta_encoding="xor",
            update_weight_delta_checksum="adler32",
            custom_update_weight_post_write_path=None,
        )
    )


def test_nvme_requires_iterator_owner_hooks(tmp_path):
    protocol = _protocol(tmp_path)
    with pytest.raises(ValueError, match="direct Megatron"):
        protocol.bind_iterator(SimpleNamespace())
    iterator = SimpleNamespace(quantization_config={"quant_method": "nvfp4"}, set_local_expert_transform=MagicMock())
    protocol.bind_iterator(iterator)
    assert protocol._nvme_quantization_config == iterator.quantization_config
    iterator.set_local_expert_transform.assert_called_once_with(
        prefetch=protocol._prefetch_expert, transform=protocol._process_expert
    )


def test_nvme_capture_excludes_handled_tensors_from_cpu_snapshot(tmp_path):
    tensors = {"router": torch.ones(2), "expert.weight": torch.zeros(4, dtype=torch.uint8)}
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    protocol = _protocol(tmp_path)
    protocol.is_sender = True
    protocol._nvme = MagicMock()
    # Owner processing removed expert.weight before the ordinary iterator yielded.
    iterator = MagicMock(return_value=iter([[("router", tensors["router"])]]))
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_rank.return_value = 1
        protocol._capture_baseline(iterator)

    assert set(protocol._snapshot) == {"router"}
    protocol._nvme.finish.assert_called_once_with()
    protocol._nvme.commit.assert_called_once_with()


@pytest.mark.parametrize("is_sender", [False, True])
def test_nvme_owner_results_are_merged_on_every_rank(tmp_path, is_sender):
    protocol = _protocol(tmp_path)
    protocol.is_sender = is_sender
    protocol._nvme = MagicMock()
    payload = np.array([1, 2, 3], dtype=np.uint8)
    protocol._nvme.finish.return_value = SimpleNamespace(
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


def test_nvme_failure_is_reported_after_cpu_pool_is_drained(tmp_path):
    protocol = _protocol(tmp_path)
    protocol.is_sender = False
    protocol._nvme = MagicMock()
    protocol._nvme.finish.side_effect = RuntimeError("NVMe read failed")
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
        with pytest.raises(RuntimeError, match="update validation failed on rank 1: RuntimeError: NVMe read failed"):
            protocol.after_base_weights()

    protocol._nvme.commit.assert_not_called()
    assert not list((tmp_path / "deltas").rglob("*.json"))


@pytest.mark.parametrize("commit_fails", [False, True])
def test_nvme_commit_follows_publication_and_precedes_receiver_reload(tmp_path, commit_fails):
    protocol = _protocol(tmp_path)
    protocol._nvme = MagicMock()
    calls = []
    protocol._write_delta_files = lambda version: calls.append("publish")
    protocol._reload_engines = lambda version: calls.append("reload")
    protocol._record_metrics = lambda version: calls.append("metrics")

    def commit():
        calls.append("commit")
        if commit_fails:
            raise OSError("baseline rename failed")

    protocol._nvme.commit.side_effect = commit
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        if commit_fails:
            with pytest.raises(RuntimeError, match="NVMe baseline commit validation failed"):
                protocol.finalize(1)
        else:
            protocol.finalize(1)

    assert calls == (["publish", "commit"] if commit_fails else ["publish", "commit", "reload", "metrics"])


def test_nvme_begin_failure_is_collective_before_iteration(tmp_path):
    protocol = _protocol(tmp_path)
    protocol._nvme = MagicMock()
    protocol._nvme.begin.side_effect = RuntimeError("NVMe staging unavailable")
    iterator = MagicMock()
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        with pytest.raises(RuntimeError, match="NVMe setup validation failed.*NVMe staging unavailable"):
            protocol.begin_sync(1, iterator)
    iterator.assert_not_called()
    protocol._nvme.begin.assert_called_once_with(capture_baseline=True, weight_version=0)


def test_nvme_directory_overlap_fails_collectively_before_creating_baseline(tmp_path):
    protocol = _protocol(tmp_path)
    protocol._nvme_dir = str(tmp_path / "deltas" / "baseline")
    protocol._nvme = MagicMock()
    iterator = MagicMock()
    with patch(f"{_MODULE}.dist") as distributed, patch(f"{_MODULE}.get_gloo_group", return_value=None):
        distributed.get_world_size.return_value = 1
        distributed.all_gather_object.side_effect = lambda output, message, **kwargs: output.__setitem__(0, message)
        with pytest.raises(RuntimeError, match="NVMe setup validation failed.*must not overlap"):
            protocol.begin_sync(1, iterator)
    iterator.assert_not_called()
    protocol._nvme.begin.assert_not_called()


@pytest.mark.parametrize("failing_suffix", [None, ".safetensors", ".json"])
def test_nvme_non_sender_publishes_or_fails_before_commit(tmp_path, failing_suffix):
    protocol = _protocol(tmp_path)
    protocol.is_sender = False
    protocol._nvme = MagicMock()
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
            with pytest.raises(RuntimeError, match="NVMe publication validation failed.*publication failed"):
                protocol.finalize(1)
        else:
            protocol.finalize(1)

    if failing_suffix is not None:
        protocol._nvme.commit.assert_not_called()
        protocol._reload_engines.assert_not_called()
    else:
        assert (Path(protocol._version_dir) / "model.safetensors.index.json").is_file()
        assert protocol.wire_bytes > 0
        protocol._nvme.commit.assert_called_once_with()
        protocol._reload_engines.assert_called_once_with(1)
