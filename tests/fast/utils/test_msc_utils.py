"""Checkpoint checks that run before Megatron's load must accept an msc:// (e.g. S3-backed) checkpoint.

An msc:// profile backed by a local directory stands in for object storage."""

import json
import uuid

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

pytest.importorskip("multistorageclient")

from miles.backends.megatron_utils.checkpoint import (  # noqa: E402
    LINEAR_ATTN_LAYOUT_KEY,
    _is_megatron_checkpoint,
    assert_linear_attn_checkpoint_is_current,
)
from miles.backends.megatron_utils.megatron_config import has_megatron_checkpoint  # noqa: E402
from miles.backends.megatron_utils.model import _has_loadable_ckpt  # noqa: E402
from miles_plugins.models.deepseek_v4.arguments import assert_checkpoint_is_current  # noqa: E402

LAYER = "decoder.layers.0.self_attention.linear_attn"


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A local directory and the msc:// URL that reaches it.

    MSC caches a client per profile for the life of the process, so every test gets its own profile."""
    root = tmp_path / "bucket"
    root.mkdir()
    profile = f"ckpt-{uuid.uuid4().hex}"
    config = tmp_path / "msc_config.json"
    config.write_text(
        json.dumps(
            {"profiles": {profile: {"storage_provider": {"type": "file", "options": {"base_path": str(root)}}}}}
        )
    )
    monkeypatch.setenv("MSC_CONFIG", str(config))
    return root, f"msc://{profile}"


def _write_torch_dist(root, step, keys):
    iteration = root / "run" / ("release" if step == "release" else f"iter_{int(step):07d}")
    iteration.mkdir(parents=True)
    dcp.save({key: torch.zeros(2) for key in keys}, storage_writer=FileSystemWriter(iteration), no_dist=True)
    (root / "run" / "latest_checkpointed_iteration.txt").write_text(str(step))


class TestMscCheckpointDiscovery:
    def test_a_tracked_checkpoint_is_found(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])

        assert has_megatron_checkpoint(f"{url}/run")
        assert _is_megatron_checkpoint(f"{url}/run")
        assert _is_megatron_checkpoint(f"{url}/run/iter_0000003")
        assert _has_loadable_ckpt(f"{url}/run")

    def test_a_selected_step_without_a_tracker_is_found(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 3, ["w"])
        (root / "run" / "latest_checkpointed_iteration.txt").unlink()

        assert has_megatron_checkpoint(f"{url}/run", ckpt_step=3)
        assert not has_megatron_checkpoint(f"{url}/run")

    def test_a_missing_or_empty_location_is_not_a_checkpoint(self, bucket):
        root, url = bucket
        (root / "empty").mkdir()

        assert not has_megatron_checkpoint(f"{url}/missing")
        assert not _is_megatron_checkpoint(f"{url}/empty")
        assert not _has_loadable_ckpt(f"{url}/empty")
        assert not _has_loadable_ckpt(f"{url}/missing")


class TestMscCheckpointLayoutGuards:
    def test_a_linear_attention_checkpoint_without_the_layout_marker_is_rejected(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 12, [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.conv1d.weight"])

        with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
            assert_linear_attn_checkpoint_is_current(f"{url}/run")

    def test_a_linear_attention_checkpoint_with_the_layout_marker_passes(self, bucket):
        root, url = bucket
        _write_torch_dist(root, 12, [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.{LINEAR_ATTN_LAYOUT_KEY}"])

        assert_linear_attn_checkpoint_is_current(f"{url}/run")

    def test_a_deepseek_v4_checkpoint_with_the_old_names_is_rejected(self, bucket):
        root, url = bucket
        _write_torch_dist(root, "release", ["decoder.layers.0.self_attention.wq_a.weight"])

        with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
            assert_checkpoint_is_current(f"{url}/run")
