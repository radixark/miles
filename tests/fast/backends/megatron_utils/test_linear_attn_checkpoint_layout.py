"""assert_linear_attn_checkpoint_is_current: a torch_dist checkpoint without the linear-attention layout
marker is rejected before Megatron would load it half-initialized."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from miles.backends.megatron_utils.checkpoint import (
    LINEAR_ATTN_LAYOUT_KEY,
    _has_linear_attention,
    assert_linear_attn_checkpoint_is_current,
    publish_as_release,
)

LAYER = "decoder.layers.0.self_attention.linear_attn"


def _write(path, keys):
    path.mkdir(parents=True)
    dcp.save({key: torch.zeros(2) for key in keys}, storage_writer=FileSystemWriter(path), no_dist=True)


def test_release_checkpoint_with_the_marker_passes(tmp_path):
    _write(tmp_path / "release", [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.{LINEAR_ATTN_LAYOUT_KEY}"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    assert_linear_attn_checkpoint_is_current(str(tmp_path))


def test_iteration_checkpoint_without_the_marker_is_rejected(tmp_path):
    _write(tmp_path / "iter_0000012", [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.conv1d.weight"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("12")
    with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
        assert_linear_attn_checkpoint_is_current(str(tmp_path))


def test_only_models_with_linear_attention_are_checked():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(LINEAR_ATTN_LAYOUT_KEY, torch.ones(1))

    with_layer = torch.nn.Sequential(torch.nn.Linear(2, 2), Core())
    assert _has_linear_attention([with_layer])
    assert not _has_linear_attention([torch.nn.Sequential(torch.nn.Linear(2, 2))])


def _converted(save_dir, marker):
    iteration = save_dir / "iter_0000001"
    iteration.mkdir()
    (iteration / "weights.txt").write_text(marker)


def test_a_first_conversion_becomes_the_release(tmp_path):
    _converted(tmp_path, "new")

    publish_as_release(str(tmp_path))

    assert (tmp_path / "release" / "weights.txt").read_text() == "new"
    assert not (tmp_path / "iter_0000001").exists()


def test_a_reconversion_replaces_the_previous_release(tmp_path):
    """shutil.move into an existing directory nests, which would leave the old weights in place."""
    stale = tmp_path / "release"
    stale.mkdir()
    (stale / "weights.txt").write_text("stale")
    (stale / "only-in-the-old-one.txt").write_text("stale")
    _converted(tmp_path, "new")

    publish_as_release(str(tmp_path))

    assert (tmp_path / "release" / "weights.txt").read_text() == "new"
    assert not (tmp_path / "release" / "only-in-the-old-one.txt").exists()
    assert not (tmp_path / "release" / "iter_0000001").exists()
