from pathlib import Path

import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.snapshot_publisher import _partial_export_warning


def _write_base(tmp_path: Path, names: list[str]) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_file({name: torch.zeros(2) for name in names}, base / "model.safetensors")
    return base


def test_partial_export_warns_with_counts_and_remedy(tmp_path: Path) -> None:
    base = _write_base(tmp_path, ["model.a", "model.visual.b", "mtp.c"])

    warning = _partial_export_warning(tmp_path / "export", ["model.a"], str(base))

    assert warning is not None
    assert "1 of the 3 tensors" in warning
    assert "model.visual.b" in warning and "mtp.c" in warning
    assert "tools/complete_hf_export.py" in warning


def test_complete_export_does_not_warn(tmp_path: Path) -> None:
    base = _write_base(tmp_path, ["model.a", "model.b"])

    assert _partial_export_warning(tmp_path / "export", ["model.a", "model.b"], str(base)) is None


def test_missing_base_checkpoint_does_not_warn(tmp_path: Path) -> None:
    assert _partial_export_warning(tmp_path / "export", ["model.a"], str(tmp_path / "gone")) is None


def test_unreadable_base_weights_do_not_warn(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")  # a dir without weights or an index

    assert _partial_export_warning(tmp_path / "export", ["model.a"], str(base)) is None
