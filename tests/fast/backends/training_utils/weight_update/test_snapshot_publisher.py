import logging
from pathlib import Path

import pytest
import safetensors.torch
import torch

import miles.backends.training_utils.weight_update.snapshot_publisher as snapshot_publisher
from miles.backends.training_utils.weight_update.snapshot_publisher import _log_partial_export_warning


@pytest.fixture(autouse=True)
def _fresh_warning_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot_publisher, "_partial_export_warnings_issued", set())


def _write_base(tmp_path: Path, names: list[str]) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_file({name: torch.zeros(2) for name in names}, base / "model.safetensors")
    (base / "config.json").write_text("{}")
    return base


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.levelno >= logging.WARNING]


def test_partial_export_warns_once_with_counts_and_remedy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _write_base(tmp_path, ["model.a", "model.visual.b", "mtp.c"])

    with caplog.at_level(logging.WARNING):
        _log_partial_export_warning(tmp_path / "export", ["model.a"], str(base))
        _log_partial_export_warning(tmp_path / "export2", ["model.a"], str(base))  # e.g. an eval snapshot

    assert len(_warnings(caplog)) == 1, "the same missing set warns once per process"
    warning = _warnings(caplog)[0]
    assert "holds 1 tensors" in warning and "2 tensors" in warning
    assert "model.visual.b" in warning and "mtp.c" in warning
    assert "tools/complete_hf_export.py" in warning


def test_complete_export_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = _write_base(tmp_path, ["model.a", "model.b"])

    with caplog.at_level(logging.WARNING):
        _log_partial_export_warning(tmp_path / "export", ["model.a", "model.b"], str(base))

    assert _warnings(caplog) == []


def test_missing_base_checkpoint_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _log_partial_export_warning(tmp_path / "export", ["model.a"], str(tmp_path / "gone"))

    assert _warnings(caplog) == []


def test_corrupt_base_weights_never_raise(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"not a safetensors file")
    (base / "config.json").write_text("{}")

    with caplog.at_level(logging.WARNING):
        _log_partial_export_warning(tmp_path / "export", ["model.a"], str(base))

    assert ["Could not check" in w for w in _warnings(caplog)] == [True]


def test_any_check_failure_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The invariant, independent of today's exception types: the advisory check cannot fail an export.
    base = _write_base(tmp_path, ["model.a"])

    def boom(exported_names, base_checkpoint):
        raise RuntimeError("boom")

    monkeypatch.setattr(snapshot_publisher, "missing_base_tensors", boom)
    with caplog.at_level(logging.WARNING):
        _log_partial_export_warning(tmp_path / "export", ["model.a"], str(base))
        _log_partial_export_warning(tmp_path / "export", ["model.a"], str(base))

    assert ["Could not check" in w for w in _warnings(caplog)] == [True], "failures are logged once, never raised"
