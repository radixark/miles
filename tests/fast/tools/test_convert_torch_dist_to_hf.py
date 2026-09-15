"""tools/convert_torch_dist_to_hf.py: bridge mode routes through AutoBridge.export_ckpt (#634)."""

import importlib.util
import pickle
import sys
import types
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "convert_torch_dist_to_hf.py"


@pytest.fixture
def tool(monkeypatch):
    """Import the tool as a module; importing must leave pickle.Unpickler alone."""
    before = pickle.Unpickler
    spec = importlib.util.spec_from_file_location("convert_torch_dist_to_hf_under_test", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert pickle.Unpickler is before
    return module


@pytest.fixture
def fake_bridge(monkeypatch):
    calls: dict[str, object] = {}

    class AutoBridge:
        @classmethod
        def from_hf_pretrained(cls, hf_dir, **kwargs):
            calls["from_hf_pretrained"] = (hf_dir, kwargs)
            return cls()

        def export_ckpt(self, *, megatron_path, hf_path):
            calls["export_ckpt"] = (megatron_path, hf_path)

    bridge_pkg = types.ModuleType("megatron.bridge")
    bridge_pkg.AutoBridge = AutoBridge
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_pkg)
    return calls


def test_bridge_mode_exports_through_autobridge(tool, fake_bridge, monkeypatch, tmp_path):
    copied = {}
    monkeypatch.setattr(tool, "copy_assets", lambda src, dst: copied.update(src=src, dst=dst))
    monkeypatch.setattr(tool, "save_tensors", lambda *a, **k: pytest.fail("raw path must not run in bridge mode"))

    tool.main(
        [
            "--input-dir",
            "ckpt",
            "--output-dir",
            str(tmp_path / "out"),
            "--origin-hf-dir",
            "hf",
            "--megatron-to-hf-mode",
            "bridge",
        ]
    )

    assert fake_bridge["from_hf_pretrained"] == ("hf", {"trust_remote_code": True})
    assert fake_bridge["export_ckpt"] == ("ckpt", str(tmp_path / "out"))
    assert copied == {"src": "hf", "dst": str(tmp_path / "out")}
    assert pickle.Unpickler.__name__ != "UnpicklerWrapper"


def test_bridge_mode_needs_the_origin_hf_dir(tool, fake_bridge, tmp_path):
    with pytest.raises(ValueError, match="--origin-hf-dir"):
        tool.main(["--input-dir", "ckpt", "--output-dir", str(tmp_path / "out"), "--megatron-to-hf-mode", "bridge"])
    assert "export_ckpt" not in fake_bridge


def test_bridge_mode_without_the_package_says_so(tool, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "megatron.bridge", None)
    with pytest.raises(RuntimeError, match="megatron.bridge"):
        tool.main(
            [
                "--input-dir",
                "ckpt",
                "--output-dir",
                str(tmp_path / "out"),
                "--origin-hf-dir",
                "hf",
                "--megatron-to-hf-mode",
                "bridge",
            ]
        )


def test_default_mode_is_raw_and_the_flag_is_constrained(tool):
    parser = tool.build_parser()
    assert parser.parse_args(["--input-dir", "a", "--output-dir", "b"]).megatron_to_hf_mode == "raw"
    with pytest.raises(SystemExit):
        parser.parse_args(["--input-dir", "a", "--output-dir", "b", "--megatron-to-hf-mode", "direct"])
