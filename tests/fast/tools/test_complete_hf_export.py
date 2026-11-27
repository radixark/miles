import json
import runpy
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch
from safetensors import safe_open
from tools.complete_hf_export import complete_export

import miles.utils.hf_utils.weight_mapping as weight_mapping


def _write_shard(directory: Path, shard_name: str, tensors: dict[str, torch.Tensor]) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, directory / shard_name)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _write_index(directory: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


def _read_all(directory: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name, shard in weight_map.items():
        with safe_open(directory / shard, framework="pt") as reader:
            out[name] = reader.get_tensor(name)
    return out


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """A base checkpoint of 6 tensors across 2 shards, with an index and a config."""
    base_dir = tmp_path / "base"
    torch.manual_seed(0)
    size = _write_shard(
        base_dir,
        "model-00001-of-00002.safetensors",
        {f"model.layers.{i}.weight": torch.randn(4, 3) for i in range(3)},
    )
    size += _write_shard(
        base_dir,
        "model-00002-of-00002.safetensors",
        {
            "model.visual.patch_embed.weight": torch.randn(8, 2),
            "mtp.fc.weight": torch.randn(5),
            "mtp.norm.weight": torch.randn(5),
        },
    )
    _write_index(
        base_dir,
        {
            **{f"model.layers.{i}.weight": "model-00001-of-00002.safetensors" for i in range(3)},
            "model.visual.patch_embed.weight": "model-00002-of-00002.safetensors",
            "mtp.fc.weight": "model-00002-of-00002.safetensors",
            "mtp.norm.weight": "model-00002-of-00002.safetensors",
        },
        size,
    )
    (base_dir / "config.json").write_text("{}")
    return base_dir


@pytest.fixture
def export(tmp_path: Path, base: Path) -> Path:
    """A raw export holding only the 3 trained tensors, with different values than the base."""
    export_dir = tmp_path / "export"
    trained = {f"model.layers.{i}.weight": torch.full((4, 3), float(i + 1)) for i in range(3)}
    size = _write_shard(export_dir, "model-00001.safetensors", trained)
    _write_index(export_dir, {name: "model-00001.safetensors" for name in trained}, size)
    return export_dir


def test_copies_missing_base_tensors(base: Path, export: Path) -> None:
    exported_shard = export / "model-00001.safetensors"
    exported_bytes = exported_shard.read_bytes()

    copied = complete_export(export, base)

    assert copied == ["model.visual.patch_embed.weight", "mtp.fc.weight", "mtp.norm.weight"]
    completed, original = _read_all(export), _read_all(base)
    assert set(completed) == set(original)
    for name in copied:
        torch.testing.assert_close(completed[name], original[name])
    # Trained tensors keep the export's values and the exported shard is untouched byte for byte.
    for i in range(3):
        torch.testing.assert_close(completed[f"model.layers.{i}.weight"], torch.full((4, 3), float(i + 1)))
    assert exported_shard.read_bytes() == exported_bytes
    # The missing tensors all come from one base shard, so they land in one completion shard, and
    # total_size now covers every tensor.
    index = json.loads((export / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {"model-00001.safetensors", "model-completion-00001.safetensors"}
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in completed.values())


def test_second_run_is_a_noop(base: Path, export: Path) -> None:
    complete_export(export, base)
    files_before = sorted(p.name for p in export.iterdir())

    assert complete_export(export, base) == []
    assert sorted(p.name for p in export.iterdir()) == files_before


def test_completion_from_a_second_base_is_refused(tmp_path: Path, base: Path, export: Path) -> None:
    complete_export(export, base)
    other_base = tmp_path / "other-base"
    size = _write_shard(other_base, "model.safetensors", {"model.layers.0.weight": torch.randn(4, 3)})
    _write_shard(other_base, "extra.safetensors", {"model.extra.weight": torch.randn(2)})
    _write_index(
        other_base, {"model.layers.0.weight": "model.safetensors", "model.extra.weight": "extra.safetensors"}, size
    )
    (other_base / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="completed from a different base"):
        complete_export(export, other_base)


def test_unrelated_namespaces_are_refused(tmp_path: Path, export: Path) -> None:
    native_base = tmp_path / "native-base"
    size = _write_shard(native_base, "model.safetensors", {"embed.weight": torch.randn(4, 3)})
    _write_index(native_base, {"embed.weight": "model.safetensors"}, size)
    (native_base / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="share no tensor names"):
        complete_export(export, native_base)


def test_native_base_names_are_remapped(tmp_path: Path, export: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A native-format base (e.g. DeepSeek V4) names tensors differently than the HF-style export;
    # the diff and the copies must use the remapped names.
    native_base = tmp_path / "native-base"
    torch.manual_seed(1)
    tensors = {"native.layers.0.weight": torch.randn(4, 3), "native.mtp.weight": torch.randn(5)}
    size = _write_shard(native_base, "model.safetensors", tensors)
    _write_index(native_base, {name: "model.safetensors" for name in tensors}, size)
    (native_base / "config.json").write_text("{}")
    monkeypatch.setattr(
        weight_mapping,
        "get_param_name_remap",
        lambda config, weight_map: lambda name: name.replace("native.", "model."),
    )

    copied = complete_export(export, native_base)

    assert copied == ["model.mtp.weight"]
    completed = _read_all(export)
    torch.testing.assert_close(completed["model.mtp.weight"], tensors["native.mtp.weight"])
    # The trained tensor the export already holds is recognized under its remapped name and not copied.
    torch.testing.assert_close(completed["model.layers.0.weight"], torch.full((4, 3), 1.0))


def test_base_without_config_is_completed_under_its_own_names(tmp_path: Path, export: Path) -> None:
    # A weights-only base (no config.json) cannot resolve a name remap; its names are used as is.
    bare_base = tmp_path / "bare-base"
    extra = torch.randn(2)
    size = _write_shard(
        bare_base, "model.safetensors", {"model.layers.0.weight": torch.randn(4, 3), "model.extra.weight": extra}
    )
    _write_index(
        bare_base, {"model.layers.0.weight": "model.safetensors", "model.extra.weight": "model.safetensors"}, size
    )

    copied = complete_export(export, bare_base)

    assert copied == ["model.extra.weight"]
    torch.testing.assert_close(_read_all(export)["model.extra.weight"], extra)


def test_export_without_index_is_refused(tmp_path: Path, base: Path) -> None:
    export_dir = tmp_path / "not-an-export"
    _write_shard(export_dir, "model.safetensors", {"model.layers.0.weight": torch.randn(4, 3)})

    with pytest.raises(FileNotFoundError, match="not a raw --save-hf export"):
        complete_export(export_dir, base)


def test_cli(base: Path, export: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["complete_hf_export.py", "--export-dir", str(export), "--hf-checkpoint", str(base)]
    )
    runpy.run_module("tools.complete_hf_export", run_name="__main__")

    assert set(_read_all(export)) == set(_read_all(base))
