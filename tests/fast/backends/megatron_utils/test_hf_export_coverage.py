"""CPU tests for the HF export coverage check: a partial export must not be marked complete."""

import ast
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture(scope="module")
def missing_hf_weights() -> Callable:
    # Execute the production helper without importing GPU-only Megatron modules.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/hf_export.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name == "missing_hf_weights") or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SAFETENSORS_INDEX" for t in node.targets))]
    namespace = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["missing_hf_weights"]


SOURCE = {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "mtp.fc.weight": "model-00002-of-00002.safetensors",
}


def _checkpoint(root: Path, weight_map: dict[str, str] | None, shards: set[str]) -> Path:
    root.mkdir(parents=True)
    if weight_map is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in shards:
        (root / shard).write_bytes(b"")
    return root


def test_complete_export(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", SOURCE, set(SOURCE.values()))
    assert missing_hf_weights(source, export) == []


def test_dropped_shard_takes_its_mapped_neighbours_with_it(tmp_path, missing_hf_weights):
    """Qwen3.8-27B: unmapped MTP tensors kept shard 2 from being written, and lm_head with it."""
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(
        tmp_path / "out",
        {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"},
        {"model-00001-of-00002.safetensors"},
    )
    assert missing_hf_weights(source, export) == ["lm_head.weight", "mtp.fc.weight"]


def test_indexed_shard_that_was_never_written(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", SOURCE, {"model-00001-of-00002.safetensors"})
    assert missing_hf_weights(source, export) == ["lm_head.weight", "mtp.fc.weight"]


def test_no_export_index(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", None, {"model.safetensors"})
    assert missing_hf_weights(source, export) == sorted(SOURCE)


def test_source_without_a_local_index_is_not_checked(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", None, {"model.safetensors"})
    export = _checkpoint(tmp_path / "out", None, {"model.safetensors"})
    assert missing_hf_weights(source, export) == []
    assert missing_hf_weights("Qwen/Qwen3-8B", export) == []
