import json
import multiprocessing
import shlex
import struct
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils import checkpoint_cache as cache


def _weights(path, value=b"ab"):
    path.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + value)
    (path / "config.json").write_text('{"model_type": "kimi_k3"}')
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))


def _recipe(layout="original"):
    return {"converter": "convert_mxfp4_to_bf16.py", "layout": layout}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("MILES_CI_CHECKPOINT_CACHE", "1")
    monkeypatch.setenv("MILES_CI_OVERWRITE_CHECKPOINT", "0")
    cache._close_leases()
    yield
    cache._close_leases()


def test_identical_recipe_reuses_the_same_files_without_conversion(tmp_path):
    path = tmp_path / "checkpoint"
    cache.cached_checkpoint(path, _recipe(), _weights)
    before = cache.snapshot(path)
    inode = (path / "model.safetensors").stat().st_ino
    cache._close_leases()
    cache.cached_checkpoint(path, _recipe(), lambda _: pytest.fail("unexpected conversion"))
    assert cache.snapshot(path) == before
    assert (path / "model.safetensors").stat().st_ino == inode


def test_unmerged_layout_cannot_replace_main_cache_without_label(tmp_path):
    path = tmp_path / "checkpoint"
    cache.cached_checkpoint(path, _recipe(), _weights)
    cache._close_leases()
    before = cache.snapshot(path)
    with pytest.raises(RuntimeError, match="overwrite-checkpoint"):
        cache.cached_checkpoint(path, _recipe("linear_attn"), lambda _: pytest.fail("must not write"))
    assert cache.snapshot(path) == before
    assert json.loads((path / cache.MANIFEST).read_text())["recipe"] == _recipe()


def test_permission_replaces_only_an_incompatible_cache(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    cache.cached_checkpoint(path, _recipe(), _weights)
    cache._close_leases()
    monkeypatch.setenv("MILES_CI_OVERWRITE_CHECKPOINT", "1")
    cache.cached_checkpoint(path, _recipe("linear_attn"), lambda p: _weights(p, b"cd"))
    cache.cached_checkpoint(path, _recipe("linear_attn"), lambda _: pytest.fail("label is not force-rebuild"))
    assert (path / "model.safetensors").read_bytes().endswith(b"cd")
    assert not list(tmp_path.glob(".checkpoint.building-*"))


def test_failed_rebuild_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    cache.cached_checkpoint(path, _recipe(), _weights)
    cache._close_leases()
    before = cache.snapshot(path)
    monkeypatch.setenv("MILES_CI_OVERWRITE_CHECKPOINT", "1")

    def fail(staging):
        _weights(staging, b"cd")
        raise ValueError("conversion failed")

    with pytest.raises(ValueError, match="conversion failed"):
        cache.cached_checkpoint(path, _recipe("changed"), fail)
    assert cache.snapshot(path) == before
    assert json.loads((path / cache.MANIFEST).read_text())["recipe"] == _recipe()
    assert not list(tmp_path.glob(".checkpoint.building-*"))


def test_legacy_or_modified_cache_cannot_be_silently_adopted(tmp_path):
    path = tmp_path / "checkpoint"
    _weights(path)
    with pytest.raises(RuntimeError, match="no compatibility manifest"):
        cache.cached_checkpoint(path, _recipe(), _weights)
    assert not (path / cache.MANIFEST).exists()


def test_modified_shard_is_rejected(tmp_path):
    path = tmp_path / "checkpoint"
    cache.cached_checkpoint(path, _recipe(), _weights)
    cache._close_leases()
    (path / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises((RuntimeError, AssertionError), match="(overwrite-checkpoint|Invalid safetensors)"):
        cache.cached_checkpoint(path, _recipe(), _weights)


def _reader(path, acquired, release):
    cache.cached_checkpoint(Path(path), _recipe(), _weights)
    acquired.set()
    assert release.wait(15)
    cache._close_leases()


def _writer(path, started, finished):
    import os

    os.environ["MILES_CI_OVERWRITE_CHECKPOINT"] = "1"
    started.set()
    cache.cached_checkpoint(Path(path), _recipe("changed"), _weights)
    cache._close_leases()
    finished.set()


def test_overwrite_waits_until_the_running_test_releases_its_reader_lease(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    acquired, release, started, finished = (ctx.Event() for _ in range(4))
    path = str(tmp_path / "checkpoint")
    reader = ctx.Process(target=_reader, args=(path, acquired, release))
    writer = ctx.Process(target=_writer, args=(path, started, finished))
    reader.start()
    try:
        assert acquired.wait(15)
        writer.start()
        assert started.wait(15)
        assert not finished.wait(0.5)
    finally:
        release.set()
        reader.join(15)
        if writer.pid is not None:
            writer.join(15)
        for process in (reader, writer):
            if process.is_alive():
                process.kill()
                process.join()
    assert reader.exitcode == writer.exitcode == 0
    assert finished.is_set()


def test_command_guard_preserves_environment_and_ignores_topology(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "result"
    _weights(source)
    monkeypatch.setattr(cache, "_conversion_code", lambda *args: {"model": "original"})
    monkeypatch.setattr(cache.importlib.metadata, "version", lambda _: "1.0")
    commands = []

    def execute(command):
        commands.append(command)
        tokens = shlex.split(command)
        _weights(Path(tokens[tokens.index("--save-dir") + 1]))

    tool = Path(cache.__file__).resolve().parents[4] / "tools/convert_mxfp4_to_bf16.py"
    command = (
        f"PYTHONPATH=/checkout-a NVTE_USE_FAST_MATH=0 python {tool} --model-dir {source} --save-dir {destination}"
    )
    cache.run_conversion(command + " --device cuda --max-workers 4", execute)
    cache._close_leases()
    cache.run_conversion(command.replace("/checkout-a", "/checkout-b") + " --device cuda --max-workers 8", execute)
    assert len(commands) == 1
    assert commands[0].startswith("PYTHONPATH=/checkout-a NVTE_USE_FAST_MATH=0 python ")
    assert str(destination) not in shlex.split(commands[0])
    with pytest.raises(RuntimeError, match="already in use"):
        cache.run_conversion(command + " --bf16", execute)


def test_changed_source_or_conversion_code_invalidates_the_cache(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "converted"
    _weights(source)
    monkeypatch.setattr(cache.importlib.metadata, "version", lambda _: "1")
    monkeypatch.setattr(cache, "_conversion_code", lambda *args: {"model": "original"})
    tool = Path(cache.__file__).resolve().parents[4] / "tools/convert_mxfp4_to_bf16.py"
    command = f"python {tool} --model-dir {source} --save-dir {destination}"

    def execute(cmd):
        argv = shlex.split(cmd)
        _weights(Path(argv[argv.index("--save-dir") + 1]))

    cache.run_conversion(command, execute)
    cache._close_leases()
    monkeypatch.setattr(cache, "_conversion_code", lambda *args: {"model": "linear_attn"})
    with pytest.raises(RuntimeError, match="overwrite-checkpoint"):
        cache.run_conversion(command, execute)
    monkeypatch.setattr(cache, "_conversion_code", lambda *args: {"model": "original"})
    (source / "config.json").write_text('{"model_type": "other"}')
    with pytest.raises(RuntimeError, match="overwrite-checkpoint"):
        cache.run_conversion(command, execute)


def test_dataset_download_does_not_enter_model_cache_policy():
    command = "hf download --repo-type dataset org/data --local-dir /datasets/data"
    assert cache.run_conversion(command, lambda cmd: cmd) == command


def test_hf_legacy_download_is_adopted_without_redownloading_weights(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    import huggingface_hub

    path = tmp_path / "model"
    _weights(path)
    siblings = []
    for name in cache.snapshot(path):
        siblings.append(SimpleNamespace(rfilename=name, lfs=None, blob_id=f"etag-{name}"))
        metadata = path / ".cache/huggingface/download" / (name + ".metadata")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"commit\netag-{name}\n{time.time() + 1}\n")
    info = SimpleNamespace(sha="pinned-commit", siblings=siblings)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **kw: info))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("must reuse HF files"))
    inode = (path / "model.safetensors").stat().st_ino
    cache.download_hf_checkpoint("org/model", local_dir=path)
    assert (path / cache.MANIFEST).is_file()
    assert (path / "model.safetensors").stat().st_ino == inode
    cache._close_leases()
    siblings[0].blob_id = "different-etag"
    with pytest.raises(RuntimeError, match="overwrite-checkpoint"):
        cache.download_hf_checkpoint("org/model", local_dir=path)


@pytest.mark.parametrize("registration", ['"kimi_k3"', '["kimi_k3", "kimi_alias"]'], ids=["single", "aliases"])
def test_kimi_layout_change_invalidates_only_relevant_model_code(tmp_path, monkeypatch, registration):
    from types import SimpleNamespace

    root = tmp_path / "repo"
    source = tmp_path / "source"
    _weights(source)
    files = {
        "tools/convert_hf_to_torch_dist.py": "# converter\n",
        "miles/backends/megatron_utils/arguments.py": "",
        "miles/backends/megatron_utils/model_provider.py": "",
        "miles/backends/megatron_utils/initialize.py": "",
        "miles/backends/megatron_utils/fp32_param_utils.py": "",
        "miles_plugins/mbridge/kimi_k3.py": f"@register_model({registration})\nclass Kimi: pass\n",
        "miles_plugins/models/kimi_k3/model.py": 'state_key = "self_attention.q_proj.weight"\n',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    megatron = tmp_path / "megatron"
    (megatron / "megatron/training").mkdir(parents=True)
    for name in ("arguments.py", "checkpointing.py", "training.py"):
        (megatron / "megatron/training" / name).write_text("")
    bridge = tmp_path / "mbridge"
    bridge.mkdir()
    (bridge / "__init__.py").write_text("")
    monkeypatch.setattr(cache, "__file__", str(root / "miles/utils/external_utils/command_utils/checkpoint_cache.py"))
    monkeypatch.setenv("MEGATRON_SOURCE_ROOT", str(megatron))
    monkeypatch.setattr(
        cache.importlib.util, "find_spec", lambda _: SimpleNamespace(submodule_search_locations=[bridge])
    )
    tool = root / "tools/convert_hf_to_torch_dist.py"
    before = cache._conversion_code(tool, {}, source, {})
    (root / "miles_plugins/models/kimi_k3/model.py").write_text(
        "# Formatting and comments do not change weights.\nstate_key='self_attention.q_proj.weight'\n"
    )
    assert cache._conversion_code(tool, {}, source, {}) == before
    (root / "miles/backends/megatron_utils/actor.py").write_text("training_change = True\n")
    assert cache._conversion_code(tool, {}, source, {}) == before
    (root / "miles_plugins/models/kimi_k3/model.py").write_text(
        'state_key = "self_attention.linear_attn.q_proj.weight"\n'
    )
    assert cache._conversion_code(tool, {}, source, {}) != before


def test_hf_update_reuses_unchanged_blobs_without_exposing_partial_files(tmp_path):
    import time

    original = tmp_path / "original"
    _weights(original)
    identities = {name: f"etag-{name}" for name in cache.snapshot(original)}
    for name, etag in identities.items():
        metadata = original / ".cache/huggingface/download" / (name + ".metadata")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"commit\n{etag}\n{time.time() + 1}\n")
    staging = tmp_path / "staging"
    identities["config.json"] = "changed-config"
    cache._seed_hf_download(original, staging, identities)
    assert (staging / "model.safetensors").stat().st_ino == (original / "model.safetensors").stat().st_ino
    assert not (staging / "config.json").exists()
    (staging / "config.json").write_text('{"model_type":"changed"}')
    assert json.loads((original / "config.json").read_text())["model_type"] == "kimi_k3"
    before = cache.snapshot(original, weights_only=True)
    (original / "README.md").write_text("Unrelated documentation change")
    assert cache.snapshot(original, weights_only=True) == before


def test_tp_only_shares_when_it_cannot_change_the_global_embedding_shape():
    def identity(**flags):
        return cache._weight_options("convert_hf_to_torch_dist.py", flags, "--hf-checkpoint", "--save")

    assert identity() == identity(**{"--bf16": [], "--tensor-model-parallel-size": ["1"]})
    assert identity() != identity(**{"--tensor-model-parallel-size": ["2"]})
    assert identity(**{"--padded-vocab-size": ["1024"], "--tensor-model-parallel-size": ["1"]}) == identity(
        **{"--padded-vocab-size": ["1024"], "--tensor-model-parallel-size": ["2"]}
    )
