import hashlib
import importlib.metadata
import json
import os

import pytest

from tests.fast.launch_scripts.py_harness import REPO_ROOT, import_launch_script

_ROOT_ENV = "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"


class _Distribution:
    version = "0.2.0"

    def __init__(self, files: dict[str, str]):
        self.files = files

    def read_text(self, name: str) -> str | None:
        return self.files.get(name)


@pytest.fixture
def launcher():
    return import_launch_script(REPO_ROOT / "scripts/tpu/run_qwen3_4b_torchtitan.py")


def _install_torch_tpu(monkeypatch, files: dict[str, str]) -> None:
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Distribution(files))


def test_each_torch_tpu_build_gets_its_own_cache_directory(monkeypatch, launcher):
    roots = []
    for record in ("torch_tpu/_C.so,sha256=a,1\n", "torch_tpu/_C.so,sha256=b,1\n"):
        _install_torch_tpu(monkeypatch, {"RECORD": record})
        roots.append(launcher.ScriptArgs(compile_cache_dir="/cache").compile_cache_root)

    expected = hashlib.sha1(b"torch_tpu/_C.so,sha256=a,1\n").hexdigest()[:10]
    assert roots[0] == f"/cache/torch_tpu-0.2.0-{expected}"
    assert roots[0] != roots[1]


@pytest.mark.parametrize(
    "files",
    [
        {"RECORD": "torch_tpu/_C.so,,\n", "direct_url.json": json.dumps({"dir_info": {"editable": True}})},
        {},
    ],
    ids=["editable", "no-record"],
)
def test_an_install_without_a_build_identity_is_refused(monkeypatch, launcher, files):
    _install_torch_tpu(monkeypatch, files)

    args = launcher.ScriptArgs(compile_cache_dir="/cache")

    with pytest.raises(ValueError):
        _ = args.compile_cache_root


def test_the_cache_dir_sets_the_root_and_turns_off_torchs_device_blind_caches(monkeypatch, launcher):
    _install_torch_tpu(monkeypatch, {"RECORD": "torch_tpu/_C.so,sha256=a,1\n"})
    for name in (_ROOT_ENV, "TORCHINDUCTOR_AUTOGRAD_CACHE", "TORCHINDUCTOR_FX_GRAPH_CACHE"):
        monkeypatch.delenv(name, raising=False)

    launcher._set_compile_cache_env(launcher.ScriptArgs(compile_cache_dir="/cache"))

    assert os.environ[_ROOT_ENV].startswith("/cache/torch_tpu-0.2.0-")
    assert os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "0"
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "0"


def test_a_preset_cache_root_conflicts_with_the_cache_dir(monkeypatch, launcher):
    monkeypatch.setenv(_ROOT_ENV, "/elsewhere")

    with pytest.raises(ValueError):
        launcher._set_compile_cache_env(launcher.ScriptArgs(compile_cache_dir="/cache"))
