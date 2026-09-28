from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).parents[4] / "docker/compat/patch_sglang_moe_lora_grid.py"
)


def _load_patcher():
    spec = spec_from_file_location("patch_sglang_moe_lora_grid", PATCHER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_prefill_falls_back_instead_of_under_launching(tmp_path: Path) -> None:
    patcher = _load_patcher()
    backend = tmp_path / "base_backend.py"
    backend.write_text(patcher.BEFORE)

    patcher.patch_backend(backend)

    patched = backend.read_text()
    assert "if grid_size * block_size >= num_tokens:" in patched
    assert "assert grid_size * block_size >= num_tokens" not in patched
    assert "searchsorted implementation below" in patched


def test_patch_is_idempotent(tmp_path: Path) -> None:
    patcher = _load_patcher()
    backend = tmp_path / "base_backend.py"
    backend.write_text(patcher.BEFORE)

    patcher.patch_backend(backend)
    once = backend.read_text()
    patcher.patch_backend(backend)

    assert backend.read_text() == once
