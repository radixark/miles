from pathlib import Path

import pytest
import typer

from miles.utils.debug_utils.run_megatron.cli.path_utils import (
    resolve_megatron_path,
    resolve_model_script,
)


class TestResolveMegatronPath:
    def test_explicit_path(self) -> None:
        explicit = Path("/custom/megatron")
        assert resolve_megatron_path(explicit) is explicit

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEGATRON_PATH", "/env/megatron")
        assert resolve_megatron_path(None) == Path("/env/megatron")

    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEGATRON_PATH", raising=False)
        assert resolve_megatron_path(None) == Path("/root/Megatron-LM")

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEGATRON_PATH", "/env/megatron")
        explicit = Path("/custom/megatron")
        assert resolve_megatron_path(explicit) is explicit


class TestResolveModelScript:
    def test_returns_path_when_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripts_dir = tmp_path / "scripts" / "models"
        scripts_dir.mkdir(parents=True)
        script_file = scripts_dir / "deepseek_v3.sh"
        script_file.touch()

        monkeypatch.setattr(
            "miles.utils.debug_utils.run_megatron.cli.path_utils._resolve_repo_base",
            lambda: tmp_path,
        )
        result = resolve_model_script("deepseek_v3")
        assert result == script_file

    def test_raises_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scripts_dir = tmp_path / "scripts" / "models"
        scripts_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "miles.utils.debug_utils.run_megatron.cli.path_utils._resolve_repo_base",
            lambda: tmp_path,
        )
        with pytest.raises(typer.BadParameter, match="Model script not found"):
            resolve_model_script("nonexistent")
