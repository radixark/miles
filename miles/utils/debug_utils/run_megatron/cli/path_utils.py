"""Path resolution utilities for run_megatron CLI."""

import os
from pathlib import Path

import typer

_DEFAULT_MEGATRON_PATH: str = "/root/Megatron-LM"


def resolve_megatron_path(megatron_path: Path | None) -> str:
    if megatron_path is not None:
        return str(megatron_path)
    env_path: str | None = os.environ.get("MEGATRON_PATH")
    if env_path:
        return env_path
    return _DEFAULT_MEGATRON_PATH


def resolve_model_script(model_type: str) -> Path:
    repo_base: Path = _resolve_repo_base()
    script: Path = repo_base / "scripts" / "models" / f"{model_type}.sh"
    if not script.exists():
        raise typer.BadParameter(f"Model script not found: {script}")
    return script


def _resolve_repo_base() -> Path:
    current: Path = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Cannot find repo root (no pyproject.toml found above {current})")
