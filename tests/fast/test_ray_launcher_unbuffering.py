from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_DIRS = ("examples", "scripts", "tools", "miles/utils/external_utils")
RAY_RUNTIME_ENV_MARKERS = ("runtime-env-json", "runtime_env=", "runtime_env_json")


def launcher_files() -> list[Path]:
    return sorted(
        path
        for launcher_dir in LAUNCHER_DIRS
        for path in (REPO_ROOT / launcher_dir).rglob("*")
        if path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts
    )


def ray_launchers() -> list[Path]:
    return [path for path in launcher_files() if any(marker in path.read_text() for marker in RAY_RUNTIME_ENV_MARKERS)]


def test_the_repo_has_ray_launchers_to_check() -> None:
    """A discovery bug that finds nothing would make every other check in this file vacuous."""
    assert len(ray_launchers()) > 30


@pytest.mark.parametrize("launcher", ray_launchers(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_ray_launcher_unbuffers_python(launcher: Path) -> None:
    """Ray buffers worker stdout unless PYTHONUNBUFFERED rides along with the job it submits."""
    assert "PYTHONUNBUFFERED" in launcher.read_text()


@pytest.mark.parametrize("launcher", ray_launchers(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_no_ray_launcher_spells_the_variable_wrong(launcher: Path) -> None:
    """PYTHONBUFFERED is not a variable python reads; the typo silently buffers everything."""
    assert "PYTHONBUFFERED" not in launcher.read_text().replace("PYTHONUNBUFFERED", "")
