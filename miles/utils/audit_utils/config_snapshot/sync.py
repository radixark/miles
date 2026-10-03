"""Sync CI records to local golden snapshots.

- Run from the repo root with Miles dependencies and authenticated `gh`.
- Download to an empty directory outside the repo:
  uv run --no-project python -m miles.utils.audit_utils.config_snapshot.sync download --run-id <id> --directory <dir>
- Update golden files in this checkout:
  uv run --no-project python -m miles.utils.audit_utils.config_snapshot.sync sync --directory <dir>
- Optional: download --repo OWNER/REPO; sync --repo-root <target-repo>.
- Sync overwrites affected golden files; preserve local edits first. No commit or push.
- Review: git diff -- tests/snapshots/runtime_config
"""

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer
import yaml

from miles.utils.audit_utils.config_snapshot.compact import ConfigSnapshotBases
from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt
from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotTestRunner
from miles.utils.audit_utils.config_snapshot.serialization import dump_config_snapshot
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.file_utils import atomic_write_text

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]

app = typer.Typer(add_completion=False)


@app.command(help="Download raw snapshot artifacts of one workflow run")
def download(
    run_id: Annotated[str, typer.Option(help="GitHub Actions run id")],
    directory: Annotated[Path, typer.Option(help="Empty download directory")],
    repo: Annotated[str, typer.Option(help="GitHub repository")] = "radixark/miles",
) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Download directory must be empty: {directory}")
    argv = [
        "gh",
        "run",
        "download",
        run_id,
        "--repo",
        repo,
        "--pattern",
        "snapshot-records-*",
        "--dir",
        str(directory),
    ]
    logger.info(f"EXEC: {shlex.join(argv)}")
    subprocess.run(argv, check=True)


@app.command(help="Convert raw dumps and update local golden snapshots")
def sync(
    directory: Annotated[Path, typer.Option(help="Downloaded artifact directory")],
    repo_root: Annotated[Path, typer.Option(help="Repository receiving the snapshots")] = _REPO_ROOT,
) -> None:
    snapshots = _collect_snapshots(directory=directory, repo_root=repo_root)
    for target, content in sorted(snapshots.items()):
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=target, text=content)
    print(f"Updated {len(snapshots)} golden file(s)", flush=True)


def _collect_snapshots(*, directory: Path, repo_root: Path) -> dict[Path, str]:
    bases = ConfigSnapshotBases.model_validate(
        yaml.safe_load((repo_root / "tests/snapshots/runtime_config/base.yaml").read_text())
    )
    completed: dict[str, list[Path]] = {}
    incomplete: set[str] = set()
    for path in sorted(directory.rglob("attempt.json")):
        attempt = ConfigSnapshotTestAttempt.model_validate_json(path.read_text())
        if attempt.completed:
            completed.setdefault(attempt.test, []).append(path.parent / "records")
        elif any((path.parent / "records").glob("*.json")):
            incomplete.add(attempt.test)
    if missing := incomplete - completed.keys():
        raise ValueError(f"No completed attempt for: {', '.join(sorted(missing))}")

    snapshots: dict[Path, str] = {}
    for test, directories in sorted(completed.items()):
        target = ConfigSnapshotTestRunner.golden_path(test=test, repo_root=repo_root)
        cases = [ConfigSnapshotConverter.convert(ConfigSnapshotStorage(directory=path).read()) for path in directories]
        contents = [dump_config_snapshot(case, bases=bases) for case in cases]
        if any(content != contents[0] for content in contents[1:]):
            raise ValueError(f"Completed attempts disagree for {test}: {directories}")
        if not cases[0].processes:
            continue
        if target in snapshots and snapshots[target] != contents[0]:
            raise ValueError(f"Tests disagree on golden file {target}")
        snapshots[target] = contents[0]
    if not snapshots:
        raise ValueError(f"No completed snapshot records found under {directory}")
    return snapshots


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
