import difflib
import os
from pathlib import Path

SNAPSHOT_UPDATE_ENV_VAR = "MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS"


def assert_scenario_snapshots(*, snapshots: dict[str, str], bases: dict[str, str], directory: Path) -> None:
    for name, snapshot in snapshots.items():
        suffix = ".yaml"
        if base := bases.get(name):
            snapshot = "".join(
                difflib.unified_diff(
                    snapshots[base].splitlines(keepends=True),
                    snapshot.splitlines(keepends=True),
                    fromfile=base,
                    tofile=name,
                    n=0,
                )
            )
            suffix = ".diff"
        assert_matches_snapshot(
            snapshot=directory / f"{name}{suffix}",
            actual=snapshot,
            subject=f"argument scenario {name}",
        )


def assert_matches_snapshot(snapshot: Path, actual: str, subject: str) -> None:
    if os.environ.get(SNAPSHOT_UPDATE_ENV_VAR):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual)
        return

    assert snapshot.exists(), f"missing snapshot for {subject}; regenerate with {SNAPSHOT_UPDATE_ENV_VAR}=1"
    expected = snapshot.read_text()
    if actual != expected:
        # a bare equality on a file-sized string reports nothing a reader can act on, and the
        # machine that regenerated the snapshot is rarely the one that fails on it
        diff = difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), fromfile=f"{snapshot}", tofile="actual", lineterm=""
        )
        raise AssertionError(f"{subject} does not match its snapshot:\n" + "\n".join(diff))
