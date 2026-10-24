from tests.fast.argument_snapshot.scenarios import capture_scenarios
from tests.fast.argument_snapshot.schema import dump_snapshot
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "argument_snapshot"


class TestArgumentSnapshots:
    def test_parser_scenarios_match_snapshots(self) -> None:
        """Parser schemas and parsed values match their reviewed snapshots."""
        snapshots = capture_scenarios()

        for name, snapshot in snapshots.items():
            assert_matches_snapshot(
                snapshot=_SNAPSHOT_DIR / f"{name}.yaml",
                actual=dump_snapshot(snapshot),
                subject=f"argument scenario {name}",
            )
