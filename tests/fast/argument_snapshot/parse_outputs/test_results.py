from pathlib import Path

from miles.utils.test_utils.snapshot import assert_scenario_snapshots, dump_snapshot
from tests.fast.argument_snapshot.parse_outputs.results import ResultScenario, capture_result
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


class TestArgumentResults:
    def test_final_configs_match_snapshots(self, tmp_path: Path) -> None:
        """Real parsing and validation preserve the complete effective configuration."""
        scenarios = {backend: ResultScenario(backend=backend) for backend in ("megatron", "fsdp")}
        snapshots = {
            name: dump_snapshot(capture_result(scenario=scenario, directory=tmp_path))
            for name, scenario in scenarios.items()
        }
        assert_scenario_snapshots(
            snapshots=snapshots,
            bases={},
            directory=REPO_ROOT / "tests" / "snapshots" / "argument_results",
        )
