from pathlib import Path

import pytest

from miles.utils.test_utils.snapshot import assert_scenario_snapshots, dump_snapshot
from tests.fast.argument_snapshot.parse_outputs.result_scenarios import result_scenarios
from tests.fast.argument_snapshot.parse_outputs.results import ResultScenario, capture_result
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


class TestArgumentResults:
    @pytest.mark.parametrize("name,scenario", list(result_scenarios().items()))
    def test_final_configs_match_snapshots(self, name: str, scenario: ResultScenario, result_files: Path) -> None:
        """Real parsing and validation preserve the complete effective configuration."""
        scenarios = {scenario.backend: ResultScenario(backend=scenario.backend), name: scenario}
        snapshots = {
            name: dump_snapshot(capture_result(scenario=scenario, directory=result_files))
            for name, scenario in scenarios.items()
        }
        assert_scenario_snapshots(
            snapshots=snapshots,
            bases={} if name == scenario.backend else {name: scenario.backend},
            directory=REPO_ROOT / "tests" / "snapshots" / "argument_results",
        )
