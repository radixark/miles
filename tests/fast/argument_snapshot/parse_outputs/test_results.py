from pathlib import Path

import pytest
from tests.fast.argument_snapshot.parse_outputs.result_scenarios import result_scenarios
from tests.fast.argument_snapshot.parse_outputs.results import ResultScenario, capture_result
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED
from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY, assert_scenario_snapshots, dump_snapshot


class TestArgumentResults:
    @pytest.mark.parametrize("name,scenario", list(result_scenarios().items()) if SNAPSHOT_TESTING_ENABLED else [])
    @pytest.mark.skip(reason="Snapshot baseline testing is disabled by op48-277")
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
            directory=REPO_ROOT / SNAPSHOT_DIRECTORY / "parsed_config",
        )


class TestRemovedArguments:
    @pytest.mark.parametrize("backend", ["megatron", "fsdp"])
    def test_unused_logprobs_batch_override_is_rejected(
        self, backend: str, result_files: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The retired log-probability batch option must not silently accept an ineffective setting."""
        scenario = ResultScenario(backend=backend, arguments=("--log-probs-max-tokens-per-gpu", "2048"))

        with pytest.raises(SystemExit) as error:
            capture_result(scenario=scenario, directory=result_files)

        assert error.value.code == 2
        assert "unrecognized arguments: --log-probs-max-tokens-per-gpu 2048" in capsys.readouterr().err
