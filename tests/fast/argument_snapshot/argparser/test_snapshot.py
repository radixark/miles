from miles.utils.test_utils.snapshot import assert_scenario_snapshots
from tests.fast.argument_snapshot.argparser.scenarios import capture_scenarios
from tests.fast.argument_snapshot.schema import dump_snapshot
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "argument_snapshot"
_SCENARIO_BASES = {
    "fully_async": "megatron",
    "legacy": "megatron",
    "custom_megatron": "megatron",
    "custom_fsdp": "fsdp",
    "hook_rollout": "megatron",
    "legacy_hook_rollout": "megatron",
    "hook_generate": "megatron",
    "legacy_hook_generate": "megatron",
    "hook_inference": "megatron",
    "legacy_hook_inference": "megatron",
    "missing_hook_module": "megatron",
    "hook_without_arguments": "megatron",
    "invalid_hook_path": "megatron",
    "hook_function": "megatron",
    "hook_fsdp": "fsdp",
    "megatron_repeat": "megatron",
}


class TestArgumentSnapshots:
    def test_parser_scenarios_match_snapshots(self) -> None:
        """Parser schemas and parsed values match their reviewed snapshots."""
        snapshots = {name: dump_snapshot(snapshot) for name, snapshot in capture_scenarios().items()}

        assert_scenario_snapshots(snapshots=snapshots, bases=_SCENARIO_BASES, directory=_SNAPSHOT_DIR)
