import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from tests.fast.argument_snapshot.argparser.scenarios import capture_scenarios
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY, assert_scenario_snapshots, dump_snapshot

_SNAPSHOT_DIR = REPO_ROOT / SNAPSHOT_DIRECTORY / "argument_schema"
_SCENARIO_BASES = {
    "fsdp": "megatron",
    "fully_async": "megatron",
    "legacy": "megatron",
    "hook_rollout": "megatron",
    "legacy_hook_rollout": "megatron",
    "hook_generate": "megatron",
    "legacy_hook_generate": "megatron",
    "hook_inference": "megatron",
    "legacy_hook_inference": "megatron",
    "hook_without_arguments": "megatron",
    "hook_function": "megatron",
    "hook_fsdp": "fsdp",
    "megatron_repeat": "megatron",
}


class TestArgumentSnapshots:
    @pytest.mark.skip(reason="Snapshot baseline testing is disabled by op48-277")
    def test_parser_scenarios_match_snapshots(self) -> None:
        """Parser schemas and parsed values match their reviewed snapshots."""
        snapshots = {name: dump_snapshot(snapshot) for name, snapshot in capture_scenarios().items()}

        assert_scenario_snapshots(snapshots=snapshots, bases=_SCENARIO_BASES, directory=_SNAPSHOT_DIR)
