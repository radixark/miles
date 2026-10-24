import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import shlex
from pathlib import Path

from tests.fast.utils.test_command_utils import _backend, _submit, bare_environment, commands

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


__all__ = ["bare_environment", "commands"]


class TestSnapshotLaunchIdentity:
    @pytest.mark.parametrize(
        ("train_args", "expected_names"),
        [
            ("--ci-test", ["tests/e2e/snapshot_launch/run-0000", "tests/e2e/snapshot_launch/run-0001"]),
            ("--ci-test --config-snapshot-name explicit", ["explicit", "explicit"]),
            ("--ci-test --ci-disable-config-snapshot", [None, None]),
        ],
    )
    def test_submitted_training_commands_retain_their_snapshot_identity(
        self,
        commands: list[str],
        monkeypatch: pytest.MonkeyPatch,
        train_args: str,
        expected_names: list[str | None],
    ) -> None:
        """The actual launcher assigns stable names while preserving explicit and disabled capture."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        monkeypatch.setattr(base_backend, "_CONFIG_SNAPSHOT_LAUNCH_COUNTS", {})
        monkeypatch.setattr("sys.argv", [str(Path(base_backend.repo_base_dir) / "tests/e2e/snapshot_launch.py")])

        observed = []
        for _ in range(2):
            _backend().execute_train(train_args=train_args, num_gpus_per_node=1, megatron_model_type="qwen3-4B")
            argv = shlex.split(_submit(commands))
            observed.append(
                argv[argv.index("--config-snapshot-name") + 1] if "--config-snapshot-name" in argv else None
            )

        assert observed == expected_names
        if "--ci-disable-config-snapshot" in train_args:
            assert "--ci-disable-config-snapshot" in argv
