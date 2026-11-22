import shlex

from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_realistic_gsm8k import _build_train_args


class TestSnapshotIdentity:
    def test_repeated_launches_keep_snapshot_identity_independent_of_run_directory(self) -> None:
        """Takeover launch commands retain one snapshot identity despite different temporary directories."""
        commands = [
            shlex.split(_build_train_args(directory, wandb_run_id=run_id))
            for directory, run_id in [("/first", "first-run"), ("/second", "second-run")]
        ]

        for command in commands:
            assert command[command.index("--config-snapshot-name") + 1] == "hot_restart_realistic_gsm8k"
            assert "--enable-event-analyzer" in command
            assert "--enable-witness" in command
        assert commands[0][commands[0].index("--wandb-run-id") + 1] == "first-run"
        assert commands[1][commands[1].index("--wandb-run-id") + 1] == "second-run"
