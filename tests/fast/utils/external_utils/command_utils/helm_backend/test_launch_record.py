import json
from pathlib import Path

from miles.utils.env_report import decode_env_report
from miles.utils.external_utils.command_utils.helm_backend import launch_record


def _record(**overrides) -> launch_record.LaunchRecord:
    fields = dict(
        run_id="260101-000000-000",
        release="miles-run-260101-000000-000",
        namespace="rl",
        train_argv=["--rollout-num-gpus", "8"],
        worker_argv=["--rollout-num-gpus", "8", "--mooncake-master", "host:9000"],
        orchestrator_command=["python", "/repo/train.py", "--rollout-num-gpus", "8"],
        env={"PYTHONUNBUFFERED": "1"},
    )
    fields.update(overrides)
    return launch_record.LaunchRecord(**fields)


class TestEnvWithLaunchRecord:
    def test_pods_receive_the_record_the_runtime_can_decode(self) -> None:
        """Pods read the record back through --env-report, which is how it reaches the wandb config."""
        env = launch_record.env_with_launch_record({"PYTHONUNBUFFERED": "1"}, record=_record())

        decoded = decode_env_report(env[launch_record.LAUNCHER_REPORT_ENV_VAR])

        assert decoded["run_id"] == "260101-000000-000"
        assert decoded["orchestrator_command"][1] == "/repo/train.py"

    def test_keeps_the_environment_it_extends(self) -> None:
        env = launch_record.env_with_launch_record({"PYTHONUNBUFFERED": "1"}, record=_record())
        assert env["PYTHONUNBUFFERED"] == "1"


class TestWriteLaunchRecord:
    def test_writes_the_record_of_this_generation(self, tmp_path: Path) -> None:
        path = launch_record.write_launch_record(tmp_path, record=_record(), generation=3)

        assert path == tmp_path / "launches" / "generation-3.json"
        assert json.loads(path.read_text())["release"] == "miles-run-260101-000000-000"

    def test_a_relaunch_does_not_overwrite_the_previous_record(self, tmp_path: Path) -> None:
        """A restarted run is exactly the case the record exists for, so neither launch may be lost."""
        first = launch_record.write_launch_record(tmp_path, record=_record(train_argv=["--lr", "1"]), generation=1)
        second = launch_record.write_launch_record(tmp_path, record=_record(train_argv=["--lr", "2"]), generation=2)

        assert first != second
        assert json.loads(first.read_text())["train_argv"] == ["--lr", "1"]
        assert json.loads(second.read_text())["train_argv"] == ["--lr", "2"]
