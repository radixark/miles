import json
from pathlib import Path

from miles.utils.env_report.launcher_report import read_launcher_report
from miles.utils.external_utils.command_utils.helm_backend.launcher.launch_record import LaunchRecord
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles

RUN_ID = "260101-000000-000"
VALUES_FILE = Path("/shared/miles-runs") / RUN_ID / "values" / "values-1.yaml"


def _plan(**overrides) -> LaunchPlan:
    fields = dict(
        run_id=RUN_ID,
        release=f"miles-run-{RUN_ID}",
        namespace="rl",
        state_file=str(Path("/shared/miles-runs") / RUN_ID / "state" / "orchestrator-1.state"),
        worker_argv=["--rollout-num-gpus", "8"],
        orchestrator_command=["python", "/repo/train.py", "--rollout-num-gpus", "8"],
        env={"PYTHONUNBUFFERED": "1"},
    )
    fields.update(overrides)
    return LaunchPlan(**fields)


def _record(*, reachable_at=None, **overrides):
    return LaunchRecord.compute(plan=_plan(**overrides), values_file=VALUES_FILE, reachable_at=reachable_at or {})


class TestComputeLaunchRecord:
    def test_names_the_files_this_launch_writes(self) -> None:
        """The record is only joinable with a run's state and values if it names both."""
        record = _record()

        assert record.state_file.endswith("orchestrator-1.state")
        assert record.values_file == str(VALUES_FILE)

    def test_hides_a_secret_flag_in_the_recorded_argv(self) -> None:
        """The record lands on a shared disk and in the wandb config, so a key in argv would leak twice."""
        record = _record(
            worker_argv=["--wandb-key", "s3cret"],
            orchestrator_command=["python", "/repo/train.py", "--wandb-key=s3cret"],
        )

        assert "s3cret" not in json.dumps([*record.worker_argv, *record.orchestrator_command])
        assert "redacted-sha256:" in record.worker_argv[1]

    def test_hides_a_secret_environment_variable(self) -> None:
        record = _record(env={"HF_TOKEN": "t0ken", "PYTHONUNBUFFERED": "1"})

        assert "t0ken" not in record.env["HF_TOKEN"]
        assert record.env["PYTHONUNBUFFERED"] == "1"

    def test_the_runtime_reads_back_the_record_it_is_pointed_at(self, tmp_path: Path) -> None:
        """Pods are handed the path through --env-report, which is how it reaches the wandb config."""
        path = tmp_path / "launches" / "launch-1.json"
        _record().write(path=path)

        report = read_launcher_report(str(path))

        assert report["run_id"] == RUN_ID
        assert report["orchestrator_command"][1] == "/repo/train.py"


class TestWriteLaunchRecord:
    def test_writes_the_record_where_it_is_told_to(self, tmp_path: Path) -> None:
        path = tmp_path / "launches" / "launch-1.json"

        _record().write(path=path)

        assert json.loads(path.read_text())["release"] == f"miles-run-{RUN_ID}"

    def test_a_relaunch_does_not_overwrite_the_previous_record(self, tmp_path: Path) -> None:
        """A restarted run is exactly the case the record exists for, so neither launch may be lost."""
        first, second = tmp_path / "launches" / "launch-1.json", tmp_path / "launches" / "launch-2.json"

        _record(worker_argv=["--lr", "1"]).write(path=first)
        _record(worker_argv=["--lr", "2"]).write(path=second)

        assert json.loads(first.read_text())["worker_argv"] == ["--lr", "1"]
        assert json.loads(second.read_text())["worker_argv"] == ["--lr", "2"]

    def test_the_launcher_names_a_fresh_record_for_every_launch(self, tmp_path: Path) -> None:
        """One file per launch is what keeps a relaunched run's earlier launches readable."""
        assert RunFiles.new_record_file(run_directory=tmp_path) != RunFiles.new_record_file(run_directory=tmp_path)
        assert RunFiles.new_record_file(run_directory=tmp_path).parent == tmp_path / "launches"


class TestRecordsHowOtherDeploymentsReachThisOne:
    def test_a_trainer_release_records_the_address_the_driving_launch_has_to_be_given(self) -> None:
        """Nothing else prints it, and the digest in the name makes it impossible to derive by hand."""
        record = _record(reachable_at={"actor": "host-0.host.rl.svc.cluster.local:8000"})

        assert record.reachable_at == {"actor": "host-0.host.rl.svc.cluster.local:8000"}

    def test_a_release_nobody_dials_records_no_address(self) -> None:
        """A primary release is reached by nothing, so an address there would be a name without a reader."""
        assert _record().reachable_at == {}
