import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("monarch")

from monarch._src.job import slurm as monarch_slurm

from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR
from miles.utils.workers.monarch.slurm import SlurmSubmission, main, submit

_COMMAND = ["python", "scripts/run_qwen3_4b.py", "train", "--num-gpus-per-node", "8"]


@dataclass
class RecordedSbatch:
    scripts: list[str] = field(default_factory=list)

    def run(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert argv == ["sbatch"], argv
        self.scripts.append(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 4242\n", stderr="")


@pytest.fixture
def recorded_sbatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RecordedSbatch:
    recorded = RecordedSbatch()
    monkeypatch.setattr(monarch_slurm.subprocess, "run", recorded.run)
    monkeypatch.chdir(tmp_path)
    for name in (
        MONARCH_JOB_STATE_ENV_VAR,
        "MILES_SCRIPT_CLUSTER_BACKEND",
        "SBATCH_EXPORT",
        "MASTER_ADDR",
    ):
        monkeypatch.delenv(name, raising=False)
    return recorded


def _submission(**overrides) -> SlurmSubmission:
    return SlurmSubmission(
        **{
            "command": _COMMAND,
            "num_nodes": 2,
            "gpus_per_node": 8,
            "slurm_args": ["--partition=gpu", "--time=1:00:00"],
            "log_dir": "logs",
            "job_name": "miles",
            **overrides,
        }
    )


def _client_command(script: str) -> list[str]:
    """The command Monarch's batch runner runs, split as it splits it."""
    [runner_line] = [line for line in script.splitlines() if "monarch._src.job._slurm_batch" in line]
    return shlex.split(shlex.split(runner_line)[-1])


class TestSubmittingARun:
    def test_one_batch_job_holds_every_node_and_runs_the_launcher_command(self, recorded_sbatch: RecordedSbatch):
        """The job must ask for all the run's nodes and gpus, and run the launcher command unchanged in this checkout."""
        assert submit(_submission()) == "4242"

        [script] = recorded_sbatch.scripts
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gpus-per-node=8" in script
        assert f"#SBATCH --chdir={os.getcwd()}" in script
        assert "#SBATCH --partition=gpu" in script
        assert "monarch._src.job._slurm_batch" in script
        assert _client_command(script)[-len(_COMMAND) :] == _COMMAND
        assert "#SBATCH --exclusive" in script

    def test_a_shared_job_leaves_room_for_other_jobs_on_its_nodes(self, recorded_sbatch: RecordedSbatch):
        """A partly used node only fits a job that asks for its own cpus and memory instead of the whole node."""
        submit(_submission(exclusive=False, slurm_args=["--cpus-per-task=32", "--mem=200G"]))

        [script] = recorded_sbatch.scripts
        assert "#SBATCH --exclusive" not in script
        assert "#SBATCH --cpus-per-task=32" in script

    def test_the_job_finds_its_saved_state_and_runs_on_monarch(self, recorded_sbatch: RecordedSbatch):
        """The job inherits this environment, which is how the driver learns where to attach to its hosts."""
        submit(_submission())

        [script] = recorded_sbatch.scripts
        env, *assignments = _client_command(script)[: -len(_COMMAND)]
        job_env = dict(assignment.split("=", 1) for assignment in assignments)
        state = Path(job_env[MONARCH_JOB_STATE_ENV_VAR])
        assert env == "env" and job_env["MILES_SCRIPT_CLUSTER_BACKEND"] == "monarch"
        assert state.is_file() and state.is_relative_to(Path("logs").resolve())
        assert MONARCH_JOB_STATE_ENV_VAR not in os.environ

    def test_a_head_address_only_the_job_can_decide_is_refused(self, recorded_sbatch: RecordedSbatch, monkeypatch):
        """The job inherits this shell, so a stale head address would win over the job's own."""
        monkeypatch.setenv("MASTER_ADDR", "10.9.9.9")

        with pytest.raises(AssertionError, match="MASTER_ADDR is set in the submitting environment"):
            submit(_submission())
        assert recorded_sbatch.scripts == []

    def test_a_command_reaches_the_job_through_the_batch_script_unchanged(self, recorded_sbatch: RecordedSbatch):
        """The command is quoted into the script and split again by monarch's batch runner, inner -- included."""
        command = ["python", "scripts/run x.py", "train", "--extra-args", "--name 'a b' -- --c", "--"]

        submit(_submission(command=command))

        [script] = recorded_sbatch.scripts
        assert _client_command(script)[-len(command) :] == command

    def test_an_export_that_drops_the_environment_is_refused(self, recorded_sbatch: RecordedSbatch, monkeypatch):
        """Without the submitting environment the job could not find its saved state."""
        monkeypatch.setenv("SBATCH_EXPORT", "NONE")

        with pytest.raises(AssertionError, match="SBATCH_EXPORT"):
            submit(_submission())

    def test_the_worker_loops_listen_where_they_are_told(self, recorded_sbatch: RecordedSbatch):
        """On a dual-stack cluster a loop on its hostname's IPv6 address never sees the driver's IPv4 dial."""
        submit(_submission(bind_to="::"))

        [script] = recorded_sbatch.scripts
        assert "--bind-to ::" in script

    def test_the_worker_loops_listen_on_the_port_they_are_given(self, recorded_sbatch: RecordedSbatch):
        """Two jobs sharing a node cannot both take the default port."""
        submit(_submission(monarch_port=23456))

        [script] = recorded_sbatch.scripts
        assert "--port 23456" in script


class TestTheSubmissionItself:
    @pytest.mark.parametrize(
        "slurm_arg", ["--nodes=3", "-N 3", "--partition=gpu --exclusive", "-J other", "--gres=gpu:4", "-n 16"]
    )
    def test_a_flag_the_submission_sets_is_refused(self, slurm_arg: str):
        """A second --nodes or --exclusive would contradict the one the submission writes."""
        with pytest.raises(AssertionError, match="must not set"):
            _submission(slurm_args=[slurm_arg])

    @pytest.mark.parametrize("slurm_arg", ["partition=gpu", " --time=4:00:00"])
    def test_an_entry_sbatch_would_never_see_is_refused(self, slurm_arg: str):
        """Monarch writes only entries that start with a dash, so any other would silently not apply."""
        with pytest.raises(AssertionError, match="no sbatch flag"):
            _submission(slurm_args=[slurm_arg])

    def test_a_bind_address_that_is_no_ip_is_refused_before_anything_is_made(self):
        """Monarch takes only an ip there, and would refuse a hostname once the job's directory exists."""
        with pytest.raises(ValueError, match="node-a"):
            _submission(bind_to="node-a")

    def test_a_submission_without_a_command_is_refused(self):
        with pytest.raises(AssertionError, match="launcher command"):
            _submission(command=[])

    def test_the_command_after_the_separator_reaches_the_job_verbatim(self, recorded_sbatch: RecordedSbatch, capsys):
        """--dry-run shows what would run and submits nothing."""
        assert (
            main(["--nodes", "2", "--gpus-per-node", "8", "--slurm-arg=--partition=gpu", "--dry-run", "--", *_COMMAND])
            is None
        )

        assert f"command={_COMMAND!r}" in capsys.readouterr().out
        assert recorded_sbatch.scripts == []

    def test_a_dry_run_refuses_what_the_submission_would(self, recorded_sbatch: RecordedSbatch, monkeypatch):
        """A dry run that passes where the real submission fails would vouch for nothing."""
        monkeypatch.setenv("SBATCH_EXPORT", "NONE")

        with pytest.raises(AssertionError, match="SBATCH_EXPORT"):
            main(["--nodes", "2", "--gpus-per-node", "8", "--dry-run", "--", *_COMMAND])
