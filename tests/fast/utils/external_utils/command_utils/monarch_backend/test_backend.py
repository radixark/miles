import os
import shlex
from dataclasses import dataclass, field

import pytest

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.monarch_backend import backend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX
from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR, MONARCH_RUN_ENV_VARS_ENV_VAR
from miles.utils.workers.types import ClusterBackend

_HEAD_IP = "10.0.0.1"
_NODES = {"node-a": _HEAD_IP, "node-b": "10.0.0.2"}


@dataclass
class RecordedMonarchLaunch:
    cpu_commands: list[str] = field(default_factory=list)
    node_commands: list[str] = field(default_factory=list)

    def run_on_node(self, cmd: str, capture_output: bool = False) -> str | None:
        if cmd.startswith("scontrol show hostnames"):
            return "\n".join(_NODES) + "\n"
        self.node_commands.append(cmd)
        return f"ran {len(self.node_commands)}" if capture_output else None

    @property
    def driver_env(self) -> dict[str, str]:
        [driver] = [cmd for cmd in self.cpu_commands if cmd.startswith("env ")]
        argv = shlex.split(driver)[1:]
        return dict(arg.split("=", 1) for arg in argv[: argv.index("python3")])


@pytest.fixture
def recorded_launch(monkeypatch: pytest.MonkeyPatch) -> RecordedMonarchLaunch:
    recorded = RecordedMonarchLaunch()
    for name in [name for name in os.environ if name.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)
    for name in ("MASTER_ADDR", "MOONCAKE_MASTER", MONARCH_JOB_STATE_ENV_VAR, "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        base_backend, "run_shell_command", lambda cmd, capture_output=False: recorded.cpu_commands.append(cmd)
    )
    monkeypatch.setattr(backend, "run_shell_command", recorded.run_on_node)
    monkeypatch.setattr(backend, "start_mooncake_master", lambda **kwargs: None)
    monkeypatch.setattr(backend, "get_current_node_ip", lambda: _HEAD_IP)
    monkeypatch.setattr(backend.socket, "gethostbyname", _NODES.__getitem__)
    return recorded


@pytest.fixture
def in_slurm_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MONARCH_JOB_STATE_ENV_VAR, "/shared/logs/miles-abc123/.monarch/job_state.pkl")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")
    monkeypatch.setattr(backend, "is_port_available", lambda port: True)


def _launch(*, num_nodes: int, **kwargs) -> None:
    ExecuteTrainConfig(cluster_backend=ClusterBackend.MONARCH, num_nodes=num_nodes).create_backend().execute_train(
        train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None, **kwargs
    )


class TestTheDriverCommand:
    def test_train_py_is_told_it_runs_on_monarch(self, recorded_launch: RecordedMonarchLaunch):
        """train.py defaults to ray, and this launch starts no ray cluster for it to join."""
        _launch(num_nodes=1)

        [driver] = [cmd for cmd in recorded_launch.cpu_commands if cmd.startswith("env ")]
        assert driver.endswith("--cluster-backend monarch")

    def test_an_explicit_monarch_flag_is_not_repeated(self, recorded_launch: RecordedMonarchLaunch):
        ExecuteTrainConfig(cluster_backend=ClusterBackend.MONARCH).create_backend().execute_train(
            train_args="--train-backend fsdp --cluster-backend monarch", num_gpus_per_node=8, megatron_model_type=None
        )

        [driver] = [cmd for cmd in recorded_launch.cpu_commands if cmd.startswith("env ")]
        assert driver.count("--cluster-backend") == 1


class TestTheHostsOfAMultiNodeRun:
    def test_a_multi_node_run_without_hosts_is_refused(self, recorded_launch: RecordedMonarchLaunch):
        """Without a job or worker loops every worker would land on the driver's one host."""
        with pytest.raises(AssertionError, match="monarch.slurm"):
            _launch(num_nodes=2)

    def test_a_run_in_a_submitted_job_reaches_the_head_node_by_its_address(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None
    ):
        """Workers on the other hosts cannot reach a loopback master or Mooncake master."""
        _launch(num_nodes=2)

        env = recorded_launch.driver_env
        assert env["MASTER_ADDR"] == _HEAD_IP
        assert env["MOONCAKE_MASTER"] == f"{_HEAD_IP}:50051"

    def test_an_exported_mooncake_master_wins(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")

        _launch(num_nodes=2)

        assert "MOONCAKE_MASTER" not in recorded_launch.driver_env

    @pytest.mark.parametrize("num_nodes", [1, 3])
    def test_a_run_that_asks_for_other_nodes_than_its_job_holds_is_refused(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None, num_nodes: int
    ):
        """Fewer would idle nodes the job holds; more would only fail at the gpu layout, after the prepare steps."""
        with pytest.raises(AssertionError, match="the monarch job it runs in holds 2"):
            _launch(num_nodes=num_nodes)
        assert recorded_launch.cpu_commands == []

    def test_a_run_in_a_job_kills_nothing_on_its_head_node(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None
    ):
        """On a shared node the sweep would kill the same user's other runs; a fresh allocation has none of its own."""
        _launch(num_nodes=2)

        assert not any("pkill" in cmd for cmd in recorded_launch.cpu_commands)

    def test_a_job_whose_master_port_is_taken_is_refused(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None, monkeypatch: pytest.MonkeyPatch
    ):
        """Another job's master on that port would otherwise be reused as this run's own."""
        monkeypatch.setattr(backend, "is_port_available", lambda port: False)

        with pytest.raises(AssertionError, match="port 50051 of the job's head node is taken"):
            _launch(num_nodes=2)

    def test_a_single_node_run_stays_on_loopback(self, recorded_launch: RecordedMonarchLaunch):
        _launch(num_nodes=1)

        env = recorded_launch.driver_env
        assert env["MASTER_ADDR"] == "127.0.0.1"
        assert "MOONCAKE_MASTER" not in env

    def test_the_driver_learns_which_variables_every_worker_needs(self, recorded_launch: RecordedMonarchLaunch):
        """A worker on another host only gets the launcher's variables when the driver passes them on."""
        _launch(num_nodes=1, extra_env_vars={"NCCL_DEBUG": "INFO"})

        env = recorded_launch.driver_env
        names = env[MONARCH_RUN_ENV_VARS_ENV_VAR].split(",")
        assert {"PYTHONPATH", "MASTER_ADDR", "NCCL_DEBUG"} <= set(names)
        assert set(names) == set(env) - {MONARCH_RUN_ENV_VARS_ENV_VAR}


class TestACommandOnEveryNode:
    def test_each_node_of_the_job_runs_it_with_its_own_placeholders(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None
    ):
        """A conversion spread over nodes needs each node's rank and address, as the ray launcher substitutes them."""
        outputs = (
            ExecuteTrainConfig(cluster_backend=ClusterBackend.MONARCH, num_nodes=2)
            .create_backend()
            .exec_command_multi_node("convert --rank {{node_rank}}/{{nnodes}} --ip {{node_ip}}", capture_output=True)
        )

        assert len(outputs) == 2
        assert sorted(recorded_launch.node_commands) == [
            "srun --overlap --nodes=1 --ntasks=1 -w node-a bash -c "
            "'unset CUDA_VISIBLE_DEVICES; convert --rank 0/2 --ip 10.0.0.1'",
            "srun --overlap --nodes=1 --ntasks=1 -w node-b bash -c "
            "'unset CUDA_VISIBLE_DEVICES; convert --rank 1/2 --ip 10.0.0.2'",
        ]

    def test_every_node_names_the_head_node_by_the_address_the_driver_exports(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None, monkeypatch: pytest.MonkeyPatch
    ):
        """On a dual-stack cluster the head's hostname can resolve to another address than the one the run uses."""
        monkeypatch.setattr(backend.socket, "gethostbyname", {**_NODES, "node-a": "fd00::1"}.__getitem__)

        ExecuteTrainConfig(
            cluster_backend=ClusterBackend.MONARCH, num_nodes=2
        ).create_backend().exec_command_multi_node("join {{master_addr}}")

        assert all(f"join {_HEAD_IP}" in cmd for cmd in recorded_launch.node_commands)

    def test_the_head_node_is_rank_0_wherever_slurm_lists_it(
        self, recorded_launch: RecordedMonarchLaunch, in_slurm_job: None, monkeypatch: pytest.MonkeyPatch
    ):
        """Rank 0 of a rendezvous listens where {{master_addr}} points, the head node the launcher runs on."""
        monkeypatch.setenv("SLURMD_NODENAME", "node-b")
        monkeypatch.setattr(backend, "get_current_node_ip", lambda: _NODES["node-b"])

        ExecuteTrainConfig(
            cluster_backend=ClusterBackend.MONARCH, num_nodes=2
        ).create_backend().exec_command_multi_node("rank {{node_rank}} at {{node_ip}}")

        assert sorted(recorded_launch.node_commands) == [
            "srun --overlap --nodes=1 --ntasks=1 -w node-a bash -c 'unset CUDA_VISIBLE_DEVICES; rank 1 at 10.0.0.1'",
            "srun --overlap --nodes=1 --ntasks=1 -w node-b bash -c 'unset CUDA_VISIBLE_DEVICES; rank 0 at 10.0.0.2'",
        ]
