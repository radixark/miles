import os
import shlex
from dataclasses import dataclass, field

import pytest

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.monarch_backend import backend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX
from miles.utils.workers.types import ClusterBackend


@dataclass
class RecordedMonarchLaunch:
    cpu_commands: list[str] = field(default_factory=list)

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
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.setattr(
        base_backend, "run_shell_command", lambda cmd, capture_output=False: recorded.cpu_commands.append(cmd)
    )
    monkeypatch.setattr(backend, "start_mooncake_master", lambda **kwargs: None)
    return recorded


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

    def test_the_run_stays_on_loopback(self, recorded_launch: RecordedMonarchLaunch):
        _launch(num_nodes=1)

        assert recorded_launch.driver_env["MASTER_ADDR"] == "127.0.0.1"

    def test_a_multi_node_run_is_refused(self, recorded_launch: RecordedMonarchLaunch):
        """Every worker would land on the driver's one host."""
        with pytest.raises(AssertionError, match="one host"):
            _launch(num_nodes=2)
