import dataclasses
import inspect
import os

import pytest
import typer

from miles.utils.external_utils.command_utils import CommandUtilConfig
from miles.utils.external_utils.command_utils.base_backend import (
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    default_config,
)
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX, dataclass_cli
from miles.utils.workers.types import ClusterBackend, DeployComponent


@pytest.fixture(autouse=True)
def bare_environment(monkeypatch):
    """No test may read a variable the shell that started pytest happened to export."""
    for name in [name for name in os.environ if name.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


class TestAScriptReadsItsLauncherConfigFromTheEnvironment:
    def test_an_unset_environment_leaves_every_default_alone(self, monkeypatch):
        """A ray run sets nothing, so reading the environment must not change what it used to get."""
        monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "pinned")

        assert default_config() == ExecuteTrainConfig(run_id="pinned")

    def test_each_launch_is_stamped_with_its_own_run_id(self):
        """Two runs installed from one namespace under one name would be one run."""
        assert default_config().run_id != default_config().run_id

    def test_the_backend_is_chosen_by_one_variable(self, monkeypatch):
        """This is the whole point: the same e2e script has to reach either cluster without being edited."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")

        assert default_config().cluster_backend is ClusterBackend.KUBERNETES

    def test_a_repeatable_option_is_split_the_way_the_command_line_splits_it(self, monkeypatch):
        """The variable feeds the same click option the script path parses, so one list has one shape."""
        monkeypatch.setenv("MILES_SCRIPT_HELM_VALUES", "/a/infra.yaml /b/infra.yaml")

        assert default_config().helm_values == ("/a/infra.yaml", "/b/infra.yaml")

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_a_flag_is_true_for_every_spelling_a_command_line_accepts(self, value: str, monkeypatch):
        """An operator who types what click accepts must not silently get the opposite."""
        monkeypatch.setenv("MILES_SCRIPT_CI_RUN", value)

        assert default_config().ci_run is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_a_flag_is_false_for_every_spelling_a_command_line_accepts(self, value: str, monkeypatch):
        """`MILES_SCRIPT_CI_RUN=false` reading as true is the failure this rules out."""
        monkeypatch.setenv("MILES_SCRIPT_CI_RUN", value)

        assert default_config().ci_run is False

    def test_a_flag_that_is_neither_is_refused_rather_than_guessed(self, monkeypatch):
        """Guessing would install a backend nobody asked for, and a run is expensive to discover that on."""
        monkeypatch.setenv("MILES_SCRIPT_CI_RUN", "maybe")

        with pytest.raises(typer.BadParameter):
            default_config()

    def test_a_number_arrives_as_a_number(self, monkeypatch):
        """num_nodes reaches arithmetic, and a string would only fail much later."""
        monkeypatch.setenv("MILES_SCRIPT_NUM_NODES", "4")

        assert default_config().num_nodes == 4

    @pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(ExecuteTrainConfig)])
    def test_every_field_answers_to_the_variable_the_command_line_binds(self, field: str):
        """The variable a test sets and the one a script's own option reads are the same one."""

        @dataclass_cli
        def train(args: ExecuteTrainConfig) -> None: ...

        bound = inspect.signature(train).parameters[field].annotation.__metadata__[0].envvar

        assert bound == f"{SCRIPT_ENV_VAR_PREFIX}{field.upper()}"


def _refuse_shell_command(*args, **kwargs):
    raise AssertionError(f"this test must never reach a shell ({args=}, {kwargs=})")


class TestAHotRestartIsRefusedOutsideKubernetes:
    def test_the_ray_backend_refuses_a_hot_restart_before_it_cleans_anything_up(self, monkeypatch):
        """Its first act is to pkill every sglang, miles and ray process, i.e. exactly what the flag keeps alive."""
        backend = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.RAY, hot_restart="orchestration,rollout_executor"
        ).create_backend()
        for name in ["exec_command_cpu", "exec_command_gpu", "exec_command_multi_node"]:
            monkeypatch.setattr(type(backend), name, _refuse_shell_command)

        with pytest.raises(AssertionError, match="only supported on the kubernetes backend"):
            backend.execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

    def test_an_ordinary_ray_config_asks_for_no_hot_restart_at_all(self):
        """The refusal reads this answer, so a launch without the flag must not present a restart to it."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)

        assert config.parsed_hot_restart == []


class TestCommandUtilConfig:
    def test_backend_config_only_contains_connection_fields(self):
        """Launch-only fields must remain on ExecuteTrainConfig rather than every command backend."""
        assert [field.name for field in dataclasses.fields(CommandUtilConfig)] == [
            "cluster_backend",
            "namespace",
            "helm_values",
            "ci_run",
        ]

    def test_backend_rejects_an_unrelated_config_type(self):
        """A backend must not retain an object that lacks its cluster connection fields."""
        with pytest.raises(AssertionError, match="CommandUtilConfig"):
            RayCommandBackend(object())


class TestExecuteTrainConfigSelection:
    def test_explicit_config_overrides_the_backend_default(self, monkeypatch):
        """A caller can launch a different deployment through an existing cluster backend."""
        recorded: list[tuple[ExecuteTrainRequest, ExecuteTrainConfig]] = []
        monkeypatch.setattr(
            RayCommandBackend,
            "_execute_train_inner",
            lambda self, *, request, config: recorded.append((request, config)),
        )
        backend_config = ExecuteTrainConfig()
        launch_config = ExecuteTrainConfig(deploy_component=DeployComponent.TRAINER)

        backend_config.create_backend().execute_train(
            train_args="--train-backend fsdp",
            num_gpus_per_node=8,
            megatron_model_type=None,
            config=launch_config,
        )

        assert recorded[0][1] is launch_config

    def test_omitted_config_uses_the_backend_execute_train_config(self, monkeypatch):
        """Existing launchers can keep constructing a backend and calling execute_train without config."""
        recorded: list[tuple[ExecuteTrainRequest, ExecuteTrainConfig]] = []
        monkeypatch.setattr(
            RayCommandBackend,
            "_execute_train_inner",
            lambda self, *, request, config: recorded.append((request, config)),
        )
        config = ExecuteTrainConfig(deploy_component=DeployComponent.TRAINER)

        config.create_backend().execute_train(
            train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None
        )

        assert recorded[0][1] is config

    def test_omitted_config_refuses_a_connection_only_backend(self):
        """A backend without launch fields cannot guess the execute_train configuration."""
        backend = CommandUtilConfig().create_backend()

        with pytest.raises(AssertionError, match="ExecuteTrainConfig"):
            backend.execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)
