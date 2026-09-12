import dataclasses
import inspect
import os
from dataclasses import dataclass
from typing import Literal

import pytest
import typer

from miles.utils.external_utils.command_utils import CommandUtilConfig, base_backend
from miles.utils.external_utils.command_utils.base_backend import (
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    default_config,
    resolve_extra_env_vars,
    resolve_hardware,
)
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX, dataclass_cli
from miles.utils.workers.types import ClusterBackend, DeployComponent


@pytest.fixture(autouse=True)
def bare_environment(monkeypatch):
    """No test may read a variable the shell that started pytest happened to export."""
    for name in [name for name in os.environ if name.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@dataclass
class _HardwareConfig(ExecuteTrainConfig):
    hardware: Literal["auto", "H100"] = "auto"


class TestResolveHardware:
    def test_supported_explicit_value_bypasses_detection_while_auto_uses_it(self, monkeypatch):
        """Explicit hardware bypasses detection, while auto resolves to a supported detected profile."""
        detected: list[None] = []

        def detect_hardware() -> str:
            detected.append(None)
            return "H100"

        monkeypatch.setattr(base_backend, "detect_hardware", detect_hardware)

        assert resolve_hardware(_HardwareConfig(hardware="H100")) == "H100"
        assert detected == []
        assert resolve_hardware(_HardwareConfig(hardware="auto")) == "H100"
        assert detected == [None]

    @pytest.mark.parametrize(
        ("configured_hardware", "detected_hardware"),
        [("unsupported", "H100"), ("auto", "unsupported")],
    )
    def test_unsupported_explicit_or_detected_value_is_rejected(
        self, configured_hardware: str, detected_hardware: str, monkeypatch
    ):
        """Neither explicit nor detected hardware may escape the config's supported profile literal."""
        monkeypatch.setattr(base_backend, "detect_hardware", lambda: detected_hardware)

        with pytest.raises(AssertionError, match="has no verified profile"):
            resolve_hardware(_HardwareConfig(hardware=configured_hardware))


class TestResolveExtraEnvVars:
    def test_config_extra_env_vars_override_the_callers_values(self):
        """Parsed config variables override duplicates while preserving caller-only variables."""
        config = ExecuteTrainConfig(extra_env_vars="SHARED=from_config CONFIG_ONLY=kept")

        resolved = resolve_extra_env_vars(
            extra_env_vars={"SHARED": "from_caller", "CALLER_ONLY": "kept"},
            config=config,
        )

        assert resolved == {
            "SHARED": "from_config",
            "CALLER_ONLY": "kept",
            "CONFIG_ONLY": "kept",
        }


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
    def test_an_explicit_config_for_another_backend_is_refused_before_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A launch for another cluster must be refused before the backend performs any action."""
        commands: list[str] = []
        monkeypatch.setattr(base_backend, "run_shell_command", lambda command, **kwargs: commands.append(command))
        backend = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY).create_backend()
        launch_config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES)

        with pytest.raises(AssertionError, match="built to talk to ray"):
            backend.execute_train(
                train_args="--train-backend fsdp",
                num_gpus_per_node=8,
                megatron_model_type=None,
                config=launch_config,
            )

        assert commands == []

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


def _launched_train_argv(monkeypatch, *, train_args: str, config: ExecuteTrainConfig) -> list[str]:
    recorded: list[ExecuteTrainRequest] = []
    monkeypatch.setattr(
        RayCommandBackend, "_execute_train_inner", lambda self, *, request, config: recorded.append(request)
    )

    config.create_backend().execute_train(train_args=train_args, num_gpus_per_node=8, megatron_model_type=None)

    return recorded[0].train_args.split()


class TestTheRunUuidALaunchDrives:
    def test_the_configured_run_uuid_reaches_the_pods(self, monkeypatch):
        """Only --deploy-component and --deploy-instance-id were appended, so an unsplit run minted a second uuid."""
        argv = _launched_train_argv(
            monkeypatch,
            train_args="--train-backend fsdp",
            config=ExecuteTrainConfig(run_uuid="0123456789abcdef"),
        )

        assert argv[argv.index("--run-uuid") + 1] == "0123456789abcdef"

    def test_a_launch_that_names_no_run_leaves_the_arguments_alone(self, monkeypatch):
        """Every existing ray launch names none, and an empty flag would be worse than no flag."""
        argv = _launched_train_argv(monkeypatch, train_args="--train-backend fsdp", config=ExecuteTrainConfig())

        assert "--run-uuid" not in argv

    def test_train_arguments_that_already_name_this_run_are_accepted(self, monkeypatch):
        """The helm launcher sets the flag itself, and a launch agreeing with it is not a conflict."""
        argv = _launched_train_argv(
            monkeypatch,
            train_args="--train-backend fsdp --run-uuid 0123456789abcdef",
            config=ExecuteTrainConfig(run_uuid="0123456789abcdef"),
        )

        assert argv.count("--run-uuid") == 2

    def test_refuses_train_arguments_that_name_another_run(self, monkeypatch):
        """The uuid is what joins the parts of a split run, so two of them are two runs."""
        with pytest.raises(AssertionError, match="--run-uuid"):
            _launched_train_argv(
                monkeypatch,
                train_args="--train-backend fsdp --run-uuid fedcba9876543210",
                config=ExecuteTrainConfig(run_uuid="0123456789abcdef"),
            )

    def test_the_component_and_instance_flags_are_still_appended_beside_it(self, monkeypatch):
        """The run uuid joins a split run, and these two are what tell its halves apart."""
        argv = _launched_train_argv(
            monkeypatch,
            train_args="--train-backend fsdp",
            config=ExecuteTrainConfig(
                run_uuid="0123456789abcdef", deploy_component=DeployComponent.TRAINER, deploy_instance_id="actor"
            ),
        )

        assert argv[argv.index("--deploy-component") + 1] == "trainer"
        assert argv[argv.index("--deploy-instance-id") + 1] == "actor"
        assert argv[argv.index("--run-uuid") + 1] == "0123456789abcdef"


class TestApiServerHost:
    def test_the_ray_api_server_is_reached_on_localhost(self) -> None:
        """The default backend must keep fault-tolerance clients on the local API server."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)

        assert config.create_backend().api_server_host(config) == "localhost"
