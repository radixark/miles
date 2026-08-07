import dataclasses
import inspect

import pytest

from miles.utils.external_utils.command_utils import env_config
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX, dataclass_cli
from miles.utils.workers.types import ClusterBackend

CHART_COMPONENTS = ("orchestrator", "mooncake-master", "static-workers", "inference-engines")


class TestAScriptReadsItsLauncherConfigFromTheEnvironment:
    def test_an_unset_environment_leaves_every_default_alone(self):
        """A ray run sets nothing, so reading the environment must not change what it used to get."""
        assert env_config.config_from_env(environ={}) == ExecuteTrainConfig()

    def test_the_backend_is_chosen_by_one_variable(self):
        """This is the whole point: the same e2e script has to reach either cluster without being edited."""
        config = env_config.config_from_env(environ={"MILES_SCRIPT_CLUSTER_BACKEND": "kubernetes"})

        assert config.cluster_backend == ClusterBackend.KUBERNETES.value

    def test_a_repeatable_option_is_split_the_way_click_splits_it(self):
        """The same variable feeds typer on the script path, and one list must not become two shapes."""
        config = env_config.config_from_env(environ={"MILES_SCRIPT_INFRA_VALUES": "/a/infra.yaml /b/infra.yaml"})

        assert config.infra_values == ("/a/infra.yaml", "/b/infra.yaml")

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_a_flag_is_true_for_every_spelling_a_command_line_accepts(self, value: str):
        """An operator who types what click accepts must not silently get the opposite."""
        assert env_config.config_from_env(environ={"MILES_SCRIPT_CI_RUN": value}).ci_run is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_a_flag_is_false_for_every_spelling_a_command_line_accepts(self, value: str):
        """`MILES_SCRIPT_CI_RUN=false` reading as true is the failure this rules out."""
        assert env_config.config_from_env(environ={"MILES_SCRIPT_CI_RUN": value}).ci_run is False

    def test_a_flag_that_is_neither_is_refused_rather_than_guessed(self):
        """Guessing would install a backend nobody asked for, and a run is expensive to discover that on."""
        with pytest.raises(AssertionError, match="neither true nor false"):
            env_config.config_from_env(environ={"MILES_SCRIPT_CI_RUN": "maybe"})

    def test_a_number_arrives_as_a_number(self):
        """num_nodes reaches arithmetic, and a string would only fail much later."""
        assert env_config.config_from_env(environ={"MILES_SCRIPT_NUM_NODES": "4"}).num_nodes == 4

    @pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(ExecuteTrainConfig)])
    def test_every_field_answers_to_the_variable_the_command_line_binds(self, field: str):
        """A script path and a test path reading different variables is a difference nobody would look for."""

        @dataclass_cli
        def train(args: ExecuteTrainConfig) -> None: ...

        bound = inspect.signature(train).parameters[field].annotation.__metadata__[0].envvar

        assert bound == env_config.env_var_name(field) == f"{SCRIPT_ENV_VAR_PREFIX}{field.upper()}"


class TestARunIsNamedAfterTheTestThatLaunchedIt:
    def test_a_ray_run_is_given_no_run_id(self, monkeypatch):
        """Only a release needs a name, and inventing one for ray would change what ray runs record."""
        monkeypatch.delenv("MILES_SCRIPT_CLUSTER_BACKEND", raising=False)
        monkeypatch.delenv("MILES_SCRIPT_RUN_ID", raising=False)

        assert env_config.config_from_env(environ={}).run_id == ""
        assert env_config.default_config().run_id == ""

    def test_an_explicit_run_id_is_never_overwritten(self, monkeypatch):
        """Relaunching one run id in place is how a run grows, so what the operator names must survive."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")
        monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "nightly-soak")

        assert env_config.default_config().run_id == "nightly-soak"

    def test_the_same_test_file_derives_the_same_run_id_twice(self):
        """A relaunch has to upgrade the release it launched before rather than open a second one."""
        first = env_config.derive_run_id(entry_script="tests/e2e/short/test_gsm8k_short.py", launch_index=1)
        second = env_config.derive_run_id(entry_script="/repo/tests/e2e/short/test_gsm8k_short.py", launch_index=1)

        assert first == second

    def test_two_test_files_of_the_same_name_derive_different_run_ids(self):
        """A suite installs them into one namespace, where the same release name would be one run."""
        common = env_config.derive_run_id(entry_script="tests/e2e/megatron/_common.py", launch_index=1)
        other = env_config.derive_run_id(entry_script="tests/e2e/fsdp/_common.py", launch_index=1)

        assert common != other

    def test_a_second_launch_of_one_test_gets_its_own_run_id(self, monkeypatch):
        """A checkpoint test launches four differently shaped runs, which one release could not hold."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")

        assert env_config.default_config().run_id != env_config.default_config().run_id

    def test_a_derived_run_id_is_a_legal_kubernetes_object_name(self):
        """It names the release, and a dot or an underscore would be refused by the api server."""
        run_id = env_config.derive_run_id(entry_script="tests/e2e/short/test_qwen2.5_0.5B_gsm8k.py", launch_index=1)

        assert naming.RUN_ID_PATTERN.fullmatch(run_id), run_id

    @pytest.mark.parametrize("component", CHART_COMPONENTS)
    def test_a_derived_run_id_leaves_room_for_every_object_the_chart_names(self, component: str):
        """Past the budget the names are truncated, and two tests would then install over each other."""
        run_id = env_config.derive_run_id(entry_script="tests/e2e/short/test_qwen2.5_0.5B_gsm8k.py", launch_index=1)
        release = naming.release_name(run_id)

        assert naming.component_name(release, component).startswith(f"{release}-")

    def test_a_derived_run_id_still_names_the_test_it_came_from(self):
        """An operator reads `kubectl get pods` to find their run, and a bare hash tells them nothing."""
        run_id = env_config.derive_run_id(entry_script="tests/e2e/short/test_gsm8k_short.py", launch_index=1)

        assert "test-gsm8k" in run_id
