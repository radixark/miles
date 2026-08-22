import ast
import shlex
from pathlib import Path

import pytest
from tests.fast.cluster_backends import both_backends, require_backend

import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.workers.types import ClusterBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_ROOT = REPO_ROOT / "tests" / "e2e"

TRAIN_ARGS_OF_A_TRIVIAL_RUN = "--train-backend fsdp"


def _config(cluster_backend: str, namespace: str) -> command_utils.ExecuteTrainConfig:
    return command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend(cluster_backend),
        namespace=namespace,
        run_id="260101-000000-000",
    )


class TestEveryBackendHonoursTheSameLauncherContract:
    @both_backends
    def test_the_config_creates_the_backend_it_names(self, cluster_backend):
        """The backend is a run's property, so choosing it in the config has to be what actually takes effect."""
        namespace = require_backend(cluster_backend)

        backend = _config(cluster_backend, namespace).create_backend()

        assert (
            backend.__class__.__name__
            == {
                ClusterBackend.RAY.value: "RayCommandBackend",
                ClusterBackend.KUBERNETES.value: "KubernetesCommandBackend",
            }[cluster_backend]
        )

    @both_backends
    def test_a_cpu_command_runs_without_reaching_the_cluster(self, cluster_backend, monkeypatch):
        """Both backends run cpu work where the launcher already sits, so a script reads the same either way."""
        namespace = require_backend(cluster_backend)
        backend = _config(cluster_backend, namespace).create_backend()

        recorded: list[str] = []
        monkeypatch.setattr(backend, "exec_command_cpu", lambda cmd, capture_output=False: recorded.append(cmd))
        backend.exec_command_cpu("echo hello")

        assert recorded == ["echo hello"]

    @both_backends
    def test_the_worker_flag_and_the_config_must_agree(self, cluster_backend):
        """The config drives the launcher and the flag drives the workers; disagreeing installs one and drives the other."""
        require_backend(cluster_backend)
        other = next(backend.value for backend in ClusterBackend if backend.value != cluster_backend)

        try:
            _config(cluster_backend, "miles-e2e").create_backend().execute_train(
                train_args=f"--train-backend fsdp --cluster-backend {other}",
                num_gpus_per_node=1,
                megatron_model_type=None,
            )
        except AssertionError as error:
            assert shlex.quote(other).strip("'") in str(error)
        else:
            raise AssertionError("a config and a train flag naming different backends must be refused")


def _e2e_modules() -> list[Path]:
    return sorted(path for path in E2E_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _constructs_a_launcher_config(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, ast.Call) and _called_name(node.func) == command_utils.ExecuteTrainConfig.__name__
        for node in ast.walk(tree)
    )


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    return getattr(func, "attr", "")


def _capture_backend(monkeypatch) -> list[ClusterBackend]:
    chosen: list[ClusterBackend] = []

    def _execute_train_inner(self, *, request, config) -> None:
        chosen.append(config.cluster_backend)

    for backend_cls in (RayCommandBackend, KubernetesCommandBackend):
        monkeypatch.setattr(backend_cls, "_execute_train_inner", _execute_train_inner)
    return chosen


class TestATrainingE2eTestRunsOnWhicheverBackendItsEnvironmentNames:
    @pytest.mark.parametrize("backend", sorted(backend.value for backend in ClusterBackend))
    def test_a_test_that_passes_no_config_follows_the_environment(self, backend: str, monkeypatch):
        """Every e2e script calls execute_train without a config, so the environment is the only dial there is."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", backend)
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-e2e")
        chosen = _capture_backend(monkeypatch)

        command_utils.default_config().create_backend().execute_train(
            train_args=TRAIN_ARGS_OF_A_TRIVIAL_RUN, num_gpus_per_node=1, megatron_model_type=None
        )

        assert chosen == [ClusterBackend(backend)]

    def test_an_unset_environment_still_means_ray(self, monkeypatch):
        """Every existing ray runner sets nothing, and reading the environment must not have moved them."""
        monkeypatch.delenv("MILES_SCRIPT_CLUSTER_BACKEND", raising=False)
        chosen = _capture_backend(monkeypatch)

        command_utils.default_config().create_backend().execute_train(
            train_args=TRAIN_ARGS_OF_A_TRIVIAL_RUN, num_gpus_per_node=1, megatron_model_type=None
        )

        assert chosen == [ClusterBackend.RAY]

    def test_a_command_run_before_the_launch_reaches_the_same_backend(self, monkeypatch):
        """prepare() downloads and converts before execute_train, and under kubernetes those are Jobs too."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", ClusterBackend.KUBERNETES.value)
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-e2e")

        backend = command_utils.default_config().create_backend()

        assert backend.__class__.__name__ == "KubernetesCommandBackend"

    @pytest.mark.parametrize("path", _e2e_modules(), ids=lambda path: path.relative_to(E2E_ROOT).as_posix())
    def test_no_e2e_module_pins_a_backend_of_its_own(self, path: Path):
        """A config built in the test would outrank the environment and pin that test to one cluster forever."""
        assert not _constructs_a_launcher_config(path)
