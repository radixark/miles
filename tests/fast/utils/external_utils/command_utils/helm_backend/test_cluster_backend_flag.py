from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from miles.ray import wiring
from miles.utils.external_utils.command_utils import helm_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import launcher
from miles.utils.external_utils.command_utils.helm_backend.values import RunLayout, build_values
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec

NAMESPACE = "rl"
RUN_ID = "260101-000000-000"


def _request(train_args: str) -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        config=ExecuteTrainConfig(cluster_backend="kubernetes", namespace=NAMESPACE, run_id=RUN_ID),
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
    )


def _refuse_ray(args: Any) -> None:
    raise AssertionError("a pod told it runs on kubernetes reached for a ray cluster that is not there")


def _router() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=30000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def launch_argv(monkeypatch: pytest.MonkeyPatch, *, train_args: str) -> list[str]:
    entrypoint = ModuleType("miles.ray.specs.entrypoint")
    entrypoint.compute_specs = lambda args: [_router()]
    monkeypatch.setitem(sys.modules, "miles.ray.specs.entrypoint", entrypoint)

    arguments = ModuleType("miles.utils.arguments")
    arguments.parse_args_from_argv = lambda argv: SimpleNamespace(colocate=False, argv=argv)
    monkeypatch.setitem(sys.modules, "miles.utils.arguments", arguments)

    recorded: list[list[str]] = []

    def fake_launch(request: ExecuteTrainRequest, **kwargs: Any) -> Any:
        recorded.append(list(kwargs["train_argv"]))
        return SimpleNamespace()

    monkeypatch.setattr(launcher, "launch", fake_launch)
    monkeypatch.setattr(launcher, "follow_until_finished", lambda run: 0)

    helm_backend.KubernetesCommandBackend().execute_train(_request(train_args))
    assert len(recorded) == 1
    return recorded[0]


def values_of(train_argv: list[str]) -> dict[str, Any]:
    return build_values(
        [_router()],
        RunLayout(
            run_id=RUN_ID,
            release="miles-run-260101-000000-000",
            orchestrator_command=launcher.orchestrator_command(_request(""), train_argv),
            worker_argv=train_argv,
            num_gpus_per_node=8,
        ),
    )


class TestWithClusterBackend:
    def test_appends_the_flag_a_run_launched_onto_kubernetes_needs(self):
        """A pod that is not told its backend takes the ray branch and looks for a cluster that is not there."""
        argv = helm_backend.with_cluster_backend(["--rollout-num-gpus", "8"], cluster_backend="kubernetes")

        assert argv == ["--rollout-num-gpus", "8", "--cluster-backend", "kubernetes"]

    def test_leaves_an_agreeing_flag_alone_rather_than_repeating_it(self):
        """argparse would take the last of two, so a duplicate is a silent way to change the backend."""
        argv = helm_backend.with_cluster_backend(["--cluster-backend", "kubernetes"], cluster_backend="kubernetes")

        assert argv == ["--cluster-backend", "kubernetes"]

    def test_leaves_an_agreeing_equals_form_alone_too(self):
        """`--flag=value` is one token, so a substring search for the space form would miss it and duplicate it."""
        argv = helm_backend.with_cluster_backend(["--cluster-backend=kubernetes"], cluster_backend="kubernetes")

        assert argv == ["--cluster-backend=kubernetes"]

    def test_rejects_train_args_that_name_another_backend(self):
        """Overwriting the user's own flag would run something other than what they asked for."""
        with pytest.raises(AssertionError, match="ray"):
            helm_backend.with_cluster_backend(["--cluster-backend", "ray"], cluster_backend="kubernetes")

    def test_rejects_the_equals_form_of_another_backend(self):
        """The conflict check has to see the same tokens argparse will."""
        with pytest.raises(AssertionError, match="ray"):
            helm_backend.with_cluster_backend(["--cluster-backend=ray"], cluster_backend="kubernetes")

    def test_does_not_mistake_another_flags_value_for_a_declaration(self):
        """A substring search would find the flag inside `--rollout-function-path=...--cluster-backend ray`."""
        argv = helm_backend.with_cluster_backend(["--data-path=--cluster-backend ray"], cluster_backend="kubernetes")

        assert argv == ["--data-path=--cluster-backend ray", "--cluster-backend", "kubernetes"]

    def test_rejects_a_trailing_flag_that_names_nothing(self):
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last train arg"):
            helm_backend.with_cluster_backend(["--cluster-backend"], cluster_backend="kubernetes")


class TestExecuteTrainTellsThePodsItsBackend:
    def test_the_train_argv_the_launcher_receives_names_kubernetes(self, monkeypatch: pytest.MonkeyPatch):
        """This argv is the only thing the pods are told about the run, so the backend has to be in it."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert helm_backend.declared_cluster_backends(argv) == ["kubernetes"]

    def test_the_orchestrator_command_and_the_worker_argv_both_carry_it(self, monkeypatch: pytest.MonkeyPatch):
        """The orchestrator and its workers have to agree, and each reads its own copy of the argv."""
        run = values_of(launch_argv(monkeypatch, train_args="--rollout-num-gpus 8"))["run"]

        assert helm_backend.declared_cluster_backends(run["orchestrator"]["command"]) == ["kubernetes"]
        assert helm_backend.declared_cluster_backends(run["staticWorkers"][0]["command"]) == []

    def test_a_user_supplied_agreeing_flag_is_not_repeated(self, monkeypatch: pytest.MonkeyPatch):
        """A run relaunched from a recorded command line already carries the flag the launcher would add."""
        argv = launch_argv(monkeypatch, train_args="--cluster-backend kubernetes --rollout-num-gpus 8")

        assert argv.count("--cluster-backend") == 1

    def test_a_user_supplied_conflicting_flag_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        """Launching a ray-flagged run onto kubernetes would install pods nothing ever drives."""
        with pytest.raises(AssertionError, match="ray"):
            launch_argv(monkeypatch, train_args="--cluster-backend ray")


class TestThePodDispatchesOnThatFlag:
    def test_that_argv_selects_the_kubernetes_branch_in_the_pod(self, monkeypatch: pytest.MonkeyPatch):
        """The flag is only worth adding if the in-pod dispatch reads it and skips the ray worker manager."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")
        declared = helm_backend.declared_cluster_backends(argv)

        installed: list[Any] = []
        sentinel = object()
        monkeypatch.setattr(
            wiring, "_kubernetes_backend_capability_from_args", lambda args: installed.append(args) or sentinel
        )
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        args = SimpleNamespace(cluster_backend=declared[0])
        assert wiring.get_backend_capability(args) is sentinel
        assert installed == [args]
        assert ClusterBackend(declared[0]) is ClusterBackend.KUBERNETES
