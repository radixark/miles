import json
from typing import Any

from tests.fast.charts.utils import render_run, render_run_error, requires_helm, sole_container_of, with_object_names

ORCHESTRATOR = "myrun-miles-run-orchestrator"
ORCHESTRATOR_IDENTITY = {"MILES_K8S_NAMESPACE": "myns", "MILES_K8S_RELEASE": "myrun"}

ALL_REPOS = (
    "--set",
    "infra.paths.repos.miles=myuser/miles",
    "--set",
    "infra.paths.repos.megatron=myuser/Megatron-LM",
    "--set",
    "infra.paths.repos.sglang=myuser/sglang",
)


def orchestrator_container(*args: str) -> dict[str, Any]:
    return sole_container_of(render_run(*args), "StatefulSet", ORCHESTRATOR)


def environment(container: dict[str, Any]) -> dict[str, str]:
    return {entry["name"]: entry["value"] for entry in container.get("env", [])}


@requires_helm
class TestCodeRepositoryOverrides:
    def test_a_configured_repo_mounts_over_the_copy_baked_into_the_image(self):
        """Editing code on shared storage is only picked up if the mount lands on the path the image imports from."""
        mounts = orchestrator_container(*ALL_REPOS)["volumeMounts"]

        assert mounts == [
            {"name": "shared-storage", "mountPath": "/cluster-storage"},
            {"name": "shared-storage", "mountPath": "/root/miles", "subPath": "myuser/miles"},
            {"name": "shared-storage", "mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
            {"name": "shared-storage", "mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"},
            {"name": "uninstall-manifest", "mountPath": "/etc/miles-uninstall", "readOnly": True},
        ]

    def test_every_overridden_repo_joins_the_python_path_by_its_in_image_location(self):
        """The mount alone does not reorder sys.path, so an installed copy would still win without this."""
        assert environment(orchestrator_container(*ALL_REPOS))["PYTHONPATH"] == (
            "/root/miles:/root/Megatron-LM:/sgl-workspace/sglang"
        )

    def test_only_the_repos_that_were_named_are_overridden(self):
        """Overriding one repo must not shadow the other two with an empty directory."""
        container = orchestrator_container("--set", "infra.paths.repos.megatron=myuser/Megatron-LM")

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/root/Megatron-LM",
            "/etc/miles-uninstall",
        ]
        assert environment(container)["PYTHONPATH"] == "/root/Megatron-LM"

    def test_the_defaults_override_no_repo_at_all(self):
        """The image is self-contained, so a run that names no repo must not gain a PYTHONPATH of its own."""
        container = orchestrator_container()

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/etc/miles-uninstall",
        ]
        assert environment(container) == ORCHESTRATOR_IDENTITY


@requires_helm
class TestRunDirectory:
    def test_the_orchestrator_watches_the_state_file_the_launcher_named(self):
        """The launcher polls the path it injected, so a chart that derives its own is a run that can only hang."""
        state_file = "/mnt/x/teamdata/miles-runs/myrun/state/orchestrator.state"

        container = orchestrator_container("--set", f"run.stateFile={state_file}")

        assert state_file in container["command"]


@requires_helm
class TestClusterEnvironment:
    def test_the_cluster_environment_reaches_the_orchestrator_pod(self):
        """The orchestrator downloads datasets and reaches the api server, so it needs the cluster's proxy too."""
        container = orchestrator_container(
            "--set", "infra.env.HTTP_PROXY=http://proxy:7890", "--set", "infra.env.HF_ENDPOINT=https://mirror"
        )

        assert environment(container) == ORCHESTRATOR_IDENTITY | {
            "HTTP_PROXY": "http://proxy:7890",
            "HF_ENDPOINT": "https://mirror",
        }

    def test_the_cluster_environment_and_the_derived_python_path_both_reach_the_pod(self):
        """One of the two overwriting the other would silently drop either the proxy or the code override."""
        container = orchestrator_container(*ALL_REPOS, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")

        assert environment(container) == ORCHESTRATOR_IDENTITY | {
            "HTTP_PROXY": "http://proxy:7890",
            "PYTHONPATH": "/root/miles:/root/Megatron-LM:/sgl-workspace/sglang",
        }


@requires_helm
class TestPythonPathIsNotAnEnvironmentVariable:
    def test_the_schema_refuses_a_pythonpath_in_the_cluster_environment(self):
        """A hand-set PYTHONPATH silently outranks the repo mounts, so the values file must not carry one."""
        error = render_run_error("--set", "infra.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_the_schema_refuses_a_pythonpath_in_the_run_environment(self):
        """The launcher derives PYTHONPATH from the mounted repos; a second source could only disagree."""
        error = render_run_error("--set", "run.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_an_ordinary_cluster_variable_is_still_accepted(self):
        """Only the variables the launcher derives are reserved; refusing the rest would make infra.env useless."""
        objects = render_run("--set", "infra.env.NCCL_SOCKET_IFNAME=bond0")

        env = sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["env"]
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "bond0"} in env


STATIC_WORKERS = [{"name": "rollout-executor", "command": ["python", "-m", "miles.utils.workers.serving.serve"]}]
TRAINER_ENGINES = [
    {"name": "trainer-engine-actor", "command": ["python", "-m", "miles.utils.workers.process_supervisor"]}
]
INFERENCE_ENGINES = [{"name": "inference-engine-0-0", "command": ["python", "-m", "sglang.launch_server"]}]

WHOLE_TOPOLOGY = (
    "--set-json",
    f"run.staticWorkers={json.dumps(with_object_names(STATIC_WORKERS))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINER_ENGINES))}",
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(INFERENCE_ENGINES))}",
)


def containers_of(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [container for obj in objects for container in _containers_in(obj)]


def _containers_in(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        found = list(node.get("containers") or [])
        for key, value in node.items():
            if key != "containers":
                found.extend(_containers_in(value))
        return found
    if isinstance(node, list):
        return [container for item in node for container in _containers_in(item)]
    return []


@requires_helm
class TestCheckoutWorkingDirectory:
    def test_every_container_runs_from_the_checkout_the_image_imports_from(self):
        """A container loading a custom function or a relative config path only resolves it from the checkout root."""
        containers = containers_of(render_run(*WHOLE_TOPOLOGY))

        assert {container["name"] for container in containers} == {"orchestrator", "worker", "engine", "trainer"}
        assert {container.get("workingDir") for container in containers} == {"/root/miles"}


@requires_helm
class TestTheLaunchRecordIsNotAnEnvironmentVariable:
    def test_every_container_is_told_what_launched_this_run(self):
        """The record is how --env-report reaches each process's env report and the wandb config."""
        container = orchestrator_container("--set-string", "run.launchRecord=/shared/launches/launch-1.json")

        assert environment(container)["MILES_SCRIPT_ENV_REPORT"] == "/shared/launches/launch-1.json"

    def test_a_run_launched_without_a_record_carries_no_empty_one(self):
        """An empty record would look like a launcher that recorded nothing, rather than one that never ran."""
        assert "MILES_SCRIPT_ENV_REPORT" not in environment(orchestrator_container())

    def test_the_schema_refuses_a_record_in_the_cluster_environment(self):
        """infra.env outranks run.env, so a hand-set record would silently replace the real one."""
        assert "MILES_SCRIPT_ENV_REPORT" in render_run_error(
            "--set", "infra.env.MILES_SCRIPT_ENV_REPORT=/hijacked.json"
        )

    def test_the_schema_refuses_a_record_in_a_pool_environment(self):
        """A per-pool variable outranks everything, so those pods would report a different launch."""
        assert "MILES_SCRIPT_ENV_REPORT" in render_run_error(
            "--set-json",
            """run.staticWorkers=[{"name":"router","objectName":"myrun-router","command":["sleep"],"""
            """"env":{"MILES_SCRIPT_ENV_REPORT":"hijacked"}}]""",
        )
