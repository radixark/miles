from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.charts.utils import RUN_ID, only_container_of, render_run, render_run_error, requires_helm

from miles.utils.external_utils.command_utils.helm_backend import run_state

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
    return only_container_of(render_run(*args), "StatefulSet", ORCHESTRATOR)


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
        ]
        assert environment(container)["PYTHONPATH"] == "/root/Megatron-LM"

    def test_the_defaults_override_no_repo_at_all(self):
        """The image is self-contained, so a run that names no repo must not gain a PYTHONPATH of its own."""
        container = orchestrator_container()

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == ["/cluster-storage"]
        assert environment(container) == ORCHESTRATOR_IDENTITY


@requires_helm
class TestRunDirectory:
    def test_the_run_directory_hangs_off_the_configured_runs_subpath(self):
        """Runs live beside the other miles data on the cluster filesystem, not at its root."""
        container = orchestrator_container(
            "--set", "infra.sharedStorage.mountPath=/mnt/x", "--set", "infra.paths.runsSubPath=teamdata"
        )

        assert f"/mnt/x/teamdata/miles-runs/{RUN_ID}/state/orchestrator.exit" in container["command"]

    def test_an_empty_runs_subpath_puts_the_run_directory_at_the_mount_root(self):
        """A cluster that dedicates the whole volume to miles must not be forced into a subdirectory."""
        container = orchestrator_container("--set", "infra.paths.runsSubPath=")

        assert f"/cluster-storage/miles-runs/{RUN_ID}/state/orchestrator.exit" in container["command"]

    @pytest.mark.parametrize(
        "paths",
        [{}, {"runsSubPath": ""}, {"runsSubPath": None}, {"runsSubPath": "teamdata"}],
        ids=["absent", "empty", "null", "named"],
    )
    def test_the_chart_and_the_launcher_resolve_the_same_run_directory(self, tmp_path: Path, paths: dict[str, Any]):
        """The launcher polls the exit file the pods write, so a second answer is a run that can only hang."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": paths}}
        values_file = tmp_path / "infra.yaml"
        values_file.write_text(yaml.safe_dump(values))

        container = orchestrator_container("--values", str(values_file))
        launcher_side = run_state.orchestrator_exit_path(run_state.run_dir(run_state.shared_root_of(values), RUN_ID))

        assert str(launcher_side) in container["command"]

    def test_a_deleted_paths_section_still_agrees_with_the_launcher(self, tmp_path: Path):
        """Nulling the whole section is how helm users turn a default off, and it deletes the sub-path with it."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": None}}
        values_file = tmp_path / "infra.yaml"
        values_file.write_text(yaml.safe_dump(values))

        container = orchestrator_container("--values", str(values_file))

        assert run_state.shared_root_of(values) == "/mnt/x"
        assert f"/mnt/x/miles-runs/{RUN_ID}/state/orchestrator.exit" in container["command"]


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
        """Only PYTHONPATH is reserved; refusing the rest would make infra.env useless."""
        objects = render_run("--set", "infra.env.NCCL_SOCKET_IFNAME=bond0")

        env = only_container_of(objects, "StatefulSet", ORCHESTRATOR)["env"]
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "bond0"} in env
