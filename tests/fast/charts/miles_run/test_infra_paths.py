from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    host_path_volume,
    only_container_of,
    pod_spec_of,
    render_run,
    render_run_error,
    requires_helm,
    volumes_args,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
ORCHESTRATOR_IDENTITY = {"MILES_K8S_NAMESPACE": "myns", "MILES_K8S_RELEASE": "myrun"}

ALL_REPOS = volumes_args(
    host_path_volume(
        mounts=[
            {"mountPath": "/cluster-storage"},
            {"mountPath": "/root/miles", "subPath": "myuser/miles"},
            {"mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
            {"mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"},
        ]
    )
)


def orchestrator_container(*args: str) -> dict[str, Any]:
    return only_container_of(render_run(*args), "StatefulSet", ORCHESTRATOR)


def environment(container: dict[str, Any]) -> dict[str, str]:
    return {entry["name"]: entry["value"] for entry in container.get("env", [])}


@requires_helm
class TestCheckoutMounts:
    def test_a_configured_repo_mounts_over_the_copy_baked_into_the_image(self):
        """Editing code on shared storage is only picked up if the mount lands on the path the image imports from."""
        mounts = orchestrator_container(*ALL_REPOS)["volumeMounts"]

        assert mounts == [
            {"name": "cluster-storage", "mountPath": "/cluster-storage"},
            {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "myuser/miles"},
            {"name": "cluster-storage", "mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
            {"name": "cluster-storage", "mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"},
            {"name": "uninstall-manifest", "mountPath": "/etc/miles-uninstall", "readOnly": True},
        ]

    def test_only_the_repos_that_were_named_are_overridden(self):
        """Overriding one repo must not shadow the other two with an empty directory."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
                    ]
                )
            )
        )

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/root/Megatron-LM",
            "/etc/miles-uninstall",
        ]

    def test_the_defaults_override_no_repo_at_all(self):
        """The image is self-contained, so a run that mounts no checkout of its own gets no extra environment."""
        container = orchestrator_container()

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/etc/miles-uninstall",
        ]
        assert environment(container) == ORCHESTRATOR_IDENTITY


@requires_helm
class TestReadOnlyMounts:
    def test_a_mount_marked_read_only_reaches_the_container_that_way(self):
        """A shared model cache is everyone's, and a run that can write it can corrupt every other run."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(mounts=[{"mountPath": "/cluster-storage"}]),
                {
                    "name": "models",
                    "hostPath": {"path": "/gpfs/models", "type": "Directory"},
                    "mounts": [{"mountPath": "/models", "readOnly": True}],
                },
            )
        )

        assert {"name": "models", "mountPath": "/models", "readOnly": True} in container["volumeMounts"]

    def test_an_ordinary_mount_says_nothing_about_being_read_only(self):
        """readOnly: false is the kubernetes default, and spelling it out only makes every diff noisier."""
        container = orchestrator_container(*volumes_args(host_path_volume()))

        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container["volumeMounts"]


@requires_helm
class TestRunsRootIsOnAMountedVolume:
    def test_a_runs_root_under_no_mount_at_all_is_refused(self):
        """A run would write its state file into the container's own filesystem, where the launcher never looks."""
        error = render_run_error(*volumes_args(host_path_volume()), "--set", "infra.paths.runsRoot=/elsewhere/data")

        assert "/elsewhere/data" in error

    def test_a_runs_root_on_a_read_only_mount_is_refused(self):
        """The volume is there, but every run writes its state, values and exit file under this directory."""
        error = render_run_error(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage", "readOnly": True}]))
        )

        assert "read-only" in error

    def test_a_runs_root_that_is_a_mount_path_itself_is_accepted(self):
        """A cluster that dedicates a whole volume to miles must not be forced into a subdirectory."""
        container = orchestrator_container(
            *volumes_args(host_path_volume()), "--set", "infra.paths.runsRoot=/cluster-storage"
        )

        assert container["image"]

    def test_a_run_with_no_runs_root_at_all_is_refused(self):
        """The launcher polls the exit file under it, so a run without one can only ever look like a hang."""
        assert "runsRoot" in render_run_error("--set", "infra.paths.runsRoot=null")


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

    def test_the_cluster_environment_reaches_a_pod_that_mounts_its_own_checkouts(self):
        """Mounting code must not cost a pod the cluster's proxy, nor the proxy cost it the mounts."""
        container = orchestrator_container(*ALL_REPOS, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")

        assert environment(container) == ORCHESTRATOR_IDENTITY | {"HTTP_PROXY": "http://proxy:7890"}


@requires_helm
class TestPythonPathIsNotAnEnvironmentVariable:
    def test_the_schema_refuses_a_pythonpath_in_the_cluster_environment(self):
        """The image installs its three source trees as editable, and a hand-set PYTHONPATH can only shadow them."""
        error = render_run_error("--set", "infra.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_the_schema_refuses_a_pythonpath_in_the_run_environment(self):
        """Same shadowing, one level down: a per-run value would reach the pods just as an infra one does."""
        error = render_run_error("--set", "run.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_an_ordinary_cluster_variable_is_still_accepted(self):
        """Only the variables the platform owns are reserved; refusing the rest would make infra.env useless."""
        objects = render_run("--set", "infra.env.NCCL_SOCKET_IFNAME=bond0")

        env = only_container_of(objects, "StatefulSet", ORCHESTRATOR)["env"]
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "bond0"} in env


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


@requires_helm
class TestNamespaceInterpolation:
    def test_a_host_path_names_the_namespace_it_is_rendered_for(self):
        """Two agents in two namespaces want two directories on the host, and one values file has to give both."""
        objects = render_run(*volumes_args(host_path_volume(path="/data/${NAMESPACE}")))

        volumes = pod_spec_of(objects, "StatefulSet", ORCHESTRATOR)["volumes"]
        assert {"name": "cluster-storage", "hostPath": {"path": f"/data/{NAMESPACE}", "type": "Directory"}} in volumes

    def test_a_mount_path_names_the_namespace(self):
        """A node-local scratch disk is one path on every node, so only the namespace keeps two runs apart."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(mounts=[{"mountPath": "/cluster-storage"}, {"mountPath": "/scratch/${NAMESPACE}"}])
            )
        )

        assert {"name": "cluster-storage", "mountPath": f"/scratch/{NAMESPACE}"} in container["volumeMounts"]

    def test_a_sub_path_names_the_namespace(self):
        """This is how five agents get five checkouts of the same repo out of one shared volume."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPACE}/miles"},
                    ]
                )
            )
        )

        assert {
            "name": "cluster-storage",
            "mountPath": "/root/miles",
            "subPath": f"repos/{NAMESPACE}/miles",
        } in container["volumeMounts"]

    def test_a_path_that_names_no_variable_is_left_exactly_as_written(self):
        """Several releases sharing one directory is a first-class choice, not an escape hatch."""
        container = orchestrator_container(*volumes_args(host_path_volume(path="/cluster-storage")))

        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container["volumeMounts"]

    def test_the_default_runs_root_is_a_directory_this_namespace_has_to_itself(self):
        """Isolation has to be what you get without asking: a default without it makes it one more thing to remember."""
        error = render_run_error(*volumes_args(host_path_volume(mounts=[{"mountPath": "/elsewhere"}])))

        assert f"/cluster-storage/{NAMESPACE}/miles_data" in error

    def test_the_mount_check_compares_resolved_paths(self):
        """Comparing an unresolved runs root against a resolved mount path would refuse one that is in fact mounted."""
        container = orchestrator_container(
            *volumes_args(host_path_volume(mounts=[{"mountPath": f"/cluster-storage/{NAMESPACE}"}]))
        )

        assert container["image"]

    def test_an_unknown_variable_in_a_host_path_is_refused(self):
        """Left in place it would name one literal directory for every namespace, which is the collision itself."""
        error = render_run_error(*volumes_args(host_path_volume(path="/data/${RELEASE}")))

        assert "infra.volumes[cluster-storage].hostPath.path" in error
        assert "${RELEASE}" in error

    def test_an_unknown_variable_in_a_mount_path_is_refused(self):
        """The refusal has to say which mount of which volume, because a values file has many of both."""
        error = render_run_error(*volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage/${USER}"}])))

        assert "infra.volumes[cluster-storage].mounts[0].mountPath" in error
        assert "${USER}" in error

    def test_an_unknown_variable_in_a_sub_path_is_refused(self):
        """A misspelt ${NAMESPACE} is the likely typo, and it is the one that silently shares a directory."""
        error = render_run_error(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPCE}/miles"},
                    ]
                )
            )
        )

        assert "infra.volumes[cluster-storage].mounts[1].subPath" in error
        assert "${NAMESPCE}" in error

    def test_an_unknown_variable_in_the_runs_root_is_refused(self):
        """Every run writes its state and exit file here, so a directory shared by accident is a run lost."""
        error = render_run_error("--set", "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data")

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error

    def test_an_unknown_variable_is_refused_by_a_render_that_deploys_nothing(self):
        """Every other template of this chart renders for some topology only, so the check needs a home without one."""
        error = render_run_error(
            "--set-json",
            "run.orchestrator.command=[]",
            "--set",
            "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data",
        )

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error
