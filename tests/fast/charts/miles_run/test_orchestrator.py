from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    RUN_STATE_FILE,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    render_run_error,
    requires_helm,
    schema_error_mentions,
    sole_container_of,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
WRAPPER_MODULE = "miles.utils.external_utils.command_utils.helm_backend.orchestrator.wrapper"


@requires_helm
class TestOrchestrator:
    def test_wraps_the_training_command_so_the_exit_code_reaches_shared_storage(self):
        """The launcher polls the exit file, so the script must run under the wrapper rather than directly."""
        objects = render_run("--set-json", 'run.orchestrator.command=["python","train.py","--foo"]')

        command = sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["command"]

        assert command == [
            "python",
            "-m",
            WRAPPER_MODULE,
            "--state-file",
            RUN_STATE_FILE,
            "--uninstall-manifest",
            "/etc/miles-uninstall/uninstall-job.yaml",
            "--",
            "python",
            "train.py",
            "--foo",
        ]

    def test_keeps_arguments_with_spaces_as_single_words(self):
        """A model arg like a json blob must not split into several argv entries on its way through the chart."""
        objects = render_run("--set-json", 'run.orchestrator.command=["python","train.py","--kwargs","{\\"a\\": 1}"]')

        assert sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["command"][-1] == '{"a": 1}'

    def test_runs_exactly_one_orchestrator(self):
        """Two orchestration scripts would drive the same run twice."""
        statefulset = named_object(render_run(), "StatefulSet", ORCHESTRATOR)

        assert statefulset["spec"]["replicas"] == 1

    def test_disables_the_service_link_environment(self):
        """Kubernetes would otherwise inject <SERVICE>_PORT vars, and a prometheus service breaks arg parsing."""
        assert pod_spec_of(render_run(), "StatefulSet", ORCHESTRATOR)["enableServiceLinks"] is False

    def test_mounts_shared_storage_where_the_run_directory_lives(self):
        """The wrapper writes the exit file to that path, so the mount has to be there."""
        mounts = sole_container_of(render_run(), "StatefulSet", ORCHESTRATOR)["volumeMounts"]

        assert [mount["mountPath"] for mount in mounts] == ["/cluster-storage", "/etc/miles-uninstall"]

    def test_serves_the_orchestrator_under_a_headless_service(self):
        """Workers address each other by stable dns, which a headless service is what provides."""
        service = named_object(render_run(), "Service", ORCHESTRATOR)

        assert service["spec"]["clusterIP"] == "None"
        assert service["metadata"]["namespace"] == NAMESPACE

    def test_rejects_a_run_id_that_cannot_be_a_kubernetes_name(self):
        """The release derives object names from it, so an invalid id must fail before anything is created."""
        assert schema_error_mentions(render_run_error("--set", "run.id=Not_A_Name"), path=("run", "id"))

    def test_refuses_an_orchestrated_run_that_names_no_state_file(self):
        """The launcher polls that path to learn the run finished; without it the launcher waits forever."""
        assert "run.stateFile" in render_run_error("--set", "run.stateFile=null")

    def test_refuses_a_run_with_nowhere_to_write_its_outcome(self):
        """The launcher learns the outcome by reading the exit file, which needs a shared volume."""
        assert "sharedStorage" in render_run_error("--set", "infra.sharedStorage.type=none")

    def test_ships_no_job_in_a_normal_install(self):
        """Command jobs are applied on their own; installing one with a run would rerun it on every upgrade."""
        assert objects_of_kind(render_run(), "Job") == []


@requires_helm
class TestOrchestratorIdentity:
    def test_tells_the_orchestrator_which_release_and_namespace_hold_its_workers(self):
        """The orchestrator selects worker pods by release label, and it cannot recompute that name."""
        env = sole_container_of(render_run(), "StatefulSet", ORCHESTRATOR)["env"]

        assert {entry["name"]: entry["value"] for entry in env} | {
            "MILES_K8S_NAMESPACE": NAMESPACE,
            "MILES_K8S_RELEASE": RUN_RELEASE_NAME,
        } == {entry["name"]: entry["value"] for entry in env}

    def test_grants_the_orchestrator_the_pod_rights_observation_and_healing_need(self):
        """Healing a cell means deleting its pods, and observing one means watching them."""
        objects = render_run()
        role = named_object(objects, "Role", ORCHESTRATOR)
        binding = named_object(objects, "RoleBinding", ORCHESTRATOR)

        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
        assert binding["roleRef"]["name"] == ORCHESTRATOR
        assert binding["subjects"] == [dict(kind="ServiceAccount", name=ORCHESTRATOR, namespace=NAMESPACE)]

    def test_runs_the_orchestrator_under_the_account_that_binding_names(self):
        """A role bound to an account nobody runs as grants nothing at all."""
        assert pod_spec_of(render_run(), "StatefulSet", ORCHESTRATOR)["serviceAccountName"] == ORCHESTRATOR
        assert named_object(render_run(), "ServiceAccount", ORCHESTRATOR)["metadata"]["namespace"] == NAMESPACE
