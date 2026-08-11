import json

from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_ID,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    render_run_error,
    requires_helm,
    sole_container_of,
    with_object_names,
)


def _rollout_executor() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="rollout-executor",
        port_infos=[],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0, num_cpus_per_worker=1),
        worker_class="miles.ray.rollout.rollout_executor.RolloutExecutor",
        ctor_kwargs=lambda context: {},
    )


TWO_WORKERS = [
    {
        "name": "router",
        "command": ["python", "-m", "sglang_router.launch_router", "--port", "30000"],
        "env": {"MILES_ROLE": "router"},
        "ports": [{"name": "http", "port": 30000}],
        "resources": {"requests": {"cpu": "2"}},
    },
    {"name": "dashboard", "command": ["python", "-m", "dashboard"], "ports": [{"name": "rpc", "port": 8000}]},
]


def _render(workers=TWO_WORKERS):
    return _render_with(workers=workers)


def _render_with(*args: str, workers=TWO_WORKERS):
    return render_run("--set-json", f"run.staticWorkers={json.dumps(with_object_names(workers))}", *args)


@requires_helm
class TestStaticWorkers:
    def test_gives_each_worker_its_own_workload_and_service(self):
        """A static worker is addressed by name, so it needs a service of its own rather than a shared one."""
        objects = _render()

        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "StatefulSet")] == [
            "myrun-miles-run-orchestrator",
            "myrun-miles-run-router",
            "myrun-miles-run-dashboard",
        ]
        assert len(objects_of_kind(objects, "Service")) == 3

    def test_passes_the_command_through_unchanged(self):
        """The launcher computed this argv; the chart must not reinterpret it."""
        router = sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")

        assert router["command"] == TWO_WORKERS[0]["command"]

    def test_renders_per_worker_environment(self):
        """Each worker learns its role and addresses from its own environment, not a shared config."""
        env = sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")["env"]

        assert {"name": "MILES_ROLE", "value": "router"} in env

    def test_projects_the_labels_the_pod_joins_into_its_cell_id(self):
        """A pool of several static cells has to tell each pod which cell it is, without parsing its name."""
        env = sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")["env"]

        (entry,) = [item for item in env if item["name"] == "MILES_CELL_INDEX"]

        assert entry["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.labels['apps.kubernetes.io/pod-index']"

    def test_a_single_pod_cell_is_told_no_pod_index(self):
        """Its cell is one pod, so the default of zero is right and a second variable would only drift."""
        env = sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")["env"]

        assert [item for item in env if item["name"] == "MILES_POD_INDEX"] == []

    def test_publishes_the_declared_ports_on_the_pod_and_the_service(self):
        """A port that reaches only one of the two is unreachable or invisible."""
        container = sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")
        service = named_object(_render(), "Service", "myrun-miles-run-router")

        assert container["ports"] == [{"name": "http", "containerPort": 30000}]
        assert service["spec"]["ports"] == [{"name": "http", "port": 30000}]

    def test_applies_only_the_resources_the_launcher_computed(self):
        """Resource arithmetic belongs to the launcher, so the chart copies the block verbatim."""
        assert sole_container_of(_render(), "StatefulSet", "myrun-miles-run-router")["resources"] == {
            "requests": {"cpu": "2"}
        }
        assert sole_container_of(_render(), "StatefulSet", "myrun-miles-run-dashboard")["resources"] == {}

    def test_disables_the_service_link_environment_for_workers_too(self):
        """Workers import miles and would hit the same PROMETHEUS_PORT parsing failure as the orchestrator."""
        assert pod_spec_of(_render(), "StatefulSet", "myrun-miles-run-router")["enableServiceLinks"] is False

    def test_renders_nothing_when_no_worker_is_declared(self):
        """A disaggregated run with an external router declares none, and must still install."""
        assert [obj["metadata"]["name"] for obj in objects_of_kind(render_run(), "StatefulSet")] == [
            "myrun-miles-run-orchestrator"
        ]

    def test_runs_one_instance_per_declared_replica(self):
        """A run wanting several session servers gets several urls, and one pod could answer only one."""
        workers = [dict(TWO_WORKERS[1], name="session-server", replicas=3)]

        statefulset = named_object(_render(workers), "StatefulSet", "myrun-miles-run-session-server")

        assert statefulset["spec"]["replicas"] == 3

    def test_keeps_a_worker_that_names_no_replica_count_single(self):
        """Most static workers are one process, and a chart-side default keeps their values entry short."""
        assert named_object(_render(), "StatefulSet", "myrun-miles-run-router")["spec"]["replicas"] == 1

    def test_addresses_every_instance_through_the_workers_own_headless_service(self):
        """Each session server needs its own dns name, which only a governing headless service gives it."""
        workers = [dict(TWO_WORKERS[1], name="session-server", replicas=3)]
        objects = _render(workers)

        statefulset = named_object(objects, "StatefulSet", "myrun-miles-run-session-server")
        service = named_object(objects, "Service", "myrun-miles-run-session-server")

        assert statefulset["spec"]["serviceName"] == "myrun-miles-run-session-server"
        assert service["spec"]["clusterIP"] == "None"

    def test_rejects_a_worker_asking_for_no_instance_at_all(self):
        """A spec a run turned off is dropped before conversion, so a zero here is a values bug, not a request."""
        error = render_run_error(
            "--set-json",
            'run.staticWorkers=[{"name": "r", "objectName": "myrun-miles-run-r", "command": ["x"], "replicas": 0}]',
        )

        assert "replicas" in error

    def test_rejects_a_worker_without_a_command(self):
        """A pod with no command would run the image entrypoint and look healthy while doing nothing."""
        error = render_run_error(
            "--set-json", 'run.staticWorkers=[{"name": "router", "objectName": "myrun-miles-run-router"}]'
        )

        assert "command" in error
