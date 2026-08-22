from tests.fast.charts.utils import objects_of_kind, render_run, render_run_error, requires_helm, single_object_of_kind

ENABLE_COMMAND_JOB = (
    "--show-only",
    "templates/command-job.yaml",
    "--set",
    "commandJob.enabled=true",
    "--set",
    "commandJob.name=convert",
    "--set",
    "commandJob.objectName=myrun-miles-run-convert",
    "--set-json",
    'commandJob.command=["bash","-c","convert"]',
)


@requires_helm
class TestCommandJob:
    def test_renders_on_its_own_so_the_caller_can_apply_it_without_a_release(self):
        """Command-job work is not part of a run, so it is rendered with --show-only and applied directly."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert job["metadata"]["name"] == "myrun-miles-run-convert"
        assert job["spec"]["template"]["spec"]["containers"][0]["command"] == ["bash", "-c", "convert"]

    def test_renders_without_the_values_a_run_would_carry(self):
        """The launcher renders a command job off the bare chart, and helm evaluates every template
        before --show-only picks one, so a run's own guard would reject work that has no run."""
        job = single_object_of_kind(
            render_run(*ENABLE_COMMAND_JOB, "--set-json", "run.orchestrator.command=[]"), "Job"
        )

        assert job["metadata"]["name"] == "myrun-miles-run-convert"

    def test_names_the_container_after_the_kind_of_work_it_does(self):
        """kubectl logs picks a container by name, so the launcher needs a name that never changes."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert [container["name"] for container in job["spec"]["template"]["spec"]["containers"]] == ["command-job"]

    def test_an_eviction_does_not_spend_the_one_attempt(self):
        """Kubernetes counts a displaced pod against backoffLimit like a failed one, so without this
        a step the cluster evicted before it ran is reported as a step that ran and failed."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert job["spec"]["podFailurePolicy"]["rules"] == [
            {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}
        ]

    def test_never_retries_a_failure(self):
        """The caller reports the failure; a silent retry would hide it and double the side effects."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert job["spec"]["backoffLimit"] == 0
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_runs_one_pod_per_node_when_asked(self):
        """A multi-node command job needs a pod per node, each knowing its index."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB, "--set", "commandJob.completions=4"), "Job")

        assert (job["spec"]["completions"], job["spec"]["parallelism"]) == (4, 4)
        assert job["spec"]["completionMode"] == "Indexed"

    def test_stays_a_plain_job_for_a_single_pod(self):
        """A single-pod step has no index to consume, so Indexed mode would only add noise."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert "completionMode" not in job["spec"]

    def test_requests_gpus_only_when_asked(self):
        """Checkpoint conversion needs gpus; a download must not sit in the gpu queue."""
        with_gpus = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB, "--set", "commandJob.gpusPerPod=8"), "Job")
        without = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert with_gpus["spec"]["template"]["spec"]["containers"][0]["resources"] == {"limits": {"nvidia.com/gpu": 8}}
        assert "resources" not in without["spec"]["template"]["spec"]["containers"][0]

    def test_gives_up_on_a_job_that_never_gets_its_node(self):
        """Without a deadline a step waiting on an unavailable gpu would hold the queue forever."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert job["spec"]["activeDeadlineSeconds"] == 10800

    def test_collects_a_finished_job_by_itself(self):
        """The launcher may die before deleting the Job, and the namespace must not fill up with corpses."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB), "Job")

        assert job["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_lets_the_caller_choose_the_deadline_and_the_lifetime(self):
        """The python launcher owns the real timeout, so the chart must accept it instead of drifting."""
        job = single_object_of_kind(
            render_run(
                *ENABLE_COMMAND_JOB,
                "--set",
                "commandJob.activeDeadlineSeconds=60",
                "--set",
                "commandJob.ttlSecondsAfterFinished=0",
            ),
            "Job",
        )

        assert (job["spec"]["activeDeadlineSeconds"], job["spec"]["ttlSecondsAfterFinished"]) == (60, 0)

    def test_bounds_a_multi_pod_job_the_same_way(self):
        """A fan-out step is exactly the one most likely to wedge, so it may not escape the deadline."""
        job = single_object_of_kind(render_run(*ENABLE_COMMAND_JOB, "--set", "commandJob.completions=4"), "Job")

        assert job["spec"]["activeDeadlineSeconds"] == 10800
        assert job["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_carries_the_cluster_environment_like_every_other_pod(self):
        """A proxy the cluster needs is as necessary for a download here as it is inside the run."""
        job = single_object_of_kind(
            render_run(*ENABLE_COMMAND_JOB, "--set", "infra.env.HTTP_PROXY=http://proxy:7890"), "Job"
        )

        assert {"name": "HTTP_PROXY", "value": "http://proxy:7890"} in (
            job["spec"]["template"]["spec"]["containers"][0]["env"]
        )

    def test_refuses_to_be_enabled_without_a_name_or_a_command(self):
        """Both are schema-valid at their defaults, and would render a Job that cannot run."""
        assert "commandJob" in render_run_error("--set", "commandJob.enabled=true")
        assert "commandJob" in render_run_error("--set", "commandJob.enabled=true", "--set", "commandJob.name=conv")

    def test_is_absent_from_a_run(self):
        """Installing it with the release would rerun the whole command job on every helm upgrade."""
        assert objects_of_kind(render_run(), "Job") == []
