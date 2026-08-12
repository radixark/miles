import contextlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.helm_backend import command_job
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper

NAMESPACE = "rl"
_RELEASE = "miles-run-command"
CHART_DIR = Path("charts/miles-run")


@dataclass
class FakeKubectl:
    statuses: list[str]
    pod_indices: list[int] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        assert argv[0] == "kubectl", f"only kubectl is expected to reach the process layer, got {argv[0]}"
        arguments = argv[1:]
        self.calls.append(arguments)
        if arguments[:2] == ["get", "pods"]:
            items = [
                {
                    "metadata": {
                        "name": f"convert-{index}",
                        "uid": f"uid-{index}",
                        "labels": {command_job._COMPLETION_INDEX_KEY: str(index)},
                    }
                }
                for index in self.pod_indices
            ]
            body = json.dumps({"items": items}) if self.pod_indices else ""
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=body, stderr="")
        if arguments[0] == "get":
            status = self.statuses.pop(0) if self.statuses else "running"
            body = {
                "running": '{"status": {}}',
                "complete": '{"status": {"conditions": [{"type": "Complete", "status": "True"}]}}',
                "failed": '{"status": {"conditions": [{"type": "Failed", "status": "True"}]}}',
                "absent": "",
            }[status]
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=body, stderr="")
        if arguments[0] == "logs":
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout=f"the output of {arguments[1]}", stderr=""
            )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    def verbs(self) -> list[str]:
        return [call[0] for call in self.calls]

    def targets(self) -> list[str]:
        return [" ".join(call[:2]) for call in self.calls]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    kubectl: FakeKubectl,
    completions: int = 1,
    capture_output: bool = False,
    timeout_seconds: float = 3600.0,
) -> list[str | None]:
    monkeypatch.setattr(command_job, "_render_job", lambda job, command: "kind: Job\n")
    _stub_cluster(monkeypatch, kubectl)
    return command_job._run_job(
        _job(completions=completions, timeout_seconds=timeout_seconds),
        command=["bash", "-c", "convert"],
        capture_output=capture_output,
    )


def _stub_cluster(monkeypatch: pytest.MonkeyPatch, kubectl: FakeKubectl) -> None:
    monkeypatch.setattr(command_wrapper, "run_process", kubectl)
    monkeypatch.setattr(command_job.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(command_job, "with_observability", lambda **kwargs: contextlib.nullcontext())


def _job(**overrides: Any) -> command_job._CommandJob:
    context_fields = {
        key: overrides.pop(key) for key in ("helm_values_files", "timeout_seconds", "chart_dir") if key in overrides
    }
    fields: dict[str, Any] = {
        "context": _context(gpus_per_node=8, **context_fields),
        "step": "convert",
        "completions": 1,
        "gpus_per_pod": 8,
        **overrides,
    }
    return command_job._CommandJob(**fields)


def _context(**overrides: Any) -> command_job.CommandJobContext:
    return command_job.CommandJobContext(
        namespace=NAMESPACE, **{"chart_dir": CHART_DIR, "gpus_per_node": 1, **overrides}
    )


def _gpu_backend() -> Any:
    pytest.importorskip("torch")
    from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
    from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend

    return KubernetesCommandBackend(ExecuteTrainConfig(namespace=NAMESPACE))


def _record_run_job(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run_job(job: command_job._CommandJob, **kwargs: Any) -> list[str | None]:
        calls.append({"job": job, **kwargs})
        return [None] * job.completions

    monkeypatch.setattr(command_job, "_run_job", fake_run_job)
    return calls


def _render_calls(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="kind: Job\n", stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    command = overrides.pop("command", ["bash", "-c", "convert"])
    command_job._render_job(
        _job(timeout_seconds=overrides.pop("timeout_seconds", 10800.0), **overrides), command=command
    )
    return captured


def _render(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[str]:
    templates = [call for call in _render_calls(monkeypatch, **overrides) if call[:2] == ["helm", "template"]]
    assert len(templates) == 1, f"expected one helm template call, got {templates}"
    return templates[0]


class TestNaming:
    def test_names_the_job_the_way_the_chart_does(self):
        """The launcher has to address an object it did not name itself."""
        assert _job().object_name == "miles-run-command-convert"

    def test_addresses_rank_zero_through_the_headless_service(self):
        """A multi-node step needs one address every pod agrees on before any of them is scheduled."""
        assert _job().master_address == ("miles-run-command-convert-0.miles-run-command-convert.rl.svc.cluster.local")

    def test_installs_command_jobs_under_their_own_release(self):
        """A step sharing a training run's release would be torn down with it, or collide with its objects."""
        assert command_job._RELEASE == "miles-run-command"


class TestRenderJob:
    def test_builds_the_chart_dependencies_before_rendering(self, monkeypatch, tmp_path):
        """A checkout carries the lock, not the subchart it pins, so a run whose first step is a
        command job renders against a chart helm calls incomplete."""
        monkeypatch.setattr(command_wrapper, "_locked_dependency_names", lambda chart: ["miles-common"])
        calls = _render_calls(monkeypatch, chart_dir=tmp_path)

        assert [call[:3] for call in calls][0] == ["helm", "dependency", "build"]
        assert any(call[:2] == ["helm", "template"] for call in calls)

    def test_renders_only_the_command_job_out_of_the_run_chart(self, monkeypatch):
        """The chart also holds the run's own workloads, and applying those would start a training run."""
        arguments = _render(monkeypatch)

        assert arguments[arguments.index("--show-only") + 1] == "templates/command-job.yaml"

    def test_turns_the_command_job_on_because_a_run_leaves_it_off(self, monkeypatch):
        """The template renders nothing by default, so a step forgetting the flag would apply an empty manifest."""
        assert "commandJob.enabled=true" in _render(monkeypatch)

    def test_passes_the_step_shape_through_the_command_job_values(self, monkeypatch):
        """The pod count and the gpus each pod claims are what make a step single-node or multi-node."""
        arguments = _render(monkeypatch, step="convert", completions=4, gpus_per_pod=8)

        assert "commandJob.name=convert" in arguments
        assert "commandJob.completions=4" in arguments
        assert "commandJob.gpusPerPod=8" in arguments

    def test_gives_the_job_the_same_deadline_the_launcher_waits_for(self, monkeypatch):
        """A Job outliving the waiter would keep a gpu busy for a launch that has already given up."""
        assert "commandJob.activeDeadlineSeconds=60" in _render(monkeypatch, timeout_seconds=60)

    def test_sends_the_command_as_json_so_helm_never_splits_it(self, monkeypatch):
        """A plain --set would split the command on commas, wrecking any argument holding one."""
        arguments = _render(monkeypatch, command=["bash", "-c", "echo a,b"])

        assert arguments[arguments.index("--set-json") + 1] == 'commandJob.command=["bash", "-c", "echo a,b"]'

    def test_names_the_run_the_chart_insists_on(self, monkeypatch):
        """run.id is required by the chart, and a command job has no run to borrow an id from."""
        assert "run.id=command-job" in _render(monkeypatch)

    def test_keeps_the_cluster_values_files(self, monkeypatch):
        """The image, the storage mounts and the node selectors all come from the infra values."""
        arguments = _render(monkeypatch, helm_values_files=("/infra.yaml",))

        assert arguments[arguments.index("/infra.yaml") - 1] == "--values"


class TestRunOnNodes:
    def test_substitutes_the_rank_with_what_the_job_gives_each_pod(self, monkeypatch):
        """A command templating the pod index has no other way to learn which pod it landed in."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "torchrun --node-rank {{node_rank}}",
            capture_output=False,
            completions=2,
            step="convert",
        )

        assert calls[0]["command"] == ["bash", "-c", "torchrun --node-rank ${JOB_COMPLETION_INDEX}"]

    def test_tells_every_pod_how_many_of_them_there_are(self, monkeypatch):
        """torchrun refuses to rendezvous unless every rank agrees on the world size."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "torchrun --nnodes={{nnodes}}",
            capture_output=False,
            completions=4,
            step="convert",
        )

        assert calls[0]["command"][-1] == "torchrun --nnodes=4"

    def test_points_every_pod_at_the_headless_address_of_rank_zero(self, monkeypatch):
        """No pod knows another pod's ip before scheduling, but the service name is fixed in advance."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "--master-addr {{master_addr}}",
            capture_output=False,
            completions=2,
            step="convert",
        )

        assert calls[0]["command"][-1] == f"--master-addr {_job().master_address}"

    def test_resolves_a_pod_own_address_inside_the_pod(self, monkeypatch):
        """The launcher cannot know the ip a pod will get, so the pod has to look it up itself."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "--node-ip {{node_ip}}",
            capture_output=False,
            completions=1,
            step="convert",
        )

        assert calls[0]["command"][-1] == "--node-ip $(hostname -i)"

    def test_carries_the_context_settings_into_the_job(self, monkeypatch):
        """The namespace, the chart and the values files decide where and as what the step actually runs."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(helm_values_files=("/infra.yaml",), timeout_seconds=60.0),
            "echo hi",
            capture_output=True,
            completions=2,
            step="step",
        )

        context = calls[0]["job"].context
        assert context.namespace == NAMESPACE
        assert context.chart_dir == CHART_DIR
        assert context.helm_values_files == ("/infra.yaml",)
        assert context.timeout_seconds == 60.0

    def test_gives_every_pod_the_gpus_of_the_node_it_lands_on(self, monkeypatch):
        """These commands are written as if they owned the machine, and the context is where that count lives."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(gpus_per_node=4), "nvidia-smi", capture_output=False, completions=2, step="convert"
        )

        assert calls[0]["job"].gpus_per_pod == 4


class TestExecCommandGpu:
    def test_asks_for_a_single_pod_holding_the_whole_node(self, monkeypatch):
        """A gpu step is written as if it owned the machine, which a pod short of the node's gpus breaks."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_gpu("nvidia-smi", num_gpus_per_node=4)

        assert (calls[0]["job"].completions, calls[0]["job"].gpus_per_pod) == (1, 4)

    def test_substitutes_the_placeholders_of_a_single_node_command_too(self, monkeypatch):
        """A converter templating its rank is run on one node as often as on many."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_gpu("torchrun --node-rank {{node_rank}} --nnodes={{nnodes}}")

        assert calls[0]["command"][-1] == "torchrun --node-rank ${JOB_COMPLETION_INDEX} --nnodes=1"

    def test_returns_the_single_result_rather_than_a_list(self, monkeypatch):
        """Its callers read the output as a string, and a list would silently become the wrong argument."""
        monkeypatch.setattr(command_job, "_run_job", lambda job, **kwargs: ["the output"])

        assert _gpu_backend().exec_command_gpu("nvidia-smi", capture_output=True) == "the output"


class TestExecCommandMultiNode:
    def test_gives_a_multi_node_step_the_gpus_of_every_node_it_lands_on(self, monkeypatch):
        """The ray backend runs these commands on whole gpu nodes, and a gpu-less pod fails torchrun outright."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2, num_gpus_per_node=4)

        assert (calls[0]["job"].completions, calls[0]["job"].gpus_per_pod) == (2, 4)


class TestRunJob:
    def test_clears_a_previous_attempt_before_submitting(self, monkeypatch):
        """apply would refuse an existing Job, and its logs would describe the wrong run."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs()[0] == "delete"

    def test_submits_the_rendered_manifest(self, monkeypatch):
        """A step whose manifest never reached the cluster would wait for a Job nobody created."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs()[1] == "apply"

    def test_polls_until_the_job_finishes(self, monkeypatch):
        """An command job takes minutes, so one status read would always find it running."""
        kubectl = FakeKubectl(statuses=["absent", "running", "running", "complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.targets().count("get job") == 4

    def test_raises_with_the_logs_when_the_job_fails(self, monkeypatch):
        """A failed conversion must stop the launch, and the reason is in the pod output."""
        kubectl = FakeKubectl(statuses=["failed"])

        with pytest.raises(RuntimeError, match="the output"):
            _run(monkeypatch, kubectl)

    def test_leaves_a_failed_job_in_place(self, monkeypatch):
        """Its pods are the only evidence left, so deleting them would destroy the diagnosis."""
        kubectl = FakeKubectl(statuses=["failed"])

        with pytest.raises(RuntimeError):
            _run(monkeypatch, kubectl)

        assert kubectl.verbs().count("delete") == 1

    def test_deletes_a_successful_job(self, monkeypatch):
        """Finished Jobs otherwise pile up in the namespace until someone notices."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs().count("delete") == 2

    def test_gives_back_one_result_per_node(self, monkeypatch):
        """Callers of the multi-node helper index the result by rank."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0, 1, 2, 3])

        assert _run(monkeypatch, kubectl, completions=4, capture_output=True) == [
            f"the output of convert-{index}" for index in range(4)
        ]

    def test_reads_each_pod_own_log_in_completion_index_order(self, monkeypatch):
        """One job log repeated N times hides every rank but one, unlike the ray backend it stands in for."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[2, 0, 1])

        logs = _run(monkeypatch, kubectl, completions=3, capture_output=True)

        assert logs == ["the output of convert-0", "the output of convert-1", "the output of convert-2"]
        assert [call[1] for call in kubectl.calls if call[0] == "logs"] == ["convert-0", "convert-1", "convert-2"]

    def test_selects_the_pods_of_this_job_alone(self, monkeypatch):
        """A namespace runs several steps at once, and another step's pods would be read as this one's ranks."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        _run(monkeypatch, kubectl, completions=1, capture_output=True)

        listing = next(call for call in kubectl.calls if call[:2] == ["get", "pods"])
        assert listing[listing.index("--selector") + 1] == command_wrapper.Kubectl.job_selector(
            "miles-run-command-convert"
        )

    def test_names_the_rank_whose_pod_never_appeared(self, monkeypatch):
        """A silent empty string would read as a rank that ran and printed nothing."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs[1] == "no pod of this job reported completion index 1"

    def test_falls_back_to_the_job_log_when_no_pod_can_be_listed(self, monkeypatch):
        """A step whose pods were already garbage collected must still surface whatever the job kept."""
        kubectl = FakeKubectl(statuses=["complete"])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs == ["the output of job/miles-run-command-convert"] * 2

    def test_gives_back_nothing_when_the_output_was_not_asked_for(self, monkeypatch):
        """Most steps only care that the command worked, and a log dump would drown the launcher output."""
        kubectl = FakeKubectl(statuses=["complete"])

        assert _run(monkeypatch, kubectl) == [None]

    def test_hands_its_own_timeout_to_the_rendered_job(self, monkeypatch):
        """Two independent timeouts would drift, leaving either an orphan Job or a premature failure."""
        rendered: list[command_job._CommandJob] = []
        monkeypatch.setattr(command_job, "_render_job", lambda job, command: rendered.append(job) or "kind: Job\n")
        _stub_cluster(monkeypatch, FakeKubectl(statuses=["complete"]))

        command_job._run_job(_job(timeout_seconds=90.0), command=["bash", "-c", "convert"], capture_output=False)

        assert int(rendered[0].context.timeout_seconds) == 90

    def test_reports_a_job_that_never_finishes(self, monkeypatch):
        """A step waiting on a gpu that never frees must fail rather than hang the launch forever."""
        kubectl = FakeKubectl(statuses=[])

        with pytest.raises(RuntimeError, match="did not finish"):
            _run(monkeypatch, kubectl, timeout_seconds=10.0)
