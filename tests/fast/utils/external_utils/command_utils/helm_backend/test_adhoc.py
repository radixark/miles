import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.helm_backend import adhoc

NAMESPACE = "rl"
RELEASE = "miles-run-adhoc"
CHART_DIR = "charts/miles-run"


@dataclass
class FakeKubectl:
    statuses: list[str]
    pod_indices: list[int] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, arguments: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(arguments)
        if arguments[:2] == ["get", "pods"]:
            items = [
                {"metadata": {"name": f"convert-{index}", "labels": {adhoc._COMPLETION_INDEX_KEY: str(index)}}}
                for index in self.pod_indices
            ]
            body = json.dumps({"items": items}) if self.pod_indices else ""
            return subprocess.CompletedProcess(args=arguments, returncode=0, stdout=body, stderr="")
        if arguments[0] == "get":
            status = self.statuses.pop(0) if self.statuses else "running"
            body = {
                "running": '{"status": {}}',
                "complete": '{"status": {"conditions": [{"type": "Complete", "status": "True"}]}}',
                "failed": '{"status": {"conditions": [{"type": "Failed", "status": "True"}]}}',
                "absent": "",
            }[status]
            return subprocess.CompletedProcess(args=arguments, returncode=0, stdout=body, stderr="")
        if arguments[0] == "logs":
            return subprocess.CompletedProcess(
                args=arguments, returncode=0, stdout=f"the output of {arguments[1]}", stderr=""
            )
        return subprocess.CompletedProcess(args=arguments, returncode=0, stdout="", stderr="")

    def verbs(self) -> list[str]:
        return [call[0] for call in self.calls]

    def targets(self) -> list[str]:
        return [" ".join(call[:2]) for call in self.calls]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    kubectl: FakeKubectl,
    completions: int = 1,
    capture_output: bool = False,
) -> list[str | None]:
    monkeypatch.setattr(adhoc, "render_job", lambda **kwargs: "kind: Job\n")
    monkeypatch.setattr(adhoc, "_apply", lambda manifest, namespace, kubectl: None)
    return adhoc.run_job(
        command=["bash", "-c", "convert"],
        namespace=NAMESPACE,
        chart_dir=CHART_DIR,
        infra_values_files=[],
        release=RELEASE,
        step="convert",
        completions=completions,
        gpus_per_pod=8,
        capture_output=capture_output,
        timeout_seconds=100,
        poll_interval_seconds=1,
        sleep=lambda seconds: None,
        kubectl=kubectl,
    )


def _context(**overrides: Any) -> adhoc.AdhocContext:
    return adhoc.AdhocContext(namespace=NAMESPACE, chart_dir=CHART_DIR, **overrides)


def _record_run_job(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run_job(**kwargs: Any) -> list[str | None]:
        calls.append(kwargs)
        return [None] * kwargs["completions"]

    monkeypatch.setattr(adhoc, "run_job", fake_run_job)
    return calls


def _render(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[str]:
    captured: list[list[str]] = []

    def fake_helm_run(arguments: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        captured.append(arguments)
        return subprocess.CompletedProcess(args=arguments, returncode=0, stdout="kind: Job\n", stderr="")

    monkeypatch.setattr(adhoc.helm, "run", fake_helm_run)
    arguments: dict[str, Any] = {
        "command": ["bash", "-c", "convert"],
        "namespace": NAMESPACE,
        "chart_dir": CHART_DIR,
        "infra_values_files": [],
        "release": RELEASE,
        "step": "convert",
        "completions": 1,
        "gpus_per_pod": 8,
        "active_deadline_seconds": 10800,
        **overrides,
    }
    adhoc.render_job(**arguments)
    return captured[0]


class TestNaming:
    def test_names_the_job_the_way_the_chart_does(self):
        """The launcher has to address an object it did not name itself."""
        assert adhoc.job_object_name(RELEASE, "convert") == "miles-run-adhoc-convert"

    def test_addresses_rank_zero_through_the_headless_service(self):
        """A multi-node step needs one address every pod agrees on before any of them is scheduled."""
        assert adhoc.master_address(RELEASE, "convert", NAMESPACE) == (
            "miles-run-adhoc-convert-0.miles-run-adhoc-convert.rl.svc.cluster.local"
        )

    def test_installs_adhoc_steps_under_their_own_release(self):
        """A step sharing a training run's release would be torn down with it, or collide with its objects."""
        assert _context().release == "miles-run-adhoc"


class TestRenderJob:
    def test_renders_only_the_adhoc_job_out_of_the_run_chart(self, monkeypatch):
        """The chart also holds the run's own workloads, and applying those would start a training run."""
        arguments = _render(monkeypatch)

        assert arguments[arguments.index("--show-only") + 1] == "templates/adhoc-job.yaml"

    def test_turns_the_adhoc_job_on_because_a_run_leaves_it_off(self, monkeypatch):
        """The template renders nothing by default, so a step forgetting the flag would apply an empty manifest."""
        assert "adhoc.enabled=true" in _render(monkeypatch)

    def test_passes_the_step_shape_through_the_adhoc_values(self, monkeypatch):
        """The pod count and the gpus each pod claims are what make a step single-node or multi-node."""
        arguments = _render(monkeypatch, step="convert", completions=4, gpus_per_pod=8)

        assert "adhoc.name=convert" in arguments
        assert "adhoc.completions=4" in arguments
        assert "adhoc.gpusPerPod=8" in arguments

    def test_gives_the_job_the_same_deadline_the_launcher_waits_for(self, monkeypatch):
        """A Job outliving the waiter would keep a gpu busy for a launch that has already given up."""
        assert "adhoc.activeDeadlineSeconds=60" in _render(monkeypatch, active_deadline_seconds=60)

    def test_sends_the_command_as_json_so_helm_never_splits_it(self, monkeypatch):
        """A plain --set would split the command on commas, wrecking any argument holding one."""
        arguments = _render(monkeypatch, command=["bash", "-c", "echo a,b"])

        assert arguments[arguments.index("--set-json") + 1] == 'adhoc.command=["bash", "-c", "echo a,b"]'

    def test_names_the_run_the_chart_insists_on(self, monkeypatch):
        """run.id is required by the chart, and an adhoc step has no run to borrow an id from."""
        assert "run.id=adhoc" in _render(monkeypatch)

    def test_keeps_the_cluster_values_files(self, monkeypatch):
        """The image, the storage mounts and the node selectors all come from the infra values."""
        arguments = _render(monkeypatch, infra_values_files=["/infra.yaml"])

        assert arguments[arguments.index("/infra.yaml") - 1] == "--values"


class TestRunOnNodes:
    def test_substitutes_the_rank_with_what_the_job_gives_each_pod(self, monkeypatch):
        """A command templating the node rank has no other way to learn which pod it landed in."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_nodes(
            _context(),
            "torchrun --node-rank {{node_rank}}",
            capture_output=False,
            completions=2,
            gpus_per_pod=8,
            step="convert",
        )

        assert calls[0]["command"] == ["bash", "-c", "torchrun --node-rank ${JOB_COMPLETION_INDEX}"]

    def test_tells_every_pod_how_many_of_them_there_are(self, monkeypatch):
        """torchrun refuses to rendezvous unless every rank agrees on the world size."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_nodes(
            _context(),
            "torchrun --nnodes={{nnodes}}",
            capture_output=False,
            completions=4,
            gpus_per_pod=8,
            step="convert",
        )

        assert calls[0]["command"][-1] == "torchrun --nnodes=4"

    def test_points_every_pod_at_the_headless_address_of_rank_zero(self, monkeypatch):
        """No pod knows another pod's ip before scheduling, but the service name is fixed in advance."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_nodes(
            _context(),
            "--master-addr {{master_addr}}",
            capture_output=False,
            completions=2,
            gpus_per_pod=8,
            step="convert",
        )

        assert calls[0]["command"][-1] == f"--master-addr {adhoc.master_address(RELEASE, 'convert', NAMESPACE)}"

    def test_resolves_a_pod_own_address_inside_the_pod(self, monkeypatch):
        """The launcher cannot know the ip a pod will get, so the pod has to look it up itself."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_nodes(
            _context(),
            "--node-ip {{node_ip}}",
            capture_output=False,
            completions=1,
            gpus_per_pod=8,
            step="convert",
        )

        assert calls[0]["command"][-1] == "--node-ip $(hostname -i)"

    def test_carries_the_context_settings_into_the_job(self, monkeypatch):
        """The namespace, the chart and the values files decide where and as what the step actually runs."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_nodes(
            _context(infra_values_files=("/infra.yaml",), timeout_seconds=60.0),
            "echo hi",
            capture_output=True,
            completions=2,
            gpus_per_pod=0,
            step="step",
        )

        assert calls[0]["namespace"] == NAMESPACE
        assert calls[0]["chart_dir"] == CHART_DIR
        assert calls[0]["infra_values_files"] == ["/infra.yaml"]
        assert calls[0]["timeout_seconds"] == 60.0


class TestRunOnOneGpuNode:
    def test_asks_for_a_single_pod_holding_the_whole_node(self, monkeypatch):
        """A gpu step is written as if it owned the machine, which a pod short of the node's gpus breaks."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_one_gpu_node(_context(gpus_per_node=4), "nvidia-smi")

        assert (calls[0]["completions"], calls[0]["gpus_per_pod"], calls[0]["step"]) == (1, 4, "gpu")

    def test_substitutes_the_placeholders_of_a_single_node_command_too(self, monkeypatch):
        """A converter templating its rank is run on one node as often as on many."""
        calls = _record_run_job(monkeypatch)

        adhoc.run_on_one_gpu_node(_context(), "torchrun --node-rank {{node_rank}} --nnodes={{nnodes}}")

        assert calls[0]["command"][-1] == "torchrun --node-rank ${JOB_COMPLETION_INDEX} --nnodes=1"

    def test_returns_the_single_result_rather_than_a_list(self, monkeypatch):
        """Its callers read the output as a string, and a list would silently become the wrong argument."""
        monkeypatch.setattr(adhoc, "run_job", lambda **kwargs: ["the output"])

        assert adhoc.run_on_one_gpu_node(_context(), "nvidia-smi", capture_output=True) == "the output"


class TestExecCommandMultiNode:
    def test_gives_a_multi_node_step_the_gpus_of_every_node_it_lands_on(self, monkeypatch):
        """The ray backend runs these commands on whole gpu nodes, and a gpu-less pod fails torchrun outright."""
        pytest.importorskip("torch")
        from miles.utils.external_utils.command_utils.helm_backend import KubernetesCommandBackend

        calls = _record_run_job(monkeypatch)
        backend = KubernetesCommandBackend()
        backend._adhoc_context = _context(gpus_per_node=4)

        backend.exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2)

        assert (calls[0]["completions"], calls[0]["gpus_per_pod"]) == (2, 4)


class TestRunJob:
    def test_clears_a_previous_attempt_before_submitting(self, monkeypatch):
        """apply would refuse an existing Job, and its logs would describe the wrong run."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs()[0] == "delete"

    def test_polls_until_the_job_finishes(self, monkeypatch):
        """An adhoc step takes minutes, so one status read would always find it running."""
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
        assert listing[listing.index("--selector") + 1] == f"{adhoc._JOB_NAME_LABEL}=miles-run-adhoc-convert"

    def test_names_the_rank_whose_pod_never_appeared(self, monkeypatch):
        """A silent empty string would read as a rank that ran and printed nothing."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs[1] == "no pod of this job reported completion index 1"

    def test_falls_back_to_the_job_log_when_no_pod_can_be_listed(self, monkeypatch):
        """A step whose pods were already garbage collected must still surface whatever the job kept."""
        kubectl = FakeKubectl(statuses=["complete"])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs == ["the output of job/miles-run-adhoc-convert"] * 2

    def test_gives_back_nothing_when_the_output_was_not_asked_for(self, monkeypatch):
        """Most steps only care that the command worked, and a log dump would drown the launcher output."""
        kubectl = FakeKubectl(statuses=["complete"])

        assert _run(monkeypatch, kubectl) == [None]

    def test_hands_its_own_timeout_to_the_rendered_job(self, monkeypatch):
        """Two independent timeouts would drift, leaving either an orphan Job or a premature failure."""
        rendered: list[dict[str, Any]] = []
        monkeypatch.setattr(adhoc, "render_job", lambda **kwargs: rendered.append(kwargs) or "kind: Job\n")
        monkeypatch.setattr(adhoc, "_apply", lambda manifest, namespace, kubectl: None)

        adhoc.run_job(
            command=["bash", "-c", "convert"],
            namespace=NAMESPACE,
            chart_dir=CHART_DIR,
            infra_values_files=[],
            release=RELEASE,
            step="convert",
            completions=1,
            gpus_per_pod=8,
            capture_output=False,
            timeout_seconds=90.0,
            poll_interval_seconds=1,
            sleep=lambda seconds: None,
            kubectl=FakeKubectl(statuses=["complete"]),
        )

        assert rendered[0]["active_deadline_seconds"] == 90

    def test_reports_a_job_that_never_finishes(self, monkeypatch):
        """A step waiting on a gpu that never frees must fail rather than hang the launch forever."""
        kubectl = FakeKubectl(statuses=[])

        with pytest.raises(RuntimeError, match="did not finish"):
            _run(monkeypatch, kubectl)
