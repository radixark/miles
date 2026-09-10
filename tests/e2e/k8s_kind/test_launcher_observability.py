from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

import pytest
from kubernetes_asyncio import client as kubernetes_client
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.utils import BUSYBOX_IMAGE, CELL_LABEL, pod_body, wait_until
from tests.e2e.k8s_kind.kind_cluster import KindCluster
from typer.testing import CliRunner

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.cluster_info import with_cluster_info
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.diagnosis import collect_diagnosis
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.log_follower import (
    with_log_following,
)
from miles.utils.external_utils.miles_workbench.__main__ import app

register_cpu_ci(est_time=600, suite="stage-b-cpu", labels=[])

_MISSING_IMAGE = "miles.invalid/there-is-no-such-image:1"
_POLL_TIMEOUT = 60.0
_STARTUP_TIMEOUT = 180.0
_LIFECYCLE_TIMEOUT = 240.0


@pytest.fixture
def kubectl_kubeconfig(kind_cluster: KindCluster, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUBECONFIG", str(kind_cluster.kubeconfig))


class TestLogFollowing:
    async def test_the_follower_streams_the_lines_a_pod_prints(
        self,
        cluster_core_v1: kubernetes_client.CoreV1Api,
        cluster_namespace: str,
        kubectl_kubeconfig: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Following a running pod logs the lines it printed, from the first, under a pod and container prefix."""
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-logs",
                cell="cell-logs",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", "i=1; while true; do echo line-$i; i=$((i + 1)); sleep 1; done"],
            ),
        )
        await _wait_for_phase(cluster_core_v1, namespace=cluster_namespace, name="pod-logs", phase="Running")

        with caplog.at_level(logging.INFO):
            with with_log_following(namespace=cluster_namespace, selector=f"{CELL_LABEL}=cell-logs"):
                await wait_until(
                    lambda: len(_followed_line_numbers(caplog)) >= 3,
                    description="the follower to stream several lines of the pod",
                    timeout=_STARTUP_TIMEOUT,
                    interval=1.0,
                )

        numbers = _followed_line_numbers(caplog)
        assert numbers == list(range(1, len(numbers) + 1)), f"the followed lines are not the pod's own {numbers=}"


class TestClusterInfo:
    async def test_it_reports_a_pod_from_its_appearance_to_its_completion(
        self,
        cluster_core_v1: kubernetes_client.CoreV1Api,
        cluster_namespace: str,
        kubectl_kubeconfig: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The watcher logs the status changes and the pod summary of a pod that runs and then succeeds."""
        with caplog.at_level(logging.INFO):
            with with_cluster_info(namespace=cluster_namespace, selector=f"{CELL_LABEL}=cell-phases"):
                await wait_until(
                    lambda: "No pods yet" in _messages(caplog),
                    description="the watcher to report the empty namespace it starts from",
                    timeout=_POLL_TIMEOUT,
                    interval=1.0,
                )
                await cluster_core_v1.create_namespaced_pod(
                    namespace=cluster_namespace,
                    body=pod_body(
                        name="pod-phases",
                        cell="cell-phases",
                        image=BUSYBOX_IMAGE,
                        command=["sh", "-c", "sleep 25"],
                    ),
                )
                await wait_until(
                    lambda: _reported_states(caplog, pod="pod-phases")[-1:] == ["Succeeded"],
                    description="the watcher to report the pod finishing",
                    timeout=_LIFECYCLE_TIMEOUT,
                    interval=1.0,
                )

        states = _reported_states(caplog, pod="pod-phases")
        assert any(state.startswith("Running") for state in states), f"the running pod was never reported {states=}"
        assert "1 pods: 1 running" in _messages(caplog), "the pod summary never counted the running pod"

    async def test_it_reports_the_warning_events_of_a_pod_that_cannot_start(
        self,
        cluster_core_v1: kubernetes_client.CoreV1Api,
        cluster_namespace: str,
        kubectl_kubeconfig: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A pod whose image cannot be pulled has its kubelet warning events surfaced as warnings."""
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(name="pod-unpullable", cell="cell-unpullable", image=_MISSING_IMAGE),
        )

        with caplog.at_level(logging.INFO):
            with with_cluster_info(namespace=cluster_namespace, selector=f"{CELL_LABEL}=cell-unpullable"):
                await wait_until(
                    lambda: _warnings_about(caplog, pod="pod-unpullable") != [],
                    description="the watcher to report a warning event of the unpullable pod",
                    timeout=_LIFECYCLE_TIMEOUT,
                    interval=1.0,
                )

        warnings = _warnings_about(caplog, pod="pod-unpullable")
        assert any(
            "ErrImagePull" in line or "ImagePullBackOff" in line or _MISSING_IMAGE in line for line in warnings
        ), f"the warnings do not say the image could not be pulled {warnings=}"


class TestCollectDiagnosis:
    async def test_it_captures_the_output_and_the_exit_code_of_a_failing_pod(
        self,
        cluster_core_v1: kubernetes_client.CoreV1Api,
        cluster_namespace: str,
        kubectl_kubeconfig: None,
        tmp_path: Path,
    ) -> None:
        """The diagnosis of a pod that ran a bad script holds its output, its exit code and the namespace events."""
        marker = f"BOOM-{uuid.uuid4().hex}"
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-bad-script",
                cell="cell-bad-script",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", f"echo {marker}; exit 3"],
            ),
        )
        await _wait_for_phase(cluster_core_v1, namespace=cluster_namespace, name="pod-bad-script", phase="Failed")

        diagnosis = collect_diagnosis(
            namespace=cluster_namespace, output_dir=tmp_path, selector=f"{CELL_LABEL}=cell-bad-script"
        )

        assert diagnosis.is_complete, f"the diagnosis of a reachable failing pod must be complete {diagnosis=}"
        assert marker in (diagnosis.directory / "pod-bad-script.log").read_text()
        assert "Exit Code:    3" in (diagnosis.directory / "pod-bad-script.describe.txt").read_text()
        assert (diagnosis.directory / "events.txt").read_text().strip() != ""

    async def test_the_command_reports_a_run_directory_that_holds_no_verdict(
        self,
        cluster_core_v1: kubernetes_client.CoreV1Api,
        cluster_namespace: str,
        kubectl_kubeconfig: None,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The CLI fails and names the missing verdict when the run directory it is pointed at holds none."""
        await cluster_core_v1.create_namespaced_pod(
            namespace=cluster_namespace,
            body=pod_body(
                name="pod-verdictless",
                cell="cell-verdictless",
                image=BUSYBOX_IMAGE,
                command=["sh", "-c", "exit 3"],
            ),
        )
        await _wait_for_phase(cluster_core_v1, namespace=cluster_namespace, name="pod-verdictless", phase="Failed")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with caplog.at_level(logging.INFO):
            result = CliRunner().invoke(
                app,
                [
                    "collect-diagnosis",
                    "--namespace",
                    cluster_namespace,
                    "--output-dir",
                    str(tmp_path),
                    "--run-dir",
                    str(run_dir),
                ],
            )

        assert result.exit_code == 1, f"an incomplete diagnosis must fail the command {result.output=}"
        reported = [message for message in _messages(caplog) if "the diagnosis is incomplete" in message]
        assert reported == [
            f"FAIL  the diagnosis is incomplete, these could not be collected: a verdict under {run_dir}"
        ], f"the verdict must be the only thing reported missing {reported=}"
        assert Path(result.output.strip().splitlines()[-1]).is_dir()

    async def test_it_says_which_pods_were_missing_when_the_selector_matches_none(
        self, cluster_namespace: str, kubectl_kubeconfig: None, tmp_path: Path
    ) -> None:
        """A selector that matches nothing degrades to an incomplete diagnosis naming the namespace."""
        diagnosis = collect_diagnosis(
            namespace=cluster_namespace, output_dir=tmp_path, selector=f"{CELL_LABEL}=cell-nothing-matches"
        )

        assert diagnosis.missing == (f"pods of the run in namespace {cluster_namespace}",)
        assert (diagnosis.directory / "events.txt").exists()

    async def test_it_says_which_pods_were_missing_when_the_namespace_does_not_exist(
        self, kubectl_kubeconfig: None, tmp_path: Path
    ) -> None:
        """A namespace that does not exist degrades to an incomplete diagnosis instead of raising."""
        namespace = f"miles-absent-{uuid.uuid4().hex[:8]}"

        diagnosis = collect_diagnosis(namespace=namespace, output_dir=tmp_path, selector=f"{CELL_LABEL}=cell-anything")

        assert diagnosis.missing == (f"pods of the run in namespace {namespace}",)


async def _wait_for_phase(
    core_v1_api: kubernetes_client.CoreV1Api,
    *,
    namespace: str,
    name: str,
    phase: str,
    timeout: float = _STARTUP_TIMEOUT,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        pod = await core_v1_api.read_namespaced_pod(namespace=namespace, name=name)
        if pod.status.phase == phase:
            return
        assert time.monotonic() < deadline, f"timed out after {timeout}s waiting for {name} to reach {phase}"
        await asyncio.sleep(1.0)


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in list(caplog.records)]


def _followed_line_numbers(caplog: pytest.LogCaptureFixture) -> list[int]:
    prefix = "[pod-logs/main] line-"
    return [int(message.removeprefix(prefix)) for message in _messages(caplog) if message.startswith(prefix)]


def _reported_states(caplog: pytest.LogCaptureFixture, *, pod: str) -> list[str]:
    states = []
    for message in _messages(caplog):
        for opening in (f"pod {pod} appeared: ", f"pod {pod} is now "):
            if message.startswith(opening):
                states.append(message.removeprefix(opening))
    return states


def _warnings_about(caplog: pytest.LogCaptureFixture, *, pod: str) -> list[str]:
    return [
        record.getMessage()
        for record in list(caplog.records)
        if record.levelno == logging.WARNING and record.getMessage().startswith(f"{pod}: ")
    ]
