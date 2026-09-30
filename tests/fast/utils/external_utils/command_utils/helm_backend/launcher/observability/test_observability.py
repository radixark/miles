import logging
import threading
from collections.abc import Iterator

import pytest

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher import observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import log_follower
from miles.utils.workers.k8s_types import EventList, Pod, PodList
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL


class TestWithObservability:
    def test_reports_cluster_status_and_follows_logs_for_the_context_lifetime(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both views of a run stay active only while the launcher watches it."""
        calls: list[tuple[str, dict[str, object]]] = []
        pod_queries = threading.Barrier(2)
        pod: Pod = make_pod(
            name="trainer-0",
            containers=(make_container(container_id="docker://a", running=True),),
        )

        def get_json(kind: str, **kwargs: object) -> PodList | EventList:
            calls.append((kind, kwargs))
            if kind == "pods":
                pod_queries.wait()
                return PodList(items=[pod])
            assert kind == "events"
            return EventList(items=[])

        class FakeProcess:
            def __init__(self) -> None:
                self.killed = False
                self.returncode = 0
                self.terminated = threading.Event()
                self.stdout = self._lines()

            def poll(self) -> int | None:
                return 0 if self.killed else None

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                self.killed = True
                self.terminated.set()

            def kill(self) -> None:
                self.killed = True
                self.terminated.set()

            def _lines(self) -> Iterator[str]:
                yield "2026-08-10T00:00:00.1Z training\n"
                self.terminated.wait()

        processes: list[FakeProcess] = []

        def popen(_command: list[str], **_kwargs: object) -> FakeProcess:
            process = FakeProcess()
            processes.append(process)
            return process

        monkeypatch.setattr(Kubectl, "get_json", get_json)
        monkeypatch.setattr(log_follower.subprocess, "Popen", popen)

        with caplog.at_level(logging.INFO):
            with observability.with_observability(namespace="rl", selector="app=x"):
                wait_for(lambda: "1 pods: 1 running" in caplog.text and "[trainer-0/app] training" in caplog.text)

        pod_calls = [kwargs for kind, kwargs in calls if kind == "pods"]
        assert pod_calls == [
            {"return_type": PodList, "namespace": "rl", "selector": "app=x"},
            {"return_type": PodList, "namespace": "rl", "selector": "app=x"},
        ]
        assert (
            "events",
            {
                "return_type": EventList,
                "namespace": "rl",
                "field_selector": "involvedObject.kind=Pod,type=Warning",
            },
        ) in calls
        assert processes and all(process.killed for process in processes)


class TestFarewell:
    def test_tells_the_user_how_to_look_again_and_how_to_stop(self):
        """The release outlives the launcher, so both commands are the only way back to the run."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "kubectl logs" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message

    def test_the_farewell_offers_the_whole_release_as_well_as_the_orchestrator(self):
        """Sending a user to one statefulset makes every other pod of their run invisible."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "statefulset/r-miles-run-orchestrator" in message
        assert f"--selector {INSTANCE_LABEL}=miles-run-x" in message

    def test_the_farewell_always_says_where_this_summary_stops(self):
        """A user who mistakes a vanilla launcher for monitoring will not notice what it never reported."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert observability._OBSERVABILITY_BOUNDARY in message
        assert "observability stack" in message

    def test_says_the_release_removes_itself_when_it_will(self):
        """A user told to uninstall by hand would keep doing it, and wonder why nothing was there."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "uninstalls itself about two minutes" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message
