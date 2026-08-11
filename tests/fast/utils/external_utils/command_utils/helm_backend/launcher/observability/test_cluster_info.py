import logging

import pytest

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_event,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import cluster_info, polling
from miles.utils.workers.k8s_types import Event, Pod


class TestPodSummary:
    def test_says_nothing_is_up_before_the_firstmake_pod(self):
        """helm returns before the scheduler has acted, and an empty summary would read as a hang."""
        assert cluster_info._pod_summary([]) == "No pods yet"

    def test_separates_a_serving_pod_from_one_still_starting(self):
        """A Running pod that is not ready is loading a model, which is the phase users wait through."""
        assert cluster_info._pod_summary([make_pod(), make_pod(ready=False)]) == "2 pods: 1 running, 1 starting"

    def test_counts_a_gated_pod_apart_from_a_pending_one(self):
        """A colocate pod waits for its trainer's node on purpose; calling that pending looks like a stall."""
        pods = [make_pod(phase="Pending"), make_pod(phase="Pending", scheduling_gated=True)]

        assert cluster_info._pod_summary(pods) == "2 pods: 1 pending, 1 gated"

    def test_surfaces_failures_and_restarts(self):
        """A crash looping pod is the one thing a user must not have to go looking for."""
        summary = cluster_info._pod_summary([make_pod(phase="Failed", ready=False), make_pod(restarts=3)])

        assert "1 failed" in summary and "1 restarted" in summary

    def test_omits_the_categories_that_are_empty(self):
        """A healthy run should read as one short line, not a table of zeroes."""
        assert cluster_info._pod_summary([make_pod()]) == "1 pods: 1 running"


class TestWarningLines:
    def test_names_the_reason_a_pod_cannot_be_scheduled(self):
        """An unschedulable pod stays Pending forever, and its phase alone never says why."""
        events = [
            make_event(reason="FailedScheduling", message="0/8 nodes are available: insufficient nvidia.com/gpu")
        ]

        lines = cluster_info._warning_lines(events)

        assert "FailedScheduling" in lines[0]
        assert "insufficient nvidia.com/gpu" in lines[0]

    def test_names_the_image_a_pod_could_not_pull(self):
        """An image pull failure is the other startup hang users hit, and it is invisible in the pod phase."""
        lines = cluster_info._warning_lines([make_event(reason="Failed", message="pull access denied", count=4)])

        assert "Failed x4" in lines[0]
        assert "pull access denied" in lines[0]

    def test_keeps_quiet_about_the_ordinary_events_of_a_healthy_run(self):
        """Pulled, Created and Started arrive for every pod and would bury the one line that matters."""
        assert cluster_info._warning_lines([make_event(reason="Pulled", event_type="Normal")]) == []

    def test_reports_the_busiest_events_first_and_says_how_many_it_left_out(self):
        """A large run repeats one failure per pod, and printing all of them hides the summary line."""
        events = [make_event(pod_name=f"p{index}", count=index) for index in range(1, 9)]

        lines = cluster_info._warning_lines(events)

        assert lines[0].startswith("p8: FailedScheduling x8")
        assert lines[-1] == "... and 3 more warning events"


class TestScaleHint:
    def test_stays_quiet_for_a_run_a_user_can_read(self):
        """Advice nobody needs trains people to skip the output that matters."""
        assert cluster_info._scale_hint([make_pod() for _ in range(10)]) is None

    def test_points_a_large_run_at_the_cluster_dashboards(self):
        """A summary of hundreds of pods is worse than the tool built for it."""
        hint = cluster_info._scale_hint([make_pod() for _ in range(200)])

        assert "200 pods" in hint


class TestEventKey:
    def test_tells_two_pods_of_one_name_apart(self):
        """A pod rebuilt under its ordinal name is a different pod, and its warning has not been reported."""
        old = make_event(pod_name="trainer-0", uid="uid-old")
        new = make_event(pod_name="trainer-0", uid="uid-new")

        assert cluster_info._event_key(old) != cluster_info._event_key(new)


class TestClusterInfoWatcher:
    def test_does_not_repeat_an_unchanged_pod_summary(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Polling an unchanged pod snapshot twice emits its summary only once."""
        pods = [make_pod(name="trainer-0")]

        def selected_pods(namespace: str, selector: str) -> list[Pod]:
            return pods

        def pod_events(*, namespace: str, pods: list[Pod]) -> list[Event]:
            return []

        monkeypatch.setattr(cluster_info, "selected_pods", selected_pods)
        monkeypatch.setattr(cluster_info, "pod_events", pod_events)
        watcher = cluster_info._ClusterInfoWatcher(namespace="rl", selector="app=x")

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            watcher.report_changes()
            watcher.report_changes()

        assert [record.message for record in caplog.records].count("1 pods: 1 running") == 1

    def test_reports_the_large_run_hint_only_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Polling the same large run twice emits the scale hint only once."""
        pods = [make_pod(name=f"trainer-{index}", uid=str(index)) for index in range(101)]

        def selected_pods(namespace: str, selector: str) -> list[Pod]:
            return pods

        def pod_events(*, namespace: str, pods: list[Pod]) -> list[Event]:
            return []

        monkeypatch.setattr(cluster_info, "selected_pods", selected_pods)
        monkeypatch.setattr(cluster_info, "pod_events", pod_events)
        watcher = cluster_info._ClusterInfoWatcher(namespace="rl", selector="app=x")

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            watcher.report_changes()
            watcher.report_changes()

        assert sum("cluster's own observability stack" in record.message for record in caplog.records) == 1


class TestWithClusterInfo:
    def test_reports_what_it_finds_and_stops_when_the_caller_leaves(self, monkeypatch, caplog):
        """A watcher that outlived its run would keep talking over whatever the user does next."""
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: [make_pod(name="trainer-0")])
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "trainer-0" in caplog.text)

        assert "pod trainer-0 appeared: Running" in caplog.text
        assert "1 pods: 1 running" in caplog.text

    def test_reports_a_warning_event_only_once(self, monkeypatch, caplog):
        """A pending pod carries the same event every poll, and repeating it would scroll the rest away."""
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: [make_pod(name="trainer-0")])
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [make_event(pod_name="trainer-0")])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "FailedScheduling" in caplog.text)

        assert caplog.text.count("FailedScheduling") == 1

    def test_keeps_watching_when_a_poll_fails(self, monkeypatch, caplog):
        """An api server hiccup must not silence the run's status for as long as the run lasts."""
        calls = []

        def flaky(namespace, selector):
            calls.append(namespace)
            if len(calls) == 1:
                raise RuntimeError("api server said no")
            return [make_pod(name="trainer-0")]

        monkeypatch.setattr(cluster_info, "selected_pods", flaky)
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "trainer-0" in caplog.text)

        assert "pod trainer-0 appeared" in caplog.text
