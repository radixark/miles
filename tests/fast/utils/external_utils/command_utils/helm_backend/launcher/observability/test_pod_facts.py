import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import pod_facts


class TestStatusChanges:
    def test_announces_a_pod_the_scheduler_has_just_created(self):
        """Pods appear one by one over minutes, and a silent launcher reads as a hung one."""
        assert pod_facts.status_changes([], [make_pod(phase="Pending", ready=False)]) == ["pod p appeared: Pending"]

    def test_announces_the_step_from_scheduled_to_serving(self):
        """Loading a model takes minutes, so the move out of that state is the progress users wait for."""
        assert pod_facts.status_changes([make_pod(ready=False)], [make_pod(ready=True)]) == ["pod p is now Running"]

    def test_says_nothing_when_a_poll_finds_the_same_pods(self):
        """A line every few seconds for an unchanged run would bury the pod logs it sits between."""
        assert pod_facts.status_changes([make_pod()], [make_pod()]) == []

    def test_reports_a_pod_that_disappeared(self):
        """A deleted or evicted pod stops producing logs, and its silence otherwise looks like a hang."""
        assert pod_facts.status_changes([make_pod()], []) == ["pod p is gone"]

    def test_separates_a_gated_pod_from_a_merely_pending_one(self):
        """A colocate pod waits for its trainer's node on purpose; calling that pending looks like a stall."""
        changes = pod_facts.status_changes([], [make_pod(phase="Pending", ready=False, scheduling_gated=True)])

        assert changes == ["pod p appeared: Pending (scheduling gated)"]

    def test_reports_a_restart_a_phase_alone_would_hide(self):
        """A crash looping pod stays Running between crashes, so only the count says anything is wrong."""
        assert pod_facts.status_changes([make_pod()], [make_pod(restarts=2)]) == [
            "pod p is now Running, restarted 2 times"
        ]


class TestContainerRuns:
    def test_finds_the_container_a_running_pod_is_serving_from(self):
        """A pod with no container id has not started yet, and following it would attach to nothing."""
        pods = [make_pod(containers=(make_container(container_id="docker://a", running=True),))]

        runs = pod_facts.container_runs(pods)

        assert [(run.key.pod, run.key.container, run.running) for run in runs.values()] == [("p", "app", True)]

    def test_treats_a_replaced_container_as_a_new_one_to_follow(self):
        """A container that crashed and came back writes a fresh log the old stream would never show."""
        before = pod_facts.container_runs([make_pod(containers=(make_container(container_id="docker://a"),))])
        after = pod_facts.container_runs([make_pod(containers=(make_container(container_id="docker://b"),))])

        assert set(before) == set(after)
        assert [run.container_id for run in before.values()] != [run.container_id for run in after.values()]

    def test_does_not_confuse_a_rebuilt_pod_with_the_one_it_replaced(self):
        """A pod recreated under the same name is a different pod, and its logs start over."""
        before = pod_facts.container_runs(
            [make_pod(uid="u1", containers=(make_container(container_id="docker://a"),))]
        )
        after = pod_facts.container_runs([make_pod(uid="u2", containers=(make_container(container_id="docker://a"),))])

        assert [run.container_id for run in before.values()] != [run.container_id for run in after.values()]

    def test_keeps_the_life_a_container_already_ended(self):
        """A container that crashed on startup is the case this whole layer exists to make readable."""
        container = make_container(container_id="docker://b", previous_container_id="docker://a")

        runs = pod_facts.container_runs([make_pod(containers=(container,))])

        assert sorted(run.key.previous for run in runs.values()) == [False, True]
        assert all(not run.running for run in runs.values() if run.key.previous)

    def test_ignores_a_container_that_never_ran(self):
        """A pod waiting on an image pull has no logs at all, and asking for them only prints an error."""
        assert pod_facts.container_runs([make_pod(containers=(make_container(),))]) == {}


class FakeKubectl:
    def __init__(self) -> None:
        self.answers: dict[str, dict] = {}
        self.calls: list[dict] = []

    def get_json(self, kind: str, *, return_type, **kwargs):
        self.calls.append({"kind": kind, **kwargs})
        answer = self.answers.get(kind)
        return return_type.model_validate(answer) if answer is not None else None


@pytest.fixture
def kubectl(monkeypatch) -> FakeKubectl:
    fake = FakeKubectl()
    monkeypatch.setattr(pod_facts, "Kubectl", fake)
    return fake


def _raw_pod(name: str, *, phase="Running", ready=True, containers=()) -> dict:
    return {
        "metadata": {"name": name, "uid": f"uid-{name}"},
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": list(containers),
        },
    }


def _raw_event(pod_name: str, reason="FailedScheduling", message="no node", count=1, event_type="Warning") -> dict:
    return {
        "involvedObject": {"name": pod_name, "kind": "Pod"},
        "reason": reason,
        "message": message,
        "count": count,
        "type": event_type,
    }


class TestSelectedPods:
    def test_reads_the_fields_the_watchers_report_on(self, kubectl):
        """Every status line and every attach decision is built out of exactly these fields."""
        container = {
            "name": "app",
            "containerID": "docker://b",
            "restartCount": 2,
            "state": {"running": {}},
            "lastState": {"terminated": {"containerID": "docker://a"}},
        }
        kubectl.answers["pods"] = {"items": [_raw_pod("trainer-0", containers=[container])]}

        pod = pod_facts.selected_pods("rl", "app=x")[0]

        assert (pod.metadata.name, pod.metadata.uid, pod.status.phase) == ("trainer-0", "uid-trainer-0", "Running")
        assert (pod_facts.is_pod_ready(pod), pod_facts.restarts_of_pod(pod)) == (True, 2)
        assert pod.status.container_statuses[0].container_id == "docker://b"
        assert pod.status.container_statuses[0].last_state.terminated.container_id == "docker://a"
        assert pod.status.container_statuses[0].state.running is not None

    def test_survives_a_pod_the_api_has_barely_filled_in(self, kubectl):
        """A pod one poll old carries almost no status, and crashing here would end the run's reporting."""
        kubectl.answers["pods"] = {"items": [{"metadata": {"name": "trainer-0", "uid": "u"}, "status": {}}]}

        pod = pod_facts.selected_pods("rl", "app=x")[0]

        assert pod_facts.phase_of_pod(pod) == "Unknown"
        assert (pod_facts.is_pod_ready(pod), pod_facts.restarts_of_pod(pod)) == (False, 0)
        assert pod.status.container_statuses == []

    def test_reports_no_pods_before_the_scheduler_has_acted(self, kubectl):
        """helm returns before any pod exists, and kubectl answers that with nothing at all."""
        assert pod_facts.selected_pods("rl", "app=x") == []


class TestPodEvents:
    def test_reads_the_events_of_the_pods_the_release_owns(self, kubectl):
        """The launcher watches one release, and another run's failures would be a false alarm."""
        kubectl.answers["events"] = {"items": [_raw_event("mine-0"), _raw_event("someone-elses-0")]}

        events = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0")])

        assert [event.involved_object.name for event in events] == ["mine-0"]

    def test_asks_the_api_only_for_warnings_about_pods(self, kubectl):
        """A namespace also records node and job events, none of which explain a pod that will not start."""
        kubectl.answers["events"] = {"items": [_raw_event("mine-0")]}

        pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0")])

        assert kubectl.calls[0]["field_selector"] == "involvedObject.kind=Pod,type=Warning"

    def test_carries_the_reason_and_the_message_the_summary_prints(self, kubectl):
        """A reason without its message says a pod failed but not what a user has to fix."""
        kubectl.answers["events"] = {"items": [_raw_event("mine-0", reason="Failed", message="denied", count=3)]}

        event = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0")])[0]

        assert (event.reason, event.message, event.count, event.type) == ("Failed", "denied", 3, "Warning")

    def test_survives_an_event_the_api_left_half_empty(self, kubectl):
        """A freshly created event has no count yet, and a crash here would hide the pod it describes."""
        kubectl.answers["events"] = {"items": [_raw_event("mine-0", count=None)]}

        event = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0")])[0]

        assert event.count == 1

    def test_does_not_ask_about_a_release_with_no_pods_yet(self, kubectl):
        """Before the scheduler acts there is nothing to explain, and the call would only cost a round trip."""
        assert pod_facts.pod_events(namespace="rl", pods=[]) == []
        assert kubectl.calls == []


class TestPodPhase:
    def test_reads_the_phase_of_the_orchestrator_pod(self, kubectl):
        """The launcher decides a run lost its orchestrator from this, so it must read the real phase."""
        kubectl.answers["pod"] = _raw_pod("r-orchestrator-0", phase="Failed")

        assert pod_facts.pod_phase("rl", "r-orchestrator") == "Failed"

    def test_reports_a_pod_that_is_not_there_as_absent(self, kubectl):
        """A pod between deletion and recreation must read as gone, not as an error the launcher dies on."""
        assert pod_facts.pod_phase("rl", "r-orchestrator") is None
