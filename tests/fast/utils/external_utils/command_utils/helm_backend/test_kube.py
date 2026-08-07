from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend import kube, observe


@dataclass
class FakeEventsApi:
    events: list[SimpleNamespace]
    field_selectors: list[str] = field(default_factory=list)

    def list_namespaced_event(self, *, namespace: str, field_selector: str) -> SimpleNamespace:
        self.field_selectors.append(field_selector)
        return SimpleNamespace(items=self.events)


def _raw_event(
    pod_name: str,
    reason: str = "FailedScheduling",
    message: str = "no node",
    count: int | None = 1,
    event_type: str = "Warning",
) -> SimpleNamespace:
    return SimpleNamespace(
        involved_object=SimpleNamespace(name=pod_name), reason=reason, message=message, count=count, type=event_type
    )


def _pod(name: str) -> observe.PodStatus:
    return observe.PodStatus(name=name, phase="Pending", ready=False, restarts=0)


@pytest.fixture
def api(monkeypatch) -> FakeEventsApi:
    fake = FakeEventsApi(events=[])
    monkeypatch.setattr(kube, "_core_api", lambda: fake)
    return fake


class TestPodEvents:
    def test_reads_the_events_of_the_pods_the_release_owns(self, api):
        """The launcher polls one release, and another run's failures would be a false alarm."""
        api.events = [_raw_event("mine-0"), _raw_event("someone-elses-0")]

        events = kube.pod_events(namespace="rl", pods=[_pod("mine-0")])

        assert [event.pod_name for event in events] == ["mine-0"]

    def test_asks_the_api_only_for_events_about_pods(self, api):
        """A namespace also records node and job events, none of which explain a pod that will not start."""
        api.events = [_raw_event("mine-0")]

        kube.pod_events(namespace="rl", pods=[_pod("mine-0")])

        assert api.field_selectors == ["involvedObject.kind=Pod"]

    def test_carries_the_reason_and_the_message_the_summary_prints(self, api):
        """A reason without its message says a pod failed but not what a user has to fix."""
        api.events = [_raw_event("mine-0", reason="Failed", message="pull access denied", count=3)]

        event = kube.pod_events(namespace="rl", pods=[_pod("mine-0")])[0]

        assert (event.reason, event.message, event.count, event.type) == ("Failed", "pull access denied", 3, "Warning")

    def test_survives_an_event_the_api_left_half_empty(self, api):
        """A freshly created event has no count yet, and a crash here would hide the pod it describes."""
        api.events = [_raw_event("mine-0", message="  spaced  ", count=None)]

        event = kube.pod_events(namespace="rl", pods=[_pod("mine-0")])[0]

        assert (event.count, event.message) == (1, "spaced")

    def test_does_not_call_the_api_for_a_release_with_no_pods_yet(self, api):
        """Before the scheduler acts there is nothing to explain, and the call would only cost a round trip."""
        assert kube.pod_events(namespace="rl", pods=[]) == []
        assert api.field_selectors == []
