from __future__ import annotations

import subprocess
import threading

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    Status,
    expand_resource_verbs,
    parallel_execute_checkers,
)

NAMESPACE = "rl"

_RULES = {
    "pods": ("create", "delete", "get", "list", "patch", "update", "watch"),
    "services": ("create", "delete", "get"),
    "pods/exec": ("create",),
}


class _RecordingKubectl:
    def __init__(self, *, meeting: int) -> None:
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self._meeting = threading.Barrier(meeting, timeout=30.0)

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        with self._lock:
            self.calls.append(list(args))
        self._meeting.wait()
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="yes\n", stderr="")


def _verb_checkers(rules: dict[str, tuple[str, ...]]) -> list[ResourceVerbAvailabilityChecker]:
    return [ResourceVerbAvailabilityChecker(NAMESPACE, verb) for verb in expand_resource_verbs(rules)]


def _record(monkeypatch: pytest.MonkeyPatch, *, meeting: int) -> _RecordingKubectl:
    kubectl = _RecordingKubectl(meeting=meeting)
    monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(kubectl))
    return kubectl


class TestAPhaseIsAskedConcurrently:
    def test_a_phase_is_asked_in_one_parallel_pass(self, monkeypatch):
        """A plan runs to several hundred rules, and asking them in turn makes the wait a round trip
        each; the barrier only opens once that many callers are in flight together."""
        checkers = _verb_checkers(_RULES)
        _record(monkeypatch, meeting=len(checkers))

        outcomes = parallel_execute_checkers(checkers)

        assert [outcome.result.status for outcome in outcomes] == [Status.PASS] * len(checkers)

    def test_a_phase_of_n_checkers_makes_exactly_one_round_of_calls(self, monkeypatch):
        """Nothing is remembered between checkers, so a phase must be one round: N checkers, N calls, all in flight."""
        checkers = _verb_checkers(_RULES)
        kubectl = _record(monkeypatch, meeting=len(checkers))

        outcomes = parallel_execute_checkers(checkers)

        assert len(kubectl.calls) == len(checkers)
        assert len(outcomes) == len(checkers)

    def test_every_rule_is_still_asked_of_the_cluster(self, monkeypatch):
        """Answering in parallel must not drop or merge a rule: each verb needs its own answer."""
        checkers = _verb_checkers(_RULES)
        kubectl = _record(monkeypatch, meeting=len(checkers))

        parallel_execute_checkers(checkers)

        asked = {(call[2], call[3]) for call in kubectl.calls}
        assert asked == {(verb, resource.partition("/")[0]) for resource, verbs in _RULES.items() for verb in verbs}

    def test_a_subresource_is_asked_for_by_name(self, monkeypatch):
        """kubectl reads pods/exec as the pods resource unless the subresource is named separately."""
        checker = ResourceVerbAvailabilityChecker(NAMESPACE, ResourceVerb(verb="create", resource="pods/exec"))
        kubectl = _record(monkeypatch, meeting=1)

        parallel_execute_checkers([checker])

        assert "--subresource=exec" in kubectl.calls[0]

    def test_an_empty_phase_asks_nothing(self, monkeypatch):
        """A plan that grants nothing leaves a phase empty, and an empty pool must not be opened for it."""
        kubectl = _record(monkeypatch, meeting=1)

        assert parallel_execute_checkers([]) == []
        assert kubectl.calls == []
