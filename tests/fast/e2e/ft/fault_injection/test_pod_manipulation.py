import random
import subprocess

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import pod_manipulation

from miles.utils.test_utils import kubectl_reads

_CELL_ID = "actor-3"
_RELEASE = "miles-run-abc123"
_OTHER_RELEASE = "miles-run-def456"
_NAMESPACE = "miles-e2e"

_PODS_OF_RELEASE: dict[str, str] = {
    _RELEASE: "actor-3-0 actor-3-1",
    _OTHER_RELEASE: "other-actor-3-0 other-actor-3-1",
}


def _completed(stdout: str = "", *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _pods_matching(selector: str) -> str:
    for release, pods in _PODS_OF_RELEASE.items():
        if f"{kubectl_reads.INSTANCE_LABEL}={release}" in selector:
            return pods
    return ""


def _fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    if argv[1] != "get":
        return _completed("")
    return _completed(_pods_matching(argv[argv.index("--selector") + 1]))


def _patch_run_process(monkeypatch, fake) -> None:
    monkeypatch.setattr(pod_manipulation, "run_process", fake)
    monkeypatch.setattr(kubectl_reads, "run_process", fake)


class TestComputeCellPodSelector:
    def test_the_selector_names_the_release_the_pool_and_the_cell_index(self) -> None:
        """Deleting by cell id needs the labels the run's pods actually carry, not a name guess."""
        selector = kubectl_reads.compute_release_selector(
            release=_RELEASE, extra_labels=pod_manipulation._compute_cell_labels(_CELL_ID)
        )

        assert f"app.kubernetes.io/instance={_RELEASE}" in selector
        assert "miles.radixark.io/pool=actor" in selector
        assert "group-index=3" in selector


class TestDeleteOnePodOfCell:
    def test_only_one_of_the_cells_pods_is_deleted(self, monkeypatch) -> None:
        """An outsider kills a pod, not the whole cell: deleting them all would mimic the heal path."""
        calls: list[list[str]] = []

        def fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return _fake_run_process(argv, **kwargs)

        _patch_run_process(monkeypatch, fake_run_process)

        deleted = pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert deleted in ("actor-3-0", "actor-3-1")
        assert [call for call in calls if call[1] == "delete"] == [
            ["kubectl", "delete", "pod", "--namespace", _NAMESPACE, "--wait=false", deleted]
        ]

    def test_a_second_release_with_the_same_topology_is_never_touched(self, monkeypatch) -> None:
        """Regression: a pool-and-index-only selector deletes another run's pod and injects nothing here."""
        _patch_run_process(monkeypatch, _fake_run_process)

        deleted = pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert deleted in _PODS_OF_RELEASE[_RELEASE].split()
        assert deleted not in _PODS_OF_RELEASE[_OTHER_RELEASE].split()

    def test_a_kubectl_call_that_never_returns_cannot_outlive_the_soak(self, monkeypatch) -> None:
        """An unbounded kubectl would let the injector thread survive the join and race the witness."""
        timeouts: list[object] = []

        def fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            timeouts.append(kwargs.get("timeout"))
            return _completed(_PODS_OF_RELEASE[_RELEASE] if argv[1] == "get" else "")

        _patch_run_process(monkeypatch, fake_run_process)

        pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert timeouts and all(isinstance(timeout, float) for timeout in timeouts), timeouts

    def test_a_cell_without_pods_fails_loudly(self, monkeypatch) -> None:
        """Silently injecting nothing would let the soak pass while proving nothing."""
        _patch_run_process(monkeypatch, lambda argv, **kwargs: _completed(""))

        with pytest.raises(AssertionError, match="nothing to delete"):
            pod_manipulation.delete_one_pod_of_cell(
                namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
            )


_POD_NAME = "rollout-engine-0-0"
_PATTERN = "sglang::"


def _fake_kubectl(monkeypatch: pytest.MonkeyPatch, respond) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
        calls.append(argv)
        return respond(argv)

    _patch_run_process(monkeypatch, fake_run_process)
    return calls


class TestSigkillProcessPatternsInPod:
    def test_it_signals_a_process_inside_the_pod_rather_than_deleting_it(self, monkeypatch: pytest.MonkeyPatch):
        """Deleting the pod is a different fault: this one leaves the pod and crashes what runs in it."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_manipulation.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][:2] == ["kubectl", "exec"]
        assert _PATTERN in calls[0]
        assert "delete" not in calls[0]

    def test_it_names_the_container_it_reaches_into(self, monkeypatch: pytest.MonkeyPatch):
        """A pod can hold sidecars, and killing a process in the wrong one is not the fault under test."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_manipulation.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][calls[0].index("--container") + 1] == "engine"

    def test_matching_no_process_is_a_failure_rather_than_a_crash_nobody_caused(self, monkeypatch: pytest.MonkeyPatch):
        """pkill exits 1 when it matched nothing, which is indistinguishable from a kill that worked."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(returncode=1))

        with pytest.raises(AssertionError, match="No process matching"):
            pod_manipulation.sigkill_process_patterns_in_pod(
                namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
            )
