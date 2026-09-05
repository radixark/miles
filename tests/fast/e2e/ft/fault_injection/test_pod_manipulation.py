import random
import subprocess

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import pod_manipulation

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
        if f"{pod_manipulation.INSTANCE_LABEL}={release}" in selector:
            return pods
    return ""


def _fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    if argv[1] != "get":
        return _completed("")
    return _completed(_pods_matching(argv[argv.index("--selector") + 1]))


class TestComputeCellPodSelector:
    def test_the_selector_names_the_release_the_pool_and_the_cell_index(self) -> None:
        """Deleting by cell id needs the labels the run's pods actually carry, not a name guess."""
        selector = pod_manipulation._compute_cell_pod_selector(release=_RELEASE, cell_id=_CELL_ID)

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

        monkeypatch.setattr(pod_manipulation, "run_process", fake_run_process)

        deleted = pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert deleted in ("actor-3-0", "actor-3-1")
        assert [call for call in calls if call[1] == "delete"] == [
            ["kubectl", "delete", "pod", "--namespace", _NAMESPACE, "--wait=false", deleted]
        ]

    def test_a_second_release_with_the_same_topology_is_never_touched(self, monkeypatch) -> None:
        """Regression: a pool-and-index-only selector deletes another run's pod and injects nothing here."""
        monkeypatch.setattr(pod_manipulation, "run_process", _fake_run_process)

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

        monkeypatch.setattr(pod_manipulation, "run_process", fake_run_process)

        pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert timeouts and all(isinstance(timeout, float) for timeout in timeouts), timeouts

    def test_a_cell_without_pods_fails_loudly(self, monkeypatch) -> None:
        """Silently injecting nothing would let the soak pass while proving nothing."""
        monkeypatch.setattr(pod_manipulation, "run_process", lambda argv, **kwargs: _completed(""))

        with pytest.raises(AssertionError, match="nothing to delete"):
            pod_manipulation.delete_one_pod_of_cell(
                namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
            )
