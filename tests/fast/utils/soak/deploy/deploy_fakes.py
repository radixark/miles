import json
import subprocess
import threading
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    NAMESPACE,
    ORCHESTRATOR,
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)
from tests.utils.deploy.hot_restart.cluster_observer import LEADER_WORKER_SET_KIND, STATEFUL_SET_KIND, ClusterSnapshot
from tests.utils.deploy.hot_restart.evidence import RunProgress
from tests.utils.soak.deploy import observers as observers_module
from tests.utils.soak.deploy.types import DeploymentTarget

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION

_WORKLOADS: tuple[str, ...] = (ORCHESTRATOR, ROLLOUT_EXECUTOR, TRAINER)
_STATE_FILE = Path("/state/generation-a.json")
_PROGRESS = RunProgress(last_saved_iteration=3, last_finished_rollout_id=5)
_UNINSTALL_JOB_STDOUT = json.dumps({"metadata": {"uid": "uid-uninstall"}})

# ============================ deployment targets ============================


def _deployment_target(
    *,
    stamps: dict[str, str | None] | None = None,
    saved_iteration: int | None = 3,
    finished_rollout_id: int | None = 5,
    **overrides: object,
) -> DeploymentTarget:
    workload_stamps = dict.fromkeys(_WORKLOADS) if stamps is None else stamps
    return DeploymentTarget(
        **{
            "identity": RELEASE,
            "incarnation": json.dumps(workload_stamps, sort_keys=True),
            "alive": True,
            "ready": True,
            "namespace": NAMESPACE,
            "release": RELEASE,
            "workload_stamps": workload_stamps,
            "workload_uids": {name: f"uid-{name}" for name in workload_stamps},
            "saved_iteration": saved_iteration,
            "finished_rollout_id": finished_rollout_id,
            "state_file": _STATE_FILE,
            "uninstall_job_uid": "uid-uninstall",
            **overrides,
        }
    )


# ============================== kubectl reads ===============================


def _workload_item(name: str, *, stamp: str | None, uid: str | None = None, deleting: bool = False) -> dict:
    metadata: dict[str, object] = {"name": name, "generation": 1, "uid": f"uid-{name}" if uid is None else uid}
    if deleting:
        metadata["deletionTimestamp"] = "2026-09-26T00:00:00Z"
    annotations = {} if stamp is None else {RESTART_AT_ANNOTATION: stamp}
    return {"metadata": metadata, "spec": {"template": {"metadata": {"annotations": annotations}}}}


class _FakeWorkloadKubectl:
    def __init__(self, *, stateful_sets: list[dict], leader_worker_sets: list[dict] | None = None) -> None:
        self._items_of_kind = {STATEFUL_SET_KIND: stateful_sets, LEADER_WORKER_SET_KIND: leader_worker_sets or []}
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: list[str], *, capture_output: bool, check: bool, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        assert argv[:2] == ["kubectl", "get"] and capture_output and check and timeout is not None
        payload = {"items": self._items_of_kind[argv[2]]}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")


class _FakeJobKubectl:
    def __init__(self, *, current_uid: str | None, failing_verb: str | None = None) -> None:
        self._current_uid = current_uid
        self._failing_verb = failing_verb
        self.calls: list[dict] = []

    def __call__(
        self,
        argv: list[str],
        *,
        capture_output: bool,
        check: bool,
        input: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, "input": input, "check": check, "timeout": timeout})
        assert argv[0] == "kubectl" and capture_output and timeout is not None
        if argv[1] == self._failing_verb:
            raise subprocess.CalledProcessError(1, argv, stderr="scripted failure")
        if argv[1] == "get":
            stdout = "" if self._current_uid is None else json.dumps({"metadata": {"uid": self._current_uid}})
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def verbs(self) -> list[str]:
        return [call["argv"][1] for call in self.calls]


# ============================ deployment reads =============================


def _release_snapshot(
    *, stamps: dict[str, str | None] | None = None, reads_missing: tuple[str, ...] = (), with_pods: bool = True
) -> ClusterSnapshot:
    stamp_of = {ORCHESTRATOR: None, ROLLOUT_EXECUTOR: None, TRAINER: None} if stamps is None else stamps
    snapshot = cluster_snapshot(
        pods=[pod_fact(f"{name}-0", uid=f"uid-{name}-0") for name in stamp_of] if with_pods else [],
        workloads=[workload_fact(name, restart_at=stamp) for name, stamp in stamp_of.items()],
        reads_missing=reads_missing,
    )
    return snapshot.model_copy(update={"orchestrator_state_file": _STATE_FILE})


class _FakeDeploymentReads:
    def __init__(
        self,
        *,
        snapshot: ClusterSnapshot,
        progress: RunProgress | BaseException = _PROGRESS,
        job_stdout: str | BaseException = _UNINSTALL_JOB_STDOUT,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._progress = progress
        self._job_stdout = job_stdout
        self._barrier = barrier
        self.cluster_kwargs: list[dict] = []
        self.progress_kwargs: list[dict] = []
        self.job_argv: list[list[str]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(observers_module, "read_cluster_snapshot", self._read_cluster)
        monkeypatch.setattr(observers_module, "read_run_progress", self._read_progress)
        monkeypatch.setattr(observers_module, "run_process", self._run_process)

    def _meet(self) -> None:
        if self._barrier is not None:
            self._barrier.wait()

    def _read_cluster(self, **kwargs: object) -> ClusterSnapshot:
        self.cluster_kwargs.append(kwargs)
        self._meet()
        return self.snapshot

    def _read_progress(self, **kwargs: object) -> RunProgress:
        self.progress_kwargs.append(kwargs)
        self._meet()
        if isinstance(self._progress, BaseException):
            raise self._progress
        return self._progress

    def _run_process(
        self, argv: list[str], *, capture_output: bool, check: bool, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.job_argv.append(argv)
        assert capture_output and check and timeout is not None
        self._meet()
        if isinstance(self._job_stdout, BaseException):
            raise self._job_stdout
        return subprocess.CompletedProcess(argv, 0, stdout=self._job_stdout, stderr="")
