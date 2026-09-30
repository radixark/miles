import json
import subprocess
import threading
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ORCHESTRATOR,
    ROLLOUT_EXECUTOR,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)
from tests.utils.deploy.hot_restart.cluster_observer import ClusterSnapshot
from tests.utils.deploy.hot_restart.evidence import RunProgress
from tests.utils.soak.deploy import observers as observers_module

_STATE_FILE = Path("/state/generation-a.json")
_PROGRESS = RunProgress(last_saved_iteration=3, last_finished_rollout_id=5)
_UNINSTALL_JOB_STDOUT = json.dumps({"metadata": {"uid": "uid-uninstall"}})


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
