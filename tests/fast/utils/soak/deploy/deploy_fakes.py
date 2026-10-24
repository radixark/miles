import asyncio
import json
import shutil
import subprocess
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime
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
from tests.fast.utils.soak.soak_fakes import _at
from tests.utils.deploy.hot_restart.cluster_observer import LEADER_WORKER_SET_KIND, STATEFUL_SET_KIND, ClusterSnapshot
from tests.utils.deploy.hot_restart.evidence import TRAIN_STEP_METRIC_KEY, RunProgress
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    LaunchOutcome,
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakLaunchFinishedEvent,
    SoakObservationEvent,
)
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.deploy import observers as observers_module
from tests.utils.soak.deploy.actions import hot_restart as hot_restart_module
from tests.utils.soak.deploy.actions.hot_restart import HotRestartForm
from tests.utils.soak.deploy.session import LauncherChain
from tests.utils.soak.deploy.types import DeploymentTarget, HotRestartDetails, HotRestartTakeOverEvidence
from tests.utils.soak.recipes.gsm8k import Gsm8kLaunchSpec

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, LaunchGuard
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION
from miles.utils.workers.types import ClusterBackend

_WORKLOADS: tuple[str, ...] = (ORCHESTRATOR, ROLLOUT_EXECUTOR, TRAINER)
_STATE_FILE = Path("/state/generation-a.json")
_RUN_ID = "demo"
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


def _restamped(
    target: DeploymentTarget,
    stamp: str,
    *,
    names: tuple[str, ...] = (ORCHESTRATOR, ROLLOUT_EXECUTOR),
    **overrides: object,
) -> DeploymentTarget:
    stamps = target.workload_stamps | dict.fromkeys(names, stamp)
    return target.model_copy(
        update={"workload_stamps": stamps, "incarnation": json.dumps(stamps, sort_keys=True), **overrides}
    )


def _hot_restart_request(target: DeploymentTarget, *, request_id: str = "req-1") -> SoakActionRequest:
    return SoakActionRequest(
        request_id=request_id, target=target, form_name="hot_restart", details=HotRestartDetails()
    )


def _requested_take_over(request: SoakActionRequest, *, at: datetime) -> SoakActionRequestedEvent:
    return SoakActionRequestedEvent(timestamp=at, request=request)


def _landed_take_over(request: SoakActionRequest, *, after: DeploymentTarget, at: datetime) -> SoakActionAppliedEvent:
    return SoakActionAppliedEvent(
        timestamp=at, request_id=request.request_id, evidence=HotRestartTakeOverEvidence(after=after)
    )


def _deployment_observation(
    targets: list[DeploymentTarget] | None, *, at: datetime, errors: dict[str, str] | None = None
) -> SoakObservationEvent:
    return SoakObservationEvent(timestamp=at, targets=targets, errors=errors or {})


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


# ================================ launchers =================================


def _launch_spec(**config_overrides: object) -> Gsm8kLaunchSpec:
    config = ExecuteTrainConfig(
        **{"cluster_backend": ClusterBackend.KUBERNETES, "namespace": NAMESPACE, "run_id": _RUN_ID, **config_overrides}
    )
    return Gsm8kLaunchSpec(config=config, train_args="--save /ckpt ", fully_async=True)


class _FakeLauncher:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.finish = asyncio.Event()
        self.calls: list[tuple[Gsm8kLaunchSpec, LaunchGuard | None]] = []
        self.cancelled = False

    async def __call__(self, spec: Gsm8kLaunchSpec, *, guard: LaunchGuard | None = None) -> None:
        self.calls.append((spec, guard))
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error


class _HotRestartHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, launcher: _FakeLauncher) -> None:
        self.launcher = launcher
        self.checked: list[DeploymentTarget] = []
        self.stale: BaseException | None = None
        self.event_log = EventLog(tmp_path / "events.jsonl")
        self.chain = LauncherChain()
        self.reported: list[SoakActionEvidence] = []
        monkeypatch.setattr(hot_restart_module, "launch", launcher)
        monkeypatch.setattr(hot_restart_module, "assert_workloads_unchanged", self._check)
        monkeypatch.setattr(hot_restart_module, "TAKE_OVER_POLL_INTERVAL_SECONDS", 0)

    def form(self, **spec_overrides: object) -> HotRestartForm:
        return HotRestartForm(launch_spec=_launch_spec(**spec_overrides), event_log=self.event_log, chain=self.chain)

    def start(self, request: SoakActionRequest, **spec_overrides: object) -> asyncio.Task[None]:
        return asyncio.create_task(self.form(**spec_overrides).execute(request, report_applied=self.reported.append))

    def observe(self, *targets: DeploymentTarget, seconds: float = 1) -> None:
        self.event_log.append(_deployment_observation(list(targets), at=_at(seconds)))

    def launch_outcomes(self) -> list[tuple[str | None, LaunchOutcome]]:
        return [
            (event.request_id, event.outcome)
            for event in self.event_log.events
            if isinstance(event, SoakLaunchFinishedEvent)
        ]

    def _check(self, target: DeploymentTarget) -> None:
        self.checked.append(target)
        if self.stale is not None:
            raise self.stale


# ============================ rolled-aside logs =============================


def _write_finished_steps(events_dir: Path, rollout_ids: Iterable[int]) -> None:
    for rollout_id in rollout_ids:
        logger = EventLogger(
            log_dir=events_dir, file_name=f"step-{rollout_id}.jsonl", source=SimpleProcessIdentity(component="main")
        )
        logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {TRAIN_STEP_METRIC_KEY: 1.0}}, print_log=False)


def _roll_log_aside(dump_dir: Path, *, rolled_aside_at: str, kept: Iterable[int]) -> None:
    events_dir = dump_dir / EVENTS_DIRNAME
    replaced = dump_dir / f".trash_{rolled_aside_at}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(events_dir), str(replaced))
    events_dir.mkdir(parents=True)
    for rollout_id in kept:
        shutil.copy(replaced / f"step-{rollout_id}.jsonl", events_dir / f"step-{rollout_id}.jsonl")
