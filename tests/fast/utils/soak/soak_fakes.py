import asyncio
import builtins
import json
import random
import subprocess
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakCollectionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
    SoakRunContext,
    SoakRunContextEvent,
    StoredEvent,
)
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.types import (
    BaseSoakActionForm,
    SoakActionEvidence,
    SoakActionRequest,
    SoakForms,
    SoakObserver,
    SoakTarget,
)
from tests.utils.soak.core.views import SoakActionRecord
from tests.utils.soak.ft.types import CellTarget, PodDetails
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence, SoakPodTarget
from tests.utils.soak.k8s_utils.pod_processes import ProcessIdentity, ProcessTarget

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    Event,
    TrainGroupStepEndEvent,
    WeightUpdateResultEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.ft_utils.api_server.models import (
    CELL_TYPE_LABEL,
    Cell,
    CellCondition,
    CellList,
    CellMetadata,
    CellSpec,
    CellStatus,
    TriState,
)
from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import ObservedFaultHookTarget
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

_BASE = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)

_ExecuteBehavior = Callable[[SoakActionRequest, Callable[[SoakActionEvidence], None]], Awaitable[None]]


def _at(seconds: float) -> datetime:
    return _BASE + timedelta(seconds=seconds)


# =============================== sut events ===============================


def _step_end(
    rollout_id: int,
    *,
    at: datetime,
    outcomes: list[TrainStepOutcome] | None = None,
    trainer_id: str = ACTOR_ROLE,
) -> TrainGroupStepEndEvent:
    return TrainGroupStepEndEvent(
        timestamp=at,
        source=TrainerControllerProcessIdentity(trainer_id=trainer_id),
        rollout_id=rollout_id,
        attempt=0,
        role="actor",
        cell_outcomes={0: [TrainStepOutcome.NORMAL] if outcomes is None else outcomes},
    )


def _reconfigure(
    *, at: datetime, healed_cell_indices: list[int], cell_incarnations_after: dict[str, str]
) -> CellReconfigureEvent:
    return CellReconfigureEvent(
        timestamp=at,
        source=TrainerControllerProcessIdentity(trainer_id=ACTOR_ROLE),
        rollout_id=0,
        quorum_id=1,
        src_cell_index=0,
        healed_cell_indices=healed_cell_indices,
        alive_cell_indices_after=[0, *healed_cell_indices],
        cell_incarnations_after=cell_incarnations_after,
    )


def _weight_update_result(
    update_id: str,
    *,
    at: datetime,
    cell_hashes: dict[str, str],
    updated: list[str],
    failed: list[str] | None = None,
    candidate_version: int | None = 1,
    published_version: int | None = None,
) -> WeightUpdateResultEvent:
    return WeightUpdateResultEvent(
        timestamp=at,
        source=TrainerControllerProcessIdentity(trainer_id=ACTOR_ROLE),
        debug_weight_update_id=update_id,
        debug_trainer_load_state_timestamp=0.0,
        rollout_id=0,
        candidate_version=candidate_version,
        published_version=(candidate_version if updated else None) if published_version is None else published_version,
        snapshot_cell_id_to_hashes=cell_hashes,
        updated_cell_ids=updated,
        failed_cell_ids=failed or [],
    )


def _sut_main_source() -> SimpleProcessIdentity:
    return SimpleProcessIdentity(component="main")


def _write_sut_lines(path: Path, events: list[Event], *, trailing: str = "") -> None:
    with path.open("a") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")
        stream.write(trailing)


# ============================= soak events ==============================


def _cell_target(
    *,
    kind: str = "actor",
    cell_index: int = 0,
    incarnation: str = "inc-a",
    alive: bool = True,
    ready: bool = True,
) -> CellTarget:
    return CellTarget(
        kind=kind,
        identity=compute_cell_id(pool_id=kind, cell_index=cell_index),
        incarnation=incarnation,
        alive=alive,
        ready=ready,
    )


def _pod_target(name: str = "pod-a") -> SoakPodTarget:
    return SoakPodTarget(namespace="ns", release="rel", name=name, uid=f"uid-{name}")


def _pod_evidence() -> PodDeletedEvidence:
    return PodDeletedEvidence(namespace="ns", pod_name="pod-a", pod_uid="uid-pod-a")


def _request(target: SoakTarget, *, form_name: str = "fake", request_id: str = "req-1") -> SoakActionRequest:
    return SoakActionRequest(
        request_id=request_id, target=target, form_name=form_name, details=PodDetails(pod=_pod_target())
    )


def _observation(
    targets: list[SoakTarget] | None,
    *,
    at: datetime,
    errors: dict[str, str] | None = None,
    new_sut_events: list[Event] | None = None,
) -> SoakObservationEvent:
    return SoakObservationEvent(
        timestamp=at, targets=targets, errors=errors or {}, new_sut_events=new_sut_events or []
    )


def _requested(request: SoakActionRequest, *, at: datetime) -> SoakActionRequestedEvent:
    return SoakActionRequestedEvent(timestamp=at, request=request)


def _applied(request: SoakActionRequest, *, at: datetime) -> SoakActionAppliedEvent:
    return SoakActionAppliedEvent(timestamp=at, request_id=request.request_id, evidence=_pod_evidence())


def _result(
    request: SoakActionRequest, *, at: datetime, returned: bool = True, error: str | None = None
) -> SoakActionResultEvent:
    return SoakActionResultEvent(timestamp=at, request_id=request.request_id, returned=returned, error=error)


# ============================== fake forms ===============================


async def _apply_and_return(request: SoakActionRequest, report_applied: Callable[[SoakActionEvidence], None]) -> None:
    report_applied(_pod_evidence())


class _FakeForm(BaseSoakActionForm):
    def __init__(
        self,
        *,
        name: str = "fake",
        harms_target: bool = False,
        recovered: bool = True,
        creates_request: bool = True,
        execute: _ExecuteBehavior = _apply_and_return,
    ) -> None:
        self._name = name
        self._harms_target = harms_target
        self.recovered = recovered
        self.creates_request = creates_request
        self._execute = execute
        self.created: list[SoakActionRequest] = []
        self.executed: list[SoakActionRequest] = []
        self.recovery_checks: list[SoakActionRecord] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def harms_target(self) -> bool:
        return self._harms_target

    def maybe_create_request(
        self,
        *,
        target: SoakTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if not self.creates_request:
            return None
        request = _request(target, form_name=self._name, request_id=f"{self._name}-{len(self.created)}")
        self.created.append(request)
        return request

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        self.executed.append(request)
        await self._execute(request, report_applied)

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        self.recovery_checks.append(action)
        return self.recovered


# ============================ fake observers =============================


class _ScriptedObserver(SoakObserver):
    def __init__(
        self,
        make_observation: Callable[[int], SoakObservationEvent],
        *,
        hang_from_call: int | None = None,
    ) -> None:
        self._make_observation = make_observation
        self._hang_from_call = hang_from_call
        self.calls = 0

    async def observe(self) -> SoakObservationEvent:
        index = self.calls
        self.calls += 1
        if self._hang_from_call is not None and index >= self._hang_from_call:
            await asyncio.Event().wait()
        return self._make_observation(index)


# ============================= stored evidence =============================


def _stored_line(sequence: int, event: SoakEvent) -> str:
    return StoredEvent(sequence=sequence, event=event).model_dump_json() + "\n"


def _run_context(sources: dict[str, Path], *, at: datetime) -> SoakRunContextEvent:
    return SoakRunContextEvent(
        timestamp=at,
        context=SoakRunContext(
            base_url="http://localhost:18080", config=SoakRunnerConfig(seed=0), form_names={}, train_config=None
        ),
        sources=sources,
    )


# ============================= teardown boundary =============================


class _RecordingProcesses:
    def __init__(self, *, stdout: str = "stopped\n", error: BaseException | None = None) -> None:
        self._stdout = stdout
        self._error = error
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: list[str], *, capture_output: bool, check: bool, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        assert capture_output and check and timeout is not None
        if self._error is not None:
            raise self._error
        return subprocess.CompletedProcess(argv, 0, stdout=self._stdout, stderr="")


class _RecordingReleaseRemoval:
    def __init__(self, *, block: threading.Event | None = None, error: BaseException | None = None) -> None:
        self._block = block
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, release: str, namespace: str) -> None:
        self.calls.append((release, namespace))
        if self._block is not None:
            self._block.wait(timeout=10)
        if self._error is not None:
            raise self._error


# ============================== runner boundary ==============================


class _ScriptedScheduler:
    def __init__(self, requests: list[SoakActionRequest | None] | None = None) -> None:
        self._requests = list(requests or [])
        self.seen_event_counts: list[int] = []

    def choose(self, *, events: list[SoakEvent], now: float) -> SoakActionRequest | None:
        self.seen_event_counts.append(len(events))
        return self._requests.pop(0) if self._requests else None


class _ScriptedSutFeed:
    def __init__(self, batches: list[list[Event]]) -> None:
        self._batches = list(batches)

    async def attach(self, observation: SoakObservationEvent) -> SoakObservationEvent:
        batch = self._batches.pop(0) if self._batches else []
        return observation.model_copy(update={"new_sut_events": batch})


class _RecordingTeardown:
    def __init__(self, event_log: EventLog) -> None:
        self._event_log = event_log
        self.calls = 0
        self.closed_when_called: list[bool] = []

    async def __call__(self) -> None:
        self.calls += 1
        self.closed_when_called.append(
            any(isinstance(event, SoakCollectionClosedEvent) for event in self._event_log.events)
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def _flatten_errors(error: BaseException) -> list[BaseException]:
    if isinstance(error, builtins.BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _flatten_errors(inner)]
    return [error]


def _healthy_observer() -> _ScriptedObserver:
    return _ScriptedObserver(lambda index: _observation([_cell_target()], at=_now()))


def _runner_config(**overrides: object) -> SoakRunnerConfig:
    return SoakRunnerConfig(seed=0, poll_interval_seconds=0.001, **overrides)


def _make_runner(
    tmp_path: Path,
    *,
    observer: SoakObserver,
    forms: SoakForms | None = None,
    scheduler: _ScriptedScheduler | None = None,
    config: SoakRunnerConfig | None = None,
    sut_events: _ScriptedSutFeed | None = None,
) -> SoakRunner:
    return SoakRunner(
        observer=observer,
        scheduler=scheduler or _ScriptedScheduler(),
        forms={"actor": [_FakeForm()]} if forms is None else forms,
        event_log=EventLog(tmp_path / "evidence" / "events.jsonl"),
        config=config or _runner_config(),
        sut_events=sut_events,
    )


# ============================== cell api boundary ==============================


def _cell(
    name: str,
    *,
    cell_type: str,
    workers_hash: str = "hash-a",
    phase: str = "Running",
    healthy: TriState = TriState.TRUE,
    serving: TriState = TriState.TRUE,
) -> Cell:
    return Cell(
        metadata=CellMetadata(name=name, labels={CELL_TYPE_LABEL: cell_type}),
        spec=CellSpec(),
        status=CellStatus(
            phase=phase,
            conditions=[CellCondition(type="Healthy", status=healthy), CellCondition(type="Serving", status=serving)],
            workers_hash=workers_hash,
        ),
    )


def _fault_target(cell_id: str, *, workers_hash: str = "hash-a", rank: int = 0) -> ObservedFaultHookTarget:
    return ObservedFaultHookTarget(cell_id=cell_id, rank=rank, workers_hash=workers_hash)


_CellReply = int | Cell | Exception


class _FakeCellApi:
    def __init__(self, cells: list[Cell]) -> None:
        self.cells = {cell.metadata.name: cell for cell in cells}
        self.list_reply: int | dict | None = None
        self.fault_targets: dict[str, ObservedFaultHookTarget | int] = {}
        self.cell_replies: dict[str, list[_CellReply]] = {}
        self.hook_status = 200
        self.hook_posts: list[tuple[str, FaultHookCommand]] = []
        self.paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(f"{request.method} {path}?{request.url.query.decode()}".rstrip("?"))
        parts = path.removeprefix("/api/v1/cells").strip("/").split("/")

        if parts == [""]:
            if isinstance(self.list_reply, int):
                return httpx.Response(self.list_reply)
            if isinstance(self.list_reply, dict):
                return httpx.Response(200, json=self.list_reply)
            return httpx.Response(200, json=CellList(items=list(self.cells.values())).model_dump(mode="json"))

        name = parts[0]
        if parts[1:] == ["fault-target"]:
            match self.fault_targets.get(name, 404):
                case int() as status:
                    return httpx.Response(status)
                case target:
                    return httpx.Response(200, json=target.model_dump(mode="json"))
        if parts[1:] == ["fault-hook"]:
            self.hook_posts.append((name, FaultHookCommand.model_validate_json(request.content)))
            return httpx.Response(self.hook_status)

        replies = self.cell_replies.get(name)
        reply: _CellReply = (
            (replies.pop(0) if len(replies) > 1 else replies[0]) if replies else self.cells.get(name, 404)
        )
        match reply:
            case int() as status:
                return httpx.Response(status)
            case Exception() as error:
                raise error
            case cell:
                return httpx.Response(200, json=cell.model_dump(mode="json"))


def _patch_http(monkeypatch: pytest.MonkeyPatch, api: _FakeCellApi) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(api.handle), **kwargs),
    )


# ============================== kubectl boundary ==============================


def _pod_json(name: str, *, pool_id: str, cell_index: int | None) -> dict:
    labels = {DEFAULT_LABEL_KEYS.pool_id: pool_id}
    if cell_index is not None:
        labels[DEFAULT_LABEL_KEYS.cell_index] = str(cell_index)
    return {"metadata": {"name": name, "uid": f"uid-{name}", "labels": labels}}


def _process_target(*, pod_uid: str, pattern: str) -> ProcessTarget:
    return ProcessTarget(
        pod_uid=pod_uid,
        boot_id="boot",
        pid_namespace="pidns",
        init_start_ticks=1,
        pattern=pattern,
        processes=[ProcessIdentity(pid=42, start_ticks=7)],
    )


class _FakeKubectl:
    def __init__(
        self,
        *,
        pods: list[dict],
        get_error: BaseException | None = None,
        process_targets: dict[tuple[str, str], ProcessTarget | BaseException] | None = None,
    ) -> None:
        self._pods = pods
        self._get_error = get_error
        self._process_targets = process_targets or {}
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: list[str], *, capture_output: bool, check: bool, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        assert argv[0] == "kubectl" and capture_output and check and timeout is not None
        if argv[1] == "get":
            if self._get_error is not None:
                raise self._get_error
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"items": self._pods}), stderr="")
        pod_name, container = argv[4], argv[6]
        reply = self._process_targets[(pod_name, container)]
        if isinstance(reply, BaseException):
            raise reply
        return subprocess.CompletedProcess(argv, 0, stdout=reply.model_dump_json(), stderr="")


def _injected(request: SoakActionRequest, *, start: float, returned: bool = True) -> list[SoakEvent]:
    return [
        _requested(request, at=_at(start)),
        _applied(request, at=_at(start + 1)),
        _result(request, at=_at(start + 2), returned=returned),
    ]


def _with_fault_target(target: CellTarget) -> CellTarget:
    return target.model_copy(update={"fault_target": _fault_target(target.identity, workers_hash=target.incarnation)})


def _raising_hook_transport(api: _FakeCellApi) -> _FakeCellApi:
    handle = api.handle

    def raise_on_hook(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/fault-hook"):
            handle(request)
            raise httpx.ReadTimeout("lost reply", request=request)
        return handle(request)

    api.handle = raise_on_hook
    return api


def _healed_injection(
    form_name: str,
    *,
    kind: str,
    cell_index: int,
    start: float,
    request_id: str,
    new_incarnation: str = "inc-b",
    ready: bool = True,
    healed_cell_indices: list[int] | None = None,
    step_after: bool = True,
) -> list[SoakEvent]:
    target = _cell_target(kind=kind, cell_index=cell_index)
    request = _request(target, form_name=form_name, request_id=request_id)
    healed = target.model_copy(update={"incarnation": new_incarnation, "ready": ready})
    reconfigurations = (
        [
            _reconfigure(
                at=_at(start + 3),
                healed_cell_indices=[cell_index] if healed_cell_indices is None else healed_cell_indices,
                cell_incarnations_after={target.identity: new_incarnation},
            )
        ]
        if kind == "actor" and healed_cell_indices != []
        else []
    )
    events = [
        *_injected(request, start=start),
        _observation([healed], at=_at(start + 3), new_sut_events=reconfigurations),
    ]
    if step_after:
        events.append(_observation(None, at=_at(start + 4), new_sut_events=[_step_end(int(start), at=_at(start + 4))]))
    return events
