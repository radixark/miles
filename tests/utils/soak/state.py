# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import enum
import hashlib
import os
import shutil
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4

from pydantic import Field, field_validator
from tests.utils.soak.config import SoakPolicy
from tests.utils.soak.process_target import ProcessTarget

from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, TrainGroupStepEndEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.cell_operations.base import FaultTarget


def cell_is_alive(cell: dict) -> bool:
    return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])


class ObservedCellState(enum.Enum):
    SUSPENDED = "Suspended"  # torn down, holding no gpu
    PENDING = "Pending"  # allocated but gated: no engine serving yet
    RUNNING_NOT_SERVING = "RunningNotServing"  # engine is up but not registered in the router
    SERVING = "Serving"  # registered in the router, i.e. actually able to answer requests


def compute_observed_cell_state(cell: dict) -> ObservedCellState:
    phase = cell["status"]["phase"]
    if phase == "Suspended":
        return ObservedCellState.SUSPENDED
    if phase == "Pending":
        return ObservedCellState.PENDING
    serving = any(cond["type"] == "Serving" and cond["status"] == "True" for cond in cell["status"]["conditions"])
    return ObservedCellState.SERVING if serving else ObservedCellState.RUNNING_NOT_SERVING


class BaseEvent(FrozenStrictBaseModel):
    # Wall clock, so an event can be lined up against the timestamps the metric events carry.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InjectionEvent(BaseEvent):
    cell_name: str
    form_name: str
    succeeded: bool
    harmed: bool = True


class SoakPodTarget(FrozenStrictBaseModel):
    namespace: str
    release: str
    name: str
    uid: str
    process_targets: dict[str, ProcessTarget] = Field(default_factory=dict)


class SoakDeploymentTarget(FrozenStrictBaseModel):
    kind: Literal["deployment"] = "deployment"
    namespace: str
    release: str
    workload_stamps: dict[str, str | None]
    workload_uids: dict[str, str]
    saved_iteration: int | None
    finished_rollout_id: int | None
    state_file: Path | None = None
    uninstall_job_uid: str | None = None


class SoakObservation(BaseEvent):
    cells: list[dict] | None
    pods_of_cell: dict[str, list[SoakPodTarget]] = Field(default_factory=dict)
    deployments: list[SoakDeploymentTarget] = Field(default_factory=list)
    details: dict[str, dict] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
    fault_targets: dict[str, FaultTarget] = Field(default_factory=dict)
    training_events: list[CellReconfigureEvent | TrainGroupStepEndEvent] = Field(default_factory=list)


class SoakActionRequest(FrozenStrictBaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    target: SoakDeploymentTarget | dict
    form_name: str
    harms_cell: bool
    next_due_at: float | None = None
    pod: SoakPodTarget | None = None
    fault_target: FaultTarget | None = None

    @field_validator("target", mode="before")
    @classmethod
    def _parse_target(cls, value: SoakDeploymentTarget | dict) -> SoakDeploymentTarget | dict:
        if isinstance(value, dict) and value.get("kind") == "deployment":
            return SoakDeploymentTarget.model_validate(value)
        return value


class SoakScheduleEvent(BaseEvent):
    due_of_type: dict[str, float]
    policy: SoakPolicy | None = None


class SoakActionRequestedEvent(BaseEvent):
    request: SoakActionRequest


class SoakActionResultEvent(BaseEvent):
    request_id: str
    returned: bool
    error: str | None = None


class SoakActionAppliedEvent(BaseEvent):
    request_id: str
    evidence: dict


class SoakLauncherExitedEvent(BaseEvent):
    request_id: str | None
    returncode: int
    log_path: Path


class SoakRunContextEvent(BaseEvent):
    details: dict
    sources: dict[str, Path]


class SoakCollectionClosedEvent(BaseEvent):
    pass


class SoakAdmissionClosedEvent(BaseEvent):
    monotonic_time: float = Field(default_factory=time.monotonic)


class SoakTeardownEvent(BaseEvent):
    resource: str
    returned: bool
    error: str | None = None


class SoakEvidenceArchivedEvent(BaseEvent):
    sources: dict[str, Path]
    missing_sources: list[str]
    sha256_of_file: dict[str, str]


class CellInfo(FrozenStrictBaseModel):
    cell_type: str
    state: ObservedCellState
    alive: bool


class ObservationsEvent(BaseEvent):
    # One whole poll, so a cell that has vanished is as recorded as one that answered.
    cell_infos: dict[str, CellInfo]
    cells: list[dict] = Field(default_factory=list)


Event = (
    InjectionEvent
    | ObservationsEvent
    | SoakActionRequestedEvent
    | SoakActionResultEvent
    | SoakScheduleEvent
    | SoakObservation
    | SoakActionAppliedEvent
    | SoakLauncherExitedEvent
    | SoakRunContextEvent
    | SoakCollectionClosedEvent
    | SoakAdmissionClosedEvent
    | SoakTeardownEvent
    | SoakEvidenceArchivedEvent
)


class _StoredEvent(FrozenStrictBaseModel):
    version: Literal[1] = 1
    sequence: int = Field(ge=0)
    event_type: str
    event: dict


def read_events(path: Path, *, require_closed: bool = True) -> list[Event]:
    event_types = {event_type.__name__: event_type for event_type in get_args(Event)}
    events: list[Event] = []
    with path.open() as stream:
        for sequence, line in enumerate(stream):
            stored = _StoredEvent.model_validate_json(line)
            assert stored.sequence == sequence, f"Missing or reordered soak event at {path}:{sequence + 1}"
            assert stored.event_type in event_types, f"Unknown soak event type: {stored.event_type}"
            assert not events or not isinstance(
                events[-1], SoakCollectionClosedEvent
            ), f"Events after closure in {path}"
            events.append(event_types[stored.event_type].model_validate(stored.event))
    if require_closed:
        assert events and isinstance(events[-1], SoakCollectionClosedEvent), f"Soak evidence is incomplete: {path}"
        for event in events:
            if isinstance(event, SoakEvidenceArchivedEvent):
                for relative, expected in event.sha256_of_file.items():
                    source = path.parent / relative
                    assert source.resolve().is_relative_to(
                        path.parent.resolve()
                    ), f"Evidence path escapes its archive: {relative}"
                    assert _file_sha256(source) == expected, f"Archived evidence changed: {source}"
    return events


class EventLog:
    """The fault injector's only mutable state: what happened, in order. Every question is a view of it."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._path: Path | None = None

    def persist_to(self, path: Path) -> None:
        with self._lock:
            assert self._path is None and not self._events, "Configure evidence persistence before recording events"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x"):
                pass
            self._path = path

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def finish(self) -> None:
        if self._path is not None:
            contexts = [event for event in self.events if isinstance(event, SoakRunContextEvent)]
            if contexts:
                self._append(_archive_sources(sources=contexts[-1].sources, destination=self._path.parent / "sources"))
        self._append(SoakCollectionClosedEvent())
        if self._path is not None:
            assert read_events(self._path) == self.events, f"Persisted soak evidence differs from memory: {self._path}"

    def note_context(self, event: SoakRunContextEvent) -> None:
        self._append(event)

    def note_injection_attempt(self, *, cell_name: str, form_name: str, succeeded: bool, harmed: bool = True) -> None:
        self._append(InjectionEvent(cell_name=cell_name, form_name=form_name, succeeded=succeeded, harmed=harmed))

    def observe(self, cells: list[dict]) -> None:
        self._append(
            ObservationsEvent(
                cells=deepcopy(cells),
                cell_infos=compute_cell_infos(cells),
            )
        )

    def close_admission(self) -> None:
        self._append(SoakAdmissionClosedEvent())

    def note_action_requested(self, request: SoakActionRequest) -> bool:
        return self._append(SoakActionRequestedEvent(request=request.model_copy(deep=True)))

    def note_action_result(self, result: SoakActionResultEvent) -> None:
        self._append(result)

    def note_action_applied(self, event: SoakActionAppliedEvent) -> None:
        self._append(event.model_copy(deep=True))

    def note_launcher_exited(self, event: SoakLauncherExitedEvent) -> None:
        self._append(event)

    def note_teardown(self, event: SoakTeardownEvent) -> None:
        self._append(event)

    def note_schedule(self, schedule: SoakScheduleEvent) -> None:
        self._append(schedule)

    def note_observation(self, observation: SoakObservation) -> None:
        self._append(observation.model_copy(deep=True))

    def _append(self, event: Event) -> bool:
        with self._lock:
            assert not self._events or not isinstance(
                self._events[-1], SoakCollectionClosedEvent
            ), "Soak evidence is closed"
            if isinstance(event, (SoakActionRequestedEvent, SoakAdmissionClosedEvent)) and any(
                isinstance(previous, SoakAdmissionClosedEvent) for previous in self._events
            ):
                return False
            snapshot = type(event).model_validate(event.model_dump(mode="json"))
            if self._path is not None:
                stored = _StoredEvent(
                    sequence=len(self._events),
                    event_type=type(snapshot).__name__,
                    event=snapshot.model_dump(mode="json"),
                )
                with self._path.open("r+b") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.write((stored.model_dump_json() + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
            self._events.append(snapshot)
            return True


def compute_cell_infos(cells: list[dict]) -> dict[str, CellInfo]:
    return {
        cell["metadata"]["name"]: CellInfo(
            cell_type=cell_type_of(cell),
            state=compute_observed_cell_state(cell),
            alive=cell_is_alive(cell),
        )
        for cell in cells
    }


def cell_type_of(cell: dict) -> str:
    return cell["metadata"]["labels"]["miles.io/cell-type"]


def target_type_of(target: SoakDeploymentTarget | dict) -> str:
    return "deployment" if isinstance(target, SoakDeploymentTarget) else cell_type_of(target)


def event_source(events: list[Event], *, name: str, fallback: Path) -> Path:
    for event in reversed(events):
        if isinstance(event, SoakEvidenceArchivedEvent):
            assert name not in event.missing_sources, f"Missing archived soak evidence: {name}"
            if name in event.sources:
                return event.sources[name]
    return fallback


def _archive_sources(*, sources: dict[str, Path], destination: Path) -> SoakEvidenceArchivedEvent:
    archived: dict[str, Path] = {}
    missing: list[str] = []
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        assert name and Path(name).name == name and name not in (".", ".."), f"Invalid evidence source name: {name}"
        if not source.is_dir():
            missing.append(name)
            continue
        root = destination / name
        target = root / source.name
        shutil.copytree(source, target)
        for discarded in sorted(source.parent.glob(".trash_*")):
            if discarded.is_dir():
                shutil.copytree(discarded, root / discarded.name)
        archived[name] = target
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(destination.parent))] = _file_sha256(path)
    return SoakEvidenceArchivedEvent(sources=archived, missing_sources=missing, sha256_of_file=hashes)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
