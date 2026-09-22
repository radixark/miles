import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field
from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest, SoakObservationDetails, SoakTarget

from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class BaseEvent(FrozenStrictBaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SoakObservationEvent(BaseEvent):
    targets: list[SoakTarget] | None
    details: SoakObservationDetails | None = None
    errors: dict[str, str] = Field(default_factory=dict)
    new_sut_events: list[Event] = Field(default_factory=list)


class SoakScheduleEvent(BaseEvent):
    due_of_type: dict[str, float]


class SoakActionRequestedEvent(BaseEvent):
    request: SoakActionRequest


class SoakActionResultEvent(BaseEvent):
    request_id: str
    returned: bool
    error: str | None = None


class SoakActionAppliedEvent(BaseEvent):
    request_id: str
    evidence: SoakActionEvidence


class SoakLaunchFinishedEvent(BaseEvent):
    request_id: str | None
    outcome: Literal["finished", "replaced", "failed"]
    error: str | None = None


class SoakRunContext(FrozenStrictBaseModel):
    base_url: str
    seed: int
    config: SoakRunnerConfig
    mean_intervals: dict[str, float]
    form_names: dict[str, list[str]]
    train_config: ExecuteTrainConfig | None


class SoakRunContextEvent(BaseEvent):
    context: SoakRunContext
    sources: dict[str, Path]


class SoakCollectionClosedEvent(BaseEvent):
    pass


class SoakAdmissionClosedEvent(BaseEvent):
    pass


class SoakTeardownEvent(BaseEvent):
    resource: str
    returned: bool
    error: str | None = None


class SoakEvidenceArchivedEvent(BaseEvent):
    sources: dict[str, Path]
    missing_sources: list[str]
    sha256_of_file: dict[str, str]


SoakEvent = (
    SoakActionRequestedEvent
    | SoakActionResultEvent
    | SoakScheduleEvent
    | SoakObservationEvent
    | SoakActionAppliedEvent
    | SoakLaunchFinishedEvent
    | SoakRunContextEvent
    | SoakCollectionClosedEvent
    | SoakAdmissionClosedEvent
    | SoakTeardownEvent
    | SoakEvidenceArchivedEvent
)


# ================================== evidence ==================================


class StoredEvent(FrozenStrictBaseModel):
    version: Literal[1] = 1
    sequence: int = Field(ge=0)
    event_type: str
    event: dict


def read_events(path: Path, *, require_closed: bool = True) -> list[SoakEvent]:
    event_types = {event_type.__name__: event_type for event_type in get_args(SoakEvent)}
    events: list[SoakEvent] = []
    with path.open() as stream:
        for sequence, line in enumerate(stream):
            stored = StoredEvent.model_validate_json(line)
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
                assert file_sha256(source) == expected, f"Archived evidence changed: {source}"
    return events


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
