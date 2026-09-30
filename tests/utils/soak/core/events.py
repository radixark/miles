from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Discriminator, Field
from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest, SoakObservationDetails, SoakTarget

from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class BaseEvent(FrozenStrictBaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SoakObservationEvent(BaseEvent):
    kind: Literal["observation"] = "observation"
    targets: list[SoakTarget] | None
    details: SoakObservationDetails | None = None
    errors: dict[str, str] = Field(default_factory=dict)
    new_sut_events: list[Event] = Field(default_factory=list)


class SoakActionRequestedEvent(BaseEvent):
    kind: Literal["action_requested"] = "action_requested"
    request: SoakActionRequest


class SoakActionResultEvent(BaseEvent):
    kind: Literal["action_result"] = "action_result"
    request_id: str
    returned: bool
    error: str | None = None


class SoakActionAppliedEvent(BaseEvent):
    kind: Literal["action_applied"] = "action_applied"
    request_id: str
    evidence: SoakActionEvidence


class LaunchOutcome(StrEnum):
    FINISHED = "finished"
    REPLACED = "replaced"
    FAILED = "failed"


class SoakLaunchFinishedEvent(BaseEvent):
    kind: Literal["launch_finished"] = "launch_finished"
    request_id: str | None
    outcome: LaunchOutcome
    error: str | None = None


class SoakRunContext(FrozenStrictBaseModel):
    base_url: str
    config: SoakRunnerConfig
    form_names: dict[str, list[str]]
    train_config: ExecuteTrainConfig | None


class SoakRunContextEvent(BaseEvent):
    kind: Literal["run_context"] = "run_context"
    context: SoakRunContext
    sources: dict[str, Path]


class SoakCollectionClosedEvent(BaseEvent):
    kind: Literal["collection_closed"] = "collection_closed"


class SoakAdmissionClosedEvent(BaseEvent):
    kind: Literal["admission_closed"] = "admission_closed"


class SoakTeardownEvent(BaseEvent):
    kind: Literal["teardown"] = "teardown"
    resource: str
    returned: bool
    error: str | None = None


class SoakEvidenceArchivedEvent(BaseEvent):
    kind: Literal["evidence_archived"] = "evidence_archived"
    sources: dict[str, Path]
    missing_sources: list[str]
    sha256_of_file: dict[str, str]


SoakEvent = Annotated[
    SoakActionRequestedEvent
    | SoakActionResultEvent
    | SoakObservationEvent
    | SoakActionAppliedEvent
    | SoakLaunchFinishedEvent
    | SoakRunContextEvent
    | SoakCollectionClosedEvent
    | SoakAdmissionClosedEvent
    | SoakTeardownEvent
    | SoakEvidenceArchivedEvent,
    Discriminator("kind"),
]


# ================================== evidence ==================================


class StoredEvent(FrozenStrictBaseModel):
    version: Literal[1] = 1
    sequence: int = Field(ge=0)
    event: SoakEvent


def read_events(path: Path, *, require_closed: bool = True) -> list[SoakEvent]:
    events: list[SoakEvent] = []

    with path.open() as stream:
        for sequence, line in enumerate(stream):
            stored = StoredEvent.model_validate_json(line)
            assert stored.sequence == sequence, f"Missing or reordered soak event at {path}:{sequence + 1}"
            assert not events or not isinstance(
                events[-1], SoakCollectionClosedEvent
            ), f"Events after closure in {path}"
            events.append(stored.event)

    if require_closed:
        assert events and isinstance(events[-1], SoakCollectionClosedEvent), f"Soak evidence is incomplete: {path}"

    _assert_archived_files_unchanged(path, events)

    return events


def file_sha256(path: Path) -> str:
    raise NotImplementedError


def _assert_archived_files_unchanged(path: Path, events: list[SoakEvent]) -> None:
    for event in events:
        if isinstance(event, SoakEvidenceArchivedEvent):
            for relative, expected in event.sha256_of_file.items():
                source = path.parent / relative
                assert source.resolve().is_relative_to(
                    path.parent.resolve()
                ), f"Evidence path escapes its archive: {relative}"
                assert file_sha256(source) == expected, f"Archived evidence changed: {source}"
