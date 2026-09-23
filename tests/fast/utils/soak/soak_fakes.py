from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
    SoakEvent,
    SoakObservationEvent,
    SoakRunContext,
    SoakRunContextEvent,
    StoredEvent,
)
from tests.utils.soak.core.types import SoakActionRequest, SoakTarget
from tests.utils.soak.ft.types import CellTarget, PodDetails
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence, SoakPodTarget

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.utils.audit_utils.event_logger.models import CellReconfigureEvent, Event, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainerControllerProcessIdentity
from miles.utils.workers.naming import compute_cell_id

_BASE = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


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
