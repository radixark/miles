from tests.utils.deploy.hot_restart.cluster_observer import SnapshotRecorder
from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND


def project_hot_restart_evidence(events: list[SoakEvent], *, release: str) -> HotRestartEvidence:
    recorder = SnapshotRecorder(release=release)
    for event in events:
        if isinstance(event, SoakObservationEvent) and (details := event.details) is not None:
            recorder.record(details.cluster)

    applied = [
        action
        for action in project_actions(events).values()
        if action.requested.request.target.kind == DEPLOYMENT_TARGET_KIND and action.applied is not None
    ]
    records = tuple(
        HotRestartRecord(
            index=index,
            saved_iteration_at_trigger=action.requested.request.target.saved_iteration,
            frozen_rollout_id=(
                -1 if (frozen := action.requested.request.target.finished_rollout_id) is None else frozen
            ),
        )
        for index, action in enumerate(applied)
    )
    return HotRestartEvidence(
        records=records,
        snapshots=tuple(recorder.snapshots),
        release=release,
        observation_attempts=recorder.attempts,
        observation_failures=recorder.failures,
    )
