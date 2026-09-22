from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.utils.deploy.hot_restart.cluster_observer import ClusterObserver
from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND


def project_hot_restart_evidence(events: list[SoakEvent], *, release: str, namespace: str) -> HotRestartEvidence:
    observer = ClusterObserver(release=release, namespace=namespace, trainer_id=DEFAULT_TRAINER_ID)
    for event in events:
        if isinstance(event, SoakObservationEvent) and (details := event.details) is not None:
            observer.record_snapshot(details.cluster)
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
        snapshots=tuple(observer.snapshots),
        release=release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )
