from tests.fast.e2e.deploy.hot_restart.cluster_facts import RELEASE
from tests.fast.utils.soak.deploy.deploy_fakes import (
    _deployment_target,
    _hot_restart_request,
    _landed_take_over,
    _release_snapshot,
    _requested_take_over,
)
from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _request, _requested
from tests.utils.deploy.hot_restart.cluster_observer import ClusterSnapshot
from tests.utils.deploy.hot_restart.evidence import HotRestartRecord
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.deploy.checkers.evidence import project_hot_restart_evidence
from tests.utils.soak.deploy.types import DeploymentObservationDetails


def _observed(snapshot: ClusterSnapshot | None, *, at: float) -> SoakObservationEvent:
    details = None if snapshot is None else DeploymentObservationDetails(cluster=snapshot)
    return SoakObservationEvent(timestamp=_at(at), targets=[], details=details)


def _take_over(
    request_id: str, *, saved: int | None, finished: int | None, at: float, landed: bool
) -> list[SoakEvent]:
    target = _deployment_target(saved_iteration=saved, finished_rollout_id=finished)
    request = _hot_restart_request(target, request_id=request_id)
    events: list[SoakEvent] = [_requested_take_over(request, at=_at(at))]
    if landed:
        events.append(_landed_take_over(request, after=target, at=_at(at + 1)))
    return events


class TestProjectHotRestartEvidence:
    def test_records_are_the_landed_take_overs_with_their_trigger_progress(self) -> None:
        """Each landed take-over becomes a record, indexed in order, carrying progress at the draw."""
        cell_request = _request(_cell_target(), request_id="cell")
        events = [
            *_take_over("a", saved=None, finished=None, at=0, landed=True),
            *_take_over("x", saved=9, finished=9, at=2, landed=False),
            _requested(cell_request, at=_at(3)),
            _applied(cell_request, at=_at(4)),
            *_take_over("b", saved=3, finished=5, at=5, landed=True),
        ]

        evidence = project_hot_restart_evidence(events, release=RELEASE)

        assert evidence.release == RELEASE
        assert evidence.records == (
            HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=-1),
            HotRestartRecord(index=1, saved_iteration_at_trigger=3, frozen_rollout_id=5),
        )

    def test_snapshots_are_recorded_once_the_release_settled_and_failures_counted(self) -> None:
        """Installing, failed and detail-less observations never become snapshots of a settled run."""
        settled = _release_snapshot()
        failed = _release_snapshot(reads_missing=("pods",))
        events = [
            _observed(settled, at=0),
            _observed(None, at=1),
            _observed(settled, at=2),
            _observed(failed, at=3),
            _observed(settled, at=4),
        ]

        evidence = project_hot_restart_evidence(events, release=RELEASE)

        assert evidence.snapshots == (settled, settled)
        assert evidence.observation_failures == 1
        assert evidence.observation_attempts == 3
