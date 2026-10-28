import asyncio
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakActionAppliedEvent, SoakActionRequestedEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionRequest
from tests.utils.soak.core.views import SoakActionRecord
from tests.utils.soak.ft.actions.resize import ResizePoolForm, _compute_step_state
from tests.utils.soak.ft.types import PoolResizedEvidence, PoolTarget, ResizeDetails, ResizeStep
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity


class TestResizeTiming:
    @pytest.mark.parametrize(
        "last,expected", [(-1, "pending"), (0, "pending"), (1, "due"), (2, "missed"), (3, "missed")]
    )
    def test_only_the_rollout_before_the_scheduled_step_is_due(self, last: int, expected: str) -> None:
        """A resize fires between training steps and detects an already missed window."""
        events = (
            []
            if last < 0
            else [
                SoakObservationEvent(
                    targets=[],
                    new_sut_events=[
                        TrainGroupStepEndEvent(
                            timestamp=datetime.now(timezone.utc),
                            source=TrainerControllerProcessIdentity(trainer_id="actor"),
                            rollout_id=last,
                            attempt=0,
                            role="actor",
                            cell_outcomes={},
                        )
                    ],
                )
            ]
        )
        assert _compute_step_state(events, step=ResizeStep(at_rollout=2, replicas=4)) == expected


class TestResizeRecovery:
    @pytest.mark.parametrize(
        "offset,identity,ready,replicas,expected",
        [
            (1, "train", True, 4, True),
            (0, "train", True, 4, False),
            (-1, "train", True, 4, False),
            (1, "other", True, 4, False),
            (1, "train", False, 4, False),
            (1, "train", True, 2, False),
        ],
    )
    def test_recovery_requires_a_new_ready_observation_of_the_requested_pool(
        self, tmp_path: Path, offset: int, identity: str, ready: bool, replicas: int, expected: bool
    ) -> None:
        """Only later evidence of the requested pool at its target size establishes recovery."""
        form = _form(tmp_path)
        request = _request(form)
        now = datetime.now(timezone.utc)
        action = SoakActionRecord(
            requested=SoakActionRequestedEvent(request=request),
            applied=SoakActionAppliedEvent(
                timestamp=now, request_id=request.request_id, evidence=PoolResizedEvidence()
            ),
        )
        observation = SoakObservationEvent(
            timestamp=now + timedelta(seconds=offset),
            targets=[PoolTarget(identity=identity, alive=True, ready=ready, replicas=replicas)],
        )
        assert form.is_recovered(action=action, events=[observation]) is expected
        assert not form.is_recovered(action=SoakActionRecord(requested=action.requested), events=[observation])

    def test_completed_schedule_and_unrelated_pools_produce_no_request(self, tmp_path: Path) -> None:
        """A completed resize cannot repeat and another pool cannot consume its schedule."""
        form = _form(tmp_path)
        request = _request(form)
        observation = SoakObservationEvent(targets=[])
        assert (
            form.maybe_create_request(
                target=request.target,
                observation=observation,
                events=[SoakActionRequestedEvent(request=request)],
                rng=random.Random(0),
            )
            is None
        )
        assert (
            form.maybe_create_request(
                target=request.target.model_copy(update={"identity": "other"}),
                observation=observation,
                events=[],
                rng=random.Random(0),
            )
            is None
        )


def _form(tmp_path: Path) -> ResizePoolForm:
    return ResizePoolForm(
        namespace="rl",
        workload="train",
        cell_type="actor",
        schedule=(ResizeStep(at_rollout=0, replicas=4),),
        relaunch_at_size=_relaunch,
        event_log=EventLog(tmp_path / "events.jsonl"),
        chain=asyncio.Queue(),
    )


def _request(form: ResizePoolForm) -> SoakActionRequest:
    return SoakActionRequest(
        target=PoolTarget(identity="train", alive=True, ready=True, replicas=2),
        form_name=form.name,
        details=ResizeDetails(step=form.schedule[0]),
    )


async def _relaunch(replicas: int) -> None:
    assert replicas == 4


class TestResizeExecution:
    @pytest.mark.parametrize("launcher_fails", [False, True])
    async def test_application_waits_for_observed_size_and_rejects_an_early_failed_launcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher_fails: bool
    ) -> None:
        """A pending relaunch applies only at the requested size while an early failure cannot report success."""
        from dataclasses import replace
        from tests.utils.soak.ft.actions import resize

        release = asyncio.Event()
        started = asyncio.Event()
        applied = []
        launchers = []
        reads = []

        async def relaunch(replicas: int) -> None:
            assert replicas == 4
            started.set()
            if launcher_fails:
                raise RuntimeError("launch failed")
            await release.wait()

        async def read_size(form: ResizePoolForm) -> int:
            if not launchers:
                launchers.append(await form.chain.get())
            await started.wait()
            assert not applied
            if launcher_fails:
                await asyncio.gather(launchers[0], return_exceptions=True)
                return 2
            reads.append(2 if not reads else 4)
            return reads[-1]

        form = replace(_form(tmp_path), relaunch_at_size=relaunch)
        monkeypatch.setattr(ResizePoolForm, "_read_replicas", read_size)
        monkeypatch.setattr(resize, "_RELAUNCH_POLL_INTERVAL_SECONDS", 0)
        try:
            if launcher_fails:
                with pytest.raises(AssertionError, match="ended while the pool still reads 2"):
                    await form.execute(_request(form), report_applied=applied.append)
                assert applied == []
            else:
                await form.execute(_request(form), report_applied=applied.append)
                assert reads == [2, 4]
                assert applied == [PoolResizedEvidence()]
                assert not launchers[0].done()
        finally:
            release.set()
            await asyncio.gather(*launchers, return_exceptions=True)
