from __future__ import annotations

import pytest

from miles.utils.ft.models import ActionType, Decision, TriggerType
from tests.fast.utils.ft.helpers import (
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    FixedDecisionDetector,
    make_detector_context,
    make_test_controller,
)


async def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("notifier broken")


class TestTickEmptyDetectorChain:
    @pytest.mark.asyncio
    async def test_tick_succeeds_with_no_detectors(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_tick_returns_none_decision(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
        assert decision.action == ActionType.NONE


class TestDetectorChain:
    def test_first_non_none_wins(self) -> None:
        none_detector = AlwaysNoneDetector()
        bad_detector = AlwaysMarkBadDetector()
        harness = make_test_controller(
            detectors=[none_detector, bad_detector],
        )

        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert none_detector.call_count == 1
        assert bad_detector.call_count == 1

    def test_short_circuit_after_non_none(self) -> None:
        bad_detector = AlwaysMarkBadDetector()
        trailing_detector = AlwaysNoneDetector()
        harness = make_test_controller(
            detectors=[bad_detector, trailing_detector],
        )

        ctx = make_detector_context(metric_store=harness.metric_store, mini_wandb=harness.mini_wandb)
        decision = harness.controller._evaluate_detectors(ctx)
        assert decision.action == ActionType.MARK_BAD_AND_RESTART
        assert bad_detector.call_count == 1
        assert trailing_detector.call_count == 0


class TestExecuteDecision:
    @pytest.mark.asyncio
    async def test_none_decision_is_noop(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()

    @pytest.mark.asyncio
    async def test_mark_bad_and_restart_does_not_raise(self) -> None:
        harness = make_test_controller(
            detectors=[AlwaysMarkBadDetector()],
        )
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_enter_recovery_does_not_raise(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=TriggerType.CRASH,
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_notify_human_sends_notification(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="test notify",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.controller._tick_count == 1
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, content, severity = harness.notifier.calls[0]
        assert title == "Fault Alert"
        assert content == "test notify"
        assert severity == "critical"

    @pytest.mark.asyncio
    async def test_notify_human_without_notifier(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="test notify no notifier",
        ))
        harness = make_test_controller(detectors=[detector], notifier=None)
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_none_decision_does_not_notify(self) -> None:
        harness = make_test_controller(detectors=[AlwaysNoneDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_mark_bad_does_not_notify(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_enter_recovery_does_not_notify(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.ENTER_RECOVERY,
            trigger=TriggerType.CRASH,
            reason="test recovery",
        ))
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.asyncio
    async def test_notify_human_notifier_exception_does_not_crash(self) -> None:
        harness = make_test_controller(
            detectors=[FixedDecisionDetector(decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="test with broken notifier",
            ))],
        )
        assert harness.notifier is not None
        harness.notifier.send = _raise_runtime_error
        await harness.controller._tick()
        assert harness.controller._tick_count == 1

    @pytest.mark.asyncio
    async def test_notify_human_sends_on_every_tick(self) -> None:
        detector = FixedDecisionDetector(decision=Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="persistent fault",
        ))
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 2


class TestMarkBadAndRestartReal:
    @pytest.mark.asyncio
    async def test_marks_bad_nodes(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.node_manager.is_node_bad("node-1")

    @pytest.mark.asyncio
    async def test_stops_and_submits_training(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.training_job._stopped
        assert harness.training_job._submitted

    @pytest.mark.asyncio
    async def test_clears_mini_wandb_before_submit(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        harness.mini_wandb.log_step(
            run_id="test", rank=0, step=1, metrics={"loss": 1.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 1.0

        await harness.controller._tick()

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) is None
