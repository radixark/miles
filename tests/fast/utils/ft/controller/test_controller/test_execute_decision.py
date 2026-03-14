from __future__ import annotations

import logging

import pytest
from tests.fast.utils.ft.conftest import (
    AlwaysEnterRecoveryDetector,
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    CrashingDetector,
    FixedDecisionDetector,
    make_detector_context,
    make_test_controller,
)
from tests.fast.utils.ft.utils.controller_fakes import get_training_subsystem_state

from miles.utils.ft.controller.state_machines.subsystem import RecoveringSt
from miles.utils.ft.controller.state_machines.subsystem.utils import run_detectors
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType

_TEST_ACTIVE_NODE_IDS: set[str] = {"node-0", "node-1"}


async def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("notifier broken")


class TestTickEmptyDetectorChain:
    @pytest.mark.anyio
    async def test_tick_succeeds_with_no_detectors(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 1

    @pytest.mark.anyio
    async def test_tick_returns_none_decision(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()
        ctx = make_detector_context(
            metric_store=harness.time_series_store,
            mini_wandb=harness.mini_wandb,
            active_node_ids=_TEST_ACTIVE_NODE_IDS,
        )
        result = run_detectors(detectors=[], ctx=ctx)
        assert result.recovery_decision is None


class TestDetectorChain:
    def test_first_non_none_wins(self) -> None:
        none_detector = AlwaysNoneDetector()
        bad_detector = AlwaysMarkBadDetector()

        ctx = make_detector_context(active_node_ids=_TEST_ACTIVE_NODE_IDS)
        result = run_detectors(detectors=[none_detector, bad_detector], ctx=ctx)
        assert result.recovery_decision is not None
        assert result.recovery_decision.action == ActionType.ENTER_RECOVERY
        assert none_detector.call_count == 1
        assert bad_detector.call_count == 1

    def test_all_detectors_evaluated_even_after_recovery(self) -> None:
        bad_detector = AlwaysMarkBadDetector()
        trailing_detector = AlwaysNoneDetector()

        ctx = make_detector_context(active_node_ids=_TEST_ACTIVE_NODE_IDS)
        result = run_detectors(detectors=[bad_detector, trailing_detector], ctx=ctx)
        assert result.recovery_decision is not None
        assert result.recovery_decision.action == ActionType.ENTER_RECOVERY
        assert bad_detector.call_count == 1
        assert trailing_detector.call_count == 1


class TestDetectorExceptionIsolation:
    def test_crashing_detector_does_not_block_others(self) -> None:
        crashing = CrashingDetector()
        good = AlwaysMarkBadDetector()

        ctx = make_detector_context(active_node_ids=_TEST_ACTIVE_NODE_IDS)
        result = run_detectors(detectors=[crashing, good], ctx=ctx)

        assert crashing.call_count == 1
        assert good.call_count == 1
        assert result.recovery_decision is not None
        assert result.recovery_decision.action == ActionType.ENTER_RECOVERY

    def test_all_detectors_crash_returns_none(self) -> None:
        d1 = CrashingDetector()
        d2 = CrashingDetector()

        ctx = make_detector_context(active_node_ids=_TEST_ACTIVE_NODE_IDS)
        result = run_detectors(detectors=[d1, d2], ctx=ctx)

        assert d1.call_count == 1
        assert d2.call_count == 1
        assert result.recovery_decision is None

    @pytest.mark.anyio
    async def test_tick_survives_exception_in_tick_inner(self) -> None:
        harness = make_test_controller()
        harness.main_job.get_status = _raise_runtime_error  # type: ignore[assignment]

        await harness.controller._tick()
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 2


class TestTickFailureTracker:
    @pytest.mark.anyio
    async def test_persistent_failure_triggers_notification(self) -> None:
        harness = make_test_controller()
        harness.main_job.get_status = _raise_runtime_error  # type: ignore[assignment]
        harness.controller._tick_loop._tick_failure_tracker._threshold = 3

        for _ in range(3):
            await harness.controller._tick()

        assert harness.notifier is not None
        titles = [title for title, _, _ in harness.notifier.calls]
        assert any("persistently failing" in t for t in titles)

    @pytest.mark.anyio
    async def test_sporadic_failure_does_not_trigger_notification(self) -> None:
        harness = make_test_controller()
        harness.main_job.get_status = _raise_runtime_error  # type: ignore[assignment]
        harness.controller._tick_loop._tick_failure_tracker._threshold = 5

        await harness.controller._tick()

        assert harness.notifier is not None
        titles = [title for title, _, _ in harness.notifier.calls]
        assert not any("persistently failing" in t for t in titles)


class TestAllDetectorsCrashSilentPass:
    @pytest.mark.anyio
    async def test_tick_with_all_crashing_detectors_logs_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        d1 = CrashingDetector()
        d2 = CrashingDetector()
        harness = make_test_controller(detectors=[d1, d2])

        with caplog.at_level(logging.ERROR):
            await harness.controller._tick()

        assert d1.call_count == 1
        assert d2.call_count == 1
        assert harness.controller._tick_loop.tick_count == 1
        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("detector_evaluate_failed" in m for m in error_messages)

    @pytest.mark.anyio
    async def test_tick_with_all_crashing_detectors_results_in_none_action(self) -> None:
        harness = make_test_controller(detectors=[CrashingDetector(), CrashingDetector()])
        await harness.controller._tick()

        assert not harness.node_manager._bad_nodes
        assert not harness.main_job._stopped
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)


class TestExecuteDecision:
    @pytest.mark.anyio
    async def test_none_decision_is_noop(self) -> None:
        harness = make_test_controller()
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0
        assert not harness.node_manager._bad_nodes
        assert not harness.main_job._stopped
        assert not harness.main_job._submitted
        assert not isinstance(get_training_subsystem_state(harness.controller), RecoveringSt)

    @pytest.mark.anyio
    async def test_mark_bad_and_restart_does_not_raise(self) -> None:
        harness = make_test_controller(
            detectors=[AlwaysMarkBadDetector()],
        )
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 1

    @pytest.mark.anyio
    async def test_enter_recovery_does_not_raise(self) -> None:
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 1

    @pytest.mark.anyio
    async def test_notify_human_sends_notification(self) -> None:
        detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="test notify",
                trigger=TriggerType.MISC,
            )
        )
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.controller._tick_loop.tick_count == 1
        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, content, severity = harness.notifier.calls[0]
        assert title == "Fault Alert"
        assert content == "test notify"
        assert severity == "critical"

    @pytest.mark.anyio
    async def test_notify_human_without_notifier(self) -> None:
        detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="test notify no notifier",
                trigger=TriggerType.MISC,
            )
        )
        harness = make_test_controller(detectors=[detector], notifier=None)
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 1

    @pytest.mark.anyio
    async def test_none_decision_does_not_notify(self) -> None:
        harness = make_test_controller(detectors=[AlwaysNoneDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.anyio
    async def test_mark_bad_sends_eviction_success_notification(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 1
        title, _, severity = harness.notifier.calls[0]
        assert title == "Nodes evicted"
        assert severity == "warning"

    @pytest.mark.anyio
    async def test_enter_recovery_does_not_notify(self) -> None:
        detector = AlwaysEnterRecoveryDetector()
        harness = make_test_controller(detectors=[detector])
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 0

    @pytest.mark.anyio
    async def test_notify_human_notifier_exception_does_not_crash(self) -> None:
        harness = make_test_controller(
            detectors=[
                FixedDecisionDetector(
                    decision=Decision(
                        action=ActionType.NOTIFY_HUMAN,
                        reason="test with broken notifier",
                        trigger=TriggerType.MISC,
                    )
                )
            ],
        )
        assert harness.notifier is not None
        harness.notifier.send = _raise_runtime_error
        await harness.controller._tick()
        assert harness.controller._tick_loop.tick_count == 1

    @pytest.mark.anyio
    async def test_notify_human_sends_on_every_tick(self) -> None:
        detector = FixedDecisionDetector(
            decision=Decision(
                action=ActionType.NOTIFY_HUMAN,
                reason="persistent fault",
                trigger=TriggerType.MISC,
            )
        )
        harness = make_test_controller(detectors=[detector])

        await harness.controller._tick()
        await harness.controller._tick()

        assert harness.notifier is not None
        assert len(harness.notifier.calls) == 2


class TestMarkBadAndRestartReal:
    @pytest.mark.anyio
    async def test_marks_bad_nodes(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.node_manager.is_node_bad("node-1")

    @pytest.mark.anyio
    async def test_stops_and_submits_training(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        await harness.controller._tick()

        assert harness.main_job._stopped
        assert harness.main_job._submitted

    @pytest.mark.anyio
    async def test_new_run_isolates_mini_wandb_from_old_data(self) -> None:
        harness = make_test_controller(detectors=[AlwaysMarkBadDetector()])
        harness.mini_wandb.log_step(
            run_id="dummy-run",
            step=1,
            metrics={"loss": 1.0},
        )
        assert harness.mini_wandb.latest(metric_name="loss") == 1.0

        await harness.controller._tick()

        assert harness.mini_wandb.latest(metric_name="loss") is None
