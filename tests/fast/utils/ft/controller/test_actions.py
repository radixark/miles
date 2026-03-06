"""Tests for controller/actions.py."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.actions import (
    PlatformDeps,
    handle_mark_bad_and_restart,
    handle_notify_human,
)
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb

from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticOrchestrator,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    make_failing_node_manager,
    make_failing_training_job,
)


def _make_decision(
    bad_node_ids: list[str],
    action: ActionType = ActionType.MARK_BAD_AND_RESTART,
    trigger: TriggerType = TriggerType.NONE,
) -> Decision:
    return Decision(
        action=action,
        bad_node_ids=bad_node_ids,
        reason="test fault",
        trigger=trigger,
    )


def _make_deps(
    *,
    node_manager: FakeNodeManager | None = None,
    training_job: FakeTrainingJob | None = None,
    mini_wandb: MiniWandb | None = None,
    notifier: FakeNotifier | None = None,
) -> PlatformDeps:
    return PlatformDeps(
        node_manager=node_manager or FakeNodeManager(),
        training_job=training_job or FakeTrainingJob(),
        metric_store=None,  # type: ignore[arg-type]
        mini_wandb=mini_wandb or MiniWandb(),
        notifier=notifier,
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        controller_exporter=None,
    )


# ===================================================================
# handle_mark_bad_and_restart
# ===================================================================


class TestMarkBadHappyPath:
    """All mark_node_bad calls succeed and restart succeeds."""

    @pytest.mark.anyio
    async def test_marks_all_nodes_and_restarts(self) -> None:
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()
        deps = _make_deps(
            node_manager=node_manager, training_job=training_job, notifier=notifier,
        )

        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0", "node-1"]),
            deps=deps,
        )

        assert node_manager.is_node_bad("node-0")
        assert node_manager.is_node_bad("node-1")
        assert training_job._submitted
        assert len(notifier.calls) == 0

    @pytest.mark.anyio
    async def test_empty_bad_nodes_still_restarts(self) -> None:
        training_job = FakeTrainingJob()
        notifier = FakeNotifier()
        deps = _make_deps(training_job=training_job, notifier=notifier)

        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=[]),
            deps=deps,
        )

        assert training_job._submitted
        assert len(notifier.calls) == 0


class TestMarkBadPartialFailure:
    """mark_node_bad fails for some nodes but restart still proceeds."""

    @pytest.mark.anyio
    async def test_partial_mark_bad_failure_notifies_and_continues_restart(self) -> None:
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        call_count = 0

        async def mark_bad_fail_once(node_id: str, reason: str = "") -> None:
            nonlocal call_count
            call_count += 1
            if node_id == "node-bad":
                raise RuntimeError("K8s API unreachable")
            node_manager._bad_nodes.add(node_id)

        node_manager.mark_node_bad = mark_bad_fail_once  # type: ignore[assignment]

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb, notifier=notifier,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-ok", "node-bad"]),
            deps=deps,
        )

        assert len(notifier.calls) == 0

        assert training_job._stopped
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_all_mark_bad_fail_still_restarts(self) -> None:
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def always_fail(node_id: str, reason: str = "") -> None:
            raise RuntimeError("total failure")

        node_manager.mark_node_bad = always_fail  # type: ignore[assignment]

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb, notifier=notifier,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0", "node-1"]),
            deps=deps,
        )

        assert all(c[0] != "Mark-Bad Failure" for c in notifier.calls)
        assert training_job._submitted


class TestRestartFailure:
    """stop_and_submit fails after mark_bad succeeds."""

    @pytest.mark.anyio
    async def test_restart_failure_notifies(self) -> None:
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()
        training_job = make_failing_training_job(fail_submit=True)
        mini_wandb = MiniWandb()

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb, notifier=notifier,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0"]),
            deps=deps,
        )

        assert node_manager.is_node_bad("node-0")
        restart_notifications = [c for c in notifier.calls if c[0] == "Restart Failure"]
        assert len(restart_notifications) == 1

    @pytest.mark.anyio
    async def test_double_failure_only_notifies_restart(self) -> None:
        """Both mark_node_bad and submit_training fail — only restart failure notified."""
        node_manager = make_failing_node_manager()
        notifier = FakeNotifier()
        training_job = make_failing_training_job(fail_submit=True)
        mini_wandb = MiniWandb()

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb, notifier=notifier,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0"]),
            deps=deps,
        )

        titles = [c[0] for c in notifier.calls]
        assert "Mark-Bad Failure" not in titles
        assert "Restart Failure" in titles


class TestMarkBadSkipsAlreadyBadNodes:
    @pytest.mark.anyio
    async def test_already_bad_node_not_remarked(self) -> None:
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-0", reason="previous")
        training_job = FakeTrainingJob()
        deps = _make_deps(node_manager=node_manager, training_job=training_job)

        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0", "node-1"]),
            deps=deps,
        )

        assert node_manager.is_node_bad("node-0")
        assert node_manager.is_node_bad("node-1")
        assert training_job._submitted

    @pytest.mark.anyio
    async def test_all_already_bad_still_restarts(self) -> None:
        node_manager = FakeNodeManager()
        await node_manager.mark_node_bad("node-0", reason="previous")
        training_job = FakeTrainingJob()
        deps = _make_deps(node_manager=node_manager, training_job=training_job)

        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0"]),
            deps=deps,
        )

        assert training_job._submitted


class TestMarkBadWithoutNotifier:
    """handle_mark_bad_and_restart with notifier=None should not crash."""

    @pytest.mark.anyio
    async def test_partial_failure_without_notifier(self) -> None:
        node_manager = FakeNodeManager()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def always_fail(node_id: str, reason: str = "") -> None:
            raise RuntimeError("fail")

        node_manager.mark_node_bad = always_fail  # type: ignore[assignment]

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0"]),
            deps=deps,
        )

        assert training_job._submitted


# ===================================================================
# handle_notify_human
# ===================================================================


class TestHandleNotifyHuman:
    @pytest.mark.anyio
    async def test_sends_notification(self) -> None:
        notifier = FakeNotifier()
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="something bad happened",
        )

        await handle_notify_human(decision=decision, notifier=notifier)

        assert len(notifier.calls) == 1
        title, content, _ = notifier.calls[0]
        assert title == "Fault Alert"
        assert "something bad happened" in content

    @pytest.mark.anyio
    async def test_none_notifier_does_not_crash(self) -> None:
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="should not crash",
        )

        await handle_notify_human(decision=decision, notifier=None)
