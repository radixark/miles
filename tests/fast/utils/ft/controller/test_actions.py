"""Tests for controller/actions.py — handle_mark_bad_and_restart failure paths."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.actions import PlatformDeps, handle_mark_bad_and_restart
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticScheduler,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
)


def _make_decision(bad_node_ids: list[str]) -> Decision:
    return Decision(
        action=ActionType.MARK_BAD_AND_RESTART,
        bad_node_ids=bad_node_ids,
        reason="test fault",
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
        diagnostic_scheduler=FakeDiagnosticScheduler(),
        controller_exporter=None,
    )


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

        mark_bad_notifications = [c for c in notifier.calls if c[0] == "Mark-Bad Failure"]
        assert len(mark_bad_notifications) == 1
        assert "node-bad" in mark_bad_notifications[0][1]

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

        mark_bad_notifications = [c for c in notifier.calls if c[0] == "Mark-Bad Failure"]
        assert len(mark_bad_notifications) == 1
        assert training_job._submitted


class TestRestartFailure:
    """stop_clear_submit fails after mark_bad succeeds."""

    @pytest.mark.anyio
    async def test_restart_failure_notifies(self) -> None:
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def failing_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed permanently")

        training_job.submit_training = failing_submit  # type: ignore[assignment]

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
    async def test_double_failure_sends_both_notifications(self) -> None:
        """Both mark_node_bad and submit_training fail — two notifications expected."""
        node_manager = FakeNodeManager()
        notifier = FakeNotifier()
        training_job = FakeTrainingJob()
        mini_wandb = MiniWandb()

        async def always_fail_mark(node_id: str, reason: str = "") -> None:
            raise RuntimeError("mark failed")

        async def always_fail_submit(excluded_node_ids: list[str] | None = None) -> str:
            raise RuntimeError("submit failed")

        node_manager.mark_node_bad = always_fail_mark  # type: ignore[assignment]
        training_job.submit_training = always_fail_submit  # type: ignore[assignment]

        deps = _make_deps(
            node_manager=node_manager, training_job=training_job,
            mini_wandb=mini_wandb, notifier=notifier,
        )
        await handle_mark_bad_and_restart(
            decision=_make_decision(bad_node_ids=["node-0"]),
            deps=deps,
        )

        titles = [c[0] for c in notifier.calls]
        assert "Mark-Bad Failure" in titles
        assert "Restart Failure" in titles


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
