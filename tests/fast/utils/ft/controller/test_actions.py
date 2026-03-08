"""Tests for controller/actions.py."""

from __future__ import annotations

import pytest
from tests.fast.utils.ft.conftest import FakeDiagnosticOrchestrator, FakeNodeManager, FakeNotifier, FakeTrainingJob

from miles.utils.ft.controller.state_machines.main.utils import PlatformDeps, handle_notify_human
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType


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
# handle_notify_human
# ===================================================================


class TestHandleNotifyHuman:
    @pytest.mark.anyio
    async def test_sends_notification(self) -> None:
        notifier = FakeNotifier()
        decision = Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="something bad happened",
            trigger=TriggerType.MISC,
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
            trigger=TriggerType.MISC,
        )

        await handle_notify_human(decision=decision, notifier=None)
