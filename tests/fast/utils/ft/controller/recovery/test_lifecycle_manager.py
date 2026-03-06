from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.ft.controller.actions import PlatformDeps
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.controller.lifecycle_manager import RecoveryLifecycleManager
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.recovery import RecoveryPhase
from tests.fast.utils.ft.conftest import (
    FakeDiagnosticOrchestrator,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
)


def _make_manager(
    *,
    window_minutes: float = 30.0,
    max_count: int = 3,
    on_recovery_duration: MagicMock | None = None,
) -> RecoveryLifecycleManager:
    cooldown = SlidingWindowThrottle(window_minutes=window_minutes, max_count=max_count)
    return RecoveryLifecycleManager(
        cooldown=cooldown,
        on_recovery_duration=on_recovery_duration,
    )


def _make_deps() -> PlatformDeps:
    return PlatformDeps(
        node_manager=FakeNodeManager(),
        training_job=FakeTrainingJob(),
        metric_store=None,  # type: ignore[arg-type]
        mini_wandb=MiniWandb(),
        notifier=FakeNotifier(),
        diagnostic_orchestrator=FakeDiagnosticOrchestrator(),
        controller_exporter=None,
    )


def _make_decision(trigger: TriggerType = TriggerType.CRASH) -> Decision:
    return Decision(
        action=ActionType.ENTER_RECOVERY,
        trigger=trigger,
        reason="test",
    )


def _make_fake_orchestrator(
    *, is_done: bool = False, phase: RecoveryPhase = RecoveryPhase.CHECK_ALERTS,
) -> MagicMock:
    orch = MagicMock()
    orch.step = AsyncMock()
    orch.is_done.return_value = is_done
    orch.phase = phase
    orch.trigger = TriggerType.CRASH
    orch.bad_node_ids = []
    orch.phase_history = [RecoveryPhase.CHECK_ALERTS]
    orch.add_bad_nodes = MagicMock()
    return orch


class TestRecoveryLifecycleManagerInitialState:
    def test_not_in_progress_initially(self) -> None:
        manager = _make_manager()
        assert not manager.in_progress
        assert manager._orchestrator is None
        assert manager.phase is None
        assert manager._last_phase_history is None


class TestRecoveryLifecycleManagerStart:
    @pytest.mark.anyio
    async def test_start_creates_orchestrator(self) -> None:
        manager = _make_manager()
        deps = _make_deps()

        result = await manager.start(decision=_make_decision(), deps=deps)

        assert result is True
        assert manager.in_progress
        assert manager._orchestrator is not None

    @pytest.mark.anyio
    async def test_start_throttled_returns_false(self) -> None:
        manager = _make_manager(max_count=2)
        deps = _make_deps()

        await manager.start(decision=_make_decision(), deps=deps)
        assert manager.in_progress
        manager._orchestrator = None

        result = await manager.start(decision=_make_decision(), deps=deps)

        assert result is False
        assert not manager.in_progress

    @pytest.mark.anyio
    async def test_different_triggers_not_throttled(self) -> None:
        manager = _make_manager(max_count=2)
        deps = _make_deps()

        result_crash = await manager.start(
            decision=_make_decision(TriggerType.CRASH), deps=deps,
        )
        assert result_crash is True
        manager._orchestrator = None

        result_hang = await manager.start(
            decision=_make_decision(TriggerType.HANG), deps=deps,
        )
        assert result_hang is True


class TestRecoveryLifecycleManagerStep:
    @pytest.mark.anyio
    async def test_step_advances_orchestrator(self) -> None:
        manager = _make_manager()
        fake_orch = _make_fake_orchestrator()
        manager._orchestrator = fake_orch

        await manager.step()

        fake_orch.step.assert_awaited_once()
        assert manager.in_progress

    @pytest.mark.anyio
    async def test_step_auto_cleans_on_done(self) -> None:
        duration_cb = MagicMock()
        manager = _make_manager(on_recovery_duration=duration_cb)
        fake_orch = _make_fake_orchestrator(is_done=True)
        fake_orch.phase_history = [RecoveryPhase.CHECK_ALERTS, RecoveryPhase.DONE]
        manager._orchestrator = fake_orch
        manager._start_time = 100.0

        with patch("miles.utils.ft.controller.recovery_lifecycle.time") as mock_time:
            mock_time.monotonic.return_value = 145.0
            await manager.step()

        assert not manager.in_progress
        assert manager._last_phase_history == [RecoveryPhase.CHECK_ALERTS, RecoveryPhase.DONE]
        duration_cb.assert_called_once_with(45.0)

    @pytest.mark.anyio
    async def test_step_noop_when_no_orchestrator(self) -> None:
        manager = _make_manager()
        await manager.step()
        assert not manager.in_progress


class TestRecoveryLifecycleManagerAddBadNodes:
    def test_add_bad_nodes_delegates_to_orchestrator(self) -> None:
        manager = _make_manager()
        fake_orch = _make_fake_orchestrator()
        manager._orchestrator = fake_orch

        manager.add_bad_nodes(["node-1", "node-2"])
        fake_orch.add_bad_nodes.assert_called_once_with(["node-1", "node-2"])

    def test_add_bad_nodes_noop_without_orchestrator(self) -> None:
        manager = _make_manager()
        manager.add_bad_nodes(["node-1"])


