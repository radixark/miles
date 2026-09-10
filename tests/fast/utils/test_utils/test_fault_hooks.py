from pathlib import Path

import pytest
from tests.fast.utils.test_utils.conftest import ControlledFaultTimer

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.test_utils import fault_hooks
from miles.utils.test_utils.fault_hooks import FaultHookRegistry, FaultHookRequest


class TestFaultHookRegistry:
    def test_weight_update_scope_records_version_and_unwinds_after_failure(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A failed update preserves its version without leaking its registry into later work."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="scoped", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda **kwargs: None)

        fault_hooks.reach_fault_hook(request.hook)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id).status == "armed"
        with pytest.raises(RuntimeError, match="unexpectedly returned"):
            with registry.weight_update_scope(weight_version=23):
                fault_hooks.reach_fault_hook(request.hook)

        record = registry.read(request_id=request.request_id, instance_id=registry.instance_id)
        assert record.weight_version == 23
        assert record.rollout_id is None
        assert record.attempt is None
        events = [event for event in read_events(tmp_path) if isinstance(event, FaultHookEvent)]
        assert events[-1].weight_version == 23

        next_request = request.model_copy(update={"request_id": "next"})
        registry.arm(next_request)
        fault_hooks.reach_fault_hook(request.hook)
        assert registry.read(request_id=next_request.request_id, instance_id=registry.instance_id).status == "armed"

    @pytest.mark.parametrize("outcome", ["fire", "cancel", "expire"])
    def test_delayed_dispatch_respects_cancellation_and_expiry(
        self,
        fault_hook_registry: FaultHookRegistry,
        fault_timers: list[ControlledFaultTimer],
        monkeypatch: pytest.MonkeyPatch,
        outcome: str,
        tmp_path: Path,
    ) -> None:
        """Arrival returns before dispatch and a winning cancellation or expiry prevents injection."""
        registry = fault_hook_registry
        now = 10.0
        monkeypatch.setattr(fault_hooks.time, "monotonic", lambda: now)
        request = FaultHookRequest(
            request_id="delayed",
            instance_id=registry.instance_id,
            hook="trainer_before_all_gather",
            mode="exit",
            delay_ms=500,
            lifetime_seconds=2,
        )
        injections: list[str] = []

        def terminate(*, mode: str, request_id: str, receipt_url: str | None) -> None:
            injections.append(request_id)
            assert registry.cancel(request).status == "fired"
            raise SystemExit(1)

        monkeypatch.setattr(fault_hooks, "inject_fault", terminate)
        registry.arm(request)
        now = 11.0
        registry.reach(hook=request.hook, rollout_id=7, attempt=2)
        registry.reach(hook=request.hook, rollout_id=8, attempt=0)
        assert injections == []
        assert len(fault_timers) == 1
        assert fault_timers[0].interval == 0.5
        assert fault_timers[0].started
        scheduled = registry.read(request_id=request.request_id, instance_id=registry.instance_id)
        assert scheduled.status == "scheduled"
        assert scheduled.reached_at == 11.0
        assert scheduled.due_at == 11.5
        assert registry.arm(request) == scheduled

        now = 11.5
        if outcome == "fire":
            with pytest.raises(SystemExit):
                fault_timers[0].dispatch()
            assert injections == [request.request_id]
        else:
            if outcome == "cancel":
                assert registry.cancel(request).status == "cancelled"
            else:
                now = 12.0
            fault_timers[0].dispatch()
            assert injections == []

        expected = {"fire": "fired", "cancel": "cancelled", "expire": "expired"}[outcome]
        final = registry.read(request_id=request.request_id, instance_id=registry.instance_id)
        assert final.status == expected
        assert final.rollout_id == 7
        assert final.attempt == 2
        assert fault_timers[0].cancelled
        events = [event for event in read_events(tmp_path) if isinstance(event, FaultHookEvent)]
        assert events[-1].status == expected
        assert events[-1].due_at == 11.5

    def test_delayed_thread_deadlock_is_rejected(self, fault_hook_registry: FaultHookRegistry) -> None:
        """A timer-thread deadlock must not be presented as a training-thread deadlock."""
        with pytest.raises(ValueError, match="immediate"):
            FaultHookRequest(
                request_id="wrong-thread",
                instance_id=fault_hook_registry.instance_id,
                hook="trainer_before_all_gather",
                mode="thread_deadlock",
                delay_ms=1,
            )

    @pytest.mark.parametrize("arm_first", [False, True])
    @pytest.mark.parametrize("mode", ["exit", "sigstop", "deadlock", "thread_deadlock"])
    def test_cancelled_request_cannot_rearm_or_fire(
        self, fault_hook_registry: FaultHookRegistry, arm_first: bool, mode: str
    ) -> None:
        """Retrying a cancelled request cannot revive a fault."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="cancelled", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode=mode
        )
        if arm_first:
            registry.arm(request)
        cancelled = registry.cancel(request)

        assert cancelled.status == "cancelled"
        assert registry.arm(request) == cancelled
        registry.reach(hook=request.hook, rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id) == cancelled

    def test_replacement_process_rejects_old_request(self, fault_hook_registry: FaultHookRegistry) -> None:
        """A recreated worker cannot accept an earlier incarnation's command."""
        request = FaultHookRequest(
            request_id="stale",
            instance_id=FaultHookRegistry().instance_id,
            hook="trainer_before_all_gather",
            mode="exit",
        )
        with pytest.raises(ValueError, match="incarnation"):
            fault_hook_registry.arm(request)

    def test_expired_arm_cannot_fire(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unobserved expiry still prevents a later hook hit."""
        registry = fault_hook_registry
        monkeypatch.setattr(fault_hooks.time, "monotonic", lambda: 10.0)
        request = FaultHookRequest(
            request_id="expired", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)
        monkeypatch.setattr(fault_hooks.time, "monotonic", lambda: 70.0)

        registry.reach(hook=request.hook, rollout_id=2, attempt=1)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id).status == "expired"

    def test_firing_wins_cancellation_and_is_recorded_before_fault(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cancellation after dispatch cannot claim to have prevented the fault."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="fired", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)

        def terminate(*, mode: str, request_id: str, receipt_url: str | None) -> None:
            assert mode == "exit"
            assert request_id == request.request_id
            record = registry.cancel(request)
            assert record.status == "fired"
            events = [event for event in read_events(tmp_path) if isinstance(event, FaultHookEvent)]
            assert events[-1].status == "fired"
            assert events[-1].rollout_id == 3
            assert events[-1].attempt == 2
            raise SystemExit(1)

        monkeypatch.setattr(fault_hooks, "inject_fault", terminate)
        with pytest.raises(SystemExit):
            registry.reach(hook=request.hook, rollout_id=3, attempt=2)
        registry.reach(hook=request.hook, rollout_id=4, attempt=0)

    def test_failed_fault_dispatch_is_not_reported_as_applied(
        self, fault_hook_registry: FaultHookRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A returning fault action leaves an explicit failure record."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="failed", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        registry.arm(request)
        monkeypatch.setattr(fault_hooks, "inject_fault", lambda **kwargs: None)
        with pytest.raises(RuntimeError, match="unexpectedly returned"):
            registry.reach(hook=request.hook, rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id).status == "failed"

    def test_second_arm_and_conflicting_retries_are_rejected(self, fault_hook_registry: FaultHookRegistry) -> None:
        """A live arm cannot be overwritten by another request or changed parameters."""
        registry = fault_hook_registry
        request = FaultHookRequest(
            request_id="first", instance_id=registry.instance_id, hook="trainer_before_all_gather", mode="exit"
        )
        original = registry.arm(request)
        assert registry.arm(request) == original
        with pytest.raises(ValueError, match="already armed"):
            registry.arm(request.model_copy(update={"request_id": "second"}))
        with pytest.raises(ValueError, match="reused"):
            registry.arm(request.model_copy(update={"mode": "sigkill"}))
        registry.reach(hook="trainer_before_weight_send", rollout_id=1, attempt=0)
        assert registry.read(request_id=request.request_id, instance_id=registry.instance_id) == original
